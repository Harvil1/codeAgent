"""工作流引擎——把一串步骤写成一个受限 Python 脚本，一次跑完（R28 落地，设计蓝图在 .superpowers/sdd/r28-workflow-blueprint.md）。

在项目里的位置：由 tools/workflow_tool.py 包成工具暴露给 LLM，跑腿的
子代理走 tools/delegate_tool.py 的 _run_child；执行日志（journal）由
agent/workflow_journal.py 负责。

workflow（工作流）和 goal（目标驱动）的分工：goal 是主循环多轮对话慢慢推，
workflow 是"一次工具调用内按写好的脚本一口气跑完"，确定性更强。

脚本长什么样：一段受约束的 Python，必须定义 ``async def main()``。引擎在
执行时塞给它 7 个现成的名字直接用：agent（调子代理干一件事）/ parallel
（几件事同时跑）/ pipeline（流水线：上一环的输出喂下一环）/ phase（给日志
打阶段标记）/ log（打日志）/ args（调用时传进来的参数）/ budget（预算池）。

安全声明：跑之前先做 AST 白名单校验——禁止 import、exec、eval、open、
__import__、双下划线名字（dunder），内置函数也只放开一小撮。注意这是
"防手滑"不是"防坏人"（对齐参考实现 script.ts 的自我声明）：想逃逸的写法
在校验层就直接拒绝，但别指望它当安全边界用。

预算口径：只统计子代理最终产出的文本量（按 len(text)//4 粗算 token）；
workflow 里子代理的花销不往 goal 的账上累加，两边天然隔离（蓝图 §3）。
"""
import ast
import asyncio
import builtins
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class WorkflowBudgetExceeded(Exception):
    """工作流的 token 预算花光了。这是个硬上限，防止脚本失控一直烧钱。"""


@dataclass
class WorkflowBudget:
    """整个工作流共享的一个 token 预算池（估算口径，蓝图 §1）。

    历史踩坑（R30c-C3 修复）：以前是纯事后记账——子代理跑完才扣钱，
    一次远超剩余额度的调用会先把 token 真实花掉才报超限；而且几路
    agent 并发时，大家能同时通过"余额还大于 0"的检查，各自都超额。
    现在改成"事前预留 + 事后结算"：调用前先 reserve（预留）一笔额度，
    能同时开几路受剩余额度约束；跑完 settle（结算）实际花费、多退少补。
    因为输出长度没法提前精确知道，超支上限约等于单次预留额度——这已经
    是估算口径下能做到的最紧约束。
    """
    total: int
    spent: int = 0        # 已结算的花费
    reserved: int = 0     # 在途预留：已经 reserve 出去、还没 settle 回来的部分

    @property
    def remaining(self) -> int:
        """还剩多少可用（总额减已花减在途；负数按 0 算）。"""
        return max(0, self.total - self.spent - self.reserved)

    def reserve(self, tokens: int) -> int:
        """跑之前先预留一笔额度，返回实际批下来的数（可能比要的少）。

        背景：有了预留，并发开的路数自然被剩余额度卡住，也不会"先花后报"。

        参数：
            tokens：想预留的额度。
        返回：实际授予的额度（剩余不够时只批剩余的，最少 1）。
        异常：剩余额度为 0 时抛 WorkflowBudgetExceeded——开跑之前就拒，
            不再出现"钱花完了才发现超支"。
        """
        want = max(1, tokens)
        avail = self.total - self.spent - self.reserved
        if avail <= 0:
            raise WorkflowBudgetExceeded(
                f"预算耗尽：已花 {self.spent} + 在途 {self.reserved} / {self.total}")
        granted = min(want, avail)
        self.reserved += granted
        return granted

    def settle(self, granted: int, actual: int) -> None:
        """跑完之后结算：把预留还回来，按实际花费记账。

        参数：
            granted：当初 reserve 批下来的额度（原数归还）。
            actual：这次实际花了多少（按产出文本估算）。
        异常：结算后发现总花费超过总额度时抛 WorkflowBudgetExceeded——
            钱已经花掉了，抛错是为了让脚本立刻停下来（跟原来 spend() 的
            语义一致）。调用方在异常/收尾路径用 actual=0 结算不会误触发。
        """
        self.reserved = max(0, self.reserved - granted)
        self.spent += max(0, actual)
        if self.spent > self.total:
            raise WorkflowBudgetExceeded(
                f"预算耗尽：{self.spent}/{self.total}")

    def spend(self, tokens: int) -> None:
        """兼容旧写法的入口（脚本或旧测试会直接调）：效果等于 settle(0, tokens)。

        参数：
            tokens：要记的花费。
        """
        self.settle(0, tokens)


# 脚本可以用的内置函数白名单（危险的那批——exec/eval/open/__import__ 等——不在里面）
SAFE_BUILTINS: Dict[str, Any] = {}
for _k in (
    "len", "range", "str", "repr", "int", "float", "bool", "list", "dict",
    "set", "tuple", "sum", "min", "max", "sorted", "reversed", "enumerate",
    "zip", "abs", "round", "any", "all", "isinstance", "type",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "Exception", "StopIteration",
):
    _v = getattr(builtins, _k, None)
    if _v is not None:
        SAFE_BUILTINS[_k] = _v
SAFE_BUILTINS["print"] = lambda *a, **kw: logger.info("[workflow] %s", " ".join(map(str, a)))

_FORBIDDEN_CALLS = {"exec", "eval", "open", "__import__", "compile", "input", "breakpoint"}


def _is_dunder(name: str) -> bool:
    """判断名字是不是双下划线包裹的形式（如 __import__）——这类名字常是逃生门。"""
    return name.startswith("__") and name.endswith("__")


def _json_parseable(text: str) -> bool:
    """结构化输出的最低门槛：文本能当 JSON 解析就算过关。

    参数：
        text：子代理的输出文本。
    返回：True=是合法 JSON。
    """
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def validate_script(source: str) -> Optional[str]:
    """跑之前的"安检"：用 AST 白名单扫一遍脚本，不合规矩的提前拦下。

    检查内容：不许 import、不许碰双下划线名字/属性、不许调 exec/eval/
    open/__import__/compile/input/breakpoint、必须定义 async def main()。

    参数：
        source：脚本文本。
    返回：合法返回 None；有问题返回一句中文原因（可直接展示给用户）。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"脚本语法错误: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "禁止 import"
        if isinstance(node, ast.Name) and _is_dunder(node.id):
            return f"禁止访问 dunder 名字: {node.id}"
        if isinstance(node, ast.Attribute) and _is_dunder(node.attr):
            return f"禁止访问 dunder 属性: {node.attr}"
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FORBIDDEN_CALLS):
            return f"禁止调用: {node.func.id}"
    has_main = any(
        isinstance(n, ast.AsyncFunctionDef) and n.name == "main"
        for n in ast.walk(tree)
    )
    if not has_main:
        return "脚本必须定义 async def main()"
    return None


async def run_workflow(
    source: str,
    *,
    args: Optional[dict] = None,
    agent_runner: Callable[[str], Any],
    validator: Optional[Callable[[str], bool]] = None,
    budget_total: int = 500_000,
    reserve_per_call: int = 50_000,   # 每次调子代理前预留的额度（R30c-C3 的事前预留机制）
    max_concurrency: int = 5,
    journal=None,          # 可选的执行日志（WorkflowJournal）；None = 不记日志
    cancel_event=None,     # run 级取消开关（kill 动作会触发它）
) -> dict:
    """执行一段受约束的工作流脚本，从校验、预算到执行全包，返回统一格式的结果。

    参数：
        source：脚本文本（必须定义 async def main()）。
        args：传给脚本的参数，脚本里用 args 名字取。
        agent_runner：真正去调子代理的函数（prompt 进、产出文本出）；
            由 make_agent_runner 造。
        validator：输出校验函数（无 schema 时的兜底检查）；不传则不校验。
        budget_total：总 token 预算（默认 50 万）。
        reserve_per_call：每次调用前预留多少额度。
        max_concurrency：同时最多几路子代理。
        journal：WorkflowJournal 实例，用来记执行日志、支持断点恢复。
        cancel_event：外部取消事件，置位后脚本尽快停。
    返回：统一结果 dict——成功 {"ok": True, "return": main 的返回值, "stats": 统计,
        "budget_spent"/"budget_total": 花销}；失败 {"ok": False, "error": 原因,
        "error_type": invalid_script/budget_exceeded/cancelled/script_error 之一}。
    """
    err = validate_script(source)
    if err:
        return {"ok": False, "error": err, "error_type": "invalid_script"}

    budget = WorkflowBudget(total=budget_total)
    sem = asyncio.Semaphore(max(1, max_concurrency))
    stats = {"calls": 0, "cached": 0, "dead": 0}

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    async def call_agent(prompt: str, schema: Optional[dict] = None):
        """脚本里的 agent() 原语（脚本调子代理干活的唯一入口）。

        流水线：先查 journal 缓存（同样的调用跑过就直接用旧结果）→ 抢并发
        槽 → 交给 runner 真跑 → 校验输出 → 预算结算。

        参数：
            prompt：给子代理的指令。
            schema：可选的 JSON Schema，给了就要求子代理只输出符合它的 JSON。
        返回：子代理的产出文本；跑挂了返回 None（记为 dead）。
        """
        key = None
        if journal is not None:
            from agent.workflow_journal import call_key  # W2 模块，journal 路径才触达
            key = call_key(prompt, schema)
            cached = journal.lookup(key)
            if cached is not None:
                out = cached.get("output")  # lookup 拿到的就是当初记下的 result dict
                # 缓存回放也要过校验：带 schema 的调用结果必须仍是合法 JSON，
                # 防止一条坏缓存借 resume（断点恢复）混过校验这一关
                if schema is None or _json_parseable(out):
                    stats["cached"] += 1
                    return out
                # 校验不过就当没命中，重新跑一遍

        if _cancelled():
            raise asyncio.CancelledError()

        # 先预留再跑（R30c-C3）：预算耗尽在开跑前就拒；并发路数也被剩余额度卡住
        granted = budget.reserve(reserve_per_call)

        # 结构化输出：把 schema 的 JSON 拼在 prompt 末尾，让子代理照着格式答（蓝图 §5）
        if schema is not None:
            full = prompt + (
                "\n\n最终回答必须只输出符合此 JSON Schema 的 JSON（不要其他文字）：\n"
                + json.dumps(schema, ensure_ascii=False))
            check = _json_parseable
        else:
            full = prompt
            check = validator

        try:
            async with sem:
                out = await agent_runner(full)
                if out is not None and check is not None and not check(out):
                    # 输出不合格就自动重试一次（这次重试不再单独扣预算）
                    out2 = await agent_runner(full)
                    out = out2 if (out2 is not None and check(out2)) else None
        except BaseException:
            budget.settle(granted, 0)  # 异常/取消：预留全退，按 0 花费结算（不会误报超限）
            raise

        if _cancelled():  # runner 内部可能已经请求了取消（比如 kill 动作的回调）
            budget.settle(granted, 0)
            raise asyncio.CancelledError()

        if out is None:
            budget.settle(granted, 0)
            stats["dead"] += 1
            if journal is not None:
                journal.append(key, {"kind": "dead", "output": None})
            return None
        stats["calls"] += 1
        budget.settle(granted, max(1, len(out) // 4))
        if journal is not None:
            journal.append(key, {"kind": "ok", "output": out})
        return out

    async def parallel(factories: list) -> list:
        """几件事同时跑（脚本里的 parallel() 原语）。

        契约：某一项挂了不连累别人，结果在对应位置放 None（null-on-error）；
        但预算耗尽和取消这两种要整体停，会往上抛。

        参数：
            factories：零参协程工厂的列表，每项是"待执行的一件事"。
        返回：与输入顺序对应的结果列表（失败的位置是 None）。
        """
        async def _one(f):
            try:
                return await f()
            except WorkflowBudgetExceeded:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[workflow] parallel 项失败（置 None）: %s", e)
                return None
        return list(await asyncio.gather(*[_one(f) for f in factories]))

    async def pipeline(items: list, stages: list) -> list:
        """流水线（脚本里的 pipeline() 原语）：每个条目依次过一遍每个加工环节。

        打个比方：洗菜 → 切菜 → 炒菜三道工序，每颗菜都要按顺序过完；
        不同的菜之间互不等待、同时进行。第 N 环拿到的是第 N-1 环的输出。

        参数：
            items：要加工的条目列表。
            stages：加工环节列表，每环是个函数（同步或异步均可），
                收上一环的值、返回本环的值。
        返回：每个条目走完全部环节后的最终结果列表（某条目中途挂了对应位置是 None）。
        """
        async def _item(v):
            for stage in stages:
                r = stage(v)
                if asyncio.iscoroutine(r) or hasattr(r, "__await__"):
                    v = await r
                else:
                    v = r
            return v
        return await parallel([lambda v=v: _item(v) for v in items])

    @contextmanager
    def phase(name: str):
        """阶段标记（脚本里的 phase() 原语）：with 块开始/结束各打一条日志，方便看进度。

        参数：
            name：阶段名。
        """
        logger.info("[workflow] ▶ 阶段: %s", name)
        yield
        logger.info("[workflow] ✓ 阶段: %s", name)

    def log(msg, *a):
        """打日志（脚本里的 log() 原语），支持 printf 风格的 % 格式化参数。"""
        logger.info("[workflow] %s", msg if not a else msg % a)

    ns: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "agent": call_agent, "parallel": parallel, "pipeline": pipeline,
        "phase": phase, "log": log,
        "args": args or {}, "budget": budget,
    }
    try:
        exec(compile(source, "<workflow>", "exec"), ns)  # noqa: S102 —— exec 本身危险，但这之前已过 AST 安检且 builtins 被换成白名单
        main = ns.get("main")
        if not asyncio.iscoroutinefunction(main):
            return {"ok": False, "error": "main 不是 async 函数",
                    "error_type": "invalid_script"}
        ret = await main()
        return {
            "ok": True, "return": ret, "stats": dict(stats),
            "budget_spent": budget.spent, "budget_total": budget.total,
        }
    except WorkflowBudgetExceeded as e:
        return {"ok": False, "error": str(e), "error_type": "budget_exceeded",
                "stats": dict(stats), "budget_spent": budget.spent}
    except asyncio.CancelledError:
        return {"ok": False, "error": "已取消", "error_type": "cancelled",
                "stats": dict(stats), "budget_spent": budget.spent}
    except Exception as e:
        logger.warning("[workflow] 脚本异常: %s", e)
        return {"ok": False, "error": str(e), "error_type": "script_error",
                "stats": dict(stats), "budget_spent": budget.spent}


def _strip_json_fence(text: str) -> str:
    """把 LLM 输出外面裹的 ```json ... ``` 围栏剥掉（模型偶尔不听话非要多包一层）。

    参数：
        text：原始输出。
    返回：剥掉围栏、去首尾空白后的文本。
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    return t.strip()


def make_validator():
    """造一个结构化输出的校验函数：(文本) -> 是否合格。

    设计：schema 的具体内容已经拼进 prompt 让 LLM 自律遵守，且 call_key
    （journal 的调用键）也把它算进去了，所以这里只把"必须是合法 JSON"这
    个形态关把住。jsonschema 库没装时降级成只验可解析（fail-open——坏不到
    哪去，但也不硬拦，记条 warning）。

    参数：无。
    返回：校验函数，吃文本、返回 bool。
    """
    try:
        import jsonschema  # noqa: F401
        _have = True
    except ImportError:
        _have = False
        logger.warning("jsonschema 未安装，结构化校验降级为仅 JSON 解析")

    import json

    def _validate(text: str) -> bool:
        try:
            json.loads(_strip_json_fence(text))
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    if not _have:
        return _validate

    def _full(text: str) -> bool:
        try:
            parsed = json.loads(_strip_json_fence(text))
        except (json.JSONDecodeError, TypeError):
            return False
        try:
            import jsonschema
            jsonschema.validate(parsed, {"type": "object"})  # 基础形态
            return True
        except Exception:
            return False

    return _full


def make_agent_runner(delegate_kwargs: dict):
    """造 agent() 背后真正干活的 runner：把调用转交给现成的子代理机制 _run_child。

    好处是零侵入——子代理的完整生命周期（中断、清理等）全部白拿现成的。
    两个关键设定：
    - role="leaf"（叶子角色，只给最小工具集）+ summary_only=False（结构化
      输出要原文，不能被摘要压缩）
    - 额外禁掉 subagent/workflow 两个工具，防止工作流里再开工作流/子代理
      无限套娃（蓝图 §6 的递归禁令）

    参数：
        delegate_kwargs：传给 _run_child 的基础参数（含 config 等），会被复制后修改。
    返回：异步 runner 函数——吃 prompt，返回子代理的最终产出文本；子代理
        本身挂了返回 None（记为 dead）。
    """
    import asyncio

    async def runner(prompt: str):
        from tools.delegate_tool import _run_child
        cfg = dict(delegate_kwargs.get("config") or {})
        disabled = list(cfg.get("disabled_tools") or [])
        for t in ("subagent", "workflow"):
            if t not in disabled:
                disabled.append(t)
        cfg["disabled_tools"] = disabled
        kwargs = dict(delegate_kwargs)
        kwargs["config"] = cfg
        kwargs["summary_only"] = False
        try:
            return await asyncio.to_thread(
                _run_child, prompt, "", "leaf", **kwargs)
        except Exception as e:
            logger.warning("workflow agent() 子代理失败（dead）: %s", e)
            return None

    return runner
