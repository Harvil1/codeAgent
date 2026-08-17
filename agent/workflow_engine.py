"""确定性工作流编排引擎（R28，蓝图 .superpowers/sdd/r28-workflow-blueprint.md）。

脚本形态：受约束 Python（必须定义 ``async def main()``），引擎注入 7 个名字：
agent / parallel / pipeline / phase / log / args / budget。

安全声明：AST 白名单校验（禁 import/exec/eval/open/__import__/dunder）+ 受限
builtins——这是**防误用不是安全边界**（对齐参考实现 script.ts 的自我声明），
逃逸尝试在校验层直接拒。

预算口径：只计子代理最终产出（len(text)//4 估算）；workflow 子代理 usage
不回累 goal（天然隔离，见蓝图 §3）。
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
    """workflow 预算耗尽（硬 cap，防失控烧钱）。"""


@dataclass
class WorkflowBudget:
    """共享 token 预算池（估算口径，蓝图 §1）。"""
    total: int
    spent: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.spent)

    def spend(self, tokens: int) -> None:
        self.spent += tokens
        if self.spent > self.total:
            raise WorkflowBudgetExceeded(
                f"预算耗尽：{self.spent}/{self.total}")


# 受限 builtins（脚本可用；exec/eval/open/__import__ 等不在此列）
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
    return name.startswith("__") and name.endswith("__")


def _json_parseable(text: str) -> bool:
    """结构化输出的最低校验：JSON 可解析即算通过。"""
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def validate_script(source: str) -> Optional[str]:
    """AST 白名单校验。合法返回 None；违规返回中文原因。"""
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
    max_concurrency: int = 5,
    journal=None,          # W2 接入：WorkflowJournal（None = 不持久化）
    cancel_event=None,     # run 级取消（kill action 触发）
) -> dict:
    """执行受约束脚本，返回统一结果 dict（见 tests 的形态）。"""
    err = validate_script(source)
    if err:
        return {"ok": False, "error": err, "error_type": "invalid_script"}

    budget = WorkflowBudget(total=budget_total)
    sem = asyncio.Semaphore(max(1, max_concurrency))
    stats = {"calls": 0, "cached": 0, "dead": 0}

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    async def call_agent(prompt: str, schema: Optional[dict] = None):
        """agent() 原语：journal 缓存 → 并发槽 → runner → 校验 → 预算。"""
        key = None
        if journal is not None:
            from agent.workflow_journal import call_key  # W2 模块，journal 路径才触达
            key = call_key(prompt, schema)
            cached = journal.lookup(key)
            if cached is not None:
                out = cached.get("result", {}).get("output")
                # 缓存回放重新过校验（蓝图 §4，对齐 hooks.ts:90）
                if validator is None or schema is None or validator(out):
                    stats["cached"] += 1
                    return out
                # 校验不过 → 视为 miss 重跑

        if _cancelled():
            raise asyncio.CancelledError()

        if budget.remaining <= 0:
            raise WorkflowBudgetExceeded(f"预算耗尽：{budget.spent}/{budget.total}")

        # 结构化输出：schema JSON 拼进 prompt 尾部（蓝图 §5）
        if schema is not None:
            full = prompt + (
                "\n\n最终回答必须只输出符合此 JSON Schema 的 JSON（不要其他文字）：\n"
                + json.dumps(schema, ensure_ascii=False))
            check = _json_parseable
        else:
            full = prompt
            check = validator

        async with sem:
            out = await agent_runner(full)
            if out is not None and check is not None and not check(out):
                # 一次自动重试（重试不双计预算）
                out2 = await agent_runner(full)
                out = out2 if (out2 is not None and check(out2)) else None

        if _cancelled():  # runner 内部可能已请求取消（如 kill action 回调）
            raise asyncio.CancelledError()

        if out is None:
            stats["dead"] += 1
            if journal is not None:
                journal.append(key, {"kind": "dead", "output": None})
            return None
        stats["calls"] += 1
        budget.spend(max(1, len(out) // 4))
        if journal is not None:
            journal.append(key, {"kind": "ok", "output": out})
        return out

    async def parallel(factories: list) -> list:
        """并发执行（每项 null-on-error 契约）。"""
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
        """逐 item 链式串 stage（stage N 收 stage N-1 输出），item 间并发。"""
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
        logger.info("[workflow] ▶ 阶段: %s", name)
        yield
        logger.info("[workflow] ✓ 阶段: %s", name)

    def log(msg, *a):
        logger.info("[workflow] %s", msg if not a else msg % a)

    ns: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "agent": call_agent, "parallel": parallel, "pipeline": pipeline,
        "phase": phase, "log": log,
        "args": args or {}, "budget": budget,
    }
    try:
        exec(compile(source, "<workflow>", "exec"), ns)  # noqa: S102 —— 校验后的受限 exec
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
