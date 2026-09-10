"""中央工具注册表——整个项目所有工具（LLM 能调用的能力）的"户口本"。

打个比方：这里就像小区的物业登记处。每个工具入住时来登记（register），
之后两件事都靠这本册子：(1) 告诉 LLM 这里有哪些工具可用、怎么用（schema 说明书）；
(2) LLM 真的要用的时分发到对应干活的函数（handler）。

依赖链（无循环，下层不知道上层存在）：
    registry.py  (本文件，无依赖)
        ↑
    tools/*.py   (每个工具文件 import registry，在模块顶层调 register() 自登记)
        ↑
    model_tools.py  (import registry + 触发工具发现)

设计要点：
1. AST 自动发现：扫描 tools/ 目录，只 import 顶层真的调用了 registry.register() 的模块，
   纯辅助模块不会被误当成工具拉进来
2. check_fn 动态门控：工具登记时可以附带一个"我现在可用吗"的检查函数
   （比如检查 API key 配没配），结果缓存 30 秒，不每次都真查
3. JSON 字符串契约：所有 handler 统一返回 JSON 字符串，方便下游统一解析
4. 线程安全：用可重入锁保护，多个线程同时登记/查询不打架
"""

import ast
import importlib
import inspect
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST（语法树）检查：判断一个模块有没有调用 registry.register()
# ---------------------------------------------------------------------------

def _is_registry_register_call(node: ast.AST) -> bool:
    """判断一个语法树节点是不是形如 `registry.register(...)` 的调用语句。

    只认模块顶层直接写的那种（像"物业登记表上一眼能看到的名字"），
    函数体内部藏着的调用不算。
    """
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _module_registers_tools(module_path: Path) -> bool:
    """检查一个 .py 文件的顶层有没有 registry.register() 调用。

    自动发现工具时先"隔着门缝看一眼"，只把真正登记了工具的文件 import 进来，
    免得把只是帮忙的辅助模块也误当成工具模块加载。

    参数：
        module_path: 要检查的 .py 文件路径。

    返回：True 表示顶层有登记调用（是工具模块）；False 表示没有，
    或者文件读不了/语法有错（一律当作"不是"处理，宁可漏不可错）。
    """
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False
    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """把所有内置工具模块加载进来（它们 import 时会自动完成登记），返回加载成功的模块名列表。

    干的事：扫描 tools/ 目录，对每个 .py 文件先用语法树检查顶层有没有
    registry.register() 调用；有才 import——import 这个动作本身就会触发
    模块顶层的登记代码跑起来。某个模块加载失败只记一条警告，不影响其他模块。

    参数：
        tools_dir: 要扫描的目录；不传就用本文件所在的 tools/ 目录。

    返回：成功 import 的模块名列表（形如 "tools.terminal_tool"）。
    """
    tools_path = Path(tools_dir) if tools_dir else Path(__file__).resolve().parent
    module_names = [
        f"tools.{path.stem}"
        for path in sorted(tools_path.glob("*.py"))
        if path.name not in {"__init__.py", "registry.py"}
        and _module_registers_tools(path)
    ]

    imported = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("无法导入工具模块 %s: %s", mod_name, e)
    return imported


# ---------------------------------------------------------------------------
# 工具档案（一个工具一条记录）
# ---------------------------------------------------------------------------

class ToolEntry:
    """一个工具的全部"档案信息"：叫什么、归哪组、说明书（schema）是什么、谁干活（handler）。"""

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "schema_overrides_fn", "isConcurrencySafe",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 schema_overrides_fn=None, isConcurrencySafe=False):
        """建档案。各字段含义见 ToolRegistry.register 的参数说明，这里只负责存下来。"""
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji
        # 运行时的 schema 覆盖函数：输入 (schema字典, 运行时上下文字典)，输出新 schema 字典。
        # 用途：让 LLM 看到实时状态（比如还剩几个并发名额、当前是什么模式）。
        # 为什么需要：schema 本身是静态的，但有些信息每刻都在变。
        # 不传（None）时 schema 原样使用，老代码不受影响。
        self.schema_overrides_fn = schema_overrides_fn
        # 这个工具能不能和其他工具同时（并发）跑：
        # True = 只读/无副作用的工具（read_file/list_dir/grep 这类"只是看看不动手"的），可以几个一起跑
        # False = 有副作用的工具（write_file/terminal/memory_save 这类"真会改东西"的），必须排队一个一个来
        # 默认 False——安全第一：没显式说明"可以并发"的一律当不能并发（宁可慢一点，也别出乱子）
        self.isConcurrencySafe = isConcurrencySafe


# ---------------------------------------------------------------------------
# check_fn 的 TTL 缓存（TTL = 结果有效期，这里是 30 秒）
# ---------------------------------------------------------------------------
# check_fn 探测的是外部状态（Docker 装了吗？API key 配了吗？）。
# 这类状态不会秒变，没必要每次都真查——结果缓存 30 秒。
# 偶发抖动（比如 Docker 服务正好忙一下）不算数：刚成功过的话，短时间内的
# 失败按抖动处理，不缓存失败，避免把其实可用的工具误判成不可用。

_CHECK_FN_TTL_SECONDS = 30.0
_CHECK_FN_FAILURE_GRACE_SECONDS = 60.0  # 失败宽限期：最近成功过之后的失败先当抖动看
_check_fn_cache: Dict[Callable, tuple] = {}  # {fn: (timestamp, result)}
_check_fn_last_good: Dict[Callable, float] = {}
_check_fn_cache_lock = threading.Lock()


def _check_fn_cached(fn: Callable) -> bool:
    """带缓存地调一次可用性检查函数，返回"工具现在可用吗"。

    check_fn 查的都是慢变化的外部状态，每次真调太浪费；偶发失败宽容处理
    （刚成功过就当是抖动），所以走这个包装。

    参数：
        fn: 可用性检查函数；None 表示没有检查（永远算可用）。

    返回：True 可用 / False 不可用。
    """
    if fn is None:
        return True

    now = time.monotonic()
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get(fn)
        if cached and now - cached[0] < _CHECK_FN_TTL_SECONDS:
            return cached[1]

    try:
        value = bool(fn())
    except Exception:
        value = False

    with _check_fn_cache_lock:
        if value:
            _check_fn_last_good[fn] = now
            _check_fn_cache[fn] = (now, True)
            return True

        # 走到这里说明这次调用失败了。如果最近成功过，就当是偶发抖动：
        # 照样报"可用"，而且不把这次失败记进缓存
        last_good = _check_fn_last_good.get(fn)
        if last_good and now - last_good < _CHECK_FN_FAILURE_GRACE_SECONDS:
            return True  # 沿用上次的好结果，不缓存这次失败

        _check_fn_cache[fn] = (now, False)
        return False


# ---------------------------------------------------------------------------
# 注册表本体
# ---------------------------------------------------------------------------

# claude code 工具名 → 本项目等价工具名（dispatch 层的别名兜底）。
# 为什么要有：官方市场装来的技能正文里常写着「用 TodoWrite 建任务」
# 这类 CC 工具名；模型照着调时，这层别名让它落到本项目等价工具上，
# 而不是「未知工具」报错。只在原名不存在时才映射，绝不遮蔽真工具。
_CC_TOOL_ALIASES = {
    "Bash": "terminal",
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "str_replace",
    "Update": "str_replace",
    "Grep": "search_files",
    "Glob": "glob",
    "WebFetch": "web_fetch",
    "TodoWrite": "task_create",
    "Task": "delegate_task",
    "Agent": "delegate_task",
    "AskUserQuestion": "ask_user",
    "NotebookEdit": "notebook_edit",
    "Skill": "load_skill",
    "EnterPlanMode": "plan_mode_v2_dispatch",
    "ExitPlanMode": "exit_plan_mode",
}


class ToolRegistry:
    """工具注册表本体：所有工具的说明书（schema）和干活函数（handler）都收在这里。

    全项目只有一个实例（文件底部的 `registry` 全局变量），像一本共享户口本。
    """

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._lock = threading.RLock()
        self._generation: int = 0  # 版本号：每次登记/注销都 +1，外面的缓存靠它判断"该刷新了"

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        override: bool = False,
        schema_overrides_fn: Callable = None,
        isConcurrencySafe: bool = False,
    ):
        """登记一个新工具。通常在工具模块被 import 时（程序启动阶段）调用。

        参数：
            name: 工具名，LLM 调用时用的名字（如 "terminal"）
            toolset: 属于哪个工具集（如 "core"），用于分组控制可见性
            schema: OpenAI function calling 格式的说明书（名字、参数、描述），
                LLM 靠它知道这个工具怎么用
            handler: 真正干活的函数，签名固定是 (args: dict, **kw) -> str
            check_fn: "我现在可用吗"检查函数，返回 True/False；None 表示总是可用
                （比如没配 API key 时对应工具自动隐藏）
            requires_env: 依赖的环境变量名字列表（只用来给文档/界面展示）
            is_async: handler 是不是 async 函数（影响 dispatch 走哪条路径执行）
            override: 遇到同名工具时允不允许覆盖；默认不允许（插件替换内置工具时才开）
            schema_overrides_fn: 运行时改说明书的函数，签名
                (schema: dict, runtime_ctx: dict) -> 新 schema 或 None（不改）。
                出错会被兜住并回退用原 schema，不会炸
            isConcurrencySafe: 能不能和其他工具同时跑。
                True = 只读/无副作用（read_file/list_dir/grep 这类），可以并发；
                False = 有副作用（write_file/terminal/memory_save 这类），必须排队。
                默认 False——安全第一，没标注的一律串行。
        """
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset and not override:
                logger.error(
                    "工具注册被拒绝: '%s'（工具集 '%s'）"
                    "会覆盖已有工具集 '%s' 中的同名工具",
                    name, toolset, existing.toolset,
                )
                return

            self._tools[name] = ToolEntry(
                name=name, toolset=toolset, schema=schema,
                handler=handler, check_fn=check_fn,
                requires_env=requires_env or [],
                is_async=is_async,
                description=description or schema.get("description", ""),
                emoji=emoji,
                schema_overrides_fn=schema_overrides_fn,
                isConcurrencySafe=isConcurrencySafe,
            )
            self._generation += 1

    def unregister(self, name: str) -> bool:
        """把一个工具从户口本上划掉（注销）。

        主要是给测试用的——测试临时登记的工具跑完要清掉，免得污染其他测试；
        生产代码不该用它（工具登记是启动期一次性的事）。

        参数：
            name: 工具名。

        返回：True = 找到并移除了；False = 本来就没有这个工具。
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                self._generation += 1
                return True
            return False

    async def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """把 LLM 发起的工具调用转交给对应的干活函数，返回 JSON 字符串结果。

        主程序是异步的（async），但工具函数有同步有异步，两种都要能跑
        且不能卡住整个事件循环。

        怎么跑：
        - async handler（如 MCP / delegate 这些）：直接 await
        - 同步 handler（内置的同步工具）：丢到线程池里跑，不阻塞事件循环，
          handler 内部代码一行都不用改

        不管哪种，返回值最后都过 _normalize_result 统一成 JSON 字符串（契约不变）。

        参数：
            name: 工具名（LLM 说的要调谁）
            args: 工具参数字典（LLM 按 schema 填的）
            **kwargs: 命名上下文（如 memory_store、agent_ref 等，原样传给 handler）

        返回：JSON 字符串——成功是工具自己的结果，失败是
            {"error": ..., "error_type": ...} 格式的错误。
        """
        # CC 工具名别名兜底（见 _CC_TOOL_ALIASES 注释）：原名不存在才映射
        if name not in self._tools:
            name = _CC_TOOL_ALIASES.get(name, name)
        with self._lock:
            entry = self._tools.get(name)

        if entry is None:
            return json.dumps({
                "error": f"未知工具: {name}",
                "error_type": "unknown_tool",
            }, ensure_ascii=False)

        # permissions.deny 的第二道防线（第一道"眼不见为净"可被手动 tool_call/
        # 缓存未刷新绕过）。规则加载一次传参复用，不再每次调用都 stat settings.json。
        # 保留 fail-open，但必须大声报 ERROR——静默吞掉防线就无声消失了。
        try:
            from agent.tool_permissions import (
                is_tool_denied, load_tool_permission_rules,
            )
            if is_tool_denied(name, rules=load_tool_permission_rules()):
                return json.dumps({
                    "error": f"工具 {name} 被 settings.json permissions.deny 规则拒绝",
                    "error_type": "permission_denied",
                }, ensure_ascii=False)
        except Exception as e:
            logger.error("deny 规则加载失败，dispatch 层防御本调用失效（fail-open）: %s", e)

        handler = entry.handler
        try:
            if inspect.iscoroutinefunction(handler):
                # 真 async handler：直接 await
                result = await handler(args, **kwargs)
            else:
                # 同步 handler：丢线程池跑。
                # 必须用 asyncio.to_thread（而不是 anyio.to_thread.run_sync）：
                # 它会自动把当前 context（上下文变量，如 workspace_cwd）复制到
                # 工作线程，否则 worktree 子代理拿到的还是主进程目录而非自己的
                # 工作区目录（anyio 版本默认不带回 context）。
                import asyncio as _asyncio
                result = await _asyncio.to_thread(handler, args, **kwargs)
            return self._normalize_result(name, result)
        except Exception as e:
            logger.exception("工具 %s 执行失败", name)
            return json.dumps({
                "error": str(e),
                "error_type": "tool_exception",
                "tool": name,
            }, ensure_ascii=False)

    @staticmethod
    def _normalize_result(name: str, result) -> str:
        """把 handler 的返回值统一整理成 JSON 字符串。

        契约要求所有工具返回 JSON 字符串，这里做兜底：字符串原样过，
        dict 帮你转 JSON，别的类型报契约错误。

        参数：
            name: 工具名（出错时写进错误信息里）
            result: handler 的原始返回值

        返回：JSON 字符串。
        """
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return json.dumps({
            "error": f"工具返回了不支持的类型: {type(result).__name__}",
            "error_type": "tool_result_contract",
            "tool": name,
        }, ensure_ascii=False)

    def get(self, name: str) -> Optional[ToolEntry]:
        """按工具名查档案；查不到返回 None。

        测试和并发分组逻辑用它直接拿工具档案（如读"能不能并发跑"字段）。

        参数：
            name: 工具名。

        返回：ToolEntry（工具档案），没有这个工具就是 None。读取时加锁；
        返回的是档案本身的引用（字段创建后不再改，多线程读是安全的）。
        """
        with self._lock:
            return self._tools.get(name)

    def get_definitions(
        self,
        tool_names: List[str],
        *,
        quiet: bool = False,
        runtime_ctx: Optional[Dict] = None,
    ) -> List[dict]:
        """整理一份发给 LLM 的工具说明书列表（OpenAI 格式）。

        只包含同时满足两个条件的工具：
        1. 名字在 tool_names 名单里（本次对话允许它看见）
        2. 可用性检查通过（比如需要的 API key 已配置）

        参数：
            tool_names: 允许暴露的工具名列表。
            quiet: True 时不打印"某某工具未注册"的调试日志。
            runtime_ctx: 运行时上下文字典（如 {"agent": self}）。
                交给 schema_overrides_fn 用来动态改说明书（比如填上实时状态）。
                不传就跳过覆盖这步，说明书原样输出（老调用方式不受影响）。

        返回：schema 字典列表，每项形如 {"type": "function", "function": {...}}。
        """
        definitions = []
        with self._lock:
            entries = [(n, self._tools.get(n)) for n in tool_names]

        for name, entry in entries:
            if entry is None:
                if not quiet:
                    logger.debug("工具 %s 未注册（被忽略）", name)
                continue
            # 可用性检查（结果有 30 秒缓存，见上文说明）
            if entry.check_fn and not _check_fn_cached(entry.check_fn):
                continue  # 不可用就不发给 LLM

            # 拷贝一份 schema 再改，别把登记时的原件改脏了
            schema = dict(entry.schema)
            # 运行时覆盖（如剩余并发槽位）
            if entry.schema_overrides_fn is not None and runtime_ctx:
                try:
                    overridden = entry.schema_overrides_fn(schema, runtime_ctx)
                    if overridden:
                        schema = overridden
                except Exception as e:
                    logger.warning(
                        "schema_overrides_fn %s 失败（用原 schema）: %s",
                        name, e,
                    )

            definitions.append({
                "type": "function",
                "function": schema,
            })

        return definitions

    def get_catalog_entry(self, name: str) -> Optional[dict]:
        """给某个工具出一张"名片"：名字 + 一句话简介 + 怎么查详情的提示。

        给 ToolSearch（工具搜索）用的：MCP 外部工具数量可能很多，全部附详细
        说明书太占 token，先只发名片，LLM 需要时再调 tool_search 取完整说明书。
        和 get_definitions 的分工：
        - get_definitions 给全量说明书（含详细 parameters）—— 用于内置工具
        - get_catalog_entry 只给名片（parameters 是空壳）—— 用于 MCP 工具

        参数：
            name: 工具名。

        返回：精简条目字典；工具不存在或当前不可用时返回 None。
        """
        with self._lock:
            entry = self._tools.get(name)
        if entry is None:
            return None
        # 可用性过滤：不可用的工具连目录都不进
        if entry.check_fn and not _check_fn_cached(entry.check_fn):
            return None
        desc = (entry.schema.get("description", "") or "")[:60]
        # 提示语里的搜索关键词用短名（去掉 mcp__<server>__ 这种长前缀，好搜）
        short_name = name.split("__")[-1] if "__" in name else name
        return {
            "name": name,
            "description": f"{desc} [调 tool_search('{short_name}') 取详细参数]",
            "parameters": {"type": "object", "properties": {}},
        }

    def list_all(self) -> List[str]:
        """列出户口本上所有工具的名字。

        返回：工具名列表（拷贝，改它不影响登记表）。
        """
        with self._lock:
            return list(self._tools.keys())

    @property
    def generation(self) -> int:
        return self._generation


# 全局唯一实例——全项目都用这个 `registry` 存取工具
registry = ToolRegistry()
