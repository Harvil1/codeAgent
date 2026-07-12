"""中央工具注册表。

依赖链（无循环）：
    registry.py  (无依赖)
        ↑
    tools/*.py   (每个文件 import registry，模块顶层 register())
        ↑
    model_tools.py  (import registry + 触发现)

设计要点：
1. AST 检查自动发现：扫描 tools/ 目录，只 import 顶层调用 registry.register() 的模块
2. check_fn 动态门控：工具可注册"可用性检查"（如 API key 是否配置），带 30s 缓存
3. JSON 字符串契约：所有 handler 返回 JSON 字符串，统一序列化
4. 线程安全：RLock 保护并发访问
"""

import ast
import importlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AST 检查：判断模块是否调用了 registry.register()
# ---------------------------------------------------------------------------

def _is_registry_register_call(node: ast.AST) -> bool:
    """判断 AST 节点是否是 registry.register(...) 调用。

    只识别形如 `registry.register(...)` 的模块级表达式。
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
    """检查模块顶层是否包含 registry.register() 调用。

    只检查模块级语句，不检查函数内部，避免把辅助模块误判为工具模块。
    """
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False
    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """导入所有自注册工具模块，返回模块名列表。

    扫描 tools/ 目录，对每个 .py 文件：
    1. AST 检查是否有顶层 registry.register() 调用
    2. 有则 import（触发模块级的 register 调用）
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
# 工具条目
# ---------------------------------------------------------------------------

class ToolEntry:
    """单个工具的元数据。"""

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji


# ---------------------------------------------------------------------------
# check_fn 的 TTL 缓存
# ---------------------------------------------------------------------------
# check_fn 会探测外部状态（Docker 装了吗？API key 配置了吗？）
# 这些状态变化慢，不需要每次都真调用。缓存 30 秒。
# 瞬时失败（Docker daemon 繁忙）不缓存，避免误判工具不可用。

_CHECK_FN_TTL_SECONDS = 30.0
_CHECK_FN_FAILURE_GRACE_SECONDS = 60.0  # 最近成功后的失败宽限期
_check_fn_cache: Dict[Callable, tuple] = {}  # {fn: (timestamp, result)}
_check_fn_last_good: Dict[Callable, float] = {}
_check_fn_cache_lock = threading.Lock()


def _check_fn_cached(fn: Callable) -> bool:
    """带 TTL 缓存地调用 check_fn。"""
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

        # 瞬时失败处理：如果最近成功过，当作抖动
        last_good = _check_fn_last_good.get(fn)
        if last_good and now - last_good < _CHECK_FN_FAILURE_GRACE_SECONDS:
            return True  # 返回 last-good True，不缓存这次失败

        _check_fn_cache[fn] = (now, False)
        return False


# ---------------------------------------------------------------------------
# 注册表单例
# ---------------------------------------------------------------------------

class ToolRegistry:
    """单例注册表，收集所有工具的 schema 和 handler。"""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._lock = threading.RLock()
        self._generation: int = 0  # 每次变更递增，用于外部缓存失效

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
    ):
        """注册一个工具。通常在模块 import 时调用。

        参数：
            name: 工具名（如 "terminal"）
            toolset: 所属工具集（如 "core"）
            schema: OpenAI function calling 格式的 schema
            handler: 实际执行函数，签名 (args: dict, **kw) -> str
            check_fn: 可用性检查函数，返回 bool。None 表示总是可用
            requires_env: 依赖的环境变量列表（用于文档/UI 显示）
            override: 是否允许覆盖同名工具（插件场景）
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
            )
            self._generation += 1

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """分发工具调用，返回 JSON 字符串结果。"""
        with self._lock:
            entry = self._tools.get(name)

        if entry is None:
            return json.dumps({
                "error": f"未知工具: {name}",
                "error_type": "unknown_tool",
            }, ensure_ascii=False)

        try:
            result = entry.handler(args, **kwargs)
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
        """规范化 handler 返回值为 JSON 字符串。"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        return json.dumps({
            "error": f"工具返回了不支持的类型: {type(result).__name__}",
            "error_type": "tool_result_contract",
            "tool": name,
        }, ensure_ascii=False)

    def get_definitions(
        self, tool_names: List[str], *, quiet: bool = False
    ) -> List[dict]:
        """返回 OpenAI 格式的工具 schema 列表。

        只返回：
        1. 名字在 tool_names 中的工具
        2. check_fn 通过的工具（有 API key 等）
        """
        definitions = []
        with self._lock:
            entries = [(n, self._tools.get(n)) for n in tool_names]

        for name, entry in entries:
            if entry is None:
                if not quiet:
                    logger.debug("工具 %s 未注册（被忽略）", name)
                continue
            # check_fn 检查（带缓存）
            if entry.check_fn and not _check_fn_cached(entry.check_fn):
                continue  # 不可用，不暴露给 LLM

            definitions.append({
                "type": "function",
                "function": entry.schema,
            })

        return definitions

    def list_all(self) -> List[str]:
        """返回所有已注册的工具名。"""
        with self._lock:
            return list(self._tools.keys())

    @property
    def generation(self) -> int:
        return self._generation


# 全局单例
registry = ToolRegistry()
