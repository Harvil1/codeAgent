"""glob 工具：按文件名模式查找文件（CCAR11 Task 1，对齐 Claude Code GlobTool）。

与 search_files（grep 文件内容）正交——这个只匹配文件名，不读内容。
结果按 mtime 降序（最近改的在前），max_results 钳制 [1, 1000]。

为什么需要它：之前 LLM 找文件只能 terminal ls / dir，走权限闸门 + subprocess，
开销大且有副作用标记（不能并发）。glob 是纯只读，可并发（isConcurrencySafe=True）。
"""
import json
import logging
from pathlib import Path

from agent.permission import safe_path
from tools.registry import registry

logger = logging.getLogger(__name__)


GLOB_SCHEMA = {
    "name": "glob",
    "description": (
        "按文件名模式查找文件（只匹配名，不读内容）。"
        "常见模式：**/*.py（递归所有 py）、test_*.md、src/**/*.ts。"
        "与 search_files（grep 内容）正交。结果按修改时间降序（最近改的在前）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "glob 模式，如 **/*.py、*.md、src/**/*.ts",
            },
            "path": {
                "type": "string",
                "default": ".",
                "description": "起始目录（默认当前工作目录）",
            },
            "max_results": {
                "type": "integer",
                "default": 200,
                "minimum": 1,
                "maximum": 1000,
                "description": "最多返回条数（超出设 truncated=true）",
            },
        },
        "required": ["pattern"],
    },
}


def _handle_glob(args: dict, **dispatch_kwargs) -> str:
    """文件名模式匹配（只读）。

    流程：
      1. 参数校验（pattern 非空，max_results 钳制 [1, 1000]）
      2. safe_path 读校验（拒绝 ~/.ssh 等受保护路径）
      3. Path.glob 递归匹配
      4. mtime 降序 + 钳制 max_results + truncated 标志

    返回 JSON 字符串（统一契约）：
      成功：{"matches": [...], "count": N, "truncated": bool, "total_found": M}
      失败：{"error": "...", "error_type": "..."}
    """
    pattern = (args.get("pattern") or "").strip()
    if not pattern:
        return json.dumps(
            {"error": "pattern 不能为空", "error_type": "invalid_args"},
            ensure_ascii=False,
        )

    path = args.get("path") or "."

    # max_results 钳制 [1, 1000]（LLM 可能传 0/负数/超大值/字符串）
    try:
        max_results = int(args.get("max_results", 200) or 200)
    except (TypeError, ValueError):
        max_results = 200
    max_results = max(1, min(1000, max_results))

    # safe_path 读校验——拒绝受保护路径（~/.ssh、/etc、C:\Windows 等）
    # safe_path 返回 PermissionResult（.allowed / .reason / .gate），
    # 不是 tuple（CCAR11 Task 1 实施时确认的真实签名）
    perm = safe_path(path, write=False)
    if not perm.allowed:
        return json.dumps(
            {
                "error": f"路径被拒绝: {perm.reason}",
                "error_type": "permission_denied",
                "gate": perm.gate,
                "path": path,
            },
            ensure_ascii=False,
        )

    try:
        base = Path(path).expanduser().resolve()
        if not base.is_dir():
            return json.dumps(
                {"error": f"目录不存在: {path}", "error_type": "invalid_args"},
                ensure_ascii=False,
            )

        # Path.glob 匹配（** 支持递归）
        matches = [p for p in base.glob(pattern) if p.is_file()]

        # mtime 降序（最近改的在前）——失败容错（文件可能被并发删）
        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        matches.sort(key=_mtime, reverse=True)

        total_found = len(matches)
        truncated = total_found > max_results
        top = matches[:max_results]

        # 相对路径优先（更短、可读），跨盘 fallback 绝对路径
        result_paths = []
        for p in top:
            try:
                result_paths.append(str(p.relative_to(base)))
            except ValueError:
                result_paths.append(str(p))

        return json.dumps(
            {
                "matches": result_paths,
                "count": len(result_paths),
                "truncated": truncated,
                "total_found": total_found,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning("glob 异常: %s", e, exc_info=True)
        return json.dumps(
            {"error": str(e), "error_type": type(e).__name__},
            ensure_ascii=False,
        )


# 模块级注册（import 时自动执行，AST 检查会发现 registry.register 调用）
registry.register(
    name="glob",
    toolset="core",
    schema=GLOB_SCHEMA,
    handler=_handle_glob,
    emoji="🔎",
    isConcurrencySafe=True,  # 只读：文件名匹配，无副作用，可 asyncio.gather 并发
)
