"""glob 工具：按文件名模式（如 **/*.py）查找文件（CCAR11 Task 1 引入，做法对齐 Claude Code 的 GlobTool）。

跟 search_files 的分工：那个是 grep——在文件**内容**里找关键词；这个只看
**文件名**长什么样，不读内容。结果按修改时间倒序排（最近改过的排最前），
最多返回 1000 条。

为什么需要它：以前模型想找个文件只能用 terminal 跑 ls / dir——既要去过
权限闸门、又要启动子进程，开销大，而且 terminal 被标成"有副作用"没法并发。
这个工具是纯只读的，标了 isConcurrencySafe=True，多条查找可以同时跑。
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
    """按文件名模式找文件（纯只读操作）。

    流程：
      1. 校验参数（模式不能为空；最多返回条数夹在 1~1000 之间）
      2. 用 safe_path 做读安检（~/.ssh 这类受保护目录一律拒绝）
      3. 交给 Path.glob 做递归匹配
      4. 按修改时间倒序排，截到上限条数，并标明"有没有被截"

    参数：
        args：工具参数字典，来自模型——pattern（文件名模式，如 **/*.py）、
            path（从哪个目录开始找）、max_results（最多返回几条）。
        **dispatch_kwargs：分发器注入的运行上下文（本函数未用到，签名保持
            工具统一契约）。

    返回：JSON 字符串（项目统一契约）：
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

    # 把最多返回条数夹在 1~1000 之间——模型可能传 0、负数、超大值甚至字符串
    try:
        max_results = int(args.get("max_results", 200) or 200)
    except (TypeError, ValueError):
        max_results = 200
    max_results = max(1, min(1000, max_results))

    # 路径安检：受保护路径（~/.ssh、/etc、C:\Windows 等）一律拒绝
    # 历史确认（CCAR11 Task 1 实施时查过）：safe_path 返回的是
    # PermissionResult 对象（用 .allowed / .reason / .gate 三个字段），
    # 不是 tuple，别按 tuple 去解包
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

        # Path.glob 做匹配（模式里的 ** 表示递归进所有子目录）
        matches = [p for p in base.glob(pattern) if p.is_file()]

        # 按修改时间倒序（最近改的排最前）——取时间失败时容错当 0
        # （排序过程中文件可能正好被别的程序删掉）
        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        matches.sort(key=_mtime, reverse=True)

        total_found = len(matches)
        truncated = total_found > max_results
        top = matches[:max_results]

        # 优先给相对路径（更短、好读）；跨盘拿不到相对路径时退回绝对路径
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


# 模块级注册：这个文件一被 import 就自动登记进中央注册表
# （注册表靠 AST 扫描发现 registry.register 调用，所以这行不能塞进函数里）
registry.register(
    name="glob",
    toolset="core",
    schema=GLOB_SCHEMA,
    handler=_handle_glob,
    emoji="🔎",
    isConcurrencySafe=True,  # 纯只读：只匹配文件名无副作用，可以用 asyncio.gather 同时跑多个
)
