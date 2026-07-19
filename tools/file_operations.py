"""文件操作工具：read_file / write_file / search_files。

read_file：读文件，带行号返回，方便 LLM 引用
write_file：写文件（覆盖）
search_files：在文件内容中搜索（类似 grep）
"""

import json
import re
from pathlib import Path
from typing import Optional

from agent.output_offload import finalize_tool_output as _finalize_output
from agent.permission import safe_path
from tools.registry import registry


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "读取文件内容。支持文本文件，带行号返回。",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径（绝对或相对）",
            },
            "offset": {
                "type": "integer",
                "description": "起始行号（0-based，默认 0）",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "读取行数（默认全部）",
            },
        },
        "required": ["path"],
    },
}


def _handle_read_file(args: dict, **kwargs) -> str:
    path_str = args.get("path", "")

    # 路径检查（受保护路径拒绝，如 ~/.ssh、/etc/passwd）
    perm = safe_path(path_str, write=False)
    if not perm.allowed:
        return json.dumps(
            {"error": f"路径拒绝: {perm.reason}", "error_type": "permission_denied"},
            ensure_ascii=False,
        )

    offset = int(args.get("offset", 0) or 0)
    limit = args.get("limit")
    if limit is not None:
        limit = int(limit)

    path = Path(path_str).expanduser()
    if not path.exists():
        return json.dumps({"error": f"文件不存在: {path}"}, ensure_ascii=False)
    if not path.is_file():
        return json.dumps({"error": f"不是文件: {path}"}, ensure_ascii=False)

    try:
        # 关键：必须指定 encoding，否则 Windows 默认 cp1252 会乱码
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # 应用 offset 和 limit
        end = offset + limit if limit else None
        selected = lines[offset:end]

        # 带行号返回（方便 LLM 引用）
        numbered = [
            f"{i + offset + 1:6}\t{line}"
            for i, line in enumerate(selected)
        ]
        raw_content = "\n".join(numbered)

        # 大输出 offload（Phase 1 后始终启用）
        tool_call_id = kwargs.get("tool_call_id")
        config = kwargs.get("config")
        harvil_home = kwargs.get("harvil_home")
        final_content = _finalize_output(raw_content, tool_call_id, harvil_home, config)
        content_offloaded = final_content != raw_content

        return json.dumps({
            "path": str(path),
            "content": final_content,
            "content_offloaded": content_offloaded,
            "total_lines": len(lines),
            "shown_lines": f"{offset + 1}-{offset + len(selected)}",
        }, ensure_ascii=False)
    except UnicodeDecodeError:
        return json.dumps({"error": "无法解码为文本（可能是二进制文件）"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "写入文件（覆盖）。目录不存在会自动创建。",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径",
            },
            "content": {
                "type": "string",
                "description": "要写入的内容",
            },
            "append": {
                "type": "boolean",
                "description": "是否追加（默认 False，覆盖）",
                "default": False,
            },
        },
        "required": ["path", "content"],
    },
}


def _handle_write_file(args: dict, **kwargs) -> str:
    path_str = args.get("path", "")
    content = args.get("content", "")
    append = bool(args.get("append", False))

    if not path_str:
        return json.dumps({"error": "path 不能为空"}, ensure_ascii=False)

    # 路径白名单检查（写操作）：受保护路径 + 工作目录外拒绝
    perm = safe_path(path_str, write=True)
    if not perm.allowed:
        return json.dumps(
            {"error": f"路径拒绝: {perm.reason}", "error_type": "permission_denied"},
            ensure_ascii=False,
        )

    path = Path(path_str).expanduser()

    try:
        # 自动创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)

        # 必须指定 encoding
        if append:
            with path.open("a", encoding="utf-8") as f:
                f.write(content + "\n")
        else:
            path.write_text(content, encoding="utf-8")

        return json.dumps({
            "path": str(path),
            "bytes": len(content.encode("utf-8")),
            "appended": append,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": (
        "在目录中搜索文件内容（类似 grep）。"
        "返回匹配的行和文件路径。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "搜索正则表达式",
            },
            "path": {
                "type": "string",
                "description": "搜索目录（默认当前目录）",
            },
            "glob": {
                "type": "string",
                "description": "文件名 glob（如 '*.py'，默认所有文件）",
            },
            "max_matches": {
                "type": "integer",
                "description": "最大匹配数（默认 50）",
                "default": 50,
            },
            "include_hidden": {
                "type": "boolean",
                "description": "是否搜索隐藏目录和依赖目录（.git/.venv/__pycache__/node_modules 等），默认 False",
                "default": False,
            },
        },
        "required": ["pattern"],
    },
}

# 默认排除的目录（依赖、缓存、版本控制——不是用户代码）
_DEFAULT_EXCLUDED_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", "bower_components",
    ".idea", ".vscode",
    "dist", "build", ".eggs",
    ".codegraph",
}


def _is_in_excluded_dir(file_path: Path, search_root: Path) -> bool:
    """文件是否落在 search_root 下的某个被排除目录里。"""
    try:
        rel = file_path.relative_to(search_root)
    except ValueError:
        return False
    return any(part in _DEFAULT_EXCLUDED_DIRS for part in rel.parts)


def _handle_search_files(args: dict, **kwargs) -> str:
    pattern = args.get("pattern", "")
    search_path = Path(args.get("path") or ".").expanduser()
    file_glob = args.get("glob") or "**/*"
    max_matches = int(args.get("max_matches", 50))
    include_hidden = bool(args.get("include_hidden", False))

    if not pattern:
        return json.dumps({"error": "pattern 不能为空"}, ensure_ascii=False)

    if not search_path.exists():
        return json.dumps({"error": f"路径不存在: {search_path}"}, ensure_ascii=False)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return json.dumps({"error": f"非法正则: {e}"}, ensure_ascii=False)

    matches = []
    files_searched = 0
    files_skipped = 0
    try:
        for file_path in search_path.glob(file_glob):
            if not file_path.is_file():
                continue
            # 默认跳过 .git/.venv/__pycache__/node_modules 等
            if not include_hidden and _is_in_excluded_dir(file_path, search_path):
                files_skipped += 1
                continue
            # 跳过二进制/大文件
            if file_path.stat().st_size > 1_000_000:
                continue
            files_searched += 1
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            for line_no, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    matches.append({
                        "file": str(file_path),
                        "line": line_no,
                        "content": line[:500],  # 截断长行
                    })
                    if len(matches) >= max_matches:
                        return json.dumps({
                            "matches": matches,
                            "truncated": True,
                            "files_searched": files_searched,
                            "files_skipped_hidden": files_skipped,
                        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps({
        "matches": matches,
        "match_count": len(matches),
        "files_searched": files_searched,
        "files_skipped_hidden": files_skipped,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

registry.register(
    name="read_file",
    toolset="core",
    schema=READ_FILE_SCHEMA,
    handler=_handle_read_file,
    emoji="📄",
)

registry.register(
    name="write_file",
    toolset="core",
    schema=WRITE_FILE_SCHEMA,
    handler=_handle_write_file,
    emoji="✏️",
)

registry.register(
    name="search_files",
    toolset="core",
    schema=SEARCH_FILES_SCHEMA,
    handler=_handle_search_files,
    emoji="🔍",
)
