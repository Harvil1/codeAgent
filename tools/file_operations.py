"""文件操作工具：read_file / write_file / search_files。

read_file：读文件，带行号返回，方便 LLM 引用
write_file：写文件（覆盖）
search_files：在文件内容中搜索（类似 grep）
"""

import json
import re
from pathlib import Path
from typing import Optional

from agent.permission import safe_path
from tools.registry import registry


def _finalize_output(
    result_content: str,
    tool_call_id: Optional[str],
    harvil_home,
    config: Optional[dict],
) -> str:
    """超阈值内容走 offload（落盘 + 预览）。

    Phase 1 Commit 7 后：原 ``use_new_pipeline`` 开关已移除，offload 始终启用。
    若调用方需要关闭 offload，直接不传 ``tool_call_id`` 或 ``harvil_home`` 即可。
    """
    if not tool_call_id or not harvil_home:
        return result_content
    from agent.output_offload import maybe_offload
    return maybe_offload(
        result_content,
        tool_call_id=tool_call_id,
        agent_home=Path(harvil_home),
        threshold=(config or {}).get("context", {}).get("output_offload_threshold", 30000),
        preview_chars=(config or {}).get("context", {}).get("output_offload_preview", 2000),
    )


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
        mode = "a" if append else "w"
        path.write_text(content if not append else content + "\n",
                        encoding="utf-8") if mode == "w" else None

        if mode == "a":
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
        },
        "required": ["pattern"],
    },
}


def _handle_search_files(args: dict, **kwargs) -> str:
    pattern = args.get("pattern", "")
    search_path = Path(args.get("path") or ".").expanduser()
    file_glob = args.get("glob") or "**/*"
    max_matches = int(args.get("max_matches", 50))

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
    try:
        for file_path in search_path.glob(file_glob):
            if not file_path.is_file():
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
                        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    return json.dumps({
        "matches": matches,
        "match_count": len(matches),
        "files_searched": files_searched,
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
