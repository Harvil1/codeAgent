"""文件操作工具：read_file / write_file / search_files / str_replace。

read_file：读文件，带行号返回，方便 LLM 引用(返回 content_hash 用于 write_file 防盲写)
write_file：写文件（覆盖）,支持 read-before-write hash 校验
search_files：在文件内容中搜索（类似 grep）
str_replace：子串替换(单次/全部),避免 read 整个文件再 write 整个文件
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from agent.output_offload import finalize_tool_output as _finalize_output
from agent.permission import safe_path
from tools.registry import registry


def _get_mode_override_from_kwargs(kwargs: dict) -> Optional[str]:
    """从工具调用的 kwargs 里提取子代理 permission_mode override。

    必修 1：工具读 kwargs["agent_ref"].permission_mode，作为本次 check 的 mode override。
    线程安全：mode override 只影响本次调用，不修改全局 checker 状态。
    返回 "bypassPermissions" / "default" / None（无 agent_ref 时）。
    """
    agent_ref = kwargs.get("agent_ref")
    if agent_ref is None:
        return None
    mode = getattr(agent_ref, "permission_mode", None)
    if mode in ("default", "bypassPermissions"):
        return mode
    return None


def _content_hash(text: str) -> str:
    """计算文本的短 hash(sha256 前 16 位),用于 read-before-write 校验。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
        omnimate_home = kwargs.get("omnimate_home")
        final_content = _finalize_output(raw_content, tool_call_id, omnimate_home, config)
        content_offloaded = final_content != raw_content

        return json.dumps({
            "path": str(path),
            "content": final_content,
            "content_hash": _content_hash(content),  # 给 write_file 做 read-before-write 校验用
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
    "description": (
        "写入文件（覆盖）。目录不存在会自动创建。"
        "**read-before-write 保护**:如果传了 expected_hash,会校验文件当前 hash 是否匹配,"
        "不匹配说明文件被外部修改过,会拒绝写入(需重新 read_file)。"
    ),
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
            "expected_hash": {
                "type": "string",
                "description": (
                    "可选。read_file 返回的 content_hash。"
                    "传入则做 read-before-write 校验:文件当前 hash 不匹配时拒绝写入,"
                    "防止覆盖外部修改。不传 = 不校验(向后兼容)。"
                ),
            },
        },
        "required": ["path", "content"],
    },
}


def _track_checkpoint(path, kwargs) -> None:
    """编辑成功后通知 checkpoint 追踪该文件（对齐 Claude Code /rewind）。

    只追踪编辑工具的直接修改（write_file/str_replace）；bash 命令不追踪。
    无 checkpoint（测试/子代理）时静默跳过。
    """
    agent_ref = kwargs.get("agent_ref")
    tracker = getattr(agent_ref, "_checkpoint_track", None)
    if tracker:
        try:
            tracker(str(path))
        except Exception:
            pass


def _handle_write_file(args: dict, **kwargs) -> str:
    path_str = args.get("path", "")
    content = args.get("content", "")
    append = bool(args.get("append", False))
    expected_hash = args.get("expected_hash")

    if not path_str:
        return json.dumps({"error": "path 不能为空"}, ensure_ascii=False)

    # 路径权限检查（write=True）：
    #   - 受保护路径（~/.ssh / /etc 等）→ 硬拒
    #   - 在 cwd 或 ~/.OmniMate 白名单 → 通过
    #   - 不在白名单 → 调 approval_callback 问用户（同意后加入持久化白名单）
    # 优先用注入的 permission_checker（cli.py 注入带 callback 的），
    # 没有则用全局默认（无 callback，白名单外路径会拒绝）。
    # 必修 1：子代理 permission_mode 透传（bypassPermissions 放行白名单外路径）。
    from agent.permission import get_default_checker
    checker = kwargs.get("permission_checker") or get_default_checker()
    mode_override = _get_mode_override_from_kwargs(kwargs)
    perm = checker.check_path(path_str, write=True, mode_override=mode_override)
    if not perm.allowed:
        return json.dumps(
            {"error": f"路径拒绝: {perm.reason}", "error_type": "permission_denied"},
            ensure_ascii=False,
        )

    path = Path(path_str).expanduser()

    # read-before-write hash 校验(仅覆盖模式,且传了 expected_hash)
    if expected_hash and not append and path.exists():
        try:
            current = path.read_text(encoding="utf-8")
            current_hash = _content_hash(current)
            if current_hash != expected_hash:
                return json.dumps({
                    "error": (
                        f"read-before-write 校验失败:文件已被外部修改"
                        f"(expected={expected_hash}, current={current_hash})。"
                        f"请重新调 read_file 拿最新内容再 write。"
                    ),
                    "error_type": "stale_hash",
                    "current_hash": current_hash,
                }, ensure_ascii=False)
        except UnicodeDecodeError:
            # 二进制文件,跳过 hash 校验(向后兼容)
            pass

    try:
        # 自动创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)

        # 必须指定 encoding
        if append:
            with path.open("a", encoding="utf-8") as f:
                f.write(content + "\n")
        else:
            path.write_text(content, encoding="utf-8")

        _track_checkpoint(path, kwargs)  # /rewind 追踪该文件

        return json.dumps({
            "path": str(path),
            "bytes": len(content.encode("utf-8")),
            "appended": append,
            "content_hash": _content_hash(content) if not append else None,
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


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------

STR_REPLACE_SCHEMA = {
    "name": "str_replace",
    "description": (
        "在文件里做子串替换(单次或全部)。比 read+write 高效——"
        "改 1 行不用 read 整个文件再 write 整个文件。"
        "支持 read-before-write hash 校验(传 expected_hash)。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "old_str": {"type": "string", "description": "要被替换的子串(必须能匹配到)"},
            "new_str": {"type": "string", "description": "替换成的内容"},
            "replace_all": {
                "type": "boolean", "default": False,
                "description": "True=全部替换;False=只替换第一处(默认)",
            },
            "expected_hash": {
                "type": "string",
                "description": "可选。read_file 返回的 content_hash,做 read-before-write 校验。",
            },
        },
        "required": ["path", "old_str", "new_str"],
    },
}


def _handle_str_replace(args: dict, **kwargs) -> str:
    path_str = args.get("path", "")
    old_str = args.get("old_str")
    new_str = args.get("new_str")
    replace_all = bool(args.get("replace_all", False))
    expected_hash = args.get("expected_hash")

    if not path_str:
        return json.dumps({"error": "path 不能为空"}, ensure_ascii=False)
    if old_str is None or old_str == "":
        return json.dumps({"error": "old_str 不能为空"}, ensure_ascii=False)
    if new_str is None:
        return json.dumps({"error": "new_str 不能为空(用空串表示删除)"}, ensure_ascii=False)

    # 路径权限检查(走 write 审批)
    # 必修 1：子代理 permission_mode 透传。
    from agent.permission import get_default_checker
    checker = kwargs.get("permission_checker") or get_default_checker()
    mode_override = _get_mode_override_from_kwargs(kwargs)
    perm = checker.check_path(path_str, write=True, mode_override=mode_override)
    if not perm.allowed:
        return json.dumps(
            {"error": f"路径拒绝: {perm.reason}", "error_type": "permission_denied"},
            ensure_ascii=False,
        )

    path = Path(path_str).expanduser()
    if not path.exists():
        return json.dumps({"error": f"文件不存在: {path}"}, ensure_ascii=False)
    if not path.is_file():
        return json.dumps({"error": f"不是文件: {path}"}, ensure_ascii=False)

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return json.dumps({"error": "无法解码为文本(可能是二进制文件)"}, ensure_ascii=False)

    # read-before-write 校验
    current_hash = _content_hash(content)
    if expected_hash and expected_hash != current_hash:
        return json.dumps({
            "error": (
                f"read-before-write 校验失败:文件已被外部修改"
                f"(expected={expected_hash}, current={current_hash})。"
                f"请重新调 read_file 拿最新内容再 str_replace。"
            ),
            "error_type": "stale_hash",
            "current_hash": current_hash,
        }, ensure_ascii=False)

    # 检查 old_str 是否存在
    occurrences = content.count(old_str)
    if occurrences == 0:
        return json.dumps({
            "error": "old_str 在文件里找不到。请检查拼写或重新 read_file。",
            "error_type": "old_str_not_found",
        }, ensure_ascii=False)
    if not replace_all and occurrences > 1:
        return json.dumps({
            "error": f"old_str 在文件里有 {occurrences} 处匹配,不唯一。"
                     f"传 replace_all=true 全部替换,或把 old_str 写得更具体。",
            "error_type": "ambiguous_match",
            "occurrences": occurrences,
        }, ensure_ascii=False)

    # 执行替换
    if replace_all:
        new_content = content.replace(old_str, new_str)
        replaced = occurrences
    else:
        new_content = content.replace(old_str, new_str, 1)
        replaced = 1

    try:
        path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    _track_checkpoint(path, kwargs)  # /rewind 追踪该文件

    return json.dumps({
        "path": str(path),
        "replaced": replaced,
        "content_hash": _content_hash(new_content),
    }, ensure_ascii=False)


registry.register(
    name="str_replace",
    toolset="core",
    schema=STR_REPLACE_SCHEMA,
    handler=_handle_str_replace,
    emoji="🔄",
)
