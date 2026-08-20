"""文件操作工具集：read_file / write_file / search_files / str_replace / notebook_edit。

这是 agent 最常用的"手"——LLM 看代码、改代码全靠这几个工具。
它们在本文件模块顶层向 registry（中央工具注册表）登记自己。

- read_file：读文件，带行号返回（方便 LLM 说"第几行"）；同时返回 content_hash
    （内容的指纹），之后 write 时带上可防"盲写"——文件被别人改过了还照写不误
- write_file：写文件（整文件覆盖或追加），支持 read-before-write 指纹校验
    （写之前先确认文件还是你上次读到的样子，被外部改过就拒绝写）
- search_files：在文件内容里搜关键词（类似 grep），支持分页
- str_replace：在文件里做子串替换（单次或全部），改 1 行不用整个读再整个写
- notebook_edit：编辑 Jupyter notebook（.ipynb）的单个单元格
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from agent.output_offload import finalize_tool_output as _finalize_output
from agent.permission import safe_path
from tools._common import get_mode_override_from_kwargs
from tools.registry import registry


def _content_hash(text: str) -> str:
    """给文本算一个短指纹（sha256 取前 16 位）。

    用途：read_file 返回指纹 → write_file/str_replace 带上它 → 写前比对，
    不一致说明文件被外部改过，拒绝写入（防止基于过时内容的"盲写"）。

    参数：
        text: 文本内容。

    返回：16 个字符的十六进制指纹串。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# R30e-H2：read 结果去重（借鉴 Claude Code 的 file_unchanged 机制）——
# 同一个文件、同一段范围（range），而且修改时间+大小都没变的话，
# 不再重复把全文塞进上下文烧 token，只回一句"文件没变"。
# 为什么用两个因子（mtime_ns 纳秒级修改时间 + size 文件大小）判断"变没变"：
# Windows 上修改时间的精度只有 ~15 毫秒，只看时间一个因子会误判"没变"
# （历史上真踩过这个坑：短时间内写过的文件被误认为没变，消息丢了）。
# 本进程自己的写入（write_file/str_replace）写完主动清缓存；
# 外部写入（比如 terminal 命令改了文件）靠双因子自然对不上而失效。
# ---------------------------------------------------------------------------
from collections import OrderedDict  # noqa: E402

_READ_SEEN_LIMIT = 200
_READ_SEEN: "OrderedDict[str, dict]" = OrderedDict()
# 记账本：键是文件的绝对路径；值形如
# {"sig": (修改时间纳秒, 大小), "ranges": {范围键: (内容指纹, 总行数)}}


def _read_range_key(offset: int, limit) -> str:
    """把"从第几行开始读、读多少行"拼成一个键，用来记账去重。

    参数：
        offset: 起始行号
        limit: 读取行数（None 表示读到末尾）

    返回：形如 "0:50" 或 "0:all" 的字符串键。
    """
    return f"{offset}:{limit if limit is not None else 'all'}"


def _read_seen_invalidate(path) -> None:
    """自己改了文件后，把该文件的"读过去重"记账划掉。

    背景：刚写完的文件内容肯定变了，去重缓存必须立刻失效，
    否则下次读会被误判成"文件没变"而不返回内容。

    参数：
        path: 刚被写过的文件路径。

    返回：无。出错也静默（去重是锦上添花，不能因为它炸掉写入）。
    """
    try:
        _READ_SEEN.pop(str(Path(path).resolve()), None)
    except Exception:
        pass


def reset_read_seen() -> None:
    """清空读过去重的记账本。测试专用（每个测试要干净起点）。"""
    _READ_SEEN.clear()


# ---------------------------------------------------------------------------
# read_file：读文件工具
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
                "description": "读取行数（默认全部）。文件 >256KB 或单次读取 >25K tokens 会报错——用 offset/limit 分段",
            },
        },
        "required": ["path"],
    },
}

# R20 #34：Read 双重上限（对齐 Claude Code 的做法——超限直接报错而不是截断）
READ_MAX_FILE_BYTES = 256 * 1024   # 第一道：文件大小预检上限（256KB），不读盘就能拒
READ_MAX_OUTPUT_TOKENS = 25_000    # 第二道：输出 token 上限（字符数/3 粗略估算）


def _handle_read_file(args: dict, **kwargs) -> str:
    """读文件工具的干活函数：读文件内容，带行号返回。

    参数：
        args: LLM 按 schema 填的参数——
            path 要读的文件路径；offset 起始行（从 0 数）；
            limit 读多少行（不传 = 读到底）
        **kwargs: 命名上下文（tool_call_id、config、omnimate_home 等，
            大输出落盘时用）

    返回：JSON 字符串——成功含 content（带行号正文）、content_hash（内容指纹）、
    total_lines 等；失败是 {"error": ..., "error_type": ...}。
    文件太大（>256KB）或输出太多（约 >25K token）会报错引导分段读，不是悄悄截断；
    同文件同范围且内容没变时返回"文件未变"的省 token 提示。
    """
    path_str = args.get("path", "")

    # 先过安全闸门：受保护的路径（如 ~/.ssh、/etc/passwd）不让读
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

    # === R20 #34：超限改为报错而不是截断（对齐 Claude Code）===
    # 为什么报错更好：CC 试过截断，结果 LLM 没意识到内容不全反复重读，
    # token 反而花得更多——直接报错让它自己分段读更省。
    # 预检第一道：文件超过 256KB，连盘都不用读直接拒
    try:
        _st = path.stat()
        file_size = _st.st_size
    except OSError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    # === R30e-H2：同文件同范围且修改时间+大小都没变 → 回"文件未变"提示 ===
    try:
        _seen_key = str(path.resolve())
        _sig = (_st.st_mtime_ns, _st.st_size)
        _rkey = _read_range_key(offset, limit)
        _seen = _READ_SEEN.get(_seen_key)
        if _seen and _seen["sig"] == _sig and _rkey in _seen["ranges"]:
            _saved_hash, _saved_lines = _seen["ranges"][_rkey]
            _READ_SEEN.move_to_end(_seen_key)
            return json.dumps({
                "path": str(path),
                "content": (
                    f"(file unchanged: {path} 自上次读取后未变化"
                    f"（mtime+size 一致），不再重复返回全文省 token；"
                    f"需要重看内容请换 offset/limit)"
                ),
                "file_unchanged": True,
                "content_hash": _saved_hash,
                "total_lines": _saved_lines,
            }, ensure_ascii=False)
    except Exception:
        pass  # 去重记账出了岔子不影响正事：照常走正常读取

    if file_size > READ_MAX_FILE_BYTES:
        return json.dumps({
            "error": (
                f"文件过大: {file_size} 字节 > 上限 {READ_MAX_FILE_BYTES}（256KB）。"
                f"用 offset/limit 分段读取，或用 search/glob 定位目标区域。"
            ),
            "error_type": "file_too_large",
            "path": str(path),
            "file_size": file_size,
        }, ensure_ascii=False)

    try:
        # 必须显式指定 utf-8：不指定的话 Windows 默认用 cp1252 编码，中文全成乱码
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # 按要求截取要读的那一段
        end = offset + limit if limit else None
        selected = lines[offset:end]

        # 每行前面加上行号（LLM 说话时能精确到"第几行"）
        numbered = [
            f"{i + offset + 1:6}\t{line}"
            for i, line in enumerate(selected)
        ]
        raw_content = "\n".join(numbered)

        # === R20 #34 第二道检查：估算输出超过 25K token → 报错（不截断）===
        # 估算方法很粗：字符数除以 3（和 estimate_message_tokens 同一套口径）。
        # 超了说明分段还不够细——报错引导再切小一点，避免大内容不打招呼就涌进上下文。
        if len(raw_content) // 3 > READ_MAX_OUTPUT_TOKENS:
            est_tokens = len(raw_content) // 3
            return json.dumps({
                "error": (
                    f"读取结果过大: 约 {est_tokens} tokens > 上限 {READ_MAX_OUTPUT_TOKENS}。"
                    f"文件共 {len(lines)} 行，本次选了 {len(selected)} 行——"
                    f"用更小的 offset/limit 分段读取。"
                ),
                "error_type": "output_too_large",
                "path": str(path),
                "total_lines": len(lines),
                "est_tokens": est_tokens,
            }, ensure_ascii=False)

        # 特别大的输出不直接塞进上下文，而是存到盘上、返回存放位置（offload 落盘）
        tool_call_id = kwargs.get("tool_call_id")
        config = kwargs.get("config")
        omnimate_home = kwargs.get("omnimate_home")

        # 防"落盘套娃"：如果正在读的文件本身就存放在 offload 目录里，
        # 就不要再为它做一次落盘了——否则会无限循环：读落盘文件 → 又落盘一个
        # 新文件 → 新文件又带一遍行号（1\t1\t1\t 这样叠罗汉）。
        # 这是压力测试时真踩过的 bug
        _skip_offload = False
        if omnimate_home:
            try:
                offload_dir = Path(omnimate_home) / ".task_outputs" / "tool-results"
                if path.resolve().is_relative_to(offload_dir.resolve()):
                    _skip_offload = True
            except Exception:
                pass

        if _skip_offload:
            final_content = raw_content
            content_offloaded = False
        else:
            final_content = _finalize_output(raw_content, tool_call_id, omnimate_home, config)
            content_offloaded = final_content != raw_content

        # R30e-H2：记下这次读取的"签名"（下次同范围且没变就回省 token 提示）
        try:
            _READ_SEEN[_seen_key] = {
                "sig": _sig,
                "ranges": {_rkey: (_content_hash(content), len(lines))},
            }
            _READ_SEEN.move_to_end(_seen_key)
            while len(_READ_SEEN) > _READ_SEEN_LIMIT:
                _READ_SEEN.popitem(last=False)
        except Exception:
            pass

        return json.dumps({
            "path": str(path),
            "content": final_content,
            "content_hash": _content_hash(content),  # 内容指纹：给 write_file 写前校验"文件没被别人动过"用
            "content_offloaded": content_offloaded,
            "total_lines": len(lines),
            "shown_lines": f"{offset + 1}-{offset + len(selected)}",
        }, ensure_ascii=False)
    except UnicodeDecodeError:
        return json.dumps({"error": "无法解码为文本（可能是二进制文件）"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# write_file：写文件工具
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
    """文件改完后，通知"存档追踪器"记一笔，支撑 /rewind（回退到之前版本）功能。

    背景：对齐 Claude Code 的 /rewind——像游戏存档一样，改过的文件可以退回去。
    只追踪编辑工具自己改的文件（write_file/str_replace）；
    bash 命令改的东西不追踪（没法可靠知道它动了哪些文件）。

    参数：
        path: 刚改过的文件路径。
        kwargs: 工具调用上下文（从里面找 agent_ref，进而找追踪器）。

    返回：无。没有追踪器（测试/子代理环境）就安静跳过，出错也不炸。
    """
    agent_ref = kwargs.get("agent_ref")
    tracker = getattr(agent_ref, "_checkpoint_track", None)
    if tracker:
        try:
            tracker(str(path))
        except Exception:
            pass


def _trigger_file_changed(path, op: str, kwargs) -> None:
    """文件写入成功后，广播一条"文件变了"事件（FILE_CHANGED hook）。

    背景（Task N 新增）：对齐 Claude Code 的 file_changed 事件——
    外部可以挂监听器（hook）做 IDE 联动、自动重载、操作记录等。
    纯通知，不关心有没有人听；出问题也不影响写文件本身（fail-open）。

    参数：
        path: 被改的文件路径。
        op: 操作类型（"write" / "append" / "edit"）。
        kwargs: 工具调用上下文（从里面找 agent_ref 和 hook 注册表）。

    返回：无。没有挂 hook（测试/子代理环境）就安静跳过。
    """
    agent_ref = kwargs.get("agent_ref")
    hooks = getattr(agent_ref, "hooks_registry", None)
    if hooks is None:
        return
    try:
        hooks.run_file_changed({
            "session_id": getattr(agent_ref, "session_id", "") if agent_ref else "",
            "path": str(path),
            "op": op,
        })
    except Exception:
        pass  # fail-open


def _handle_write_file(args: dict, **kwargs) -> str:
    """写文件工具的干活函数：把内容写进文件（覆盖或追加）。

    参数：
        args: LLM 按 schema 填的参数——
            path 要写的路径；content 要写的内容；
            append True=追加到末尾（默认 False 覆盖整个文件）；
            expected_hash 可选，read_file 返回的内容指纹，用于写前校验
        **kwargs: 命名上下文（permission_checker 权限检查器、agent_ref 等）

    返回：JSON 字符串——成功含 bytes（写入字节数）、content_hash；
    失败是 {"error": ..., "error_type": ...}（路径被拒/指纹过期等）。
    """
    path_str = args.get("path", "")
    content = args.get("content", "")
    append = bool(args.get("append", False))
    expected_hash = args.get("expected_hash")

    if not path_str:
        return json.dumps({"error": "path 不能为空"}, ensure_ascii=False)

    # 写入前的路径安检（write=True 表示按"写"的严格程度来查）：
    #   - 受保护路径（~/.ssh / /etc 这些要害部位）→ 直接拒，没商量
    #   - 在当前工作目录或 ~/.OmniMate 白名单内 → 放行
    #   - 白名单之外 → 调审批回调问用户（用户同意后可加进持久白名单）
    # 优先用外面注入的检查器（cli.py 注入的带"问用户"能力）；
    # 没注入就用全局默认（没有问询能力，白名单外一律拒）。
    # 必修 1：子代理（主对话派出去的分身）的权限模式要透传——
    # 比如 bypassPermissions 模式要能放行白名单外的路径。
    from agent.permission import get_default_checker
    checker = kwargs.get("permission_checker") or get_default_checker()
    mode_override = get_mode_override_from_kwargs(kwargs)
    perm = checker.check_path(path_str, write=True, mode_override=mode_override)
    if not perm.allowed:
        return json.dumps(
            {"error": f"路径拒绝: {perm.reason}", "error_type": "permission_denied"},
            ensure_ascii=False,
        )

    path = Path(path_str).expanduser()

    # 写前指纹校验（只在覆盖模式 + 调用方传了 expected_hash 时做）
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
            # 二进制文件没法算文本指纹，跳过校验（保持老行为兼容）
            pass

    try:
        # 目录不存在就一路建出来（比如写 a/b/c.txt 时把 a/b/ 都创建好）
        path.parent.mkdir(parents=True, exist_ok=True)

        # 同样必须显式 utf-8，防 Windows 默认编码乱码
        if append:
            with path.open("a", encoding="utf-8") as f:
                f.write(content + "\n")
        else:
            # 历史踩坑（X16 修复）：改用原子写（先写临时文件、刷盘、再一步换过去）。
            # 之前直接 write_text，写到一半程序崩了会留下半截文件——
            # 原子写要么完整的新的，要么还是旧的，绝不出现半截
            from agent.atomic_io import atomic_write_text
            atomic_write_text(path, content)

        _track_checkpoint(path, kwargs)  # 给 /rewind 存档
        _read_seen_invalidate(path)  # R30e-H2：刚写过，读去重缓存立刻作废
        _trigger_file_changed(path, "append" if append else "write", kwargs)  # 广播"文件变了"事件

        return json.dumps({
            "path": str(path),
            "bytes": len(content.encode("utf-8")),
            "appended": append,
            "content_hash": _content_hash(content) if not append else None,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# search_files：文件内容搜索工具
# ---------------------------------------------------------------------------

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": (
        "在目录中搜索文件内容（类似 grep）。"
        "返回匹配的行和文件路径；支持 offset 分页翻看更多结果。"
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
                "description": "本页返回的最大匹配数（默认 50，配合 offset 翻页）",
                "default": 50,
            },
            "offset": {
                "type": "integer",
                "description": "跳过前 N 个匹配（分页翻看：第 2 页传 offset=50，第 3 页 offset=100……）",
                "default": 0,
            },
            "context": {
                "type": "integer",
                "description": "每个匹配附带前后各 N 行上下文（0-10，默认 0）",
                "default": 0,
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "大小写不敏感匹配（默认 False）",
                "default": False,
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

# 默认不搜索的目录（依赖包、缓存、版本控制——这些不是用户写的代码，搜了纯浪费）
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
    """判断某个文件是不是待在"不搜索名单"里的目录中（比如在 .git/ 或 node_modules/ 里）。

    参数：
        file_path: 要判断的文件路径。
        search_root: 本次搜索的起点目录。

    返回：True = 文件在被排除目录里（该跳过）；False = 不在（或压根不在搜索范围内）。
    """
    try:
        rel = file_path.relative_to(search_root)
    except ValueError:
        return False
    return any(part in _DEFAULT_EXCLUDED_DIRS for part in rel.parts)


def _handle_search_files(args: dict, **kwargs) -> str:
    """搜索工具的干活函数：在目录下的文件内容里找匹配（类似 grep）。

    参数：
        args: LLM 按 schema 填的参数——
            pattern 要搜的正则表达式；path 在哪个目录搜（默认当前目录）；
            glob 只搜哪些文件（如 "*.py"，默认全部）；
            max_matches 本页最多返回多少条（配合 offset 翻页）；
            offset 跳过前 N 条（翻页用）；context 每条匹配带前后各几行上下文；
            case_insensitive 是否忽略大小写；include_hidden 是否连隐藏/依赖目录也搜
        **kwargs: 命名上下文（本函数未用到，保持统一签名）

    返回：JSON 字符串——成功含 matches（匹配列表）和翻页提示；
    失败是 {"error": ...}（正则非法/路径被拒/路径不存在等）。
    """
    pattern = args.get("pattern", "")
    search_path = Path(args.get("path") or ".").expanduser()
    file_glob = args.get("glob") or "**/*"
    max_matches = max(1, int(args.get("max_matches", 50)))
    include_hidden = bool(args.get("include_hidden", False))
    # R30e-H3：加的能力——offset 翻页、匹配附带上下文行、大小写开关
    page_offset = max(0, int(args.get("offset", 0) or 0))
    context = min(10, max(0, int(args.get("context", 0) or 0)))
    ignore_case = bool(args.get("case_insensitive", False))

    if not pattern:
        return json.dumps({"error": "pattern 不能为空"}, ensure_ascii=False)

    # 历史踩坑（S3 修复）：search_files 以前没过安全闸门，
    # 能把 ~/.ssh/id_rsa 私钥的内容片段搜出来——现在必须先过 safe_path 安检
    from agent.permission import safe_path
    perm = safe_path(search_path, write=False)
    if not perm.allowed:
        return json.dumps({
            "error": f"权限拒绝: {perm.reason}",
            "error_type": "permission_denied",
            "gate": perm.gate,
            "path": str(search_path),
        }, ensure_ascii=False)

    if not search_path.exists():
        return json.dumps({"error": f"路径不存在: {search_path}"}, ensure_ascii=False)

    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return json.dumps({"error": f"非法正则: {e}"}, ensure_ascii=False)

    matches = []
    files_searched = 0
    files_skipped = 0
    has_more = False
    # 收够"跳过数+每页数+1"条就停（多拿的那 1 条只为判断"还有没有下一页"，
    # 不需要把总数全数一遍——大目录数总数太浪费）
    collect_limit = page_offset + max_matches + 1
    try:
        # R30e-H3：文件列表先排序——翻页要求两次调用之间顺序稳定，
        # 否则第 2 页可能重复或漏掉第 1 页的内容
        for file_path in sorted(search_path.glob(file_glob)):
            if not file_path.is_file():
                continue
            # 默认跳过 .git/.venv/__pycache__/node_modules 这些目录（除非显式要求搜）
            if not include_hidden and _is_in_excluded_dir(file_path, search_path):
                files_skipped += 1
                continue
            # 超过 1MB 的大文件直接跳过（多半是二进制或数据文件，搜了也没意义）
            if file_path.stat().st_size > 1_000_000:
                continue
            files_searched += 1
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            all_lines = content.splitlines()
            for line_no, line in enumerate(all_lines, 1):
                if regex.search(line):
                    m = {
                        "file": str(file_path),
                        "line": line_no,
                        "content": line[:500],  # 太长的行只留前 500 字符
                    }
                    if context > 0:
                        m["before"] = [
                            f"{n}: {all_lines[n - 1][:200]}"
                            for n in range(max(1, line_no - context), line_no)
                        ]
                        m["after"] = [
                            f"{n}: {all_lines[n - 1][:200]}"
                            for n in range(line_no + 1,
                                           min(len(all_lines), line_no + context) + 1)
                        ]
                    matches.append(m)
                    if len(matches) >= collect_limit:
                        has_more = True
                        break
            if has_more:
                break
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    total_collected = len(matches)
    page = matches[page_offset: page_offset + max_matches]
    result = {
        "matches": page,
        "match_count": len(page),
        "shown_range": (
            f"{page_offset + 1}-{page_offset + len(page)}"
            if page else "0-0"
        ),
        "files_searched": files_searched,
        "files_skipped_hidden": files_skipped,
    }
    if has_more or page_offset + max_matches < total_collected:
        result["truncated"] = True
        result["pagination_hint"] = (
            f"还有更多匹配——传 offset={page_offset + max_matches} 翻下一页"
        )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 模块加载时向中央注册表登记（import 本文件即生效）
# ---------------------------------------------------------------------------

registry.register(
    name="read_file",
    toolset="core",
    schema=READ_FILE_SCHEMA,
    handler=_handle_read_file,
    emoji="📄",
    isConcurrencySafe=True,  # 只读不动手：可以和其他工具同时跑
)

registry.register(
    name="write_file",
    toolset="core",
    schema=WRITE_FILE_SCHEMA,
    handler=_handle_write_file,
    emoji="✏️",
    isConcurrencySafe=False,  # 真会改文件：必须排队一个一个来
)

registry.register(
    name="search_files",
    toolset="core",
    schema=SEARCH_FILES_SCHEMA,
    handler=_handle_search_files,
    emoji="🔍",
    isConcurrencySafe=True,  # 只读不动手：可以和其他工具同时跑
)


# ---------------------------------------------------------------------------
# str_replace：子串替换工具
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
    """子串替换工具的干活函数：在文件里把一段文字换成另一段。

    参数：
        args: LLM 按 schema 填的参数——
            path 文件路径；old_str 要被替换的文字（必须能在文件里找到）；
            new_str 换成什么（空串表示删除）；replace_all True=全部替换/False=只换第一处；
            expected_hash 可选指纹，做写前校验
        **kwargs: 命名上下文（permission_checker、agent_ref 等）

    返回：JSON 字符串——成功含 replaced（换了几处）和新指纹；
    失败是 {"error": ..., "error_type": ...}（找不到/匹配不唯一/指纹过期等）。
    """
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

    # 路径安检（按"写"的严格程度走，和 write_file 同款）
    # 必修 1：子代理（主对话派出去的分身）的权限模式要透传。
    from agent.permission import get_default_checker
    checker = kwargs.get("permission_checker") or get_default_checker()
    mode_override = get_mode_override_from_kwargs(kwargs)
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

    # 写前指纹校验（文件被外部改过就拒绝动手）
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

    # 先看要替换的文字在不在、有几处
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

    # 真正动手替换
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

    _track_checkpoint(path, kwargs)  # 给 /rewind 存档
    _read_seen_invalidate(path)  # R30e-H2：刚改过，读去重缓存立刻作废
    _trigger_file_changed(path, "edit", kwargs)  # 广播"文件变了"事件

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
    isConcurrencySafe=False,  # 真会改文件：必须排队一个一个来
)


# ---------------------------------------------------------------------------
# NotebookEdit（R20 #35，对齐 Claude Code 的 NotebookEdit：编辑 Jupyter notebook 单元格）
# ---------------------------------------------------------------------------

NOTEBOOK_EDIT_SCHEMA = {
    "name": "notebook_edit",
    "description": (
        "编辑 Jupyter notebook（.ipynb）的单个单元格。"
        "edit_mode: replace（替换内容）/ insert（插入新格）/ delete（删除）。"
        "cell_id 匹配 cell 的 id 字段；也接受纯数字（按索引）。"
        "insert 不给 cell_id 时追加到末尾。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "notebook_path": {"type": "string", "description": "notebook 文件路径（.ipynb）"},
            "cell_id": {
                "type": "string",
                "description": "目标单元格 id 或数字索引（replace/delete 必需；insert 可选=末尾）",
            },
            "new_source": {"type": "string", "description": "新内容（replace/insert）"},
            "cell_type": {
                "type": "string",
                "enum": ["code", "markdown", "raw"],
                "description": "insert 时的单元格类型",
            },
            "edit_mode": {
                "type": "string",
                "enum": ["replace", "insert", "delete"],
                "description": "编辑模式（默认 replace）",
            },
        },
        "required": ["notebook_path"],
    },
}


def _find_cell_index(cells: list, cell_id: str) -> int:
    """按单元格的 id 或数字序号找到它在列表里的位置。

    参数：
        cells: notebook 的单元格列表。
        cell_id: 要找的 id（也接受纯数字，按第几个算，从 0 数）。

    返回：位置下标；找不到返回 -1。
    """
    for i, cell in enumerate(cells):
        if str(cell.get("id", "")) == cell_id:
            return i
    try:
        idx = int(cell_id)
        if 0 <= idx < len(cells):
            return idx
    except ValueError:
        pass
    return -1


def _handle_notebook_edit(args: dict, **kwargs) -> str:
    """notebook 编辑工具的干活函数：改 Jupyter notebook（.ipynb）的一个单元格。

    支持三种玩法：replace（换内容）/ insert（插入新格）/ delete（删格）。
    单元格定位用它的 id 字段，也接受纯数字按序号找；insert 不给位置就加到末尾。

    参数：
        args: LLM 按 schema 填的参数——
            notebook_path 文件路径；cell_id 目标单元格 id 或序号；
            new_source 新内容；cell_type 插入时的类型（code/markdown/raw）；
            edit_mode 模式（默认 replace）
        **kwargs: 命名上下文（permission_checker、agent_ref 等）

    返回：JSON 字符串——成功含 action（干了什么）和 total_cells（总格数）；
    失败是 {"error": ..., "error_type": ...}。
    """
    path_str = args.get("notebook_path", "")
    cell_id = str(args.get("cell_id", "") or "")
    new_source = args.get("new_source", "")
    cell_type = args.get("cell_type", "code")
    edit_mode = args.get("edit_mode", "replace")

    # 安检：和 write_file 同款（按"写"的严格程度过路径白名单）
    from agent.permission import get_default_checker
    checker = kwargs.get("permission_checker") or get_default_checker()
    mode_override = get_mode_override_from_kwargs(kwargs)
    perm = checker.check_path(path_str, write=True, mode_override=mode_override)
    if not perm.allowed:
        return json.dumps(
            {"error": f"路径拒绝: {perm.reason}", "error_type": "permission_denied"},
            ensure_ascii=False,
        )

    path = Path(path_str).expanduser()
    if not path.exists():
        return json.dumps({"error": f"文件不存在: {path}"}, ensure_ascii=False)
    if path.suffix.lower() != ".ipynb":
        return json.dumps({
            "error": f"不是 notebook 文件（.ipynb）: {path}", "error_type": "invalid_args",
        }, ensure_ascii=False)

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return json.dumps({"error": f"notebook 解析失败: {e}"}, ensure_ascii=False)

    cells = nb.get("cells")
    if not isinstance(cells, list):
        return json.dumps({"error": "notebook 缺 cells 数组（结构异常）"}, ensure_ascii=False)

    # 老版本 notebook 的单元格没有 id 字段，这里补上（nbformat 4.5+ 的标准要求）。
    # 起名用 cell_<序号>，避免和文件里已有的 id 撞车
    for i, cell in enumerate(cells):
        if isinstance(cell, dict) and not cell.get("id"):
            cell["id"] = f"cell_{i}"

    if edit_mode == "insert":
        if cell_type not in ("code", "markdown", "raw"):
            return json.dumps({
                "error": f"insert 需要合法 cell_type（code/markdown/raw）: {cell_type}",
                "error_type": "invalid_args",
            }, ensure_ascii=False)
        new_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": str(new_source),
        }
        if cell_type == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None
        else:
            new_cell["id"] = f"cell_new_{len(cells)}"
        # 新建的 code 单元格也统一补上 id（上面的循环只照顾了文件里原有的单元格）
        if "id" not in new_cell:
            new_cell["id"] = f"cell_new_{len(cells)}"
        if cell_id:
            idx = _find_cell_index(cells, cell_id)
            insert_at = idx if idx >= 0 else len(cells)
        else:
            insert_at = len(cells)
        cells.insert(insert_at, new_cell)
        action_desc = f"insert {cell_type} @ {insert_at} (id={new_cell['id']})"
    else:
        if not cell_id:
            return json.dumps({
                "error": f"{edit_mode} 需要 cell_id", "error_type": "invalid_args",
            }, ensure_ascii=False)
        idx = _find_cell_index(cells, cell_id)
        if idx < 0:
            return json.dumps({
                "error": f"未找到 cell: {cell_id}", "error_type": "cell_not_found",
            }, ensure_ascii=False)
        if edit_mode == "delete":
            removed = cells.pop(idx)
            action_desc = f"delete @ {idx} (id={removed.get('id', '?')})"
        else:  # replace
            cells[idx]["source"] = str(new_source)
            action_desc = f"replace @ {idx} (id={cells[idx].get('id', '?')})"

    # 原子写回（临时文件+一步替换，写一半崩了也不会毁原文件；
    # 缩进用 1 是 nbformat 的惯例，末尾带换行）
    from agent.atomic_io import atomic_write_text
    try:
        atomic_write_text(path, json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    except Exception as e:
        return json.dumps({"error": f"写回失败: {e}"}, ensure_ascii=False)

    _track_checkpoint(path, kwargs)
    _trigger_file_changed(path, "edit", kwargs)

    return json.dumps({
        "path": str(path),
        "action": action_desc,
        "total_cells": len(cells),
    }, ensure_ascii=False)


registry.register(
    name="notebook_edit",
    toolset="core",
    schema=NOTEBOOK_EDIT_SCHEMA,
    handler=_handle_notebook_edit,
    emoji="📓",
    isConcurrencySafe=False,  # 真会改文件：必须排队一个一个来
)
