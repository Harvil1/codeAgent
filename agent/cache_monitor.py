"""prompt 缓存（LLM 服务商对"开头不变的部分"打折复用的机制）的破坏检测器。

借鉴 claude-code-main 的 promptCacheBreakDetection。背景：缓存命中省钱
又提速，一旦开头内容变了缓存就全废（费用翻倍），所以要有个"哨兵"盯着
每次调用、发现缓存掉了就报告是哪里变了。

工作流程（三步）：
1. 调 API 前：record_prompt_state 给 prompt 的各个维度拍快照
2. 调 API 后：check_cache_break 对比这次实际读到的缓存 token 数，掉了就查根因
3. 确认缓存被破坏：打日志 + 记进历史 + 写 diff 文件（/cache-stats 命令展示用）

盯的 12 个维度（对齐 claude-code-main promptCacheBreakDetection.ts）：
  1. system_hash（系统提示哈希）   2. tools_hash（工具集哈希）  3. model（模型名）
  4. cache_strategy               5. betas_hash                6. max_tokens
  7. temperature                  8. stream_mode（是否流式）   9. tool_choice
  10. user_content_prefix（用户消息前缀） 11. messages_count    12. system_boundary

CCAR4 Task A 的扩展：
  - per-tool hash（每个工具单独记哈希，能指出具体是哪个工具变了）
  - diff 文件落盘（~/.OmniMate/.cache-breaks/cache-break-*.diff）
  - TTL 时长分析（区分 5 分钟 / 1 小时缓存过期）

整个模块绝不打断主流程（fail-open：任何异常只记日志不抛出）。
"""
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolHashEntry:
    """单个工具的指纹记录（per-tool hash 用的最小单元）。

    name: 工具名（function.name）
    schema_hash: 参数 schema 的哈希值（用来发现 schema 变化）
    """
    name: str = ""
    schema_hash: int = 0


@dataclass
class PromptState:
    """调 API 前给 prompt 各维度拍的快照（12 维度 + 每个工具各自的哈希）。

    有了快照，下次缓存掉了就能逐项对比找出"是谁变了"。维度清单对齐
    claude-code-main promptCacheBreakDetection.ts。
    """
    # 最早实现的 5 个核心维度
    system_hash: int = 0
    tools_hash: int = 0
    model: str = ""
    cache_strategy: str = ""
    betas_hash: int = 0
    # CCAR4 Task A 补充的 7 个维度
    max_tokens: int = 0
    temperature: Optional[float] = None
    stream_mode: bool = False
    tool_choice: Optional[str] = None  # "auto" / "none" / 或指定某个工具名
    user_content_prefix: int = 0  # 第一条 user 消息前 N 个字符的哈希
    messages_count: int = 0
    system_boundary: str = ""  # system 是单个字符串还是多块列表
    system_len: int = 0  # system prompt 长度（只存长度省内存，用于显示增减量）
    # 每个工具单独记一条哈希，缓存掉了能精准点名是哪个工具变了
    tool_hashes: List[ToolHashEntry] = field(default_factory=list)


# 模块级状态（沿用 brief 的约定：一个进程一个会话，切会话时整体重置）
_last_state: Optional[PromptState] = None
_last_cache_read: Optional[int] = None
_pending_compaction: bool = False  # 刚做过压缩，下次缓存下降是预期内的
_break_history: List[dict] = []  # 破坏事件累计（/cache-stats 展示用）
_BREAK_HISTORY_LIMIT: int = 100  # 历史上限，防长会话内存越攒越多
_last_baseline_at: Optional[float] = None  # 上次基线的设定时间（分析缓存是否过期用）
_diff_counter: int = 0  # diff 文件名的递增序号（防同一秒写文件互相覆盖）


def _hash_content(content: Any) -> int:
    """把任意内容算成一个整数哈希，之后比较"变没变"就比整数，飞快。

    参数：
        content: 任意内容（字符串、列表、字典都行）

    返回：哈希整数；算不出来时返回 0（不抛异常）。
    """
    if isinstance(content, str):
        return hash(content)
    try:
        return hash(str(content))
    except Exception:
        return 0


def record_prompt_state(
    *,
    system_prompt: Any,
    tools: list,
    model: str,
    max_tokens: int = 0,
    temperature: Optional[float] = None,
    stream_mode: bool = False,
    tool_choice: Optional[str] = None,
    user_content_prefix: str = "",
    betas: Optional[dict] = None,
    **kwargs,
) -> PromptState:
    """调 API 前给 prompt 各维度拍快照（12 维度 + 每工具哈希）。

    背景：check_cache_break 要拿"上一次的快照"和这一次对比，所以每次
    调用前都得先来这儿登记一遍。

    参数：
        system_prompt: 系统提示（字符串，或多个文本块的列表）
        tools: 工具 schema 列表（OpenAI 格式）
        model: 模型名
        max_tokens: 最大输出 token 数
        temperature: 采样温度
        stream_mode: 是否流式调用
        tool_choice: tool_choice 参数（强制模型用/不用某工具）
        user_content_prefix: 第一条 user 消息的开头文字（用来发现 user 消息被改）
        betas: beta 请求头字典（如 anthropic_beta）
        **kwargs: 兼容扩展字段（cache_strategy / messages_count 等）

    返回：填好 12 维度的 PromptState 快照。任何异常都吞掉、返回一份
    默认空快照（绝不影响主流程）。
    """
    try:
        # 给每个工具单独算哈希，之后能点名到具体哪个工具变了
        tool_hashes: List[ToolHashEntry] = []
        for t in (tools or []):
            fn = t.get("function", {}) if isinstance(t, dict) else getattr(t, "function", None)
            if not fn:
                continue
            if isinstance(fn, dict):
                name = fn.get("name", "")
                schema = fn.get("input_schema", {}) or fn.get("parameters", {})
            else:
                name = getattr(fn, "name", "")
                schema = getattr(fn, "input_schema", {}) or getattr(fn, "parameters", {})
            tool_hashes.append(ToolHashEntry(
                name=name or "",
                schema_hash=_hash_content(schema),
            ))

        # user 消息前缀只取前 500 字算哈希（够识别变化了）
        ucp_str = (user_content_prefix or "")[:500]

        # system 只记长度不存原文——够显示"变长/变短了多少"就行，省内存
        if isinstance(system_prompt, list):
            _sys_len = sum(
                len(b.get("text", "")) if isinstance(b, dict) else len(str(b))
                for b in system_prompt
            )
        else:
            _sys_len = len(system_prompt or "")

        state = PromptState(
            system_hash=_hash_content(system_prompt),
            tools_hash=_hash_content([(t.name, t.schema_hash) for t in tool_hashes]),
            model=model or "",
            cache_strategy=kwargs.get("cache_strategy", ""),
            betas_hash=_hash_content(betas or {}),
            max_tokens=max_tokens or 0,
            temperature=temperature,
            stream_mode=stream_mode,
            tool_choice=tool_choice,
            user_content_prefix=_hash_content(ucp_str),
            messages_count=kwargs.get("messages_count", 0),
            system_boundary="multi-block" if isinstance(system_prompt, list) else "single",
            system_len=_sys_len,
            tool_hashes=tool_hashes,
        )
        return state
    except Exception as e:
        logger.debug("record_prompt_state fail-open: %s", e)
        return PromptState()


def check_cache_break(
    *,
    current_state: PromptState,
    cache_read_tokens: int,
    query_source: str = "",
) -> Optional[str]:
    """调 API 后检查缓存是不是被破坏了，是就查出根因。

    背景：LLM 返回的响应里带着"这次命中了多少缓存 token"，拿它跟上一次
    对比——掉了说明缓存失效了，得找出是哪个维度的变化干的。

    判定标准：读到的缓存 token 比上次降了 5% 以上**并且**绝对量超过
    2000 tokens，才算真破坏（小抖动忽略，避免误报）。

    参数：
        current_state: 本次调用前拍的快照（record_prompt_state 的产物）
        cache_read_tokens: 本次响应实际读到的缓存 token 数
        query_source: 这次调用的来源标记（记进历史方便排查）

    返回：None 表示没破坏；否则返回根因描述字符串。任何异常都吞掉
    返回 None（绝不影响主流程）。
    """
    global _last_state, _last_cache_read, _pending_compaction, _last_baseline_at
    try:
        # 防御：缓存 token 数不可能是负数或 None，出现就当 0 处理
        if cache_read_tokens is None or cache_read_tokens < 0:
            cache_read_tokens = 0

        # 刚做过压缩（compact），缓存下降是预期内的，不算破坏
        if _pending_compaction:
            _pending_compaction = False
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            _last_baseline_at = time.time()
            return None

        # 第一次调用，还没有可比的基线，只登记不判定
        if _last_state is None or _last_cache_read is None:
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            _last_baseline_at = time.time()
            return None

        # 判定：降幅超 5% 且绝对量超 2000 tokens 才算破坏
        token_drop = _last_cache_read - cache_read_tokens
        if cache_read_tokens >= _last_cache_read * 0.95 or token_drop < 2000:
            # 不算破坏，刷新基线走人
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            _last_baseline_at = time.time()
            return None

        # 真破坏：逐维度查根因 + 找出哪个工具变了 + 分析是不是缓存过期
        reasons = _diagnose_break(current_state, _last_state, token_drop)

        # 写 diff 文件（仅当 system 或工具变了）——单独 try/except，失败也不影响后续
        diff_path = None
        try:
            diff_path = _write_break_diff(_last_state, current_state, reasons)
        except Exception as de:
            logger.debug("_write_break_diff fail-open: %s", de)

        # 记进历史（限长，防长会话内存膨胀）
        _break_history.append({
            "from": _last_cache_read,
            "to": cache_read_tokens,
            "drop": token_drop,
            "root_cause": "; ".join(reasons) if reasons else "未知原因",
            "query_source": query_source,
            "diff_path": diff_path,
        })
        if len(_break_history) > _BREAK_HISTORY_LIMIT:
            del _break_history[0:len(_break_history) - _BREAK_HISTORY_LIMIT]

        # 清理过量的 diff 文件（只在真写了 diff 时才清，省得白做 IO）
        if diff_path:
            _enforce_diff_lru_limit()

        # 刷新基线
        prev_read = _last_cache_read
        _last_state = current_state
        _last_cache_read = cache_read_tokens
        _last_baseline_at = time.time()

        root_cause = "; ".join(reasons) if reasons else "未知原因"
        logger.warning(
            "prompt cache break! cache read %d -> %d (drop %d tokens). cause: %s%s",
            prev_read, cache_read_tokens, token_drop, root_cause,
            f" diff: {diff_path}" if diff_path else "",
        )
        return root_cause
    except Exception as e:
        logger.debug("check_cache_break fail-open: %s", e)
        return None


def _diagnose_break(current: PromptState, prev: PromptState, token_drop: int) -> list:
    """对比新旧两份快照，找出缓存被破坏的可能原因。

    做法：12 个维度逐个比哈希；哪个变了就记一条原因。工具集变了再细查
    是哪个工具；所有字段都没变，就按"距上次基线过了多久"判断是不是
    缓存自己过期了（服务商缓存有 5 分钟 / 1 小时的保质期）。

    参数：
        current: 本次快照
        prev: 上次快照
        token_drop: 缓存 token 掉了多少（记录用）

    返回：根因描述列表（可能多条）。
    """
    reasons = []

    # 12 个维度逐个比
    if current.system_hash != prev.system_hash:
        delta = current.system_len - prev.system_len
        reasons.append(
            f"system prompt 变了 ({'+' if delta >= 0 else ''}{delta} chars)"
        )
    if current.tools_hash != prev.tools_hash:
        # 工具集整体哈希变了——细查具体是哪个工具
        tool_diff = _diff_tool_hashes(current.tool_hashes, prev.tool_hashes)
        reasons.append(f"工具 schema 变了 ({tool_diff})")
    if current.model != prev.model:
        reasons.append(f"model 变了 ({prev.model} → {current.model})")
    if current.max_tokens != prev.max_tokens:
        reasons.append(f"max_tokens 变了 ({prev.max_tokens} → {current.max_tokens})")
    if current.temperature != prev.temperature:
        reasons.append(f"temperature 变了 ({prev.temperature} → {current.temperature})")
    if current.stream_mode != prev.stream_mode:
        reasons.append(f"stream 模式变 ({prev.stream_mode} → {current.stream_mode})")
    if current.tool_choice != prev.tool_choice:
        reasons.append(f"tool_choice 变了 ({prev.tool_choice} → {current.tool_choice})")
    if current.user_content_prefix != prev.user_content_prefix:
        reasons.append("user content prefix 变了")
    if current.messages_count != prev.messages_count:
        reasons.append(
            f"messages count 变了 ({prev.messages_count} → {current.messages_count})"
        )
    if current.system_boundary != prev.system_boundary:
        reasons.append(
            f"system 边界变 ({prev.system_boundary} → {current.system_boundary})"
        )
    if current.betas_hash != prev.betas_hash:
        reasons.append("betas 变了")
    if current.cache_strategy != prev.cache_strategy:
        reasons.append(
            f"cache_strategy 变了 ({prev.cache_strategy} → {current.cache_strategy})"
        )

    # 缓存保质期分析：字段都没变但缓存掉了，多半是过期（5 分钟 / 1 小时档）
    global _last_baseline_at
    if not reasons:
        elapsed = time.time() - (_last_baseline_at or time.time())
        if elapsed > 3600:
            reasons.append(
                f"无字段变化但 break（>1h，可能 1h TTL 过期，elapsed={int(elapsed)}s）"
            )
        elif elapsed > 300:
            reasons.append(
                f"无字段变化但 break（>5min，可能 5min TTL 过期，elapsed={int(elapsed)}s）"
            )
        else:
            reasons.append(
                f"无字段变化（server-side 或未知，elapsed={int(elapsed)}s）"
            )

    return reasons


def _diff_tool_hashes(current: List[ToolHashEntry], prev: List[ToolHashEntry]) -> str:
    """对比两份"每工具哈希"清单，用 +新增/-删除/~修改 的形式描述差异。

    参数：
        current: 本次调用的工具哈希列表
        prev: 上次调用的工具哈希列表

    返回：如 "+2 (a,b) -1 (c)" 的差异描述；两边工具名单一样但整体哈希
    不同这种罕见情况，返回一句说明。
    """
    cur_names = {t.name: t.schema_hash for t in (current or [])}
    prev_names = {t.name: t.schema_hash for t in (prev or [])}
    added = set(cur_names) - set(prev_names)
    removed = set(prev_names) - set(cur_names)
    changed = [n for n in cur_names if n in prev_names and cur_names[n] != prev_names[n]]
    parts = []
    if added:
        parts.append(f"+{len(added)} ({','.join(sorted(added)[:3])})")
    if removed:
        parts.append(f"-{len(removed)} ({','.join(sorted(removed)[:3])})")
    if changed:
        parts.append(f"~{len(changed)} ({','.join(sorted(changed)[:3])})")
    return " ".join(parts) if parts else "schema 全等但 tools_hash 变了"


def _write_break_diff(
    prev: PromptState, cur: PromptState, reasons: List[str],
) -> Optional[str]:
    """缓存被破坏时，把前后差异写进 ~/.OmniMate/.cache-breaks/ 下的文件。

    背景：光看日志里一行原因不够直观，落个文件方便事后翻查对比。

    参数：
        prev: 破坏前的快照
        cur: 破坏后的快照
        reasons: 已查出的根因列表

    返回：写好的 diff 文件路径；没写（无根因 / system 和工具都没变 /
    写失败）返回 None。其他维度变化对排查帮助小，只在 system 或工具
    变了时才写。
    """
    global _diff_counter
    try:
        if not reasons:
            return None
        # 只有 system 或工具变了才值得写 diff
        if not any("system" in r or "工具" in r for r in reasons):
            return None

        from constants import get_omnimate_home
        diff_dir = get_omnimate_home() / ".cache-breaks"
        diff_dir.mkdir(parents=True, exist_ok=True)

        _diff_counter += 1
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        diff_path = diff_dir / f"cache-break-{ts}-{_diff_counter:04d}.diff"

        # 简化版 diff（工具级 + system 哈希对比，不是逐行文本 diff）
        lines = [
            f"# Cache break at {ts}",
            f"# Reasons: {'; '.join(reasons)}",
            f"# Token drop context: see /cache-stats",
            "",
        ]
        if prev.system_hash != cur.system_hash:
            lines.append("## system prompt")
            lines.append(f"OLD hash: {prev.system_hash}")
            lines.append(f"NEW hash: {cur.system_hash}")
            lines.append(
                f"boundary: {prev.system_boundary} → {cur.system_boundary}"
            )
            lines.append("")
        if prev.tools_hash != cur.tools_hash:
            lines.append("## tools schema")
            lines.append(
                f"OLD tools: {[t.name for t in (prev.tool_hashes or [])]}"
            )
            lines.append(
                f"NEW tools: {[t.name for t in (cur.tool_hashes or [])]}"
            )
            # 附上具体哪个工具新增/删除/修改的明细
            tool_diff = _diff_tool_hashes(cur.tool_hashes, prev.tool_hashes)
            lines.append(f"diff: {tool_diff}")
            lines.append("")

        diff_path.write_text("\n".join(lines), encoding="utf-8")
        return str(diff_path)
    except Exception as e:
        logger.debug("write_break_diff fail-open: %s", e)
        return None


def _enforce_diff_lru_limit() -> None:
    """控制 diff 文件总量不超上限（默认 100 个，超了删最旧的）。

    背景：diff 文件会越攒越多，得有人扫地，删的时候留新的删旧的。

    返回：无。删任何文件失败都静默忽略，整体绝不抛异常。
    """
    try:
        from constants import get_omnimate_home
        # 上限从配置读 max_cache_break_diff_files，默认 100
        limit = 100  # 兜底值；实际配置经 _read_diff_limit() 读
        try:
            limit = _read_diff_limit()
        except Exception:
            pass

        diff_dir = get_omnimate_home() / ".cache-breaks"
        if not diff_dir.exists():
            return
        diff_files = list(diff_dir.glob("cache-break-*.diff"))
        if len(diff_files) <= limit:
            return
        # 按修改时间排序，删最旧的
        diff_files.sort(key=lambda p: p.stat().st_mtime)
        to_delete = diff_files[:len(diff_files) - limit]
        for f in to_delete:
            try:
                f.unlink()
            except Exception:
                pass
    except Exception as e:
        logger.debug("enforce_diff_lru_limit fail-open: %s", e)


# 模块级 diff 上限（AIAgent 启动时可覆盖）
# 必须放在 _read_diff_limit 前面定义（先定义后使用）
_diff_limit: int = 100


def _read_diff_limit() -> int:
    """读 diff 文件数量上限（默认 100）。

    背景：本模块拿不到 config 实例，也不直接 import config（避免循环
    依赖），所以把上限存在模块级变量 _diff_limit 里，由 AIAgent.__init__
    读配置（约定键 DEFAULT_CONFIG["context"]["max_cache_break_diff_files"]）
    后调 set_diff_limit 写进来。

    返回：上限值；读不到时 100。
    """
    try:
        return _diff_limit
    except Exception:
        return 100


def set_diff_limit(limit: int) -> None:
    """设置 diff 文件数量上限（AIAgent.__init__ 从 config 读完后调这个）。

    参数：
        limit: 上限值（小于 1 按 1 算；转换失败静默忽略）
    """
    global _diff_limit
    try:
        _diff_limit = max(1, int(limit))
    except Exception:
        pass


def notify_compaction() -> None:
    """打个招呼：下次缓存下降是我们自己干的，别报警。

    背景：压缩（compact）会改写消息历史，下次读到的缓存 token 必然掉一截，
    但这不是"缓存被破坏"。做完压缩后调一下本函数，check_cache_break
    就会跳过下一次的破坏判定。
    """
    global _pending_compaction
    _pending_compaction = True


def get_stats() -> dict:
    """拿缓存监控的统计数字（给 /cache-stats 命令展示用）。

    返回：字典，含破坏总次数、最近一次破坏详情、最近一次缓存读取量。
    """
    return {
        "total_breaks": len(_break_history),
        "last_break": _break_history[-1] if _break_history else None,
        "last_cache_read": _last_cache_read,
    }


def reset_cache_monitor() -> None:
    """把监控状态全部清零。每个新会话开始时调（AIAgent.__init__ 末尾），
    免得上个会话的基线污染这个会话的判定。
    """
    global _last_state, _last_cache_read, _pending_compaction, _break_history
    global _last_baseline_at
    _last_state = None
    _last_cache_read = None
    _pending_compaction = False
    _break_history = []
    _last_baseline_at = None
