"""prompt cache 检测系统（借鉴 claude-code-main promptCacheBreakDetection）。

工作流程：
1. pre-call：record_prompt_state 快照 prompt 维度
2. post-call：check_cache_break 比较 cacheReadTokens + 找根因
3. break 时 log + 记入 _break_history（供 /cache-stats 展示）

不打断主流程（fail-open）。
"""
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptState:
    """API 调用前的 prompt 状态快照（12 维度，先实现核心 5 个）。

    完整 12 维度（参考 claude-code-main，剩余 7 个先留 TODO）：
      6. max_tokens  7. temperature  8. stream 模式
      9. tool_choice  10. user content prefix  11. messages count  12. system 边界
    """
    system_hash: int = 0
    tools_hash: int = 0
    model: str = ""
    cache_strategy: str = ""
    betas_hash: int = 0


# 模块级状态（对齐 brief：单进程单会话，session 切换时 reset）
_last_state: Optional[PromptState] = None
_last_cache_read: Optional[int] = None
_pending_compaction: bool = False  # compact 后预期 cache 下降
_break_history: List[dict] = []  # break 事件累计（给 /cache-stats 展示）
_BREAK_HISTORY_LIMIT: int = 100  # 防长会话内存膨胀


def _hash_content(content: Any) -> int:
    """计算内容哈希（int，用于快速比较）。fail-open：异常返回 0。"""
    if isinstance(content, str):
        return hash(content)
    try:
        return hash(str(content))
    except Exception:
        return 0


def record_prompt_state(
    *, system_prompt: str, tools: list, model: str, **kwargs,
) -> PromptState:
    """API 调用前快照 prompt 状态。

    fail-open：异常返回默认 PromptState（不抛）。
    """
    try:
        state = PromptState(
            system_hash=_hash_content(system_prompt),
            tools_hash=_hash_content([
                (t.get("function", {}) or {}).get("name", "")
                if isinstance(t, dict) else str(t)
                for t in (tools or [])
            ]),
            model=model or "",
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
    """API 调用后检查 cache 是否 break。

    判定条件：cache read 下降 > 5% **且** > 2000 tokens。
    返回 None（没 break）或 break 根因描述字符串。
    fail-open：任何异常返回 None（绝不影响主流程）。
    """
    global _last_state, _last_cache_read, _pending_compaction
    try:
        # 防御：cache_read_tokens 异常值
        if cache_read_tokens is None or cache_read_tokens < 0:
            cache_read_tokens = 0

        # compact 后预期下降，不算 break
        if _pending_compaction:
            _pending_compaction = False
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            return None

        # 首次调用（无 baseline）
        if _last_state is None or _last_cache_read is None:
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            return None

        # 判定 break：cache read 下降 > 5% 且 > 2000 tokens
        token_drop = _last_cache_read - cache_read_tokens
        if cache_read_tokens >= _last_cache_read * 0.95 or token_drop < 2000:
            # 不算 break，更新 baseline
            _last_state = current_state
            _last_cache_read = cache_read_tokens
            return None

        # 找根因：对比维度
        reasons = []
        if current_state.system_hash != _last_state.system_hash:
            reasons.append("system prompt 变了")
        if current_state.tools_hash != _last_state.tools_hash:
            reasons.append("工具 schema 变了")
        if current_state.model != _last_state.model:
            reasons.append("model 变了")

        root_cause = "; ".join(reasons) if reasons else "未知原因"

        # 记录到历史（限长，防长会话内存膨胀）
        _break_history.append({
            "from": _last_cache_read,
            "to": cache_read_tokens,
            "drop": token_drop,
            "root_cause": root_cause,
            "query_source": query_source,
        })
        if len(_break_history) > _BREAK_HISTORY_LIMIT:
            del _break_history[0:len(_break_history) - _BREAK_HISTORY_LIMIT]

        # 更新 baseline
        prev_read = _last_cache_read
        _last_state = current_state
        _last_cache_read = cache_read_tokens

        logger.warning(
            "prompt cache break! cache read %d -> %d (drop %d tokens). cause: %s",
            prev_read, cache_read_tokens, token_drop, root_cause,
        )
        return root_cause
    except Exception as e:
        logger.debug("check_cache_break fail-open: %s", e)
        return None


def notify_compaction() -> None:
    """标记：下次调用预期 cache 下降（compact / cache_edits 后调）。

    压缩会修改 messages，下次 cache read 必然下降——这不算 break，
    调本函数让 check_cache_break 跳过一次判定。
    """
    global _pending_compaction
    _pending_compaction = True


def get_stats() -> dict:
    """返回 cache 监控统计（给 /cache-stats 命令用）。"""
    return {
        "total_breaks": len(_break_history),
        "last_break": _break_history[-1] if _break_history else None,
        "last_cache_read": _last_cache_read,
    }


def reset_cache_monitor() -> None:
    """会话开始时重置（避免跨会话污染）。

    在 AIAgent.__init__ 末尾调用。
    """
    global _last_state, _last_cache_read, _pending_compaction, _break_history
    _last_state = None
    _last_cache_read = None
    _pending_compaction = False
    _break_history = []
