"""子代理轨迹（transcript——谁在什么时候说了什么的过程记录）落盘持久化（借鉴 claude-code-main）。

子代理在主对话之外独立跑（sidechain，旁路对话），跑挂了或中断后想恢复（resume）
就得有轨迹可查。本模块负责把轨迹和元数据写到磁盘：

- ~/.OmniMate/.agent-sessions/<agent_id>.jsonl  # 轨迹正文，一行一条消息
  （CCAR13 Task 3 起的口径：user 指令 + 每轮 assistant 文本）
- ~/.OmniMate/.agent-sessions/<agent_id>.meta.json  # 元数据
  （agent_id / agent_type / parent_session / status / created_at / updated_at）

agent_id 长这样：sub-{父会话id前8位}-{YYYYMMDD-HHMMSS}-{随机8位}
  例：sub-parent-s-20260811-143022-605d9a3a

status 状态流：running（在跑）→ completed（完成）/ failed（失败）/ interrupted（被中断）

历史踩坑（CCAR13 Task 3，补 CCAR5-I Phase 2）：旧版只在 on_response 回调里记
最终响应——子代理中途被打断就一点轨迹都没有，没法 resume。修复后改为每轮追加：
_run_child 给子代理挂一个独立 HookRegistry 的 POST_LLM_CALL 程序式 hook，
LLM 每回一次话就立刻落盘该轮 assistant 文本。
口径约束：轨迹只存 user 指令 + 每轮 assistant 文本；tool_calls / tool result
绝不落盘——孤儿 tool_call 没有配对 result 会让 API 报 400，
而只存文本的话，resume 时的 initial_messages 天然不会出现配对残缺。

设计约定（来自 CLAUDE.md）：
- **fail-open 硬要求**：所有持久化操作都包 try/except，轨迹系统出任何错
  都不能把主流程带崩
- **encoding="utf-8"**：所有文件读写显式指定（Windows 默认编码会乱码）
- **完全可逆**：轨迹可以删可以清（cleanup_old 定期清理）
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# agent_id 生成
# ---------------------------------------------------------------------------

def generate_agent_id(parent_session_id: str = "") -> str:
    """给新起的子代理造一个全球不撞车的 ID。

    参数：
        parent_session_id：父会话的 session id（为空时用 "orphan" 占位——查日志一眼
            就知道这个子代理没有挂靠的父会话）

    返回：
        形如 sub-{父id前8位}-{日期时间}-{随机8位} 的字符串。
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    rand = uuid.uuid4().hex[:8]
    parent_part = parent_session_id[:8] if parent_session_id else "orphan"
    return f"sub-{parent_part}-{ts}-{rand}"


# ---------------------------------------------------------------------------
# 存储目录
# ---------------------------------------------------------------------------

def _sessions_dir() -> Path:
    """拿到 .agent-sessions 存储目录的 Path（目录不存在就顺手创建）。"""
    from constants import get_omnimate_home
    d = get_omnimate_home() / ".agent-sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def write_metadata(agent_id: str, meta: dict) -> None:
    """把子代理的元数据写到 <agent_id>.meta.json（整份覆盖，自动补 agent_id 和 updated_at）。

    为什么用原子写（先写临时文件再替换，走 atomic_write_text_lite）：替换是一瞬间
    完成的，并发读的人永远不会读到写了一半的 JSON。

    参数：
        agent_id：子代理 ID
        meta：要写的元数据 dict

    返回：无。写失败只打 warning（fail-open——元数据丢了轨迹 jsonl 还在，不至于崩）。
    """
    try:
        from agent.atomic_io import atomic_write_text_lite

        path = _sessions_dir() / f"{agent_id}.meta.json"
        payload = {
            **meta,
            "agent_id": agent_id,
            "updated_at": time.time(),
        }
        atomic_write_text_lite(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("write_metadata fail-open [%s]: %s", agent_id, e)


def load_metadata(agent_id: str) -> Optional[dict]:
    """读某个子代理的元数据。

    参数：
        agent_id：子代理 ID

    返回：
        元数据 dict；文件不存在或读挂了返回 None（fail-open）。
    """
    try:
        path = _sessions_dir() / f"{agent_id}.meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("load_metadata fail-open [%s]: %s", agent_id, e)
        return None


# ---------------------------------------------------------------------------
# transcript (JSONL)
# ---------------------------------------------------------------------------

def append_message(agent_id: str, message: dict) -> None:
    """往轨迹 jsonl 文件末尾追加一条消息（每轮 LLM 响应后调用）。

    参数：
        agent_id：子代理 ID
        message：要落盘的消息 dict

    返回：无。写失败只打 warning（fail-open，不打断子代理干活）。
    """
    try:
        path = _sessions_dir() / f"{agent_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("append_message fail-open [%s]: %s", agent_id, e)


def load_transcript(agent_id: str) -> List[dict]:
    """把某个子代理的完整轨迹读回来（resume 时用）。

    参数：
        agent_id：子代理 ID

    返回：
        消息 dict 列表；文件不存在或读挂了返回空列表（fail-open）。
    """
    try:
        path = _sessions_dir() / f"{agent_id}.jsonl"
        if not path.exists():
            return []
        msgs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                msgs.append(json.loads(line))
        return msgs
    except Exception as e:
        logger.warning("load_transcript fail-open [%s]: %s", agent_id, e)
        return []


# ---------------------------------------------------------------------------
# 列表 / 状态管理
# ---------------------------------------------------------------------------

def list_resumable() -> List[dict]:
    """列出所有还标记为 running 的子代理元数据。

    用途：程序重启后挨个检查这些记录——上个进程已经退了，它们其实都是
    僵尸状态（stale running），要么标记 interrupted 要么供 resume 挑选。

    返回：
        元数据 dict 列表；目录读不了返回空列表（fail-open）。
    """
    try:
        result = []
        for meta_path in _sessions_dir().glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("status") == "running":
                    result.append(meta)
            except Exception:
                continue
        return result
    except Exception as e:
        logger.debug("list_resumable fail-open: %s", e)
        return []


def mark_completed(agent_id: str, status: str = "completed") -> None:
    """给子代理盖终态章（completed / failed / interrupted 三选一）。

    参数：
        agent_id：子代理 ID
        status：终态名，默认 "completed"

    返回：无。做法是读旧元数据 → 改 status 和 completed_at → 写回。
    """
    meta = load_metadata(agent_id) or {}
    meta["status"] = status
    meta["completed_at"] = time.time()
    write_metadata(agent_id, meta)


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

def cleanup_old(days: int = 7) -> int:
    """删除 N 天前就已进入终态（completed/failed/interrupted）的子代理记录。

    参数：
        days：保留天数，默认 7

    返回：
        int——本次清理掉的数量。只动终态记录；running 状态的一律不删
        （防止误删还在跑的子代理）。
    """
    cutoff = time.time() - days * 86400
    cleaned = 0
    try:
        sessions_dir = _sessions_dir()
    except Exception as e:
        logger.debug("cleanup_old: _sessions_dir 失败: %s", e)
        return 0

    for meta_path in sessions_dir.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") in ("completed", "failed", "interrupted"):
                if meta.get("completed_at", 0) < cutoff:
                    agent_id = meta.get("agent_id") or meta_path.stem.removesuffix(".meta")
                    meta_path.unlink(missing_ok=True)
                    jsonl_path = sessions_dir / f"{agent_id}.jsonl"
                    jsonl_path.unlink(missing_ok=True)
                    cleaned += 1
        except Exception:
            continue
    return cleaned


def cleanup_stale_subagents() -> int:
    """程序启动时把僵尸记录扶正：所有 status=running 的改成 interrupted。

    为什么能一刀切：程序刚重启，此刻不可能有真正在跑的子代理——
    上一批 running 记录对应的进程早随上个进程一起退出了。

    返回：
        int——处理掉的僵尸记录数。
    """
    cleaned = 0
    for meta in list_resumable():
        try:
            agent_id = meta.get("agent_id")
            if agent_id:
                mark_completed(agent_id, "interrupted")
                cleaned += 1
        except Exception as e:
            logger.debug("cleanup_stale_subagents 跳过 %s: %s", meta.get("agent_id"), e)
    if cleaned > 0:
        logger.info("cleanup_stale_subagents: %d 个 stale running 子代理标记为 interrupted", cleaned)
    return cleaned
