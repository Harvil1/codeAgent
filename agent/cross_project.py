"""跨项目会话恢复：复用 HandoffStore。

- auto-save on exit：会话退出时自动保存为 bundle
- list across projects：列出最近 N 个 bundle（跨所有项目）
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agent.handoff import HandoffBundleMeta, HandoffStore

logger = logging.getLogger(__name__)


def list_recent_bundles_across_projects(
    handoff_store: HandoffStore,
    limit: int = 10,
    cwd_filter: Optional[str] = None,
) -> List[HandoffBundleMeta]:
    """跨项目列出最近 bundle。

    list_bundles 已按 created_at 倒序返回，这里只做 cwd 过滤 + limit 截断。
    """
    bundles = handoff_store.list_bundles()
    if cwd_filter:
        bundles = [b for b in bundles if b.source_cwd == cwd_filter]
    return bundles[:limit]


def auto_save_current_session(
    handoff_store: HandoffStore,
    agent,
    title: Optional[str] = None,
    source_cwd: Optional[str] = None,
) -> str:
    """会话退出时自动保存为 bundle。返回 bundle_id。

    fail-open：保存失败时只记日志返回空串，不抛。
    """
    if title is None:
        cwd_name = Path(source_cwd).name if source_cwd else Path.cwd().name
        title = f"auto-save {cwd_name} {datetime.now().isoformat(timespec='minutes')}"

    try:
        bundle_id = handoff_store.save(
            transcript=agent.conversation_history,
            source_session_id=getattr(agent, "session_id", None),
            model=getattr(agent, "model", {"main": "deepseek-chat"}),
            title=title,
            source_cwd=source_cwd or str(Path.cwd()),
            auto_saved=True,
        )
        logger.info("auto-save bundle: %s", bundle_id)
        return bundle_id
    except Exception as e:
        logger.warning("auto-save 失败（fail-open）: %s", e)
        return ""
