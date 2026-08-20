"""跨项目的会话恢复：站在 HandoffStore（会话移交仓库）肩膀上加两个便利功能。

打个比方：handoff 是"打包行李"，这个文件负责两件周边杂事：
- auto-save on exit：退出会话时不用用户动手，自动把当前会话打包存一份
- list across projects：把所有项目攒下的 bundle 放在一起列出来，
  换项目/换电脑后能找回上次的现场
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
    """跨项目列出最近保存的 bundle（不分哪个项目存的，一锅端）。

    参数：
        handoff_store：HandoffStore 实例（bundle 都从它那拿）。
        limit：最多返回几条，默认 10。
        cwd_filter：不为 None 时只留来源目录等于它的 bundle（按项目筛）。

    返回：
        HandoffBundleMeta 列表，最新的在前。
    排序不用自己做——list_bundles 返回时已经按创建时间从新到旧排好了，
    这里只做"按项目筛 + 截前 N 条"两件事。
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
    """会话退出时自动把当前会话打包成 bundle 存档，返回 bundle_id。

    参数：
        handoff_store：HandoffStore 实例（往里存）。
        agent：AIAgent 实例——从它身上取对话历史、会话 ID、模型名。
        title：bundle 标题；None 时自动起一个（"auto-save 项目名 时间"）。
        source_cwd：记录是哪个项目的会话；None 时用当前目录。

    返回：
        新 bundle 的 ID；保存失败返回空串。
    保存失败绝不抛异常（fail-open）：自动存档是贴心服务，不能反过来
    把正常的退出流程搞崩，失败只记一条日志。
    """
    # 标题没给就自动起一个：项目名 + 分钟级时间，一眼能认出是哪次存档
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
