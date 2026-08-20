"""会话级 scratchpad（涂鸦区）——多个 worker 子代理共享的临时草稿区（R24 #36，对齐 CC 的 getScratchpadDir）。

干什么用：coordinator（协调者）/orchestrator（编排者）带着一群 worker
子代理干活时，需要一个大家都能写的"共享白板"——中间结论、共享状态、
还没合并的草稿都先搁这儿。文件结构随便组织（CC 的原话："怎么顺手怎么摆"，
反正是跨 worker 的持久知识）。

- 路径：``<omnimate_home>/.scratchpad/<session_id>/``——每个会话一格，互不串门
- **免权限读写**：把目录加进 safe_path（路径安全检查）的 extra_allowed_roots
  （运行时白名单）。注意只加在内存里、不写 settings.json——这白名单只活
  到进程退出，重启自动失效，正好匹配"会话级"的生命周期
- ensure 时建目录；cleanup 按保留天数清理旧会话的目录（默认 7 天）

和 CC 的差异：CC 用 0o700 权限隔离 + 内部门控；OmniMate 简化成目录白名单 +
定期清理（单机单人产品，没有多用户隔离需求）。fail-open：白名单注入失败
只记 log，涂鸦区照常能写——只是写文件要多过一次审批。
"""
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 涂鸦区目录的保留天数：超过 7 天没动过的会话目录会被清理
DEFAULT_RETENTION_DAYS = 7


def scratchpad_dir(session_id: str, omnimate_home=None) -> Path:
    """算出某个会话的涂鸦区目录路径（只算路径，不建目录）。

    参数：
        session_id：会话 id（里面的怪字符会被剔掉，防止拼出危险路径）。
        omnimate_home：数据根目录，不传用默认 ~/.OmniMate。
    返回：该会话涂鸦区的 Path。
    """
    if omnimate_home is None:
        try:
            from constants import get_omnimate_home
            omnimate_home = get_omnimate_home()
        except Exception:
            omnimate_home = Path.home() / ".OmniMate"
    safe_sid = "".join(c for c in str(session_id or "default") if c.isalnum() or c in "-_")
    return Path(omnimate_home) / ".scratchpad" / (safe_sid or "default")


def ensure_scratchpad(session_id: str, omnimate_home=None) -> Optional[Path]:
    """把会话涂鸦区建出来，并加进免权限白名单。返回目录路径。

    白名单注入说明：走 add_extra_allowed_root，只加在运行时内存里（跟
    /add-dir 是同一个通道，但这里不持久化——会话级的东西不该写进全局
    settings.json）。fail-open：白名单没加成也照常返回目录，只是往里写
    文件时会走审批流程。

    参数：
        session_id：会话 id。
        omnimate_home：数据根目录，不传用默认 ~/.OmniMate。
    返回：涂鸦区目录 Path；连目录都建不出来返回 None（fail-open）。
    """
    try:
        d = scratchpad_dir(session_id, omnimate_home)
        d.mkdir(parents=True, exist_ok=True)
        try:
            from agent.permission import add_extra_allowed_root
            add_extra_allowed_root(d)
        except Exception as e:
            logger.warning("scratchpad 白名单注入失败（写入将走审批）: %s", e)
        return d
    except Exception as e:
        logger.warning("scratchpad 创建失败（fail-open）: %s", e)
        return None


def scratchpad_context_block(session_id: str, omnimate_home=None) -> str:
    """生成一段塞进协调者上下文的"涂鸦区使用说明"（文案对齐 CC）。

    参数：
        session_id：会话 id。
        omnimate_home：数据根目录，不传用默认 ~/.OmniMate。
    返回：说明文字（含目录路径），可直接拼在 user 消息后面。
    """
    d = scratchpad_dir(session_id, omnimate_home)
    return (
        f"\n\nScratchpad 目录：{d}\n"
        "worker 子代理可在此免权限读写。用它存放跨 worker 的持久知识"
        "（中间结论/共享状态/待合并草稿）——文件结构自由组织。"
    )


def cleanup_old_scratchpads(omnimate_home=None, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """清理"超过 retention_days 天没动过"的旧会话涂鸦区目录。返回清了几个。

    ⚠️ 这是"完全可逆"铁律的一个明写例外：项目里的自动管理原则是永不真删，
    但涂鸦区是**临时草稿**不是知识库——按最后修改时间清理是 CC 同款语义；
    真正要长期保留的知识请走记忆/技能系统，别指望这里。

    参数：
        omnimate_home：数据根目录，不传用默认 ~/.OmniMate。
        retention_days：保留天数（默认 7 天）。
    返回：删掉的目录数；任何异常都吞掉返回 0（fail-open）。
    """
    try:
        if omnimate_home is None:
            try:
                from constants import get_omnimate_home
                omnimate_home = get_omnimate_home()
            except Exception:
                return 0
        root = Path(omnimate_home) / ".scratchpad"
        if not root.exists():
            return 0
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for d in root.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    import shutil
                    shutil.rmtree(d)
                    removed += 1
            except Exception:
                continue
        if removed:
            logger.info("scratchpad 清理 %d 个过期会话目录", removed)
        return removed
    except Exception:
        return 0
