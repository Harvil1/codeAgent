# -*- coding: utf-8 -*-
"""按模型分别记账的 token 用量追踪器。

干什么：像流水账一样记下每次 LLM 调用花了多少 token——按模型分开累计
五个数：prompt（输入）/ completion（输出）/ cache_read（缓存命中）/
cache_creation（缓存写入）/ calls（调用次数）。

存哪：每个会话一个文件，落在 ``~/.OmniMate/.usage/{session_id}.json``，
用原子写（读的人永远看不到半截），任何失败都静默吞掉不影响主流程
（记账不能把正事拖垮）。

只统计 token、不算钱（用户裁决：本项目不需要价格估算，价表会过期
也没人维护，删掉了 agent/pricing.py）。

接线方式：cli 的 RuntimeContext 创建后通过 set_usage_tracker 注入
AIAgent，主循环每次 LLM 调用后在 ``_record_llm_usage`` 里记账；
``/usage`` 命令按模型展示结果。
"""
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def _empty_model_row() -> dict:
    """一个模型的空白账页（五个计数全 0），新模型入账时用它开张。"""
    return {
        "calls": 0, "prompt": 0, "completion": 0,
        "cache_read": 0, "cache_creation": 0,
    }


class UsageTracker:
    """按模型累计用量的记账器：内存里一个字典 + 每会话落一个 json 文件。"""

    def __init__(self, home, session_id: str = ""):
        """初始化并读回该会话已有的账页（文件不存在就空账起步）。

        参数：
            home       —— OmniMate 主目录（~/.OmniMate），账本存在其下 .usage/
            session_id —— 会话 ID，空串时用 "default"
        """
        self._home = Path(home)
        self._session_id = str(session_id or "default")
        self._path = self._home / ".usage" / f"{self._session_id}.json"
        self._models: Dict[str, dict] = {}
        self._load()

    # ---- 记录 ----

    def record(self, *, model: str, prompt: int = 0, completion: int = 0,
               cache_read: int = 0, cache_creation: int = 0) -> None:
        """记一笔：某模型本次调用花了多少 token，然后立刻落盘。

        任何异常都吞掉只打 debug 日志（fail-open）——记账失败绝不能
        影响对话主流程。

        参数：
            model           —— 模型名（空的按 "unknown" 记）
            prompt          —— 本次输入 token 数
            completion      —— 本次输出 token 数
            cache_read      —— 其中命中缓存的 token 数
            cache_creation  —— 其中写缓存的 token 数
        """
        try:
            row = self._models.setdefault(str(model or "unknown"), _empty_model_row())
            row["calls"] += 1
            row["prompt"] += int(prompt or 0)
            row["completion"] += int(completion or 0)
            row["cache_read"] += int(cache_read or 0)
            row["cache_creation"] += int(cache_creation or 0)
            self._save()
        except Exception as e:
            logger.debug("usage_tracker.record 失败（fail-open）: %s", e)

    # ---- 查询 ----

    def summary(self) -> dict:
        """出报表：按模型汇总 + 全模型合计（只有 token 计数，没有金额）。

        返回：字典，含 session_id、models（每模型一行）、totals（合计行）。
        """
        models = {}
        totals = _empty_model_row()
        for name, row in self._models.items():
            models[name] = dict(row)
            for k in totals:
                totals[k] += row.get(k, 0)
        return {
            "session_id": self._session_id,
            "models": models,
            "totals": totals,
        }

    # ---- 持久化 ----

    def _load(self) -> None:
        """从账本文件读回上次累计（文件缺失/损坏 → 空账起步，不报错）。"""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("models"), dict):
                self._models = data["models"]
        except Exception:
            pass  # 没有文件或文件坏了 → 从零开始记

    def _save(self) -> None:
        """把账本原子落盘；失败只打 debug 日志（fail-open）。"""
        try:
            from agent.atomic_io import atomic_write_text_lite
            self._path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text_lite(
                self._path,
                json.dumps({"models": self._models}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("usage 持久化失败（fail-open）: %s", e)
