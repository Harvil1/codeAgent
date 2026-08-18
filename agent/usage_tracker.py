# -*- coding: utf-8 -*-
"""Per-model token 用量追踪（R30f-H8，对标 CCB cost-tracker 的核心子集）。

- 按 model 四维累计：prompt / completion / cache_read / cache_creation + 调用数
- 按 session 持久化到 ``~/.OmniMate/.usage/{session_id}.json``（原子写，fail-open）
- 金额复用 ``agent/pricing.py`` 价表（先按 default_provider 查，查不到按
  模型名扫全 provider）；未收录模型只报 token 不估金额（不瞎猜价格）。

接线：cli RuntimeContext 创建后注入 AIAgent（set_usage_tracker），
``_record_llm_usage`` 每次调用累计；``/usage`` 按 model 展示。
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _empty_model_row() -> dict:
    return {
        "calls": 0, "prompt": 0, "completion": 0,
        "cache_read": 0, "cache_creation": 0,
    }


class UsageTracker:
    """按 model 的用量累计器（进程内字典 + 每 session 一个 json 文件）。"""

    def __init__(self, home, session_id: str = "", *,
                 default_provider: str = ""):
        self._home = Path(home)
        self._session_id = str(session_id or "default")
        self._path = self._home / ".usage" / f"{self._session_id}.json"
        self._default_provider = default_provider or ""
        self._models: Dict[str, dict] = {}
        self._load()

    # ---- 记录 ----

    def record(self, *, model: str, prompt: int = 0, completion: int = 0,
               cache_read: int = 0, cache_creation: int = 0) -> None:
        """累计一次 LLM 调用。fail-open（异常吞掉不影响主流程）。"""
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
        """按 model 汇总（含可选 cost_usd）+ 全模型 totals。"""
        models = {}
        totals = _empty_model_row()
        for name, row in self._models.items():
            out = dict(row)
            cost = self._cost(name, row)
            if cost is not None:
                out["cost_usd"] = round(cost, 4)
            models[name] = out
            for k in totals:
                totals[k] += row.get(k, 0)
        total_cost = sum(m["cost_usd"] for m in models.values() if "cost_usd" in m)
        return {
            "session_id": self._session_id,
            "models": models,
            "totals": dict(totals, cost_usd=round(total_cost, 4))
            if total_cost else totals,
        }

    def _cost(self, model: str, row: dict) -> Optional[float]:
        """复用 agent/pricing.py 的价表与估算语义（fail-open 返回 None）。"""
        try:
            from agent.pricing import estimate_cost_usd, PRICING
            est = estimate_cost_usd(
                provider=self._default_provider, model=model,
                prompt_tokens=row["prompt"],
                completion_tokens=row["completion"],
                cache_read_tokens=row["cache_read"],
                cache_creation_tokens=row["cache_creation"],
            )
            if est is not None:
                return est["cost_usd"]
            # default_provider 没命中 → 按模型名扫全 provider（aux 模型
            # 可能与主 provider 不同家）
            for prov in PRICING:
                est = estimate_cost_usd(
                    provider=prov, model=model,
                    prompt_tokens=row["prompt"],
                    completion_tokens=row["completion"],
                    cache_read_tokens=row["cache_read"],
                    cache_creation_tokens=row["cache_creation"],
                )
                if est is not None:
                    return est["cost_usd"]
        except Exception as e:
            logger.debug("usage 成本估算失败（fail-open）: %s", e)
        return None

    # ---- 持久化 ----

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("models"), dict):
                self._models = data["models"]
        except Exception:
            pass  # 无文件/损坏 → 空表起步

    def _save(self) -> None:
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
