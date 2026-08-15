# -*- coding: utf-8 -*-
"""InstinctStore：instinct 行为记忆存储层（CCAR15 Task 1，对标 CCB instinctStore）。

核心思想：agent 在某个 trigger 下反复选择某个 action，说明这是一条值得
记住的"条件反射"。每重复观察一次，置信度按衰减累积公式上涨——

    new_confidence = old_confidence * 0.9 + 0.25（封顶 1.0）

* 重复观察单调上涨（0.3 → 0.52 → 0.718 → ...），不重复则停在原地
* 公式天然有上限：不动点在 2.5 处，但被 1.0 封顶，不会溢出

持久化布局（JSON，一文件一条）：

    <base>/instincts/<scope_dir>/<slug(trigger)>.json

* scope_dir：global → "global"；"project:<key>" → slug 化（':' 在 Windows
  目录名非法，替换为 '-'；scope 原文保存在 JSON 里，round-trip 保真）
* slug：trigger 小写后把非 [a-z0-9_-] 全替换为 '-'，截 60 字符

合并 key：normalized(trigger) + action + scope 三元组。trigger 归一化 =
lower + 压缩空白（"Run  Tests " 与 "run tests" 是同一个 trigger）。

本层不做 fail-open（存储失败可抛）；调用方接线（T3）负责包异常。
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agent.atomic_io import atomic_write_text

# confidence 累积公式系数：new = old * DECAY + BOOST
_DECAY = 0.9
_BOOST = 0.25
_CAP = 1.0

# evidence 去重后最多保留条数（留最新的）
_MAX_EVIDENCE = 10

# slug：文件名安全字符
_SLUG_RE = re.compile(r"[^a-z0-9_-]")
_SLUG_MAX = 60

# prune 默认：置信度低于此值的过期条目才删（高置信的"金科玉律"不清理）
_PRUNE_MIN_CONFIDENCE = 0.5


@dataclass
class Instinct:
    """一条 instinct 行为记忆。

    Attributes:
        trigger: 触发情境描述（如 "用户要求跑测试"）。
        action: agent 的 habitual 行为（如 "先跑 pytest 再汇报"）。
        confidence: 0.0~1.0，重复观察按 old*0.9+0.25 累积。
        evidence: 支撑证据列表（去重、截最新 10 条）。
        scope: 作用域，"global" 或 "project:<key>"（项目隔离）。
        updated_at: 最后更新时间（ISO 8601，UTC）。
    """
    trigger: str
    action: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    scope: str = "global"
    updated_at: str = ""


def _normalize_trigger(trigger: str) -> str:
    """trigger 归一化：小写 + 压缩/去除首尾空白。

    "Run  Tests " / "run tests" / " RUN tests " → "run tests"。
    内部多余空白也压缩，避免簇碎片化。
    """
    return " ".join(trigger.lower().split())


def _slug(text: str) -> str:
    """文本 → 文件系统安全的 slug：非 [a-z0-9_-] 替换为 '-'，截 60。"""
    return _SLUG_RE.sub("-", text.lower())[:_SLUG_MAX]


def _parse_iso(ts: str) -> float:
    """解析 ISO 时间戳为 epoch 秒（失败返回 0，视为最早）。"""
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return 0.0


class InstinctStore:
    """instinct 行为记忆存储：置信度累积 + scope 隔离 + 惰性 JSON 落盘。"""

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir) / "instincts"
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------

    def _scope_dir_name(self, scope: str) -> str:
        """scope → 目录名。project:<key> 的 ':' 等 Windows 非法字符 slug 化。

        注意 "project:global" 必须与 "global" 落不同目录（slug 化天然满足）。
        """
        if scope == "global":
            return "global"
        return _slug(scope)

    def _scope_dir(self, scope: str) -> Path:
        d = self._base / self._scope_dir_name(scope)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path_for(self, inst: Instinct) -> Path:
        return self._scope_dir(inst.scope) / f"{_slug(_normalize_trigger(inst.trigger))}.json"

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(inst: Instinct) -> dict:
        return {
            "trigger": inst.trigger,
            "action": inst.action,
            "confidence": inst.confidence,
            "evidence": list(inst.evidence),
            "scope": inst.scope,
            "updated_at": inst.updated_at,
        }

    @staticmethod
    def _from_dict(data: dict) -> Instinct:
        return Instinct(
            trigger=data["trigger"],
            action=data["action"],
            confidence=float(data["confidence"]),
            evidence=list(data.get("evidence") or []),
            scope=data.get("scope", "global"),
            updated_at=data.get("updated_at", ""),
        )

    def _read_file(self, path: Path) -> Optional[Instinct]:
        try:
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._from_dict(data)
        except (OSError, ValueError, KeyError):
            return None  # 损坏文件视为不存在（存储层吞读错误，不阻塞主流程）

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def upsert(self, inst: Instinct) -> Instinct:
        """写入/合并一条 instinct。

        合并条件：normalized(trigger) + action + scope 三元组相同。
        合并规则：confidence = old * 0.9 + 0.25（封顶 1.0）；
        evidence 去重后保留最新 10 条；updated_at 刷新为本次。
        返回合并后的最终条目。
        """
        import json
        from datetime import datetime, timezone

        path = self._path_for(inst)
        existing = self._read_file(path)
        # 合并 key 第三元（action）不同 → 视为该 trigger 下的另一条习惯，
        # 文件名只含 trigger，改用带 action 指纹的路径，并重读该路径下的
        # 同 action 条目（保证重复 upsert 仍走置信度累积）。
        if existing is not None and existing.action != inst.action:
            path = self._disambiguate_path(inst)
            existing = self._read_file(path)

        if existing is not None and existing.action == inst.action:
            merged = Instinct(
                trigger=existing.trigger,  # 保留首次入库的原文
                action=existing.action,
                confidence=min(existing.confidence * _DECAY + _BOOST, _CAP),
                evidence=self._merge_evidence(existing.evidence, inst.evidence),
                scope=existing.scope,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        else:
            merged = inst

        atomic_write_text(path, json.dumps(
            self._to_dict(merged), ensure_ascii=False, indent=2
        ))
        return merged

    def _disambiguate_path(self, inst: Instinct) -> Path:
        """同 trigger 不同 action：路径加 action slug 后缀避免覆盖。

        惯用场景少（一个 trigger 多个 habitual action），不建索引，
        简单地在文件名后拼 "-<action-slug-hash>"。
        """
        import hashlib
        base = _slug(_normalize_trigger(inst.trigger))
        h = hashlib.md5(f"{inst.action}|{inst.scope}".encode("utf-8")).hexdigest()[:8]
        return self._scope_dir(inst.scope) / f"{base}-{h}.json"

    @staticmethod
    def _merge_evidence(old: List[str], new: List[str]) -> List[str]:
        """evidence 合并：保序去重，保留最新 _MAX_EVIDENCE 条。"""
        seen = []
        for ev in list(old) + list(new):
            if ev not in seen:
                seen.append(ev)
        return seen[-_MAX_EVIDENCE:]

    def cluster(self, scope: str) -> Dict[str, List[Instinct]]:
        """按归一化 trigger 分簇：{trigger_norm: [Instinct...]}。

        大小写/空白差异的 trigger 归同一簇，供上层做模式聚合。
        """
        clusters: Dict[str, List[Instinct]] = {}
        for inst in self.list_all(scope=scope):
            clusters.setdefault(_normalize_trigger(inst.trigger), []).append(inst)
        return clusters

    def prune(self, days: int = 30, min_confidence: float = _PRUNE_MIN_CONFIDENCE) -> int:
        """清理过期且低置信的条目，返回删除数。

        "过期" = updated_at 距今超过 days 天；
        "低置信" = confidence < min_confidence。
        两个条件都满足才删——高置信的老习惯（金科玉律）永久保留，
        新近观察到的低置信条目也保留（还没涨起来就删会误伤）。
        """
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()
        removed = 0
        for path in self._base.rglob("*.json"):
            inst = self._read_file(path)
            if inst is None:
                continue
            if _parse_iso(inst.updated_at) < cutoff_ts and inst.confidence < min_confidence:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue  # 删除失败跳过，prune 尽力而为
        return removed

    def list_all(self, scope: Optional[str] = None) -> List[Instinct]:
        """列出条目；scope=None 时跨所有 scope 返回。"""
        if scope is not None:
            root = self._base / self._scope_dir_name(scope)
            paths = sorted(root.glob("*.json")) if root.is_dir() else []
        else:
            paths = sorted(self._base.rglob("*.json"))

        result: List[Instinct] = []
        for path in paths:
            inst = self._read_file(path)
            if inst is not None:
                # scope 过滤看 JSON 原文而非目录名（目录名被 slug 化过）
                if scope is None or inst.scope == scope:
                    result.append(inst)
        return result
