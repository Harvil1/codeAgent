# -*- coding: utf-8 -*-
"""InstinctStore：instinct 行为记忆的存储层。

（历史轮次 CCAR15 Task 1 引入，对标 CCB 项目的 instinctStore。）

核心思想：agent 在某个 trigger（触发情境）下反复选择某个 action（做法），
说明这是一条值得记住的「条件反射」。每重复观察到一次，置信度按
衰减累积公式上涨——

    新置信度 = 旧置信度 * 0.9 + 0.25（封顶 1.0）

* 只会涨不会降：重复观察时一路上涨（0.3 → 0.52 → 0.718 → ...），
  不重复就停在原地
* 公式天生有上限：数学上的稳定点在 2.5，但被 1.0 封顶，不会溢出

存盘方式（JSON，一条习惯一个文件）：

    <base>/instincts/<scope_dir>/<slug(trigger)>.json

* scope_dir（作用域目录）：global → "global"；"project:<key>" → 转成
  slug（':' 在 Windows 目录名里非法，替换成 '-'；scope 原文存在 JSON
  里，读回来还是原样，不丢信息）
* slug：trigger 小写后把 [a-z0-9_-] 之外的字符全替换成 '-'，截 60 字符

判断「是不是同一条习惯」用三元组：归一化的 trigger + action + scope。
trigger 归一化 = 转小写 + 压缩空白（"Run  Tests " 和 "run tests" 算同一个）。

本层故意不做 fail-open（存储失败可以抛异常）；调用方接线
（历史轮次 T3）负责在外面包异常兜底。
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agent.atomic_io import atomic_write_text

# 置信度累积公式的两个系数：新 = 旧 * DECAY + BOOST（0.9 / 0.25）
_DECAY = 0.9
_BOOST = 0.25
_CAP = 1.0

# 证据（evidence）去重后最多留 10 条，留最新的
_MAX_EVIDENCE = 10

# slug：文件名只许这些安全字符，其余替换掉
_SLUG_RE = re.compile(r"[^a-z0-9_-]")
_SLUG_MAX = 60

# 清理（prune）的默认门槛：置信度低于它的条目才算「低置信」，
# 高置信的「金科玉律」哪怕过期了也不清理
_PRUNE_MIN_CONFIDENCE = 0.5


@dataclass
class Instinct:
    """一条 instinct 行为记忆（一条「遇到什么 → 怎么做」的习惯）。

    字段说明：
        trigger：触发情境（如「用户要求跑测试」）。
        action：agent 的惯常做法（如「先跑 pytest 再汇报」）。
        confidence：置信度 0.0~1.0，重复观察按「旧*0.9+0.25」累积。
        evidence：支撑证据列表（去重后留最新 10 条）。
        scope：作用域——"global"（所有项目通用）或 "project:<key>"
            （只属于某个项目，物理隔离）。
        updated_at：最后更新时间（ISO 8601 格式，UTC 时区）。
    """
    trigger: str
    action: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    scope: str = "global"
    updated_at: str = ""


def _normalize_trigger(trigger: str) -> str:
    """trigger 归一化：转小写 + 压缩空白。

    为什么：大小写和空格差异会让同一条习惯裂成好几条。
    "Run  Tests " / "run tests" / " RUN tests " 统统归成 "run tests"。

    参数：
        trigger：原始触发情境文本。

    返回：归一化后的文本。
    """
    return " ".join(trigger.lower().split())


def _slug(text: str) -> str:
    """把任意文本转成能当文件名的 slug。

    参数：
        text：原始文本。

    返回：小写、[a-z0-9_-] 之外的字符全换成 '-'、截 60 字符的结果
    （中文会全变 '-'，唯一性靠 hash 兜底，见 _disambiguate_path）。
    """
    return _SLUG_RE.sub("-", text.lower())[:_SLUG_MAX]


def _parse_iso(ts: str) -> float:
    """把 ISO 格式时间戳解析成秒数（用于比较新旧）。

    参数：
        ts：ISO 8601 字符串。

    返回：对应的 epoch 秒；解析失败返回 0（当成「最早」，排序时沉底）。
    """
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return 0.0


class InstinctStore:
    """instinct 行为记忆的仓库：管置信度累积、全局/项目隔离和 JSON 落盘。

    「惰性落盘」：不做常驻内存索引，每次读直接读文件——写得少读得也少，
    简单可靠。

    参数（__init__）：
        base_dir：存放根目录，实际数据在 <base_dir>/instincts/ 下面。
    """

    def __init__(self, base_dir) -> None:
        self._base = Path(base_dir) / "instincts"
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 目录与路径计算
    # ------------------------------------------------------------------

    def _scope_dir_name(self, scope: str) -> str:
        """把 scope 转成目录名。

        参数：
            scope："global" 或 "project:<key>"。

        返回：目录名。"global" 原样；带冒号的（Windows 目录名禁字）转 slug。
        注意 "project:global" 和 "global" 必须落进不同目录——slug 化天然
        就能满足（前者变成 "project-global"）。
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
    # JSON 与对象互转、单文件读取
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
            return None  # 文件坏了就当它不存在——读错误不能阻塞主流程（写错误才上抛）

    # ------------------------------------------------------------------
    # 对外方法
    # ------------------------------------------------------------------

    def upsert(self, inst: Instinct) -> Instinct:
        """写入一条习惯；已存在同款则合并（置信度涨一截）。

        参数：
            inst：要写入的 Instinct。

        返回：最终落盘的那条（新写的或合并后的）。

        判断「同款」：归一化 trigger + action + scope 三元组都相同。
        合并规则：置信度 = 旧 * 0.9 + 0.25（封顶 1.0）；证据去重后留
        最新 10 条；更新时间刷新为现在。
        """
        import json
        from datetime import datetime, timezone

        path = self._path_for(inst)
        existing = self._read_file(path)
        # 同 trigger 但 action 不同 = 同一情境下的另一条习惯。默认文件名只含
        # trigger 会互相覆盖，所以改用带 action 指纹的路径，并重读该路径下
        # 的同款条目（保证重复 upsert 仍走置信度累积，而不是归零重写）。
        if existing is not None and existing.action != inst.action:
            path = self._disambiguate_path(inst)
            existing = self._read_file(path)

        if existing is not None and existing.action == inst.action:
            merged = Instinct(
                trigger=existing.trigger,  # 保留第一次入库时的原文写法，不被后来的变体冲掉
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
        """同 trigger 不同 action 时，算一个带指纹的文件路径防覆盖。

        参数：
            inst：要落盘的习惯条目。

        返回：文件名形如 <trigger-slug>-<8位hash>.json 的路径。

        为什么这么简单粗暴：一个 trigger 对应多个做法的场景很少见，
        不值得为此建索引，直接在文件名后拼 action+scope 的 md5 前 8 位
        就够用了。
        """
        import hashlib
        base = _slug(_normalize_trigger(inst.trigger))
        h = hashlib.md5(f"{inst.action}|{inst.scope}".encode("utf-8")).hexdigest()[:8]
        return self._scope_dir(inst.scope) / f"{base}-{h}.json"

    @staticmethod
    def _merge_evidence(old: List[str], new: List[str]) -> List[str]:
        """合并新旧证据列表。

        参数：
            old：已有的证据列表。
            new：本次新带来的证据列表。

        返回：按出现顺序去重后，只保留最新的 10 条（截尾）。
        """
        seen = []
        for ev in list(old) + list(new):
            if ev not in seen:
                seen.append(ev)
        return seen[-_MAX_EVIDENCE:]

    def cluster(self, scope: str) -> Dict[str, List[Instinct]]:
        """把同一情境的习惯归成堆。

        打个比方：把散落的习惯卡片按「情境」分堆——写法略有差异但
        实际相同的 trigger（大小写/空白不同）归进同一堆，方便上层看出
        「这个情境下攒了多少条做法」。

        参数：
            scope：作用域，只归这个 scope 的条目。

        返回：{归一化后的 trigger: [该情境下的 Instinct 列表]}。
        """
        clusters: Dict[str, List[Instinct]] = {}
        for inst in self.list_all(scope=scope):
            clusters.setdefault(_normalize_trigger(inst.trigger), []).append(inst)
        return clusters

    def prune(self, days: int = 30, min_confidence: float = _PRUNE_MIN_CONFIDENCE) -> int:
        """清理「又老又不常被验证」的习惯，给存储瘦身。

        参数：
            days：多少天没更新算「过期」，默认 30。
            min_confidence：置信度低于多少算「低置信」，默认 0.5。

        返回：实际删掉的条数。

        删除条件是「且」不是「或」——过期且低置信才删。这样高置信的老
        习惯（金科玉律）永久保留；刚观察到的低置信新习惯也保留（还没
        涨起来就删会误伤）。
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
                    continue  # 删不掉就算了——prune 是打扫卫生，尽力而为不较真
        return removed

    def list_all(self, scope: Optional[str] = None) -> List[Instinct]:
        """列出存储里的习惯条目。

        参数：
            scope：作用域。传了只列这个 scope 的；不传（None）列全部。

        返回：Instinct 列表（按文件名排序）。
        """
        if scope is not None:
            root = self._base / self._scope_dir_name(scope)
            paths = sorted(root.glob("*.json")) if root.is_dir() else []
        else:
            paths = sorted(self._base.rglob("*.json"))

        result: List[Instinct] = []
        for path in paths:
            inst = self._read_file(path)
            if inst is not None:
                # 过滤 scope 时以 JSON 里存的原文为准，不看目录名——目录名被 slug 化过，对不回原值
                if scope is None or inst.scope == scope:
                    result.append(inst)
        return result
