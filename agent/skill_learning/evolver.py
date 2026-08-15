# -*- coding: utf-8 -*-
"""SkillEvolver：instinct 簇达标 → 自动生成技能（CCAR15 Task 3，对标 CCB skillEvolver）。

演化门槛（双条件同时满足才生成）：
    * 簇成员 ≥ 3        —— 同一 trigger 下已有 3 个不同 habitual action
    * 簇平均 confidence ≥ 0.75 —— 这些习惯都被反复观察验证过

生成的技能落盘：

    <skills_dir>/learned-<slug>/SKILL.md

* frontmatter：name: learned-<slug>、description: <trigger 一句话>
* 正文三段：触发情境（trigger）/ 建议行为（action，按置信度降序）/ 证据（前 3 条）
* 幂等：SKILL.md 已存在时跳过（不重复生成、不覆盖——已生成的技能交由
  curator 后续维护，对齐"完全可逆 + 用户意图优先"的项目原则）

slug 规则：与 store 的 `_slug` 同源（lower + 非 [a-z0-9_-] → '-'），但中文
trigger 会全变 '-'（如 "项目约定" → "------"），所以加保底链：

    1. 先从 trigger 提取 ascii 词（"使用 grep" → "grep"）
    2. 一个 ascii 词都没有 → "habit-<簇成员数>-<md5(trigger) 前 8 位>"
       （成员数 + 短 hash：目录名可读、同 trigger 稳定、不同 trigger 不撞车）

fail-open：单个簇生成失败只跳过该簇；调用方接线（agent/__init__.py）再包
一层 try/except，学习链路任何异常绝不影响主对话。
"""
import hashlib
import re
from pathlib import Path
from typing import List

from agent.atomic_io import atomic_write_text

# 演化门槛
_MIN_MEMBERS = 3
_MIN_AVG_CONFIDENCE = 0.75

# 正文证据最多列 3 条（与 skill description 一样走"少而精"）
_MAX_EVIDENCE_IN_MD = 3

# slug：与 store 同源的字符白名单 + 截断长度
_SLUG_RE = re.compile(r"[^a-z0-9_-]")
_SLUG_MAX = 60
# 提取 ascii 词（比 _SLUG_RE 更严：连续的 [a-z0-9_-] 才算词，避免纯标点）
_ASCII_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _skill_slug(trigger: str, insts) -> str:
    """trigger → 技能目录 slug（保底链见模块 docstring）。

    与 InstinctStore 的 _slug 规则同源但有 ascii 提取增强：
    store 落盘 key 允许全 '-'（文件名唯一性靠 hash 兜底），技能目录名
    面向用户可见，必须可读，所以提取不到 ascii 词时用
    "habit-<成员数>-<md5 前 8 位>" 保底。
    """
    words = _ASCII_WORD_RE.findall(str(trigger or "").lower())
    if words:
        # ascii 词拼接后仍过一遍 slug 白名单（join 用 '-'，本身就是安全字符）
        return "-".join(words)[:_SLUG_MAX]
    digest = hashlib.md5(str(trigger or "").encode("utf-8")).hexdigest()[:8]
    return f"habit-{len(insts)}-{digest}"


def _skill_md(trigger: str, insts) -> str:
    """生成 SKILL.md 全文（frontmatter + trigger/action/证据三段正文）。

    insts 按置信度降序排列——最可靠的习惯排最前，LLM 加载技能时
    第一眼看到的就是验证最充分的建议。
    """
    trigger = str(trigger or "").strip()
    slug = _skill_slug(trigger, insts)
    sorted_insts = sorted(insts, key=lambda i: i.confidence, reverse=True)

    # 证据：跨簇成员收集、保序去重、截前 3 条
    evidence: List[str] = []
    for inst in sorted_insts:
        for ev in (inst.evidence or []):
            if ev and ev not in evidence:
                evidence.append(ev)
    evidence = evidence[:_MAX_EVIDENCE_IN_MD]

    lines = [
        "---",
        f"name: learned-{slug}",
        f"description: {trigger}",  # description = trigger 一句话
        "---",
        "",
        "<!-- 本技能由 skill_learning 自动生成（instinct 簇达标演化）。"
        "可通过 skill_manage 归档/pin 维护。 -->",
        "",
        "## 触发情境",
        "",
        trigger,
        "",
        "## 建议行为（按置信度降序）",
        "",
    ]
    for idx, inst in enumerate(sorted_insts, 1):
        lines.append(
            f"{idx}. **{inst.action}**（confidence={inst.confidence:.2f}）")
    lines += [
        "",
        "## 证据",
        "",
    ]
    for ev in evidence:
        lines.append(f"- {ev}")
    lines.append("")
    return "\n".join(lines)


def maybe_evolve(store, scope: str, skills_dir) -> list:
    """检查指定 scope 下的 instinct 簇，达标的生成 SKILL.md。

    Args:
        store: InstinctStore（用它的 cluster(scope) 拿归一化分簇）。
        scope: "global" 或 "project:<key>"（只演化该 scope 的簇）。
        skills_dir: 技能根目录（生成 <skills_dir>/learned-<slug>/SKILL.md）。

    Returns:
        本次生成的 SKILL.md 路径列表（Path）；已存在/不达标的簇不在其中。

    幂等：目录已存在 SKILL.md 视为已生成，跳过（不覆盖 curator 可能在
    维护的内容）。单簇失败只跳过该簇，不影响其他簇。
    """
    generated = []
    clusters = store.cluster(scope)
    for _norm, insts in clusters.items():
        try:
            if len(insts) < _MIN_MEMBERS:
                continue
            avg_conf = sum(i.confidence for i in insts) / len(insts)
            if avg_conf < _MIN_AVG_CONFIDENCE:
                continue

            # trigger 用簇内首条原文（归一化 key 是小写压空白版，原文更可读）
            trigger = insts[0].trigger
            slug = _skill_slug(trigger, insts)
            skill_md = Path(skills_dir) / f"learned-{slug}" / "SKILL.md"
            if skill_md.exists():
                continue  # 幂等：已生成，后续维护交 curator

            skill_md.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(skill_md, _skill_md(trigger, insts))
            generated.append(skill_md)
        except Exception:
            continue  # 单簇失败不连坐其他簇（fail-open）
    return generated
