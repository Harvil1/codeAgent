# -*- coding: utf-8 -*-
"""技能演化器：习惯攒够了就自动升级成正式技能。

打比方：平时记的习惯是散落的便签，攒了一沓（同一个情境下 3 条不同做法、
且都被反复验证过）就装订成一本正经的说明书（技能 MD 文件）。

升级门槛（两个条件同时满足才动手）：
    * 簇成员 ≥ 3 —— 同一 trigger（触发情境）下已攒了 3 个不同做法
    * 簇平均置信度 ≥ 0.75 —— 这些做法都被反复观察验证过

生成的技能写到：

    <skills_dir>/learned-<slug>/SKILL.md

* frontmatter（文件头元信息）：name: learned-<slug>，description = trigger 一句话
* 正文三段：触发情境 / 建议行为（按置信度从高到低排）/ 证据（前 3 条）
* 幂等（不重不漏）：SKILL.md 已存在就跳过——不重复生成也不覆盖，生成过的
  技能交给 curator（后台维护工人）接管，对齐项目「完全可逆 + 用户意图
  优先」的原则

slug（目录名）规则：与 store 的 _slug 同源（小写 + 非 [a-z0-9_-] 换 '-'），
但中文 trigger 会全变 '-'（"项目约定" → "------"），所以有保底链：

    1. 先从 trigger 里抠出英文词（"使用 grep" → "grep"）
    2. 一个英文词都没有 → "habit-<簇成员数>-<md5(trigger) 前 8 位>"
       （成员数 + 短 hash：目录名可读、同一个 trigger 结果稳定、
        不同 trigger 不会撞车）

fail-open：单个簇生成失败只跳过该簇；调用方接线（agent/__init__.py）
还会再包一层 try/except——学习链路的任何异常绝不影响主对话。
"""
import hashlib
import re
from pathlib import Path
from typing import List

from agent.atomic_io import atomic_write_text

# 升级门槛：簇里至少 3 条习惯、平均置信度至少 0.75
_MIN_MEMBERS = 3
_MIN_AVG_CONFIDENCE = 0.75

# 技能正文里证据最多列 3 条（和技能描述一样，走「少而精」路线）
_MAX_EVIDENCE_IN_MD = 3

# slug 字符白名单 + 截断长度（与 store 的 _slug 同源，行为保持一致）
_SLUG_RE = re.compile(r"[^a-z0-9_-]")
_SLUG_MAX = 60
# 抠英文词用的正则（比 _SLUG_RE 更严：连续的 [a-z0-9_-] 开头才算词，
# 避免把纯标点当词）
_ASCII_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _skill_slug(trigger: str, insts) -> str:
    """把 trigger 转成技能目录名的 slug。

    保底链：先抠英文词拼目录名；一个英文词都抠不到就用
    "habit-<成员数>-<md5 前 8 位>"（详见模块 docstring）。

    参数：
        trigger：触发情境原文。
        insts：这个簇里的习惯条目（全变 '-' 的保底命名要用成员数）。

    返回：技能目录名 slug（≤60 字符）。

    为什么和 InstinctStore 的 _slug 不完全一样：store 的文件名允许全是
    '-'（唯一性反正有 hash 兜底）；技能目录名是给用户看的，必须可读。
    """
    words = _ASCII_WORD_RE.findall(str(trigger or "").lower())
    if words:
        # 词间用 '-' 连接本身就是安全字符，不必再过白名单
        return "-".join(words)[:_SLUG_MAX]
    digest = hashlib.md5(str(trigger or "").encode("utf-8")).hexdigest()[:8]
    return f"habit-{len(insts)}-{digest}"


def _skill_md(trigger: str, insts) -> str:
    """生成 SKILL.md 的完整内容。

    参数：
        trigger：触发情境原文（写成技能的 description 和「触发情境」段）。
        insts：这个簇里的习惯条目列表。

    返回：完整的 SKILL.md 文本（frontmatter 头 + 触发情境/建议行为/证据
    三段正文）。

    条目按置信度从高到低排——最可靠的建议排最前，AI 加载技能时第一眼
    看到的就是验证最充分的做法。
    """
    trigger = str(trigger or "").strip()
    slug = _skill_slug(trigger, insts)
    sorted_insts = sorted(insts, key=lambda i: i.confidence, reverse=True)

    # 证据从各条习惯里汇总：保序去重，只留前 3 条
    evidence: List[str] = []
    for inst in sorted_insts:
        for ev in (inst.evidence or []):
            if ev and ev not in evidence:
                evidence.append(ev)
    evidence = evidence[:_MAX_EVIDENCE_IN_MD]

    lines = [
        "---",
        f"name: learned-{slug}",
        f"description: {trigger}",  # 描述就是触发情境那句话，让索引能按情境搜到
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


def maybe_evolve(store, scope: str, skills_dir, *,
                 min_members: int = _MIN_MEMBERS,
                 min_avg_confidence: float = _MIN_AVG_CONFIDENCE) -> list:
    """检查习惯簇，把达标的升级成技能文件。

    参数：
        store：InstinctStore（用它的 cluster(scope) 拿按情境分好的簇）。
        scope：作用域——"global" 或 "project:<key>"，只处理这个 scope 的簇。
        skills_dir：技能根目录（生成的文件落在
            <skills_dir>/learned-<slug>/SKILL.md）。
        min_members：簇最小成员数门槛（keyword-only；可从
            config["skill_learning"]["evolve_min_cluster"]
            传入，默认 3）。
        min_avg_confidence：簇平均置信度门槛（keyword-only；config 键
            evolve_threshold，默认 0.75）。

    返回：本次新生成的 SKILL.md 路径列表（Path）；已经生成过的和不达标
    的簇不在其中。

    幂等：目录里已有 SKILL.md 就视为生成过，跳过——绝不覆盖 curator
    （后台维护工人）可能正在维护的内容。单个簇失败只跳过它，不影响其他簇。
    """
    generated = []
    clusters = store.cluster(scope)
    for _norm, insts in clusters.items():
        try:
            if len(insts) < min_members:
                continue
            avg_conf = sum(i.confidence for i in insts) / len(insts)
            if avg_conf < min_avg_confidence:
                continue

            # 情境名用簇内第一条的原文（归一化 key 是压过空白的小写版，原文可读性更好）
            trigger = insts[0].trigger
            slug = _skill_slug(trigger, insts)
            skill_md = Path(skills_dir) / f"learned-{slug}" / "SKILL.md"
            if skill_md.exists():
                continue  # 幂等：生成过就不再动，后续维护是 curator 的事

            skill_md.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(skill_md, _skill_md(trigger, insts))
            generated.append(skill_md)
        except Exception:
            continue  # 这一簇炸了不连坐别的簇（fail-open）
    return generated
