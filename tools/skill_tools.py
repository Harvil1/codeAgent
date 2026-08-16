"""技能查看工具：skills_list + skill_view。

skills_list：列出所有可用技能
skill_view：查看某技能的完整内容
"""

import json
import re
from pathlib import Path

from tools.registry import registry
from agent.skill_commands import parse_frontmatter
from tools.skill_usage import bump_view, load_usage


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": (
        "列出所有可用技能。传 query 时按 TF-IDF 相关性排序返回 Top 匹配"
        "（推荐先 query 定向找，找不到再全量列）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词（对 name/description 建词频向量打分，中英文都支持）",
            },
        },
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "查看某技能的完整内容。",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "技能名"},
        },
        "required": ["name"],
    },
}


def _get_skills_dirs(kwargs: dict):
    """获取技能扫描目录列表（内置 + 用户 + 已启用插件）。

    顺序 = 优先级，后者覆盖前者（用户/插件可覆盖内置同名技能）。
    自定义 omnimate_home 时用自定义用户目录替换默认用户目录。
    """
    from constants import all_skills_dirs, get_omnimate_home
    dirs = list(all_skills_dirs())
    home = kwargs.get("omnimate_home")
    if home:
        home_path = Path(home)
        if home_path != get_omnimate_home():
            # 自定义 home：内置目录 + 自定义用户目录 + 插件目录
            dirs = [dirs[0], home_path / "skills"] + dirs[2:]
    return dirs


def _get_usage_dir(kwargs: dict) -> Path:
    """usage 统计写入的用户技能目录（第一个非内置目录）。"""
    dirs = _get_skills_dirs(kwargs)
    return dirs[1] if len(dirs) > 1 else dirs[0]


def _find_skill_md(name: str, dirs) -> Path:
    """跨目录按名字找 SKILL.md（用户/插件优先）。找不到返回 None。"""
    for d in reversed(dirs):
        p = Path(d) / name / "SKILL.md"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# R22 #27：TF-IDF 技能搜索（对齐 CC localSearch 的轻量版）
# ---------------------------------------------------------------------------

# 中文常用虚词 + 英文停用词（对齐 CC STOP_WORDS 精简版；中文按字切分时
# 虚词高频无区分度，直接进停用表）
_SKILL_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "for", "on", "at", "by", "with", "and", "or", "not", "no", "do",
    "does", "did", "will", "would", "could", "should", "can", "may", "how",
    "what", "when", "which", "who", "that", "this", "these", "those", "it",
    "its", "as", "from", "into", "about", "use", "using", "used",
    # 中文虚词/疑问代词
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "和", "与",
    "或", "不", "没", "有", "这", "那", "个", "什么", "怎么", "如何", "哪",
    "用", "把", "被", "给", "对", "从", "到", "以", "为", "就", "会", "能",
    "要", "可以", "一个", "一些",
})

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_\-]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list:
    """分词：英文连字符词组整体 + 部件都收（release-notes → 3 tokens，
    部件让子词查询可命中）；中文按单字（CJK 无空格，单字可匹配词首）。"""
    tokens = []
    for m in _TOKEN_RE.finditer(text or ""):
        t = m.group(0).lower()
        if t in _SKILL_STOP_WORDS:
            continue
        tokens.append(t)
        # 连字符词组拆部件（子词可查询）
        if "-" in t:
            for part in t.split("-"):
                if len(part) >= 2 and part not in _SKILL_STOP_WORDS:
                    tokens.append(part)
    return tokens


def _skill_search_rank(skills: dict, query: str, top_n: int = 10) -> list:
    """TF-IDF 打分排序（query tokens 对技能 name/description 词频向量）。

    轻量版：TF（词频）× IDF（query 词在多少技能出现——稀有词权重高）；
    name 命中加权 ×3（名字是最强信号）。无 query 或无命中返回空列表。
    """
    q_tokens = _tokenize(query)
    if not q_tokens or not skills:
        return []
    items = list(skills.values())
    # 每技能的 token 列表（name 加权 ×3）
    docs = []
    for s in items:
        toks = _tokenize(s.get("name", "")) * 3 + _tokenize(s.get("description", ""))
        docs.append(toks)
    # IDF：query token 的文档频率
    n = len(docs)
    scores = []
    for i, toks in enumerate(docs):
        tf_map = {}
        for t in toks:
            tf_map[t] = tf_map.get(t, 0) + 1
        score = 0.0
        for qt in q_tokens:
            if qt not in tf_map:
                continue
            df = sum(1 for d in docs if qt in d)
            idf = 1.0 + (n - df) / n  # 稀有词 > 1，全文档命中 → 1
            # 中文单字（len==1）TF 封顶 2——长描述里高频虚字/常用字会刷分
            tf = min(tf_map[qt], 2) if len(qt) == 1 else tf_map[qt]
            score += tf * idf
        if score > 0:
            scores.append((score, items[i]))
    scores.sort(key=lambda x: -x[0])
    return [s for _, s in scores[:top_n]]


def _handle_skills_list(args: dict, **kwargs) -> str:
    dirs = _get_skills_dirs(kwargs)
    usage = load_usage(_get_usage_dir(kwargs))
    skills = {}
    for d in dirs:  # 顺序：内置 → 用户 → 插件，后者覆盖前者
        d = Path(d)
        if not d.exists():
            continue
        for skill_md in sorted(d.glob("*/SKILL.md")):
            name = skill_md.parent.name
            rec = usage.get(name, {})
            # 跳过归档的
            if rec.get("state") == "archived":
                continue

            # 尝试从 frontmatter 读描述
            description = rec.get("description", "")
            if not description:
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    frontmatter, _ = parse_frontmatter(content)
                    description = frontmatter.get("description", "")
                except Exception:
                    pass

            skills[name] = {
                "name": name,
                "description": description,
                "use_count": rec.get("use_count", 0),
                "view_count": rec.get("view_count", 0),
                "state": rec.get("state", "active"),
            }

    # R22 #27：query 参数 → TF-IDF 相关性排序（找得到才返回，找不到全量列）
    query = (args.get("query") or "").strip()
    if query:
        ranked = _skill_search_rank(skills, query)
        if ranked:
            return json.dumps({
                "query": query,
                "matched": len(ranked),
                "total": len(skills),
                "skills": ranked,
            }, ensure_ascii=False)
        return json.dumps({
            "query": query,
            "matched": 0,
            "total": len(skills),
            "skills": [],
            "hint": "无匹配——去掉 query 参数看全量列表",
        }, ensure_ascii=False)

    return json.dumps({"skills": list(skills.values())}, ensure_ascii=False)


def _handle_skill_view(args: dict, **kwargs) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    skill_md = _find_skill_md(name, _get_skills_dirs(kwargs))
    if skill_md is None:
        return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

    content = skill_md.read_text(encoding="utf-8")
    bump_view(_get_usage_dir(kwargs), name)  # 查看计数 +1

    return json.dumps({
        "name": name,
        "content": content,
        "path": str(skill_md),
    }, ensure_ascii=False)


registry.register(
    name="skills_list",
    toolset="core",
    schema=SKILLS_LIST_SCHEMA,
    handler=_handle_skills_list,
    emoji="📋",
    isConcurrencySafe=True,  # 只读：列技能目录，无副作用，可并发
)

registry.register(
    name="skill_view",
    toolset="core",
    schema=SKILL_VIEW_SCHEMA,
    handler=_handle_skill_view,
    emoji="👁️",
    isConcurrencySafe=True,  # 只读：读技能正文（bump view 计数是小副作用，对并发不致命），可并发
)


# ---------------------------------------------------------------------------
# load_skill：LLM 主动按需加载技能正文
# ---------------------------------------------------------------------------

LOAD_SKILL_SCHEMA = {
    "name": "load_skill",
    "description": (
        "按名字加载技能的完整指令正文。system prompt 里有技能索引"
        "（名字+描述，~100 tokens/技能），当你判断需要某个技能的详细流程时"
        "调用此工具获取完整内容（~2000 tokens/技能）。"
        "区别于 skill_view：load_skill 只返回指令正文（去 frontmatter），"
        "专门给 LLM 按需读取执行。"
        "\n\n支持技能束：传 name=\"bundle:<bundle_name>\" 一次性加载多个技能"
        "（在 ~/.OmniMate/.skill-bundles.json 配置）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "技能名"},
        },
        "required": ["name"],
    },
}


def _handle_load_skill(args: dict, **kwargs) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    usage_dir = _get_usage_dir(kwargs)
    dirs = _get_skills_dirs(kwargs)

    # batch1-T3: 支持 bundle:<name> 加载技能束
    if name.startswith("bundle:"):
        bundle_name = name[len("bundle:"):]
        from agent.skill_bundle import load_bundle
        result = load_bundle(bundle_name, usage_dir)
        # 对成功加载的技能 bump view 计数
        for sname in result.get("skills_loaded", []):
            try:
                bump_view(usage_dir, sname)
            except Exception:
                pass
        return json.dumps(result, ensure_ascii=False)

    skill_md = _find_skill_md(name, dirs)
    if skill_md is None:
        return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

    content = skill_md.read_text(encoding="utf-8")
    # 去掉 frontmatter，只返回指令正文
    frontmatter, body = parse_frontmatter(content)

    # T3（核心机制对齐第 3 项）：frontmatter files: 参考文件附件
    attachments = _load_skill_attachments(skill_md.parent, frontmatter.get("files"), kwargs)

    bump_view(usage_dir, name)  # 加载也计入 view 计数

    # round3: context:fork 技能提示 LLM 用 subagent 跑
    if frontmatter.get("context") == "fork":
        return json.dumps({
            "name": name,
            "body": body.strip(),
            "path": str(skill_md),
            "attachments": attachments,
            "fork_required": True,
            "hint": ("该技能声明 context:fork，应在隔离子代理里执行。"
                     "请用 subagent 工具派生子代理，把上述技能正文作为子代理指令运行。"),
        }, ensure_ascii=False)

    # allowed-tools / disallowed-tools：技能触发时临时调整可用工具集
    allowed = frontmatter.get("allowed-tools")
    disallowed = frontmatter.get("disallowed-tools")
    agent = kwargs.get("agent_ref")
    if agent is not None and (allowed or disallowed):
        try:
            agent._skill_tool_scope = (allowed, disallowed)
        except Exception:
            pass

    return json.dumps({
        "name": name,
        "body": body.strip(),
        "path": str(skill_md),
        "attachments": attachments,
    }, ensure_ascii=False)


def _load_skill_attachments(skill_dir, files, kwargs: dict) -> list:
    """读技能附件（T3），max_chars 从 config skills.file_attachment_max_chars 取。"""
    from agent.skill_commands import read_skill_attachment_files, DEFAULT_ATTACHMENT_MAX_CHARS
    cfg = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else {}
    max_chars = (cfg.get("skills") or {}).get(
        "file_attachment_max_chars", DEFAULT_ATTACHMENT_MAX_CHARS,
    )
    try:
        return read_skill_attachment_files(str(skill_dir), files, max_chars=int(max_chars))
    except Exception:
        return []


registry.register(
    name="load_skill",
    toolset="core",
    schema=LOAD_SKILL_SCHEMA,
    handler=_handle_load_skill,
    emoji="📖",
    isConcurrencySafe=False,  # 副作用：可能改 agent._skill_tool_scope（实例级状态），保守标 False
)
