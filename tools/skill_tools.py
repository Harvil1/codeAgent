"""技能查看工具包：skills_list（列技能/搜技能）+ skill_view（看某个技能全文）+ load_skill（LLM 按需取技能正文）。

技能（skill）是存在磁盘上的 Markdown 使用说明书，agent 干活时按需翻阅。
本文件给 LLM 提供三个查询工具：
- skills_list：列出所有可用技能；带 query 关键词时按相关性排序只返回最像的几条
- skill_view：查看某个技能的完整原文（给用户视角看，含文件头元信息）
- load_skill：LLM 主动加载技能正文来照着执行（system prompt 里只放目录省 token，详细内容用这个取）

在项目里的位置：属于工具层（tools/），注册进中央工具注册表（registry）暴露给 LLM；
读写统计委托给 tools/skill_usage.py，目录发现委托给 constants.py。
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
    """收集要去哪些目录里找技能文件，返回目录路径列表。

    技能可能放在三个地方——软件自带的（内置）、用户自己的（~/.codeAgent/skills）、
    插件带来的。列表顺序就是优先级：排后面的同名技能会覆盖排前面的（所以用户能改造内置技能）。

    参数：
    - kwargs：工具调用时传进来的上下文。这里只关心 codeagent_home（自定义的数据目录），
      传了就用它下面的 skills 目录替换默认用户目录。

    返回：目录路径列表，按「内置 → 用户 → 插件」排列。
    """
    from constants import all_skills_dirs, get_codeagent_home
    dirs = list(all_skills_dirs())
    home = kwargs.get("codeagent_home")
    if home:
        home_path = Path(home)
        if home_path != get_codeagent_home():
            # 自定义 home：内置目录 + 自定义用户目录 + 插件目录
            dirs = [dirs[0], home_path / "skills"] + dirs[2:]
    return dirs


def _get_usage_dir(kwargs: dict) -> Path:
    """决定把使用统计写到哪个目录：用户技能目录（列表里第一个非内置的）。

    使用统计（查看/使用次数）不写进内置目录，写到用户自己的目录。

    参数：
    - kwargs：工具调用上下文，用来算出技能目录列表。

    返回：统计文件所在的目录路径。
    """
    dirs = _get_skills_dirs(kwargs)
    return dirs[1] if len(dirs) > 1 else dirs[0]


def _find_skill_md(name: str, dirs) -> Path:
    """按技能名在多个目录里找到它的 SKILL.md 文件（用户/插件的同名技能优先）。

    同一个技能名可能在多个目录都有，倒着遍历（从优先级高的开始）保证取到覆盖版。

    参数：
    - name：技能名（就是技能目录的文件夹名）
    - dirs：要搜的目录列表

    返回：找到的 SKILL.md 完整路径；全都找不到就返回 None。
    """
    for d in reversed(dirs):
        p = Path(d) / name / "SKILL.md"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# TF-IDF 技能搜索（轻量版）。
# TF-IDF 是搜索排序的老办法：一个词在这份文档里出现越多（TF）、同时在别的文档里越少见（IDF），
# 就越能代表这份文档，得分越高。
# ---------------------------------------------------------------------------

# 停用词表（搜索时直接忽略的词）：中文虚词 + 英文常见功能词。
# 为什么需要：「的」「如何」这类词到处都是，没有任何区分度，留着只会干扰打分。
# （取精简版）
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
    """把一段文字切成一个个「词」（token），供搜索打分用。

    中文没有空格分不出词，按单个汉字切（查「登录」也能命中「登录页」）；
    英文按单词切；带连字符的词组（如 release-notes）整体收一份、拆开的部件
    （release、notes）也各收一份——用其中一个词去搜也能找到。

    参数：
    - text：要切分的文字，可以是空串。

    返回：小写化的词列表（停用词已剔除）。
    """
    tokens = []
    for m in _TOKEN_RE.finditer(text or ""):
        t = m.group(0).lower()
        if t in _SKILL_STOP_WORDS:
            continue
        tokens.append(t)
        # 连字符词组再补上拆开的部件，方便用半个词也能搜到
        if "-" in t:
            for part in t.split("-"):
                if len(part) >= 2 and part not in _SKILL_STOP_WORDS:
                    tokens.append(part)
    return tokens


def _skill_search_rank(skills: dict, query: str, top_n: int = 10) -> list:
    """按相关性给技能打分排序，返回最匹配的前 N 个。

    打分思路（轻量版 TF-IDF）：搜索词在技能的名字/描述里出现越多得分越高（TF），
    且这个词越少见（在越少技能里出现）权重越大（IDF，稀有词更能说明相关性）；
    名字里命中按 3 倍计——名字是最强的信号。

    参数：
    - skills：技能信息字典，键是技能名，值含 name/description 等字段
    - query：用户的搜索关键词
    - top_n：最多返回几个，默认 10

    返回：按得分从高到低排的技能信息列表；query 为空、切不出词或没有任何命中时返回空列表。
    """
    q_tokens = _tokenize(query)
    if not q_tokens or not skills:
        return []
    items = list(skills.values())
    # 每个技能的词表；名字里的词重复 3 遍 = 名字命中加权 3 倍
    docs = []
    for s in items:
        toks = _tokenize(s.get("name", "")) * 3 + _tokenize(s.get("description", ""))
        docs.append(toks)
    # IDF：算每个搜索词出现在几个技能里（出现得越少越「稀有」，越值钱）
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
            idf = 1.0 + (n - df) / n  # 稀有词权重 > 1；所有技能都有这个词时就只剩 1
            # 单个汉字的词频封顶 2 次：防止长描述里反复出现的常用字刷高得分
            tf = min(tf_map[qt], 2) if len(qt) == 1 else tf_map[qt]
            score += tf * idf
        if score > 0:
            scores.append((score, items[i]))
    scores.sort(key=lambda x: -x[0])
    return [s for _, s in scores[:top_n]]


def _handle_skills_list(args: dict, **kwargs) -> str:
    """skills_list 的实际处理函数：扫描技能目录拼出技能清单，支持按关键词过滤排序。

    参数：
    - args：LLM 传的工具参数，这里只看可选的 query（搜索关键词）
    - kwargs：运行时上下文（codeagent_home 等），用来定位技能目录

    返回：JSON 字符串。带 query 且有匹配时返回按相关性排序的结果；
    有 query 但没匹配时返回空列表加提示；没 query 时返回全量列表。
    """
    dirs = _get_skills_dirs(kwargs)
    usage = load_usage(_get_usage_dir(kwargs))
    skills = {}
    for d in dirs:  # 顺序：内置 → 用户 → 插件；后扫到的同名技能覆盖先扫到的
        d = Path(d)
        if not d.exists():
            continue
        for skill_md in sorted(d.glob("*/SKILL.md")):
            name = skill_md.parent.name
            rec = usage.get(name, {})
            # 已归档的技能不出现在清单里
            if rec.get("state") == "archived":
                continue

            # 统计里没存描述的话，就读 SKILL.md 头部的 frontmatter（--- 包住的元信息区）拿一份
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

    # 带 query 参数时按 TF-IDF 相关性排序——
    # 搜得到就只返回匹配的；搜不到就返回提示让 LLM 去掉 query 看全量
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
    """skill_view 的实际处理函数：读出指定技能 SKILL.md 的完整原文。

    参数：
    - args：LLM 传的工具参数，只看必填的 name（技能名）
    - kwargs：运行时上下文（codeagent_home 等），用来定位技能目录

    返回：JSON 字符串，含技能名、完整内容、文件路径；名字为空或技能不存在时返回 error。
    """
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    skill_md = _find_skill_md(name, _get_skills_dirs(kwargs))
    if skill_md is None:
        return json.dumps({"error": f"技能不存在: {name}"}, ensure_ascii=False)

    content = skill_md.read_text(encoding="utf-8")
    bump_view(_get_usage_dir(kwargs), name)  # 顺手把「被查看次数」加一，供后续推荐/清理参考

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
    isConcurrencySafe=True,  # 只读不改动任何东西，多个并发跑也安全
)

registry.register(
    name="skill_view",
    toolset="core",
    schema=SKILL_VIEW_SCHEMA,
    handler=_handle_skill_view,
    emoji="👁️",
    isConcurrencySafe=True,  # 基本只读（只是顺手记一下查看计数，这个小副作用并发跑也无妨）
)


# ---------------------------------------------------------------------------
# load_skill：LLM 主动按需加载技能正文（相当于「翻到这一页说明书照着做」）
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
        "（在 ~/.codeAgent/.skill-bundles.json 配置）。"
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
    """load_skill 的实际处理函数：取出技能的指令正文交给 LLM 照着执行。

    system prompt 里只放技能目录（省 token），LLM 判断需要某个技能时调这里取全文。
    顺带处理几种特殊情况：技能束（一次加载一组技能）、frontmatter 声明的附件文件、
    context:fork 技能（要在隔离子代理（主对话派出去帮忙干活的分身）里跑）、
    allowed-tools/disallowed-tools（技能触发的临时工具开关）。

    参数：
    - args：LLM 传的工具参数，只看必填的 name（技能名，可以是 "bundle:<束名>"）
    - kwargs：运行时上下文（codeagent_home、config、agent_ref 等）

    返回：JSON 字符串，含技能正文、路径、附件；名字为空或技能不存在时返回 error。
    """
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name 不能为空"}, ensure_ascii=False)

    usage_dir = _get_usage_dir(kwargs)
    dirs = _get_skills_dirs(kwargs)

    # 支持传 bundle:<束名> 一次加载一整组技能
    if name.startswith("bundle:"):
        bundle_name = name[len("bundle:"):]
        from agent.skill_bundle import load_bundle
        result = load_bundle(bundle_name, usage_dir)
        # 束里每个成功加载的技能也记一次查看
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
    # 把文件头元信息区（frontmatter）剥掉，只留指令正文
    frontmatter, body = parse_frontmatter(content)

    # frontmatter 里写了 files: 时，把参考文件一起读进来当附件
    attachments = _load_skill_attachments(skill_md.parent, frontmatter.get("files"), kwargs)

    bump_view(usage_dir, name)  # 加载也算一次查看，进统计

    # 声明了 context:fork 的技能不可以在主对话里直接跑，要派子代理去跑
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

    # 技能可以通过 frontmatter 的 allowed-tools / disallowed-tools 临时收窄可用工具范围
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
    """读取技能声明的附件文件（参考文件），返回内容列表。

    技能除说明书正文外还可带几个参考文件，加载技能时一起读进来。

    参数：
    - skill_dir：技能所在目录（附件相对它找）
    - files：frontmatter 里声明的文件列表；为空则没有附件
    - kwargs：运行时上下文；从 config 的 skills.file_attachment_max_chars 读单文件最大字符数

    返回：附件内容列表；读取出任何问题都返回空列表（附件是锦上添花，失败不报错）。
    """
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
    isConcurrencySafe=False,  # 有副作用：可能改动 agent 实例的工具开关状态，保守起见不许并发
)
