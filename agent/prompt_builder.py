"""system prompt（系统提示词，模型每次对话都收到的"开场白"）的组装车间。

在项目里的位置：被 agent/__init__.py（AIAgent 主类）调用，产出发给 LLM 的
system prompt；素材来自记忆、技能目录、项目 CODEAGENT.md 等。

四条关键原则（改这里之前必须懂）：
1. 会话开始时构建一次就缓存住（_cached_system_prompt），中途不再重建
2. 同一会话内必须字节级不变——LLM 服务商按"前缀一模一样"来复用缓存，
   改一个字缓存就全废，费用直接翻倍
3. 记忆是开工那一刻拍的照片（frozen 快照），本次会话中途不刷新
4. 技能只列"名字 + 一句话描述"，正文等模型自己调 load_skill 按需取

三层结构（目的是让缓存命中率最大化）：
- stable：跨会话都不变（身份、各种指导文案）——缓存几乎永远命中
- context：单个会话内不变（记忆、技能索引、CODEAGENT.md）——会话内命中
- volatile：每轮都可能变（todo、提醒）——不指望命中缓存
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 下面几个 GUIDANCE 常量是"使用说明书"文本，拼进 system prompt，
# 教模型怎么用记忆/技能/会话搜索这些家伙什
# ---------------------------------------------------------------------------

MEMORY_GUIDANCE = (
    "## 记忆系统使用指南\n"
    "用 memory 工具保存稳定事实:用户偏好、环境细节、工具怪癖、约定。\n"
    "记忆每轮注入,保持精简——只存一周后仍重要的事实。\n\n"
    "❌ 不存任务进度/会话结果/日志/临时状态、PR/issue 号、commit SHA、"
    "'修了 bug X'、'阶段 N 完成'、文件计数等易过时内容"
    "(这些用 session_search 从历史查)。\n"
    "✅ 写成声明式事实:'项目使用 pytest' 而非 '必须用 pytest 跑'。\n"
    "新方法、解过的问题 → 用 skill_manage 存成技能。\n\n"
    "### 主动学习\n"
    "反复观察到某模式(如连续 3 次用 uv 而非 pip)但不确定是否长期偏好时,"
    "先问用户确认,再调 memory 保存(type=feedback, confidence=1.0)。"
)

SESSION_SEARCH_GUIDANCE = (
    "## 会话搜索指南\n"
    "当用户提到过去的对话内容，或你怀疑有相关的跨会话上下文时，"
    "先用 session_search 查找，再问用户。"
)

SKILLS_GUIDANCE = (
    "## 技能系统指南\n"
    "复杂任务(5+ 工具调用)、棘手错误、非平凡工作流完成后,"
    "用 skill_manage 存成技能复用;发现技能过时/不完整/错误,"
    "立即 skill_manage(action='patch') 修复。\n\n"
    "### 调用技能的 3 条铁律\n"
    "1. **load_skill 后照搬 Quick Start**:按模板改参数执行,不要自己发明命令格式。\n"
    "2. **不确定就 help**:外部 CLI 格式不确定时,先调 `<cli> --help` 一次,别瞎猜。\n"
    "3. **查完就动手**:只读命令(get/query/describe)用 1-2 次后即用写命令"
    "(add/set/update)动手;连续 3+ 次只读还在查 = 分析瘫痪,停下规划或问用户。"
)

TOOL_USAGE_GUIDANCE = (
    "## 工具使用规范\n"
    "- 工具结果都是 JSON 字符串,解析后使用;失败返回 {\"error\":...},据此自我修正\n"
    "- 文件操作必须指定 encoding='utf-8';不要假设工具可用(check_fn 可能隐藏)\n"
    "- **临时文件清理**:write_file 建的临时脚本/中间文件,执行完立即用 terminal 删除\n"
    "- **上下文占位消息识别**:看到 [snip_compact] / micro_compacted / "
    "[紧急上下文压缩] / [COMPACT_BOUNDARY] 占位消息且需更早上下文时,"
    "按占位消息里的快照路径"
    "(压缩快照在 .transcripts/ 目录,latest.txt 指向最新一份;"
    "大输出在 .task_outputs/tool-results/ 下)"
    "用 read_file 读回(在 agent_home 下,默认安全;快照文件大,用 offset/limit 分段)\n"
    "- **关键结论钉住**:长任务里给出重要决策/架构结论的回复,内容开头加 "
    "`[pinned] ` 标记——上下文折叠/压缩时钉住的消息原样保留,不被摘要改写\n"
    "- **写入白名单**:write_file 默认只允许 cwd 和 ~/.codeAgent;写其他位置会触发审批,"
    "同意后进持久化白名单。不要绕过——用户拒绝就换个白名单内位置写\n"
    "- **自身源码保护**:不要修改 CodeAgent 自身源码"
    "(开发=项目根,打包=site-packages 安装目录)。你是工具,用户用你改**他们的项目**,不是改你自己\n"
    "- **依赖安装(跟随 cwd)**:在用户当前目录用其项目环境装包:"
    "`uv pip install <pkg>`(推荐,尊重用户项目约定)或 `pip install <pkg>`;"
    "优先复用已有 .venv/pyproject.toml,不污染系统 Python;"
    "连续 2 次装失败 → 停下问用户\n"
    "- **代码输出(跟随 cwd)**:脚本写到当前目录,输出(PPT/Excel/Word 等)写到 "
    "`<cwd>/outputs/`;用户指定别的位置就照做(白名单自动处理审批)\n"
    "- **先查技能**:任何任务前先扫技能索引,判断有无适用流程技能"
    "(设计先行/调试/写计划等),有就先 load_skill('using-codeagent') 看总纲\n"
    "- **后台任务无需轮询**:bg_start / subagent(background=true) 完成时会以 "
    "<task_notification>/<delegation_completion> 通知你,主对话空闲时会自动唤醒"
    "继续处理——不要循环查 bg_status;本轮要收工时若仍有后台任务在跑"
    "(见 <background_tasks_running>),向用户说明完成后会自动跟进"
)


TASK_GUIDANCE = (
    "## 任务追踪（Task System）\n"
    "3 步以上的任务必须先调 task_create 创建任务列表，每步完成调 task_complete；"
    "任务之间的依赖用 blocked_by 字段声明；多步并行用 task_list 查看 ready 任务。"
    "Task System 跨会话持久化（~/.codeAgent/.tasks/）。"
)


DELEGATE_GUIDANCE = (
    "## 子代理委托（subagent）\n"
    "大项目探索、多个独立子任务、或单子任务预计 10+ 次工具调用,必须用 subagent "
    "省父 context(可 tasks=[...] 批量并行)。\n"
    "不要委托:强依赖父上下文(如基于已读内容做判断)、短任务(<5 次调用)、顺序型任务。\n"
    "用法:subagent(prompt=..., role='leaf', summary_only=True);"
    "summary_only 自动把结果压缩成 300 字摘要回填。\n"
    "**并行铁律**:需要同时派多个子代理(如并行探索各模块)时,必须用 tasks=[...] "
    "一次调用;禁止发多个独立 subagent——独立调用是串行执行的,逐个等待,不会并行。\n"
    "判断标准:同一领域打算连续 read_file 5+ 次,立刻停,委托子代理——"
    "父 context 留给综合判断。"
)


# ---------------------------------------------------------------------------
# 三层结构的数据容器
# ---------------------------------------------------------------------------

@dataclass
class SystemPromptLayers:
    """装 system prompt 三层文本的容器。

    - stable: 跨会话不变的部分（身份、指导、工具文档）
    - context: 单会话内不变的部分（记忆索引、技能索引、CODEAGENT.md）
    - volatile: 每轮可变的部分（todo、reminder、extra_instructions）

    为什么要分层：LLM 服务商的缓存按"消息前缀是否一致"来复用，越靠前的
    内容越稳定、缓存命中越多。把不变的内容排前面、易变的排后面，
    stable 层可以跨会话命中，context 层在会话内命中，整体省钱提速。
    """
    stable: str
    context: str
    volatile: str

    def render_flat(self) -> str:
        """把三层拼成一个大字符串。给老接口用（老接口只收一个字符串）。"""
        parts = [self.stable, self.context, self.volatile]
        return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 主构建函数（三层版，新代码都用这个）
# ---------------------------------------------------------------------------

def build_system_prompt_layers(
    *,
    memory_store=None,
    memory_manager=None,
    enabled_toolsets: Optional[List[str]] = None,
    skills_dir: Optional[Path] = None,
    context_files: Optional[List[Path]] = None,
    extra_instructions: str = "",
    include_guidance: bool = True,
    # volatile 层的素材（运行时才有的东西，从外面传进来）
    task_state: Optional[str] = None,
    reminder: Optional[str] = None,
    # === 自定义子代理可跳过项目 CODEAGENT.md 注入（省 token）===
    omit_project_memory: bool = False,
    # === 输出风格节的文本（拼进 context 层；空=未启用风格）===
    output_style_text: str = "",
) -> SystemPromptLayers:
    """组装出三层 system prompt——缓存按"前缀一致"复用，把不变的内容排前面、易变的排后面。

      - stable：跨会话不变（同版本同一台机器，几乎 100% 命中缓存）
      - context：单会话内不变（记忆/技能/CODEAGENT.md，会话内大部分轮次命中）
      - volatile：每轮可变（todo / 提醒），不指望命中缓存

    参数：
        memory_store: 记忆仓库对象。用来读记忆索引快照（snapshot_for_prompt，
            ≤200 行/25KB）拼进 context 层——会话级一次，中途写新记忆不打穿
            前缀缓存；详情仍靠每轮检索式注入
        memory_manager: 记忆管理器，能产出扩展记忆块拼进 context 层；没有就不拼
        enabled_toolsets: 当前启用的工具集名字列表（目前本函数未直接使用）
        skills_dir: 技能目录；不传就自动用内置 + 用户两个默认目录
        context_files: 额外要拼进 prompt 的文件路径列表（存在才读）
        extra_instructions: 额外指令文本，进 volatile 层
        include_guidance: 是否拼入各段"使用指南"文案；子代理可关掉省 token
        task_state: 当前任务列表的文本快照，进 volatile 层
        reminder: 给模型的提醒文本，进 volatile 层
        omit_project_memory: True 时跳过项目 CODEAGENT.md 注入——给只读/
            轻量子代理省 token 用
        output_style_text: 输出风格节文本，拼在 context 层末尾；空串表示
            未启用风格、不拼

    返回：SystemPromptLayers 三层容器。
    """
    # ---- stable 层（跨会话不变的指导文案）----
    stable_parts = []
    if include_guidance:
        stable_parts.extend([
            MEMORY_GUIDANCE, SKILLS_GUIDANCE,
            SESSION_SEARCH_GUIDANCE, TOOL_USAGE_GUIDANCE, TASK_GUIDANCE,
            DELEGATE_GUIDANCE,
        ])
    stable = "\n\n".join(stable_parts)

    # ---- context 层（单会话内不变的东西）----
    context_parts = []
    # 先注入当前工作目录——明确告诉模型当前目录是哪个、"以当前为准"
    # （防被记忆检索结果里别的项目的条目带偏，跑去翻旧项目）
    try:
        from agent.workspace_context import get_workspace_cwd
        context_parts.append(
            "## 当前工作目录\n"
            f"{get_workspace_cwd()}\n\n"
            "用户在此目录启动了会话。用户说\"这个项目\"时指当前工作目录；"
            "记忆检索结果或历史会话中出现的**其他项目路径是历史信息**，"
            "不代表用户当前所在的项目——除非用户明确点名，一律以当前目录为准。"
        )
    except Exception as e:
        logger.warning("当前工作目录注入失败(可忽略): %s", e)
    if skills_dir is None:
        try:
            from constants import all_skills_dirs as _asd
            skills_dir = _asd()  # 默认扫内置 + 用户两个技能目录
        except Exception:
            skills_dir = None
    if skills_dir:
        skill_index = _build_skill_index(skills_dir)
        if skill_index:
            context_parts.append(f"## 可用技能\n{skill_index}")
    # 记忆索引走两层（claude code 同款思想）：
    # 1. system prompt 常驻索引（本函数拼一次，会话内不重建——system
    #    prompt 本来就是会话级缓存，中途写新记忆不打穿前缀缓存）；
    # 2. 每轮按当前问题检索相关记忆的 ephemeral 注入（memory_injection）。
    # 旧顾虑「每条新记忆报废前缀缓存」只在每轮重注的方案下成立，
    # 会话级一次没有这个问题——所以下面这段真正开始注入索引。
    if memory_manager:
        try:
            ext_block = memory_manager.build_system_prompt()
            if ext_block:
                context_parts.append(ext_block)
        except Exception:
            pass
            logger.warning("异常被吞(fail-open)", exc_info=True)

    # 用户画像文件（系统自动归纳，每 5 次反思后更新一次）
    try:
        from constants import get_codeagent_home
        profile_path = get_codeagent_home() / "USER_PROFILE.md"
        if profile_path.exists():
            profile_text = profile_path.read_text(encoding="utf-8").strip()
            if profile_text:
                context_parts.append(profile_text)
    except Exception:
        pass
        logger.warning("异常被吞(fail-open)", exc_info=True)

    # MCP 路由提示：用户在 .mcp.json 里可以给 server 加
    # keywords 字段，模型一看到关键词就知道该找哪个外部工具服务器。例如：
    #   "postgres": {"keywords": ["订单", "数据库", "SQL"]}
    try:
        from agent.mcp_client import collect_routing_hints
        hints_block = collect_routing_hints()
        if hints_block:
            context_parts.append(hints_block)
    except Exception as e:
        logger.warning("MCP routing hints 收集失败(可忽略): %s", e)

    # 项目记忆：从当前目录一路向上扫到仓库根，收集沿途所有 CODEAGENT.md。
    # cwd 用 get_workspace_cwd() 而不是 Path.cwd()——后者读的是整个进程的
    # 当前目录，多个子代理并发跑会互相踩（前者是每个任务独立的上下文变量）。
    # omit_project_memory=True 时跳过这整段（子代理省 token 用）
    if not omit_project_memory:
        try:
            from agent.workspace_context import get_workspace_cwd
            scan_root = Path(get_workspace_cwd())
            project_mds = _scan_project_memory_files(scan_root)
            for pmd in project_mds:
                try:
                    content = pmd.read_text(encoding="utf-8")
                    content = _expand_imports(content, pmd.parent)
                    if content.strip():
                        # 展示用相对路径好读（绝对路径太长太吵）
                        try:
                            rel = pmd.relative_to(scan_root)
                        except ValueError:
                            rel = pmd
                        context_parts.append(f"## 项目记忆: {rel}\n{content}")
                except Exception as e:
                    logger.warning("读取项目记忆失败 %s: %s", pmd, e)
        except Exception as e:
            logger.warning("项目记忆扫描失败(可忽略): %s", e)

    # 记忆索引常驻注入（claude code 同款：索引放指令链末尾=离用户消息
    # 最近、注意力权重最高的位置）。只放 name+一句话钩子两层，详情靠
    # 每轮检索式注入或 memory 工具 load 按需取——索引只让模型"知道
    # 已经有什么"，不搬正文。会话级拼一次：中途新写的记忆不进本会话
    # 索引（下个会话才见），换来前缀缓存一份不破。
    if memory_store is not None:
        try:
            idx = memory_store.snapshot_for_prompt()
            if idx and idx.strip():
                context_parts.append(
                    "## 记忆索引（已有长期记忆清单，跨会话沉淀）\n"
                    f"{idx}\n\n"
                    "以上只列标题和一句话钩子；需要细节用 memory 工具按 id "
                    "load，或依赖每轮自动检索注入。"
                    "⚠️ 索引里出现的项目/路径是**历史信息**——用户当前消息中"
                    "明确写出的路径或项目名永远优先，不要把记忆里的项目当成"
                    "用户现在所指的项目。"
                )
        except Exception as e:
            logger.warning("记忆索引注入失败(可忽略): %s", e)

    if context_files:
        for cf in context_files:
            cf = Path(cf)
            if cf.exists():
                try:
                    content = cf.read_text(encoding="utf-8")
                    context_parts.append(f"## 上下文文件: {cf.name}\n{content}")
                except Exception as e:
                    logger.warning("读取上下文文件失败 %s: %s", cf, e)
    # C6：输出风格节放 context 层末尾。空文本 = 未启用风格，就不拼（出错也不影响主流程）
    if output_style_text:
        context_parts.append(output_style_text)
    context = "\n\n".join(context_parts)

    # ---- volatile 层（每轮都可能变的部分）----
    volatile_parts = []
    if task_state:
        volatile_parts.append(f"<current_tasks>{task_state}</current_tasks>")
    if reminder:
        volatile_parts.append(reminder)
    if extra_instructions:
        volatile_parts.append(extra_instructions)
    volatile = "\n\n".join(volatile_parts)

    return SystemPromptLayers(stable=stable, context=context, volatile=volatile)


# ---------------------------------------------------------------------------
# 老接口包装（新代码请直接用上面的三层版）
# ---------------------------------------------------------------------------

def build_system_prompt(
    *,
    memory_store=None,
    memory_manager=None,
    enabled_toolsets: Optional[List[str]] = None,
    skills_dir: Optional[Path] = None,
    context_files: Optional[List[Path]] = None,
    extra_instructions: str = "",
    include_guidance: bool = True,
    output_style_text: str = "",
) -> str:
    """组装 system prompt 并拍平成单个字符串——给只收单字符串的调用方用的兼容壳，内部调 build_system_prompt_layers 拿三层再拼平（新代码应直接调三层版，享受分层缓存的好处）。

    参数：
        memory_store: 记忆仓库（现在只是签名兼容，不注入内容）
        memory_manager: 记忆管理器，可产出扩展记忆块
        enabled_toolsets: 启用的工具集列表（透传，未直接使用）
        skills_dir: 技能目录；不传用默认目录
        context_files: 额外拼入的文件路径列表
        extra_instructions: 额外指令文本（进 volatile 层）
        include_guidance: 是否拼入各段使用指南
        output_style_text: 输出风格节文本；空串表示未启用

    返回：拼平的 system prompt 字符串。
    """
    layers = build_system_prompt_layers(
        memory_store=memory_store,
        memory_manager=memory_manager,
        enabled_toolsets=enabled_toolsets,
        skills_dir=skills_dir,
        context_files=context_files,
        extra_instructions=extra_instructions,
        include_guidance=include_guidance,
        output_style_text=output_style_text,
    )
    return layers.render_flat()


def _current_cwd() -> str:
    from agent.workspace_context import get_workspace_cwd
    return get_workspace_cwd()


def _paths_match(paths: list, cwd: str) -> bool:
    """判断当前目录是否命中技能 frontmatter 里的 paths 条件（简化版通配匹配）。

    技能可声明"只在这些路径下生效"（paths 字段），本函数检查当前
    工作目录是否匹配。支持三种写法：目录前缀（src/**）、扩展名（*.py）、
    星号通配。实现上用简化的"字符串前缀"判断，并额外
    兼容绝对路径形式的 cwd：检查目录名是否作为完整路径段出现。

    参数：
        paths: 技能声明的路径模式列表（如 ["src/**", "*.py"]）
        cwd: 当前工作目录字符串

    返回：True 表示当前目录匹配（技能该出现）；False 不匹配。
    """
    import fnmatch
    cwd_norm = cwd.replace("\\", "/")
    for pat in paths or []:
        pat = pat.replace("\\", "/")
        # src/** 形态：当前目录在 src/ 下面就算匹配
        if pat.endswith("/**"):
            base = pat[:-3]
            # 按字符串前缀匹配（会接受 srcfoo 这种
            # 边界误命中，是已知的可接受风险）
            if cwd_norm.startswith(base):
                return True
            # 绝对路径形态：要求目录名作为完整路径段出现（/src/ 或结尾 /src）
            if f"/{base}/" in cwd_norm or cwd_norm.endswith(f"/{base}"):
                return True
        # *.py 这类扩展名模式没法只看目录字符串判断——保守起见按"匹配"放行，
        # 让技能显示出来（宁可多显示不可漏掉）；用 fnmatch 通配兜底
        if fnmatch.fnmatch(cwd_norm, f"*/{pat}") or fnmatch.fnmatch(cwd_norm, pat):
            return True
    return False


def _expand_imports(
    content: str,
    base_dir: Path,
    depth: int = 0,
    _visited: Optional[set] = None,
) -> str:
    """展开 CODEAGENT.md 里的 `@path/to/file` 引用（把引用的文件内容贴进来）——写一行 @docs/api.md，读取时自动把那个文件的内容展开到这个位置，多个文件可以拼着用。

    规则：
    - `@path/to/file` 相对 base_dir 解析，引用里还有引用就递归展开（最深 5 层）
    - `@~/foo/bar` 会展开用户 home 目录
    - 代码片段（`...`）和代码块（```...```）里的 @ 不算引用（防止把
      `@anthropic-ai/sdk` 这种包名误当成引用）
    - 文件不存在或不是文件 → 原样保留那行字（不报错）
    - 同一个文件被引用多次只展开第一次（防止 A 引 B、B 引 A 无限转圈）

    参数：
        content: CODEAGENT.md 原文
        base_dir: 相对引用的基准目录（通常是被展开文件所在目录）
        depth: 当前递归深度
        _visited: 已展开过的文件集合（防环用，递归间共享）

    返回：展开后的文本。
    """
    import re
    if _visited is None:
        _visited = set()
    if depth > 5:
        logger.warning("@import 递归深度超过 5，跳过剩余展开")
        return content

    pattern = re.compile(r"@(~?[\w./-]+)")

    def _resolve_and_read(path_str: str):
        # `@foo` 这种单词（没有 / 也没有 ~）是 @用户名 不是文件引用，跳过
        if not path_str.startswith("~") and "/" not in path_str \
                and not path_str.endswith((".md", ".txt", ".rst")):
            return None
        target = Path(path_str).expanduser() if path_str.startswith("~") \
            else (base_dir / path_str).resolve()
        if not target.is_file():
            return None
        key = str(target)
        if key in _visited:
            return f"<!-- @import 已展开过: {path_str} -->"
        _visited.add(key)
        try:
            return target.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("@import 读取失败 %s: %s", target, e)
            return None

    def _expand_text_segment(text: str) -> str:
        """扫描一段纯文本里的 @path 并替换为文件内容（代码片段已被切走保护）。"""
        def replace(m):
            path_str = m.group(1)
            inner = _resolve_and_read(path_str)
            if inner is None:
                return m.group(0)  # 原样
            # 引用的文件里可能还有 @path，递归继续展开
            return _expand_imports(inner, (base_dir / path_str).resolve().parent
                                   if not path_str.startswith("~")
                                   else Path(path_str).expanduser().parent,
                                   depth + 1, _visited)
        return pattern.sub(replace, text)

    # 按行扫：围栏代码块整块跳过；行内代码片段用正则切开保护起来
    out_lines = []
    in_fence = False
    fence_marker = None
    inline_code_re = re.compile(r"(`[^`]*`)")

    for line in content.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        # 不在代码块里：把行内代码片段切出来不动，剩下的部分做 @path 展开
        segments = inline_code_re.split(line)
        processed = [
            seg if (seg.startswith("`") and seg.endswith("`") and len(seg) >= 2)
            else _expand_text_segment(seg)
            for seg in segments
        ]
        out_lines.append("".join(processed))
    return "\n".join(out_lines)


def _scan_project_memory_files(cwd: Path) -> List[Path]:
    """从当前目录一路向上扫，收集沿途所有 CODEAGENT.md（每层目录都可以有自己的说明文件，全给模型看）。

    停止规则：遇到含 .git 的目录（仓库根）就停，含这一层，不再往上。
    这是对 monorepo（一个大仓多个子项目）友好的设计：在子项目里跑时，
    扫到 monorepo 根就停，不会越过仓库跑到无关目录去。

    参数：
        cwd: 从哪个目录开始向上扫

    返回：CODEAGENT.md 路径列表，顺序是**从仓库根到当前目录**——外层的
    先注入、内层的后注入（后读的语义上覆盖先读的，跟变量作用域同理）。
    """
    found: List[Path] = []
    try:
        current = Path(cwd).resolve()
    except Exception:
        return found
    while True:
        codeagent_md = current / "CODEAGENT.md"
        if codeagent_md.exists():
            found.append(codeagent_md)
        # 有 .git 的这层就是仓库根——这层也要、但不再往上
        if (current / ".git").exists():
            break
        parent = current.parent
        if parent == current:
            break  # 到磁盘最顶层了
        current = parent
    found.reverse()  # 倒序成"从根到当前目录"
    return found


def _build_skill_index(skills_dirs) -> str:
    """构建拼进 prompt 的技能索引：每行一个"技能名 + 一句话描述"。system prompt 只放索引不放正文（省 token），模型看索引决定要不要调 load_skill 取正文。支持多个技能目录（内置 + 用户自定义）。

    规则：只列启用中的技能，归档的不列。多个目录按列表顺序扫，
    同名技能后扫的覆盖先扫的（用户目录排在后面 = 用户说了算）。

    参数：
        skills_dirs: 技能目录路径列表（也兼容传单个路径）

    返回：索引文本；一个技能都没有时返回空串（调用方就不拼这节了）。
    """
    import json

    # 调用方可能只传单个目录，包成列表统一处理
    if isinstance(skills_dirs, (str, Path)):
        skills_dirs = [skills_dirs]

    lines = ["使用 /技能名 触发对应技能。"]

    # 收集各目录的 .usage.json（使用状态记录；用户目录的覆盖内置的）
    usage = {}
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        usage_path = skills_dir / ".usage.json"
        if usage_path.exists():
            try:
                usage.update(json.loads(usage_path.read_text(encoding="utf-8")))
            except Exception:
                pass
                logger.warning("异常被吞(fail-open)", exc_info=True)

    # 扫各目录的技能，同名以后扫的为准
    seen = {}  # 技能名 → SKILL.md 路径
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        if not skills_dir.exists():
            continue
        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            name = skill_md.parent.name
            seen[name] = skill_md  # 同名技能：后扫的覆盖先扫的

    for name in sorted(seen.keys()):
        skill_md = seen[name]
        rec = usage.get(name, {})

        # 已归档的技能不进索引
        if rec.get("state") == "archived":
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, _ = parse_frontmatter(content)
            # 声明了 disable-model-invocation 的技能不进索引（只许用户手动触发）
            if frontmatter.get("disable-model-invocation", False) is True:
                continue
            # 声明了 paths 条件的技能：当前目录不匹配就不进索引
            paths = frontmatter.get("paths")
            if paths:
                if not _paths_match(paths, _current_cwd()):
                    continue
            # 描述优先取 frontmatter 里的，没有再从正文猜
            description = frontmatter.get("description", "") or _extract_description(content)
            if description:
                lines.append(f"- /{name}: {description}")
            else:
                lines.append(f"- /{name}")
        except Exception:
            lines.append(f"- /{name}")

    if len(lines) == 1:  # 只剩标题行说明一个技能都没有
        return ""

    return "\n".join(lines)


def _extract_description(content: str) -> str:
    """从 SKILL.md 的 frontmatter 里抠出 description 字段的值。

    参数：
        content: SKILL.md 全文

    返回：描述文本；没有 frontmatter 或没有该字段时返回空串。
    """
    if not content.startswith("---"):
        return ""
    parts = content.split("---", 2)
    if len(parts) < 3:
        return ""
    for line in parts[1].splitlines():
        if line.strip().startswith("description:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return ""
