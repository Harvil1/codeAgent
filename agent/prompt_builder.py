"""System prompt 组装器。

关键原则：
1. 会话开始时构建一次，后续缓存（_cached_system_prompt）
2. 必须 byte-stable（同一会话内字节级不变）
3. 记忆是 frozen 快照（本次会话不更新）
4. 技能只列名字和描述，不包含正文

05 升级：拆成 stable/context/volatile 三层，让 prompt cache 命中率最大化。
- stable：跨会话不变（身份、指导）
- context：单会话内不变（记忆、技能、CLAUDE.md）
- volatile：每轮可变（todo、reminder）
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 指导文本（注入 system prompt，告诉 LLM 怎么用记忆/技能/搜索）
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
    "[紧急上下文压缩] 占位消息且需更早上下文时,按占位消息里的路径"
    "(通常是 .transcripts/latest.jsonl 或 .task_outputs/tool-results/ 下的文件)"
    "用 read_file 读回(在 agent_home 下,默认安全)\n"
    "- **写入白名单**:write_file 默认只允许 cwd 和 ~/.OmniMate;写其他位置会触发审批,"
    "同意后进持久化白名单。不要绕过——用户拒绝就换个白名单内位置写\n"
    "- **自身源码保护**:不要修改 OmniMate 自身源码"
    "(开发=项目根,打包=site-packages 安装目录)。你是工具,用户用你改**他们的项目**,不是改你自己\n"
    "- **依赖安装(跟随 cwd)**:在用户当前目录用其项目环境装包:"
    "`uv pip install <pkg>`(推荐,尊重用户项目约定)或 `pip install <pkg>`;"
    "优先复用已有 .venv/pyproject.toml,不污染系统 Python;"
    "连续 2 次装失败 → 停下问用户\n"
    "- **代码输出(跟随 cwd)**:脚本写到当前目录,输出(PPT/Excel/Word 等)写到 "
    "`<cwd>/outputs/`;用户指定别的位置就照做(白名单自动处理审批)\n"
    "- **先查技能**:任何任务前先扫技能索引,判断有无适用流程技能"
    "(设计先行/调试/写计划等),有就先 load_skill('using-omnimate') 看总纲"
)


TODO_GUIDANCE = (
    "## 任务追踪（TodoWrite）\n"
    "3 步以上的任务必须先调 todo_write 创建清单:首项 `in_progress`,其余 `pending`;"
    "每步完成时更新状态(同时只能 1 个 `in_progress`,强制顺序聚焦);"
    "全部完成时全 `completed`。\n\n"
    "### 开放式探索任务也要先列清单\n"
    "用户给的是模糊目标(非步骤列表)时,**先拆成探索清单再动手**,不要一路 read_file 几十轮:"
    "'学习这个项目' → [后端架构, 前端结构, skills, 部署/运行, 测试组织]"
    "'排查 bug' → [复现路径, 相关代码, 数据流, 假设根因, 验证]。\n"
    "列清单:(a)不漏维度 (b)用户能纠偏 (c)有节奏推进。\n\n"
    "系统 3 轮未更新清单会自动提醒。清单仅存内存(单会话),跨会话用 task_create。"
)


DELEGATE_GUIDANCE = (
    "## 子代理委托（delegate_task）\n"
    "大项目探索、多个独立子任务、或单子任务预计 10+ 次工具调用,必须用 delegate_task "
    "省父 context(可 tasks=[...] 批量并行)。\n"
    "不要委托:强依赖父上下文(如基于已读内容做判断)、短任务(<5 次调用)、顺序型任务。\n"
    "用法:delegate_task(goal=..., role='leaf', summary_only=True);"
    "summary_only 自动把结果压缩成 300 字摘要回填。\n"
    "**并行铁律**:需要同时派多个子代理(如并行探索各模块)时,必须用 tasks=[...] "
    "一次调用;禁止发多个独立 delegate_task——独立调用是串行执行的,逐个等待,不会并行。\n"
    "判断标准:同一领域打算连续 read_file 5+ 次,立刻停,委托子代理——"
    "父 context 留给综合判断。"
)


# ---------------------------------------------------------------------------
# 身份声明（多语言：中文/英文）
# ---------------------------------------------------------------------------

IDENTITY_ZH = "你是 OmniMate,自学习 AI Agent 平台。"

IDENTITY_EN = "You are OmniMate, a self-learning AI Agent platform."

# 向后兼容别名（中文为默认）
IDENTITY = IDENTITY_ZH


OUTPUT_CONVENTION_ZH = (
    "## 输出约定\n"
    "- 使用中文回复\n"
    "- 代码标识符（变量名、函数名、类名）使用英文\n"
    "- 长输出分段，使用 markdown 格式"
)

OUTPUT_CONVENTION_EN = (
    "## Output Convention\n"
    "- Respond in English\n"
    "- Use English for code identifiers (variable/function/class names)\n"
    "- Break long output into sections; use Markdown formatting"
)


# ---------------------------------------------------------------------------
# 三层结构（05）
# ---------------------------------------------------------------------------

@dataclass
class SystemPromptLayers:
    """三层系统提示（05）。

    - stable: 跨会话不变（身份、指导、工具文档）
    - context: 单会话内不变（记忆索引、技能索引、CLAUDE.md）
    - volatile: 每轮可变（todo、reminder、extra_instructions）

    API 厂商 prompt cache 按"前缀哈希"匹配，分层能让 stable 跨会话命中、
    context 单会话命中，整体命中率上去。
    """
    stable: str
    context: str
    volatile: str

    def render_flat(self) -> str:
        """合并成单个字符串（向后兼容老接口）。"""
        parts = [self.stable, self.context, self.volatile]
        return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 主构建函数（05 三层版）
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
    language: str = "zh",
    # volatile 来源（运行时传入）
    todo_state: Optional[str] = None,
    task_state: Optional[str] = None,
    reminder: Optional[str] = None,
) -> SystemPromptLayers:
    """构建三层 system prompt（05）。

    分层动机：
      - stable：跨会话不变（同版本同一台机器，几乎 100% 命中 cache）
      - context：单会话内不变（记忆/技能/CLAUDE.md，会话内 80%+ 命中）
      - volatile：每轮可变（todo / reminder），不期望 cache 命中

    language: "zh"（默认）或 "en"。只影响身份声明和输出约定。
    """
    # ---- stable 层 ----
    stable_parts = []
    if language == "en":
        stable_parts.append(IDENTITY_EN)
        stable_parts.append(OUTPUT_CONVENTION_EN)
    else:
        stable_parts.append(IDENTITY_ZH)
        stable_parts.append(OUTPUT_CONVENTION_ZH)
    if include_guidance:
        stable_parts.extend([
            MEMORY_GUIDANCE, SKILLS_GUIDANCE,
            SESSION_SEARCH_GUIDANCE, TOOL_USAGE_GUIDANCE, TODO_GUIDANCE,
            DELEGATE_GUIDANCE,
        ])
    stable = "\n\n".join(stable_parts)

    # ---- context 层 ----
    context_parts = []
    if skills_dir is None:
        try:
            from constants import all_skills_dirs as _asd
            skills_dir = _asd()  # 内置 + 用户两个目录
        except Exception:
            skills_dir = None
    if skills_dir:
        skill_index = _build_skill_index(skills_dir)
        if skill_index:
            context_parts.append(f"## 可用技能\n{skill_index}")
    if memory_store:
        try:
            index_block = memory_store.snapshot_for_prompt()
            if index_block:
                context_parts.append(f"## 记忆索引\n{index_block}")
        except Exception as e:
            logger.warning("读取记忆索引失败: %s", e)
    if memory_manager:
        try:
            ext_block = memory_manager.build_system_prompt()
            if ext_block:
                context_parts.append(ext_block)
        except Exception:
            pass

    # 用户画像(自动归纳,每 5 次反思后更新)
    try:
        from constants import get_omnimate_home
        profile_path = get_omnimate_home() / "USER_PROFILE.md"
        if profile_path.exists():
            profile_text = profile_path.read_text(encoding="utf-8").strip()
            if profile_text:
                context_parts.append(profile_text)
    except Exception:
        pass

    # MCP routing hints(借鉴 DeerFlow):用户配 .mcp.json 时可加 keywords 字段,
    # 帮助 LLM 看到关键词就知道用哪个 MCP server。如:
    #   "postgres": {"keywords": ["订单", "数据库", "SQL"]}
    try:
        from agent.mcp_client import collect_routing_hints
        hints_block = collect_routing_hints()
        if hints_block:
            context_parts.append(hints_block)
    except Exception as e:
        logger.debug("MCP routing hints 收集失败(可忽略): %s", e)
    if context_files:
        for cf in context_files:
            cf = Path(cf)
            if cf.exists():
                try:
                    content = cf.read_text(encoding="utf-8")
                    context_parts.append(f"## 上下文文件: {cf.name}\n{content}")
                except Exception as e:
                    logger.warning("读取上下文文件失败 %s: %s", cf, e)
    context = "\n\n".join(context_parts)

    # ---- volatile 层 ----
    volatile_parts = []
    if todo_state:
        volatile_parts.append(f"<todo_state>{todo_state}</todo_state>")
    if task_state:
        volatile_parts.append(f"<current_tasks>{task_state}</current_tasks>")
    if reminder:
        volatile_parts.append(reminder)
    if extra_instructions:
        volatile_parts.append(extra_instructions)
    volatile = "\n\n".join(volatile_parts)

    return SystemPromptLayers(stable=stable, context=context, volatile=volatile)


# ---------------------------------------------------------------------------
# 主构建函数（向后兼容旧接口）
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
    language: str = "zh",
) -> str:
    """组装 system prompt（向后兼容旧接口）。

    内部走 build_system_prompt_layers 然后合并成单个字符串。
    新代码应直接调 build_system_prompt_layers 拿三层。
    """
    layers = build_system_prompt_layers(
        memory_store=memory_store,
        memory_manager=memory_manager,
        enabled_toolsets=enabled_toolsets,
        skills_dir=skills_dir,
        context_files=context_files,
        extra_instructions=extra_instructions,
        include_guidance=include_guidance,
        language=language,
    )
    return layers.render_flat()


def _build_skill_index(skills_dirs) -> str:
    """构建技能索引(名字 + 描述),支持多目录(内置 + 用户)。

    只列出 active 状态的技能,跳过归档的。
    多目录场景:按列表顺序扫描,后者覆盖前者(用户目录优先)。
    """
    import json

    # 兼容单目录输入
    if isinstance(skills_dirs, (str, Path)):
        skills_dirs = [skills_dirs]

    lines = ["使用 /技能名 触发对应技能。"]

    # 收集所有目录的 .usage.json(用户目录的覆盖内置的)
    usage = {}
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        usage_path = skills_dir / ".usage.json"
        if usage_path.exists():
            try:
                usage.update(json.loads(usage_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    # 扫描所有目录的技能,后者覆盖前者
    seen = {}  # name → skill_md 路径
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        if not skills_dir.exists():
            continue
        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            name = skill_md.parent.name
            seen[name] = skill_md  # 后扫的覆盖先扫的

    for name in sorted(seen.keys()):
        skill_md = seen[name]
        rec = usage.get(name, {})

        # 跳过归档的
        if rec.get("state") == "archived":
            continue

        # 解析 frontmatter 获取描述
        try:
            content = skill_md.read_text(encoding="utf-8")
            description = _extract_description(content)
            if description:
                lines.append(f"- /{name}: {description}")
            else:
                lines.append(f"- /{name}")
        except Exception:
            lines.append(f"- /{name}")

    if len(lines) == 1:  # 只有标题行
        return ""

    return "\n".join(lines)


def _extract_description(content: str) -> str:
    """从 SKILL.md 提取 description 字段。"""
    if not content.startswith("---"):
        return ""
    parts = content.split("---", 2)
    if len(parts) < 3:
        return ""
    for line in parts[1].splitlines():
        if line.strip().startswith("description:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return ""
