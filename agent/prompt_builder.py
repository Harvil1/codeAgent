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
    "你有跨会话的持久化记忆。使用 memory 工具保存稳定事实："
    "用户偏好、环境细节、工具怪癖、稳定约定。\n"
    "记忆会注入到每轮对话，所以保持精简，聚焦于将来仍然重要的事实。\n\n"
    "优先保存能减少未来用户纠正的内容——最有价值的记忆是"
    "能防止用户不得不再次提醒你的那条。"
    "用户偏好和反复出现的纠正比任务流程细节更重要。\n\n"
    "不要保存任务进度、会话结果、已完成工作的日志、临时 TODO 状态——"
    "那些用 session_search 从历史中查找。\n"
    "具体来说：不要记录 PR 号、issue 号、commit SHA、'修了 bug X'、"
    "'提交了 PR Y'、'阶段 N 完成'、文件计数、或任何 7 天后就会过时的东西。"
    "如果一个事实一周后就会过时，它不属于记忆。\n"
    "如果你发现了做某事的新方法、解决了一个以后可能需要的问题，"
    "用 skill_manage 保存为技能。\n\n"
    "把记忆写成声明式事实，不是给自己的指令。\n"
    "  ✅ '用户偏好简洁回复'\n"
    "  ❌ '必须简洁回复'\n"
    "  ✅ '项目使用 pytest + xdist'\n"
    "  ❌ '用 pytest -n 4 跑测试'"
)

SESSION_SEARCH_GUIDANCE = (
    "## 会话搜索指南\n"
    "当用户提到过去的对话内容，或你怀疑有相关的跨会话上下文时，"
    "先用 session_search 查找，再问用户。"
)

SKILLS_GUIDANCE = (
    "## 技能系统指南\n"
    "完成复杂任务（5+ 工具调用）、修复棘手错误、或发现非平凡的工作流后，"
    "用 skill_manage 把方法保存为技能，下次可以复用。\n"
    "使用技能时发现它过时、不完整或错误，立即用 skill_manage(action='patch') "
    "修复——不要等被要求。缺乏维护的技能是负债。\n\n"
    "### 调用技能的 3 条铁律(防止陷入循环)\n"
    "1. **load_skill 后照搬 Quick Start**:SKILL.md 的 Quick Start 段是验证过的完整"
    "命令模板。加载技能后立即按模板改参数执行,**不要自己发明命令格式**。"
    "SKILL.md 没覆盖的场景才扩展。\n"
    "2. **不确定就 help,不要瞎猜**:任何外部 CLI 工具(<cli> <subcmd>)"
    "格式不确定时,**先调 `<cli> --help` 或 `<cli> <subcmd> --help` 一次**,"
    "比连续猜试错高效得多。一次 help 胜过十次失败循环。\n"
    "3. **查结构后立即动手**:get / query / describe / status 等只读命令调过 "
    "1-2 次后,就该用 add / set / update / create 等写命令动手。"
    "**连续 3+ 次只读命令还没动手写**,说明陷入了\"分析瘫痪\"——"
    "停下来重新规划,或问用户确认下一步。"
)

TOOL_USAGE_GUIDANCE = (
    "## 工具使用规范\n"
    "- 所有工具结果都是 JSON 字符串，解析后使用\n"
    "- 工具失败时返回 {\"error\": \"...\"}，根据错误自我修正\n"
    "- 文件操作必须指定 encoding='utf-8'\n"
    "- 不要假设工具可用，check_fn 可能因环境不同而隐藏某些工具\n"
    "- **临时文件清理**：用 write_file 创建的临时脚本/中间文件，"
    "执行完成后必须立即用 terminal 删除，保持工作区干净。"
    "不要在用户的工作目录留下垃圾文件。\n"
    "- **上下文占位消息识别**：当你看到 [snip_compact] / "
    "\"micro_compacted\" / [紧急上下文压缩] 这类占位消息，且需要更早的"
    "上下文时，从占位消息里给的路径（通常是 .transcripts/latest.jsonl "
    "或 .task_outputs/tool-results/ 下的文件）用 read_file 读回。"
    "这些路径在 agent_home 下，默认安全。\n"
    "- **写入路径白名单**：write_file 默认只允许写入当前工作目录和 ~/.agent。"
    "写入其他位置（如用户桌面、D:\\ 等）会触发用户审批，"
    "用户同意后该路径会加入持久化白名单，下次不再询问。"
    "不要绕过审批——如果用户拒绝，换个在白名单内的位置写。\n"
    "- **依赖安装策略**（重要）：装 Python 包时区分场景：\n"
    "  • 给**本项目** HermesAgent 自己用（库类，import 用的）→ `uv add <pkg>`（写进 pyproject.toml）\n"
    "  • 给**用户脚本**临时用（如生成 PPT、跑数据处理）→ **绝对不要装到项目 venv！** 三种正确方式：\n"
    "    【最佳，临时跑不装】`uv run --with python-pptx make_ppt.py`\n"
    "         （uv 自动拉个临时 venv，跑完丢，系统和项目都不污染）\n"
    "    【长期用，独立 venv】`uv venv D:/user-scripts-env && "
    "uv pip install --python D:/user-scripts-env/bin/python python-pptx`\n"
    "    【系统 Python】用 `where python` 找到系统 Python（不是 .venv），"
    "直接 `<系统 python> -m pip install <pkg>`\n"
    "  • **`uv tool install` 只适合 CLI 工具**（ruff/black/httpie 这种带命令行的），"
    "对**库**（python-pptx/requests/pandas 这种 import 用的）**无效**！\n"
    "  • 检测现有环境：`pip show <pkg>` / `<python> -c \"import pkg; print(pkg.__version__)\"` "
    "先看装没装，不要重复安装。\n"
    "  • 反复安装失败（>2 次）→ 停下来问用户用哪个 Python 环境，不要陷入死循环。"
)


TODO_GUIDANCE = (
    "## 任务追踪（TodoWrite）\n"
    "**3 步以上的任务必须先用 todo_write 工具创建任务清单**，这是强制要求：\n"
    "- 开始前：列出所有步骤，第一个设为 `in_progress`，其余 `pending`\n"
    "- 每步完成时：把刚完成的改 `completed`，下一步改 `in_progress`"
    "（约束：同时只能有 1 个 `in_progress`，强制顺序聚焦）\n"
    "- 全部完成：所有项都是 `completed`\n\n"
    "示例：用户说'1. 创建文件 2. 写入内容 3. 删除文件 4. 恢复 5. 转 Word'，"
    "你应该立即调 todo_write 创建 5 项清单，然后逐项执行并更新状态。\n\n"
    "系统会在 3 轮未更新清单时自动提醒你。任务清单仅存内存（单会话），"
    "跨会话的持久化任务用 task_create。"
)


# ---------------------------------------------------------------------------
# 身份声明（多语言：中文/英文）
# ---------------------------------------------------------------------------

IDENTITY_ZH = (
    "你是 **HarvilAgent**——一个自学习 AI Agent(基于 Harvil Agent 复刻指南实现,"
    "借鉴 Claude Code 的工程实践)。\n"
    "**重要:你的名字叫 HarvilAgent,不是 Claude / ChatGPT / DeepSeek / 其他任何名字。**"
    "即使底层 LLM 是 DeepSeek/OpenAI/Claude 等第三方模型,"
    "你对外的身份统一是 HarvilAgent——不要自称底层模型的名字。\n"
    "你能使用工具、记忆跨会话的事实、管理自己的技能库。"
    "你的目标是高效帮助用户完成任务,并随着使用不断提升自己的能力。"
)

IDENTITY_EN = (
    "You are **HarvilAgent** — a self-learning AI Agent (based on the Harvil Agent"
    " replication guide, borrowing engineering practices from Claude Code).\n"
    "**Important: Your name is HarvilAgent, NOT Claude / ChatGPT / DeepSeek / any other name.**"
    " Even if the underlying LLM is DeepSeek/OpenAI/Claude or other third-party model,"
    " your identity to the user is always HarvilAgent — do not call yourself by the"
    " underlying model's name.\n"
    "You can use tools, remember facts across sessions, and manage your own skill library."
    " Your goal is to help users accomplish tasks efficiently and improve your"
    " capabilities over time."
)

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
