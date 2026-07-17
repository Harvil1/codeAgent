"""System prompt 组装器。

关键原则：
1. 会话开始时构建一次，后续缓存（_cached_system_prompt）
2. 必须 byte-stable（同一会话内字节级不变）
3. 记忆是 frozen 快照（本次会话不更新）
4. 技能只列名字和描述，不包含正文
"""

import logging
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
    "修复——不要等被要求。缺乏维护的技能是负债。"
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
    "这些路径在 agent_home 下，默认安全。"
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
# 身份声明
# ---------------------------------------------------------------------------

IDENTITY = (
    "你是一个自学习 AI Agent（基于 Harvil Agent 复刻指南实现）。\n"
    "你能使用工具、记忆跨会话的事实、管理自己的技能库。"
    "你的目标是高效帮助用户完成任务，并随着使用不断提升自己的能力。"
)


# ---------------------------------------------------------------------------
# 主构建函数
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
) -> str:
    """组装 system prompt。

    这是一次性操作：结果会被 agent 缓存，本次会话不再重建。
    """
    parts = []

    # 1. 身份
    parts.append(IDENTITY)

    # 2. 输出约定
    parts.append(
        "## 输出约定\n"
        "- 使用中文回复\n"
        "- 代码标识符（变量名、函数名、类名）使用英文\n"
        "- 长输出分段，使用 markdown 格式"
    )

    # 3. 核心指导
    if include_guidance:
        parts.append(MEMORY_GUIDANCE)
        parts.append(SKILLS_GUIDANCE)
        parts.append(SESSION_SEARCH_GUIDANCE)
        parts.append(TOOL_USAGE_GUIDANCE)
        parts.append(TODO_GUIDANCE)

    # 4. 技能索引（只列名字 + 描述，不包含正文）
    if skills_dir is None:
        try:
            from constants import skills_dir as _sd
            skills_dir = _sd()
        except Exception:
            skills_dir = None

    if skills_dir:
        skill_index = _build_skill_index(Path(skills_dir))
        if skill_index:
            parts.append(f"## 可用技能\n{skill_index}")

    # 5. 记忆索引（多文件模式，Phase 5）
    if memory_store:
        try:
            index_block = memory_store.snapshot_for_prompt()
            if index_block:
                parts.append(f"## 记忆索引\n{index_block}")
        except Exception as e:
            logger.warning("读取记忆索引失败: %s", e)

    # 6. 外部 provider 的静态块
    if memory_manager:
        try:
            ext_block = memory_manager.build_system_prompt()
            if ext_block:
                parts.append(ext_block)
        except Exception:
            pass

    # 7. 上下文文件（CLAUDE.md / AGENTS.md 等）
    if context_files:
        for cf in context_files:
            cf = Path(cf)
            if cf.exists():
                try:
                    content = cf.read_text(encoding="utf-8")
                    parts.append(f"## 上下文文件: {cf.name}\n{content}")
                except Exception as e:
                    logger.warning("读取上下文文件失败 %s: %s", cf, e)

    # 8. 额外指令
    if extra_instructions:
        parts.append(extra_instructions)

    return "\n\n".join(parts)


def _build_skill_index(skills_dir: Path) -> str:
    """构建技能索引（名字 + 描述）。

    只列出 active 状态的技能，跳过归档的。
    """
    import json

    lines = ["使用 /技能名 触发对应技能。"]

    # 读取使用统计（获取状态）
    usage_path = skills_dir / ".usage.json"
    usage = {}
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 扫描技能
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        name = skill_md.parent.name
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
