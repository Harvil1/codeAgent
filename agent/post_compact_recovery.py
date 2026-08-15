"""compact 后主动重注入关键上下文（CCAR4 Task B）。

借鉴 claude-code-main 的 createPostCompactFileAttachments + createSkillAttachmentIfNeeded，
在 compact 完成后把以下内容作为 user 消息重注入：
1. 最近 N 个 read_file 文件路径 + preview（每文件 ≤1K）
2. invoked_skills 正文重注入（每 skill ≤5K，总 ≤25K budget）

设计原则：
- **fail-open**：recovery 在 compact 末尾，任何异常只 log 不崩（compact 不能因 recovery 失败）
- **safe_path 白名单**：读文件必须走 safe_path，受保护路径（~/.ssh 等）跳过
- **prompt cache 神圣不可侵犯**：recovery 注入是 user 消息，不动 system prompt
- **会话级状态**：_recent_read_files / _recent_skills 是 AIAgent 实例属性，新会话自动重置
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import AIAgent

logger = logging.getLogger(__name__)

# ============================================================================
# 预算常量（对齐 claude-code-main + Task B brief）
# ============================================================================
MAX_RECENT_FILES = 5                      # 最近文件数上限
RECENT_FILE_PREVIEW_CHARS = 1000           # 每文件 preview 字符上限
MAX_INVOKED_SKILLS = 5                     # invoked skills 数上限
SKILL_BUDGET_CHARS = 25000                 # skill 总 budget
SKILL_PER_BUDGET_CHARS = 5000              # 每 skill budget

# 向后兼容：旧 config key
LEGACY_REINJECT_CHAR_LIMIT = 25000


def _est_tokens(text: str) -> int:
    """粗略 token 估算（chars/4，够预算统筹用）。"""
    return len(text) // 4


def build_post_compact_brief(agent: "AIAgent") -> str:
    """compact 后构建 recovery brief（fail-open，异常返回空串不崩）。

    T2（核心机制对齐第 2 项）：统一 token 预算统筹 + plan/async 状态恢复。
    - 总预算：config context.post_compact_recovery_budget（默认 40000 tokens，chars/4 估算）
    - 优先级：plan/async 状态 > 最近文件 > 技能正文（超预算按优先级截断——
      计划和在跑任务丢了最致命，技能丢了可 load_skill 重取）

    返回的字符串会被嵌入到 <post_compress_brief> 的 user 消息里。
    如果 agent 无最近文件/技能，或 config 关闭了 recovery，返回空串。

    Args:
        agent: AIAgent 实例（读 _recent_read_files / _recent_skills / config）

    Returns:
        recovery 段文本（空串表示无内容可恢复）
    """
    try:
        # config 开关检查
        ctx_cfg = (agent.config or {}).get("context", {})
        if not ctx_cfg.get("post_compact_recovery_enabled", True):
            return ""

        # 从 config 读上限（允许覆盖默认值）
        max_files = ctx_cfg.get("post_compact_recovery_max_files", MAX_RECENT_FILES)
        max_skills = ctx_cfg.get("post_compact_recovery_max_skills", MAX_INVOKED_SKILLS)
        # 向后兼容：旧 config key reinject_char_limit 映射到 skill budget
        skill_budget = ctx_cfg.get("reinject_char_limit", SKILL_BUDGET_CHARS)
        # T2：总预算（tokens）
        budget_tokens = ctx_cfg.get("post_compact_recovery_budget", 40000)

        # 按优先级收集段（高 → 低）
        sections = []

        # 1. plan / async 状态（T2 新增，最高优先——丢了最致命）
        state_brief = _build_plan_async_state_brief(agent)
        if state_brief:
            sections.append(state_brief)

        # 2. 最近文件 preview
        files_brief = _build_recent_files_brief(
            agent._recent_read_files, max_files,
        )
        if files_brief:
            sections.append(files_brief)

        # 3. invoked skills 正文
        skills_brief = _build_invoked_skills_brief(
            agent, max_skills, skill_budget,
        )
        if skills_brief:
            sections.append(skills_brief)

        if not sections:
            return ""

        # 统一预算：逐段累加，超预算截断（保留高优先级段）
        out = _apply_budget(sections, budget_tokens)
        return "\n\n".join(out)
    except Exception as e:
        logger.debug("build_post_compact_brief fail-open: %s", e)
        return ""


# 段被预算截断时至少保留的 token 数（低于此直接丢段，不输出残段）
_MIN_SECTION_TOKENS = 200


def _apply_budget(sections: list, budget_tokens: int) -> list:
    """按优先级顺序分配预算（T2）。

    每段估算 chars/4；装得下整段就整段放，装不下但剩余预算 >=
    _MIN_SECTION_TOKENS 就截断放，否则丢段（后续段自然也没预算）。
    """
    remaining = budget_tokens
    out = []
    for text in sections:
        t = _est_tokens(text)
        if t <= remaining:
            out.append(text)
            remaining -= t
            continue
        if remaining >= _MIN_SECTION_TOKENS:
            out.append(text[: remaining * 4] + "\n...[recovery 预算截断]")
            remaining = 0
        break  # 预算耗尽，后续段全丢（保持优先级语义）
    return out


def _build_plan_async_state_brief(agent: "AIAgent") -> str:
    """plan / async 执行状态段（T2 新增恢复源，fail-open）。

    - agent.plan_mode=True → 提示在 plan 调研模式
    - agent._last_approved_plan 非空 → 已批准计划全文（执行中）
    - delegate_tool._async_tasks 有 running → 列出 id/goal 摘要/已运行时长
    """
    lines = []
    try:
        if getattr(agent, "plan_mode", False):
            lines.append(
                "## 当前状态：plan 调研模式\n"
                "正在做实施前调研（只读工具集）。完成调研后必须调 "
                "exit_plan_mode(plan=...) 提交计划等待用户审批。"
            )
        plan_text = getattr(agent, "_last_approved_plan", "") or ""
        if plan_text:
            lines.append(
                "## 正在执行的计划（已获用户批准，compact 后重注入）\n"
                f"{plan_text}"
            )
    except Exception as e:
        logger.debug("plan 状态恢复失败（fail-open）: %s", e)

    try:
        import time as _time
        from tools.delegate_tool import _async_tasks
        running = []
        for tid, info in list(_async_tasks.items()):
            thread = info.get("thread")
            # thread 存在且已死 = 残留条目（finally 清理竞态），跳过
            if thread is not None and not thread.is_alive():
                continue
            started = info.get("started_at") or 0
            elapsed = int(_time.time() - started) if started else 0
            goal = str(info.get("goal", ""))[:80]
            running.append(f"- {tid}（已运行 {elapsed}s）：{goal}")
        if running:
            lines.append(
                "## 在跑的 async 子代理（compact 后仍在后台执行）\n" + "\n".join(running)
            )
    except Exception as e:
        logger.debug("async 状态恢复失败（fail-open）: %s", e)

    return "\n\n".join(lines)


def _build_recent_files_brief(paths: list, max_files: int = MAX_RECENT_FILES) -> str:
    """最近 N 个文件路径 + preview（每文件 ≤1K）。

    走 safe_path 白名单检查，受保护路径（~/.ssh / /etc 等）跳过。
    """
    if not paths:
        return ""

    from agent.permission import safe_path

    # 取最近 max_files 个（去重保序）
    recent = paths[-max_files:] if len(paths) > max_files else paths

    lines = [f"## 最近读过的文件（前 {len(recent)} 个，compact 后重注入）"]
    any_success = False
    for path in recent:
        try:
            # safe_path 白名单检查（读模式）
            perm = safe_path(path, write=False)
            if not perm.allowed:
                logger.debug("recovery 跳过受保护路径: %s (%s)", path, perm.reason)
                continue

            p = Path(path).expanduser()
            if not p.exists() or not p.is_file():
                continue

            content = p.read_text(encoding="utf-8", errors="replace")
            preview = content[:RECENT_FILE_PREVIEW_CHARS]
            if len(content) > RECENT_FILE_PREVIEW_CHARS:
                preview += "\n...[truncated]..."

            lines.append(f"### {path}\n```\n{preview}\n```")
            any_success = True
        except Exception as e:
            logger.debug("recovery 读文件失败 %s: %s", path, e)
            continue

    return "\n".join(lines) if any_success else ""


def _build_invoked_skills_brief(
    agent: "AIAgent",
    max_skills: int = MAX_INVOKED_SKILLS,
    total_budget: int = SKILL_BUDGET_CHARS,
) -> str:
    """invoked skills 正文重注入（每 skill ≤5K，总 budget 限制）。

    使用 agent._load_skill_body() 加载技能正文（跨目录 + 去 frontmatter）。
    """
    skill_names = agent._recent_skills
    if not skill_names:
        return ""

    # 取最近 max_skills 个（去重保序）
    recent_skills = skill_names[-max_skills:] if len(skill_names) > max_skills else skill_names

    lines = [f"## 最近加载的技能正文（前 {len(recent_skills)} 个，compact 后重注入）"]
    used = 0
    any_success = False

    # 逆序：最近用的优先（对齐 Claude Code 语义）
    for name in reversed(recent_skills):
        try:
            body = agent._load_skill_body(name)
            if not body:
                continue

            # 每 skill 截到 5K
            if len(body) > SKILL_PER_BUDGET_CHARS:
                body = body[:SKILL_PER_BUDGET_CHARS] + "\n...[truncated]..."

            # 总 budget 检查
            if used + len(body) > total_budget:
                break

            lines.append(f"### 技能 {name}\n{body}")
            used += len(body)
            any_success = True
        except Exception as e:
            logger.debug("recovery 加载技能失败 %s: %s", name, e)
            continue

    return "\n".join(lines) if any_success else ""
