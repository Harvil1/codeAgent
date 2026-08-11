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


def build_post_compact_brief(agent: "AIAgent") -> str:
    """compact 后构建 recovery brief（fail-open，异常返回空串不崩）。

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

        parts = []

        # 1. invoked skills 正文（优先——官方确认的核心恢复力）
        skills_brief = _build_invoked_skills_brief(
            agent, max_skills, skill_budget,
        )
        if skills_brief:
            parts.append(skills_brief)

        # 2. 最近文件 preview
        files_brief = _build_recent_files_brief(
            agent._recent_read_files, max_files,
        )
        if files_brief:
            parts.append(files_brief)

        if not parts:
            return ""

        return "\n\n".join(parts)
    except Exception as e:
        logger.debug("build_post_compact_brief fail-open: %s", e)
        return ""


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
