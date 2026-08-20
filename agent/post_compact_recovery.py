"""上下文压缩（compact，把长对话浓缩成摘要腾地方）之后的关键信息"补发"模块。

背景（CCAR4 Task B，借鉴 claude-code-main 的 createPostCompactFileAttachments
+ createSkillAttachmentIfNeeded）：压缩会把对话历史浓缩，模型刚读过的工作
材料也一起被浓缩没了，导致它"失忆"手忙脚乱。所以在压缩完成后，把下面
这些东西作为 user 消息重新塞回去：
1. 最近读过的 N 个文件路径 + 内容开头（每文件最多 1K 字符）
2. 最近加载过的技能正文（每个最多 5K，总量最多 25K）

设计原则（四条底线）：
- **fail-open**：恢复动作挂在压缩流程末尾，任何异常只记日志不崩溃
  （压缩绝不能被恢复失败连累）
- **路径安全**：读文件必须过 safe_path 白名单检查，受保护路径
  （~/.ssh 之类）直接跳过
- **前缀缓存神圣不可侵犯**：恢复内容走 user 消息，绝不碰 system prompt
- **会话级状态**：_recent_read_files / _recent_skills 是 AIAgent 实例
  属性，新会话自动从零开始
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import AIAgent

logger = logging.getLogger(__name__)

# ============================================================================
# 预算常量（对齐 claude-code-main + Task B 的约定）
# ============================================================================
MAX_RECENT_FILES = 5                      # 最多补发几个最近文件
RECENT_FILE_PREVIEW_CHARS = 1000           # 每个文件正文开头最多给多少字符
MAX_INVOKED_SKILLS = 5                     # 最多补发几个最近用过的技能
SKILL_BUDGET_CHARS = 25000                 # 技能正文总字数预算
SKILL_PER_BUDGET_CHARS = 5000              # 单个技能正文的字数上限

# 向后兼容：老的配置键名
LEGACY_REINJECT_CHAR_LIMIT = 25000


def _est_tokens(text: str) -> int:
    """粗略估一下文本有多少 token（字符数除以 4）。

    不求精确，够做预算分配用。

    参数：
        text: 文本

    返回：估算的 token 数。
    """
    return len(text) // 4


def build_post_compact_brief(agent: "AIAgent") -> str:
    """压缩完成后构建"补发摘要"（recovery brief）。

    背景（T2，核心机制对齐第 2 项）：压缩后模型丢了工作材料，这里统一
    按 token 预算把最重要的东西补回去，并新增计划/后台任务状态的恢复。
    - 总预算：config context.post_compact_recovery_budget（默认 40000
      tokens，按"字符数/4"估算）
    - 优先级：计划/后台任务状态 > 最近文件 > 技能正文（超预算按优先级
      砍尾——计划和在跑任务丢了最致命，技能丢了还能用 load_skill 重取）

    参数：
        agent: AIAgent 实例（读 _recent_read_files / _recent_skills / config）

    返回：补发段文本（会被嵌进 <post_compress_brief> 的 user 消息里）；
    没有可恢复的内容、或配置关了恢复功能时返回空串。任何异常也返回
    空串（fail-open，绝不崩）。
    """
    try:
        # 先看配置开关
        ctx_cfg = (agent.config or {}).get("context", {})
        if not ctx_cfg.get("post_compact_recovery_enabled", True):
            return ""

        # 从配置读各项上限（允许覆盖默认值）
        max_files = ctx_cfg.get("post_compact_recovery_max_files", MAX_RECENT_FILES)
        max_skills = ctx_cfg.get("post_compact_recovery_max_skills", MAX_INVOKED_SKILLS)
        # 向后兼容：老配置键 reinject_char_limit 当作技能预算用
        skill_budget = ctx_cfg.get("reinject_char_limit", SKILL_BUDGET_CHARS)
        # T2：总预算（token 数）
        budget_tokens = ctx_cfg.get("post_compact_recovery_budget", 40000)

        # 按优先级从高到低收集各段
        sections = []

        # 1. 计划 / 后台任务状态（T2 新增，最高优先——丢了最致命）
        state_brief = _build_plan_async_state_brief(agent)
        if state_brief:
            sections.append(state_brief)

        # 2. 最近文件的正文开头
        files_brief = _build_recent_files_brief(
            agent._recent_read_files, max_files,
        )
        if files_brief:
            sections.append(files_brief)

        # 3. 最近用过的技能正文
        skills_brief = _build_invoked_skills_brief(
            agent, max_skills, skill_budget,
        )
        if skills_brief:
            sections.append(skills_brief)

        if not sections:
            return ""

        # 统一过预算：逐段往里装，装不下就截断（高优先级的段保住）
        out = _apply_budget(sections, budget_tokens)
        return "\n\n".join(out)
    except Exception as e:
        logger.debug("build_post_compact_brief fail-open: %s", e)
        return ""


# 某段装不下时，剩余预算至少要有这么多 token 才值得截断着放
# （太小的残段没意义，直接整段丢）
_MIN_SECTION_TOKENS = 200


def _apply_budget(sections: list, budget_tokens: int) -> list:
    """按优先级顺序给各段分预算（T2 引入的统一统筹）。

    规则：每段按"字符数/4"估算 token；整段装得下就整段放；装不下但
    剩余预算还够 200 token 就截断着放；再不够就整段丢——后面的段自然
    也没预算了（保住排前面的高优先级段）。

    参数：
        sections: 按优先级从高到低排好的文本段列表
        budget_tokens: 总预算（token 数）

    返回：实际放入的段列表。
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
        break  # 预算花光，后面的段全丢（优先级语义就这样定的）
    return out


def _build_plan_async_state_brief(agent: "AIAgent") -> str:
    """构建"计划 / 后台任务状态"补发段（T2 新增的恢复源，fail-open）。

    三种情况各自成段：
    - agent.plan_mode 为 True → 提醒模型当前在计划调研模式
    - agent._last_approved_plan 非空 → 用户已批准的计划全文（正在执行中）
    - delegate_tool._async_tasks 里有在跑的 → 列出任务 ID/目标摘要/已跑多久

    参数：
        agent: AIAgent 实例

    返回：拼好的状态文本；全都没有时返回空串。
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
            # 线程记录还在但线程本身已死 = 清理没跑完的残留条目（竞态），跳过
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
    """构建"最近读过的文件"补发段：文件路径 + 正文开头（每个最多 1K 字符）。

    安全底线：每个文件都过 safe_path 白名单检查，受保护路径
    （~/.ssh、/etc 等）直接跳过不读。

    参数：
        paths: 最近读过的文件路径列表（按读取顺序）
        max_files: 最多取几个

    返回：拼好的文本段；一个文件都读不成时返回空串。
    """
    if not paths:
        return ""

    from agent.permission import safe_path

    # 取最近的 max_files 个（保持原顺序，重复的已经被上游去重）
    recent = paths[-max_files:] if len(paths) > max_files else paths

    lines = [f"## 最近读过的文件（前 {len(recent)} 个，compact 后重注入）"]
    any_success = False
    for path in recent:
        try:
            # 过 safe_path 路径白名单检查（读模式）
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
    """构建"最近用过的技能"补发段：把技能正文重新塞回去。

    限制：每个技能最多 5K 字符，总量受 budget 限制。技能正文用
    agent._load_skill_body() 加载（它会跨目录找并剥掉开头的 frontmatter）。

    参数：
        agent: AIAgent 实例（读 _recent_skills）
        max_skills: 最多补发几个技能
        total_budget: 技能正文的总字数预算

    返回：拼好的文本段；一个技能都加载不成时返回空串。
    """
    skill_names = agent._recent_skills
    if not skill_names:
        return ""

    # 取最近的 max_skills 个（保持原顺序）
    recent_skills = skill_names[-max_skills:] if len(skill_names) > max_skills else skill_names

    lines = [f"## 最近加载的技能正文（前 {len(recent_skills)} 个，compact 后重注入）"]
    used = 0
    any_success = False

    # 倒着取：最近用过的技能优先补（对齐 Claude Code 的语义）
    for name in reversed(recent_skills):
        try:
            body = agent._load_skill_body(name)
            if not body:
                continue

            # 单个技能超 5K 就截断
            if len(body) > SKILL_PER_BUDGET_CHARS:
                body = body[:SKILL_PER_BUDGET_CHARS] + "\n...[truncated]..."

            # 总预算检查
            if used + len(body) > total_budget:
                break

            lines.append(f"### 技能 {name}\n{body}")
            used += len(body)
            any_success = True
        except Exception as e:
            logger.debug("recovery 加载技能失败 %s: %s", name, e)
            continue

    return "\n".join(lines) if any_success else ""
