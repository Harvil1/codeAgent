"""自定义子代理 .md 定义扫描。

扫描两个目录：
  - ~/.OmniMate/agents/（用户级，跨项目）
  - <cwd>/.omnimate/agents/（项目级，入库共享，覆盖用户级）

frontmatter 字段：name / description / model / tools / disallowedTools /
permissionMode / isolation / maxTurns。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    name: str
    description: str = ""
    model: Optional[str] = None
    tools: List[str] = field(default_factory=list)          # toolset 名白名单
    disallowed_tools: List[str] = field(default_factory=list)
    permission_mode: Optional[str] = None                   # default | bypassPermissions
    isolation: Optional[str] = None                          # "worktree" | None
    max_turns: Optional[int] = None
    system_prompt: str = ""
    # Task C1 新增：memory/skills/mcpServers 三字段
    memory: bool = False                                  # frontmatter "memory: true"
    skills: List[str] = field(default_factory=list)       # frontmatter "skills: [...]"
    mcp_servers: List[str] = field(default_factory=list)  # frontmatter "mcpServers: [...]"
    effort: Optional[str] = None                           # frontmatter "effort: max|high|medium|low"
    # === Task N 新增 4 字段（借鉴 Claude Code）===
    omit_claude_md: bool = False            # frontmatter "omitClaudeMd: true" → 子代理跳过项目 OMNIMATE.md（省 token）
    initial_prompt: str = ""                # frontmatter "initialPrompt" → 首 user turn 前置（slash 风格预处理）
    required_mcp_servers: List[str] = field(default_factory=list)  # frontmatter "requiredMcpServers" → 缺失则 agent 不显示
    critical_reminder: str = ""             # frontmatter "criticalReminder" → 拼 system_prompt（cache 友好）


def _user_agents_dir() -> Path:
    from constants import get_omnimate_home
    return get_omnimate_home() / "agents"


def _project_agents_dir() -> Path:
    # Round 1 fix: Path.cwd() 是进程级（=os.getcwd），并发子代理会踩。
    # 改走 get_workspace_cwd()（线程局部 ContextVar）。
    from agent.workspace_context import get_workspace_cwd
    return Path(get_workspace_cwd()) / ".omnimate" / "agents"


def _builtin_agents_dir() -> Path:
    """内置子代理定义目录（随项目分发）：agent/builtin_agents/。"""
    return Path(__file__).parent / "builtin_agents"


def _parse_one(skill_md: Path) -> Optional[AgentDefinition]:
    try:
        content = skill_md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content)
        if not fm.get("name"):
            return None
        return AgentDefinition(
            name=fm["name"],
            description=fm.get("description", ""),
            model=fm.get("model"),
            tools=fm.get("tools") or [],
            disallowed_tools=fm.get("disallowedTools") or [],
            permission_mode=fm.get("permissionMode"),
            isolation=fm.get("isolation"),
            max_turns=fm.get("maxTurns"),
            system_prompt=body.strip(),
            memory=bool(fm.get("memory", False)),
            skills=fm.get("skills") or [],
            mcp_servers=fm.get("mcpServers") or [],
            effort=fm.get("effort"),
            # === Task N: 4 新字段 frontmatter camelCase → snake_case ===
            omit_claude_md=bool(fm.get("omitClaudeMd", False)),
            initial_prompt=str(fm.get("initialPrompt") or ""),
            required_mcp_servers=fm.get("requiredMcpServers") or [],
            critical_reminder=str(fm.get("criticalReminder") or ""),
        )
    except Exception as e:
        logger.warning("解析子代理定义失败 %s: %s", skill_md, e)
        return None


def scan_agent_defs() -> Dict[str, AgentDefinition]:
    """扫描内置 + 用户 + CLI 注入 + 项目四个来源，后者覆盖前者。

    优先级（低 → 高）：
      1. 内置（agent/builtin_agents/）
      2. 用户级（~/.OmniMate/agents/）
      3. CLI 注入（`--agents '{json}'`，对齐 Claude Code 的 --agents flag）
      4. 项目级（<cwd>/.omnimate/agents/）
    """
    defs: Dict[str, AgentDefinition] = {}
    for d in [_builtin_agents_dir(), _user_agents_dir()]:
        if not d.exists():
            continue
        for md in sorted(d.glob("*.md")):
            ad = _parse_one(md)
            if ad and ad.name:
                defs[ad.name] = ad  # 后扫的覆盖先扫的
    # 阶段 6 NEW: CLI 注入的子代理（优先级介于 user 和 project 之间）
    for name, ad in _cli_injected.items():
        defs[name] = ad
    # 项目级最高优先级
    proj_dir = _project_agents_dir()
    if proj_dir.exists():
        for md in sorted(proj_dir.glob("*.md")):
            ad = _parse_one(md)
            if ad and ad.name:
                defs[ad.name] = ad
    return defs


# ---------------------------------------------------------------------------
# 阶段 6 NEW: CLI 动态注入（--agents '{json}'）
# ---------------------------------------------------------------------------

_cli_injected: Dict[str, AgentDefinition] = {}


def inject_cli_agents(cli_agents: Dict[str, dict]) -> int:
    """注入 CLI `--agents '{json}'` 传入的子代理定义。

    参数 cli_agents：{name: {description, prompt, tools, model, ...}}，
    对齐 Claude Code `claude --agents '{json}'` 格式。
    返回成功解析的数量。同 session 内幂等：可多次调用替换。
    """
    _cli_injected.clear()
    count = 0
    if not cli_agents or not isinstance(cli_agents, dict):
        return 0
    for name, cfg in cli_agents.items():
        if not isinstance(cfg, dict):
            continue
        try:
            ad = AgentDefinition(
                name=name,
                description=cfg.get("description", ""),
                model=cfg.get("model"),
                tools=cfg.get("tools") or [],
                disallowed_tools=cfg.get("disallowedTools") or [],
                permission_mode=cfg.get("permissionMode"),
                isolation=cfg.get("isolation"),
                max_turns=cfg.get("maxTurns"),
                system_prompt=cfg.get("prompt", "").strip(),
                memory=bool(cfg.get("memory", False)),
                skills=cfg.get("skills") or [],
                mcp_servers=cfg.get("mcpServers") or [],
                effort=cfg.get("effort"),
                # === Task N: CLI 注入也接受 4 新字段 ===
                omit_claude_md=bool(cfg.get("omitClaudeMd", False)),
                initial_prompt=str(cfg.get("initialPrompt") or ""),
                required_mcp_servers=cfg.get("requiredMcpServers") or [],
                critical_reminder=str(cfg.get("criticalReminder") or ""),
            )
            _cli_injected[name] = ad
            count += 1
        except Exception as e:
            logger.warning("CLI 注入子代理 '%s' 解析失败: %s", name, e)
    return count


def get_cli_injected() -> Dict[str, AgentDefinition]:
    """测试用：返回当前 CLI 注入的子代理。"""
    return dict(_cli_injected)


def clear_cli_injected() -> None:
    """测试用：清空 CLI 注入。"""
    _cli_injected.clear()


def get_agent_def(name: str) -> Optional[AgentDefinition]:
    """按名字取单个定义。"""
    return scan_agent_defs().get(name)
