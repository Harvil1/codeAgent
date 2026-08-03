"""自定义子代理 .md 定义扫描（对齐 Claude Code .claude/agents/*.md）。

扫描两个目录：
  - ~/.OmniMate/agents/（用户级，跨项目）
  - <cwd>/.claude/agents/（项目级，入库共享，覆盖用户级）

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


def _user_agents_dir() -> Path:
    from constants import get_omnimate_home
    return get_omnimate_home() / "agents"


def _project_agents_dir() -> Path:
    return Path.cwd() / ".claude" / "agents"


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
        )
    except Exception as e:
        logger.warning("解析子代理定义失败 %s: %s", skill_md, e)
        return None


def scan_agent_defs() -> Dict[str, AgentDefinition]:
    """扫描两个目录，项目级覆盖用户级。"""
    defs: Dict[str, AgentDefinition] = {}
    for d in [_user_agents_dir(), _project_agents_dir()]:
        if not d.exists():
            continue
        for md in sorted(d.glob("*.md")):
            ad = _parse_one(md)
            if ad and ad.name:
                defs[ad.name] = ad  # 后扫的覆盖先扫的
    return defs


def get_agent_def(name: str) -> Optional[AgentDefinition]:
    """按名字取单个定义。"""
    return scan_agent_defs().get(name)
