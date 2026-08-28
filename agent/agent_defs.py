"""自定义子代理（subagent——主代理派出去干活的分身）的 .md 定义文件扫描。

用户用 Markdown 写一份"这个子代理叫什么、会什么、用什么模型、能用哪些工具"的
说明文件（顶部 frontmatter 填配置，正文当 system prompt），本模块负责把它们
找出来解析成 AgentDefinition 对象，供 delegate_tool 派活时使用。

从这几个目录扫（优先级从低到高，同名的后者覆盖前者）：
  - agent/builtin_agents/（内置，随项目分发）
  - ~/.OmniMate/agents/（用户级，自己所有项目共用）
  - CLI --agents 注入（命令行动态传入）
  - <cwd>/.omnimate/agents/（项目级，可入库跟仓库走，团队共享）

frontmatter（文件顶部 --- 包住的配置段）支持的字段：
name / description / model / tools / disallowedTools / permissionMode /
isolation / maxTurns（以及后面代码里陆续加的扩展字段）。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """一份子代理定义（从 .md 文件或 CLI 注入解析出来）。

    delegate_tool 按它起独立的 AIAgent 分身：用什么模型、能看到哪些
    工具、什么权限，全由这里的字段决定。
    """

    name: str
    description: str = ""
    model: Optional[str] = None
    tools: List[str] = field(default_factory=list)          # 工具集白名单（只列这些 toolset 名）
    disallowed_tools: List[str] = field(default_factory=list)
    permission_mode: Optional[str] = None                   # 权限模式：default | bypassPermissions
    isolation: Optional[str] = None                          # 隔离方式："worktree"（独立 git 工作树）| None
    max_turns: Optional[int] = None
    system_prompt: str = ""
    # 三个扩展字段：memory / skills / mcpServers
    memory: bool = False                                  # frontmatter 写 "memory: true" → 子代理有独立记忆目录
    skills: List[str] = field(default_factory=list)       # frontmatter "skills: [...]"
    mcp_servers: List[str] = field(default_factory=list)  # frontmatter "mcpServers: [...]"
    # 内联 mcpServers——直接在定义里写 server 配置（{名字: {command/args/url/...}}）。
    # 起子代理时临时连上、跑完就断（不写进全局配置）。与上面 mcp_servers 是互补关系：
    # mcp_servers 是"挑哪些已配置的全局 server 给它用"；inline_mcp_servers 是"现场定义新 server"
    inline_mcp_servers: dict = field(default_factory=dict)
    effort: Optional[str] = None                           # frontmatter "effort: max|high|medium|low"
    # === 4 个扩展字段 ===
    omit_claude_md: bool = False            # frontmatter "omitClaudeMd: true" → 子代理不加载项目 OMNIMATE.md（省 token）
    initial_prompt: str = ""                # frontmatter "initialPrompt" → 垫在第一条 user 消息前面（类似 slash 命令的预处理）
    required_mcp_servers: List[str] = field(default_factory=list)  # frontmatter "requiredMcpServers" → 缺这些 server 时整个 agent 不出现
    critical_reminder: str = ""             # frontmatter "criticalReminder" → 拼进 system_prompt 末尾（放尾部是为了不动前缀、保 cache）
    # 定义从哪来（builtin/user/cli/project）。为什么要记来源：项目级（project）
    # 的内联 MCP 要过首次连接审批——clone 陌生 repo 带进来的 agent .md 和 .mcp.json 是同一种威胁
    source: str = "user"


def _user_agents_dir() -> Path:
    """用户级子代理定义目录：~/.OmniMate/agents/。"""
    from constants import get_omnimate_home
    return get_omnimate_home() / "agents"


def _project_agents_dir() -> Path:
    # Path.cwd() 是进程共享的，并发子代理会互踩；get_workspace_cwd()
    # 基于 ContextVar，每个并发上下文拿到自己的工作目录。
    from agent.workspace_context import get_workspace_cwd
    return Path(get_workspace_cwd()) / ".omnimate" / "agents"


def _builtin_agents_dir() -> Path:
    """内置子代理定义所在目录（随项目代码一起分发）：agent/builtin_agents/。"""
    return Path(__file__).parent / "builtin_agents"


def _parse_inline_mcp(raw) -> dict:
    """解析 frontmatter 里的内联 mcpServers 配置。

    参数：
        raw：frontmatter 解析出来的原始值

    返回：
        {server名: 配置dict} 形式的 dict；raw 不是 dict 或子项不是 dict 时返回空 dict。
    """
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def _parse_one(skill_md: Path) -> Optional[AgentDefinition]:
    """把一个 agent .md 文件解析成 AgentDefinition。

    参数：
        skill_md：.md 文件路径

    返回：
        解析好的 AgentDefinition；文件没有 name、或解析过程出任何错，
        都返回 None（打个 warning，不让一个坏文件连累整轮扫描）。
    """
    try:
        content = skill_md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content)
        if not fm.get("name"):
            return None
        # fail-loud：tools 字段是套餐名清单，写错一个就整份定义跳过——
        # 否则子代理会静默拿不到工具（silent dead agent 比报错危险得多）
        from toolsets import TOOLSETS as _KNOWN_TOOLSETS
        _bad_ts = [
            t for t in (fm.get("tools") or [])
            if t not in _KNOWN_TOOLSETS
        ]
        if _bad_ts:
            logger.warning(
                "子代理定义 %s 的 tools 含未知工具集名 %s（合法值: %s），"
                "整份定义跳过",
                skill_md.name, _bad_ts, sorted(_KNOWN_TOOLSETS),
            )
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
            inline_mcp_servers=_parse_inline_mcp(fm.get("mcpServersInline") or fm.get("inlineMcpServers")),
            effort=fm.get("effort"),
            # === 4 个扩展字段，frontmatter 里是 camelCase，这里转成 python 的 snake_case ===
            omit_claude_md=bool(fm.get("omitClaudeMd", False)),
            initial_prompt=str(fm.get("initialPrompt") or ""),
            required_mcp_servers=fm.get("requiredMcpServers") or [],
            critical_reminder=str(fm.get("criticalReminder") or ""),
        )
    except Exception as e:
        logger.warning("解析子代理定义失败 %s: %s", skill_md, e)
        return None


def scan_agent_defs() -> Dict[str, AgentDefinition]:
    """把四个来源的子代理定义全扫一遍，合并成一份总表。

    同名冲突时后扫的覆盖先扫的。优先级（低 → 高）：
      1. 内置（agent/builtin_agents/，随代码分发）
      2. 用户级（~/.OmniMate/agents/，跨项目个人配置）
      3. CLI 注入（启动命令 --agents '{json}'）
      4. 项目级（<cwd>/.omnimate/agents/，跟仓库走，团队共享）

    返回：
        {子代理名: AgentDefinition} 字典。
    """
    defs: Dict[str, AgentDefinition] = {}
    for d, _src in [(_builtin_agents_dir(), "builtin"), (_user_agents_dir(), "user")]:
        if not d.exists():
            continue
        for md in sorted(d.glob("*.md")):
            ad = _parse_one(md)
            if ad and ad.name:
                ad.source = _src
                defs[ad.name] = ad  # 同名时后扫到的赢
    # CLI 注入的子代理（优先级排在 user 和 project 之间）
    for name, ad in _cli_injected.items():
        defs[name] = ad
    # 项目级优先级最高，最后扫、最终生效
    proj_dir = _project_agents_dir()
    if proj_dir.exists():
        for md in sorted(proj_dir.glob("*.md")):
            ad = _parse_one(md)
            if ad and ad.name:
                ad.source = "project"
                defs[ad.name] = ad
    return defs


def project_inline_mcp_servers() -> Dict[str, dict]:
    """把所有项目级 agent .md 里声明的内联 MCP server 合并成一张表。

    这些 server 来自项目目录（可能是 clone 来的陌生仓库），收拢出来
    交给首连审批统一把关。

    返回：
        {server名: 配置dict}；扫描出任何异常都返回空 dict（fail-open，不阻塞启动）。
    """
    try:
        return {
            name: cfg
            for ad in scan_agent_defs().values()
            if ad.source == "project"
            for name, cfg in (ad.inline_mcp_servers or {}).items()
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CLI 动态注入——启动命令 --agents '{json}' 传进来的定义放这里
# ---------------------------------------------------------------------------

_cli_injected: Dict[str, AgentDefinition] = {}


def inject_cli_agents(cli_agents: Dict[str, dict]) -> int:
    """把 CLI `--agents '{json}'` 参数传进来的子代理定义登记进来（不落盘）。

    参数：
        cli_agents：{名字: {description, prompt, tools, model, ...}} 形式的 dict

    返回：
        int——成功解析的数量。重复调用是替换语义（先清空再装，幂等）。
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
                # === CLI 注入同样接受那 4 个新字段 ===
                omit_claude_md=bool(cfg.get("omitClaudeMd", False)),
                initial_prompt=str(cfg.get("initialPrompt") or ""),
                required_mcp_servers=cfg.get("requiredMcpServers") or [],
                critical_reminder=str(cfg.get("criticalReminder") or ""),
                source="cli",
            )
            _cli_injected[name] = ad
            count += 1
        except Exception as e:
            logger.warning("CLI 注入子代理 '%s' 解析失败: %s", name, e)
    return count


def get_cli_injected() -> Dict[str, AgentDefinition]:
    """查看当前 CLI 注入了哪些子代理（主要给测试断言用）。"""
    return dict(_cli_injected)


def clear_cli_injected() -> None:
    """清空 CLI 注入的子代理（主要给测试用例之间隔离用）。"""
    _cli_injected.clear()


def get_agent_def(name: str) -> Optional[AgentDefinition]:
    """按名字查一个子代理定义（内部会重新扫一遍四个来源）。

    参数：
        name：子代理名

    返回：
        对应的 AgentDefinition；不存在返回 None。
    """
    return scan_agent_defs().get(name)
