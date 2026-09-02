"""输出风格（output styles）模块——让用户一键切换模型的说话/排版方式。

所谓"风格"就是一份 .md 文件：
- 文件名去掉扩展名就是风格名（如 concise.md → 风格 concise）
- 开头的 frontmatter 可选（可写 name/description，解析器和技能共用同一套）
- 正文就是给模型的提示词（比如"回答保持三句话以内"）

去哪找风格文件（项目级和用户级同名时，项目级说了算）：
- 项目级：从当前目录向上到 git 仓库根（含），沿途的
  ``.codeAgent/output-styles/*.md``（和 CODEAGENT.md 的向上收集规则一致）
- 用户级：``<agent_home>/output-styles/*.md``（全局通用）

怎么生效：注入到 system prompt 的 context 层（会话内不变，保住前缀
缓存；切风格 = 改 settings.json 顶层的 output_style 并作废旧 prompt）。

整条链 fail-open：没配置/风格名不存在/目录读失败 → 什么都不注入。
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(".codeAgent") / "output-styles"  # 项目级目录（相对项目根）
_USER_DIR = "output-styles"                          # 用户级目录（相对 agent_home）


@dataclass
class OutputStyle:
    """一个输出风格的定义（名字、描述、提示词正文、来源文件）。"""
    name: str
    description: str
    body: str
    source: str  # 来源文件路径（展示给用户看用）


def _parse_style_file(path: Path) -> Optional[OutputStyle]:
    """解析一个风格 .md 文件（拆出 frontmatter 和正文）。

    参数：
        path: 风格文件路径

    返回：OutputStyle 对象；文件读不了、正文是空的都返回 None
    （fail-open，坏一个文件不影响其他风格）。
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.debug("读取输出风格 %s 失败（忽略）: %s", path, e)
        return None
    try:
        from agent.skill_commands import parse_frontmatter
        fm, body = parse_frontmatter(content)
    except Exception:
        fm, body = {}, content
    body = (body or "").strip()
    if not body:
        return None
    return OutputStyle(
        name=str(fm.get("name") or path.stem),
        description=str(fm.get("description") or ""),
        body=body,
        source=str(path),
    )


def _project_roots(cwd: str):
    """列出从当前目录一路向上到 git 仓库根（含）的所有目录。

    和 CODEAGENT.md 的收集规则保持一致——每层目录都可能藏着
    项目级风格文件，得挨个看一遍。

    参数：
        cwd: 起始目录

    返回：目录路径列表（从当前目录到仓库根）。
    """
    roots = []
    current = Path(cwd).resolve()
    while True:
        roots.append(current)
        if (current / ".git").exists():
            break  # 到 git 仓库根（这层也要）就停
        parent = current.parent
        if parent == current:
            break
        current = parent
    return roots


def discover_output_styles(cwd: str, agent_home) -> Dict[str, OutputStyle]:
    """扫出所有可用的输出风格。

    规则：用户级先入表，项目级后扫——同名时项目级覆盖用户级
    （项目里的约定优先于个人全局偏好）。

    参数：
        cwd: 当前工作目录（向上找项目级风格用）
        agent_home: agent 根目录（找用户级风格用）

    返回：{风格名: OutputStyle} 字典；没有可用风格时为空字典。
    """
    styles: Dict[str, OutputStyle] = {}
    # 用户级先入表
    user_dir = Path(agent_home) / _USER_DIR if agent_home else None
    if user_dir and user_dir.is_dir():
        for p in sorted(user_dir.glob("*.md")):
            s = _parse_style_file(p)
            if s:
                styles[s.name] = s
    # 项目级后扫覆盖（从当前目录向上到 git 仓库根）
    try:
        roots = _project_roots(str(cwd))
    except Exception:
        roots = []
    for root in roots:
        d = root / _PROJECT_DIR
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.md")):
            s = _parse_style_file(p)
            if s:
                styles[s.name] = s
    return styles


def resolve_output_style(
    config: dict, cwd: str, agent_home,
) -> Optional[OutputStyle]:
    """按配置里指定的名字找到对应的输出风格。

    参数：
        config: 配置字典（读顶层 output_style 键）
        cwd: 当前工作目录（发现项目级风格用）
        agent_home: agent 根目录（发现用户级风格用）

    返回：找到的 OutputStyle；没配置或名字对不上任何风格时返回 None
    （fail-open，不注入任何内容）。
    """
    name = (config or {}).get("output_style")
    if not name:
        return None
    styles = discover_output_styles(cwd, agent_home)
    style = styles.get(str(name))
    if style is None:
        logger.info("输出风格 %r 不存在（可用: %s），跳过注入", name, sorted(styles))
    return style


def render_style_section(style: OutputStyle) -> str:
    """把风格渲染成拼进 system prompt context 层的一节文本。

    参数：
        style: 输出风格对象

    返回：形如 "## 输出风格：xxx" 开头、正文跟在后面的文本块。
    """
    head = f"## 输出风格：{style.name}"
    if style.description:
        head += f"\n{style.description}"
    return f"{head}\n{style.body}"
