"""输出风格（output styles）——对齐 CCB outputStyles 的极简版（C6 借鉴）。

风格 = 一个 .md 文件：
- 文件名（去扩展名）即风格名
- frontmatter 可选（name/description，与 skills 同款解析器）
- 正文即提示词（指导模型的输出形态）

目录发现（项目级覆盖用户级，同名后者胜）：
- 项目级：``<cwd 向上到 git root（含）>/.omnimate/output-styles/*.md``
  （对齐 OMNIMATE.md 的向上收集语义）
- 用户级：``<agent_home>/output-styles/*.md``

注入：system prompt 的 **context 层**（单会话内不变，保 prompt cache；
切换风格 = settings.json 顶层 ``output_style`` + invalidate prompt）。

整链 fail-open：未配置/风格名不存在/目录读失败 → 不注入任何内容。
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_DIR = Path(".omnimate") / "output-styles"  # 相对项目根
_USER_DIR = "output-styles"                          # 相对 agent_home


@dataclass
class OutputStyle:
    """一个输出风格定义。"""
    name: str
    description: str
    body: str
    source: str  # 来源文件路径（展示用）


def _parse_style_file(path: Path) -> Optional[OutputStyle]:
    """解析单个风格 .md（frontmatter + 正文）。失败返回 None（fail-open）。"""
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
    """从 cwd 向上到 git root（含）收集目录（对齐 OMNIMATE.md 收集语义）。"""
    roots = []
    current = Path(cwd).resolve()
    while True:
        roots.append(current)
        if (current / ".git").exists():
            break  # git root 层含这一层即停
        parent = current.parent
        if parent == current:
            break
        current = parent
    return roots


def discover_output_styles(cwd: str, agent_home) -> Dict[str, OutputStyle]:
    """发现全部输出风格。项目级同名覆盖用户级（后扫描覆盖）。"""
    styles: Dict[str, OutputStyle] = {}
    # 用户级先入表
    user_dir = Path(agent_home) / _USER_DIR if agent_home else None
    if user_dir and user_dir.is_dir():
        for p in sorted(user_dir.glob("*.md")):
            s = _parse_style_file(p)
            if s:
                styles[s.name] = s
    # 项目级后扫描覆盖（cwd 向上到 git root）
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
    """按 config.output_style 解析风格。未配置/未知名 → None（fail-open）。"""
    name = (config or {}).get("output_style")
    if not name:
        return None
    styles = discover_output_styles(cwd, agent_home)
    style = styles.get(str(name))
    if style is None:
        logger.info("输出风格 %r 不存在（可用: %s），跳过注入", name, sorted(styles))
    return style


def render_style_section(style: OutputStyle) -> str:
    """渲染注入 system prompt context 层的节文本。"""
    head = f"## 输出风格：{style.name}"
    if style.description:
        head += f"\n{style.description}"
    return f"{head}\n{style.body}"
