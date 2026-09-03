"""皮肤引擎——界面配色/品牌件的统一供血层（从 hermes 移植的子集）。

大白话：界面上的颜色（❯ 提示符、状态栏、回答框边框）、品牌件（agent
名字、回答框标签、告别语）、工具行前缀 ┊，全部从「当前皮肤」取——
换皮肤 = 一处换、处处变，不用满代码库找颜色字面量。

皮肤从哪来（优先级从低到高）：
1. 内置四套：default（金）/ mono（无色，老旧终端友好）/ slate（蓝灰）/
   daylight（亮底）；
2. 用户 YAML：~/.codeAgent/skins/*.yaml，以 default 为底逐节覆盖。

用法：
    init_skin_from_config(rt.config)   # 启动时按 settings 的 display.skin 激活
    get_active_skin().get_color("prompt", "#00aa88")
    set_active_skin("slate")           # /skin 命令热切换
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cli_commands import slash_command
from cli_ui import console

logger = logging.getLogger(__name__)


# =============================================================================
# 皮肤数据结构
# =============================================================================

@dataclass
class SkinConfig:
    """一套皮肤的完整配置（hermes SkinConfig 的子集，够用就好）。

    字段：
        name / description: 名字和一句话介绍（/skin list 展示用）
        colors: 颜色键值表（hex），键见 _BUILTIN_SKINS["default"]
        branding: 品牌文案（agent_name / response_label / prompt_symbol / goodbye）
        tool_prefix: 工具行前缀（默认 ┊）
        spinner_frames: spinner 动画帧（空列表 = 用 cli_layout 的盲文帧）
    """

    name: str
    description: str = ""
    colors: Dict[str, str] = field(default_factory=dict)
    branding: Dict[str, str] = field(default_factory=dict)
    tool_prefix: str = "┊"
    spinner_frames: List[str] = field(default_factory=list)

    def get_color(self, key: str, fallback: str = "") -> str:
        """取颜色值，键不存在给兜底。"""
        return self.colors.get(key, fallback)

    def get_branding(self, key: str, fallback: str = "") -> str:
        """取品牌文案，键不存在给兜底。"""
        return self.branding.get(key, fallback)


# =============================================================================
# 内置皮肤（配色抄 hermes 的四个经典款）
# =============================================================================

_BUILTIN_SKINS: Dict[str, Dict[str, Any]] = {
    "default": {
        "name": "default",
        "description": "经典金——暖色高亮，暗底终端",
        "colors": {
            "banner_border": "#CD7F32",
            "banner_title": "#FFD700",
            "banner_text": "#FFF8DC",
            "ui_accent": "#FFBF00",
            "response_border": "#FFD700",
            "prompt": "#00aa88",
            "status_bar_fg": "",
            "separator": "#555555",
            "placeholder": "#777777",
            "tool_line": "",
        },
        "branding": {
            "agent_name": "CodeAgent",
            "response_label": "⚕ CodeAgent",
            "prompt_symbol": "❯",
            "goodbye": "再见！",
        },
        "tool_prefix": "┊",
    },
    "mono": {
        "name": "mono",
        "description": "无色——不发任何颜色码，老旧终端/日志翻录友好",
        "colors": {
            "banner_border": "",
            "banner_title": "",
            "banner_text": "",
            "ui_accent": "",
            "response_border": "",
            "prompt": "",
            "status_bar_fg": "",
            "separator": "",
            "placeholder": "",
            "tool_line": "",
        },
        "branding": {
            "agent_name": "CodeAgent",
            "response_label": "CodeAgent",
            "prompt_symbol": ">",
            "goodbye": "再见！",
        },
        "tool_prefix": "|",
    },
    "slate": {
        "name": "slate",
        "description": "石板蓝灰——冷色调，长时间盯着不累",
        "colors": {
            "banner_border": "#5F7A8A",
            "banner_title": "#A8C4D4",
            "banner_text": "#D8E4EC",
            "ui_accent": "#7FA8C9",
            "response_border": "#8FB8D8",
            "prompt": "#6FA8DC",
            "status_bar_fg": "",
            "separator": "#3A4A55",
            "placeholder": "#6A7A85",
            "tool_line": "",
        },
        "branding": {
            "agent_name": "CodeAgent",
            "response_label": "▤ CodeAgent",
            "prompt_symbol": "❯",
            "goodbye": "再见！",
        },
        "tool_prefix": "┊",
    },
    "daylight": {
        "name": "daylight",
        "description": "日光亮底——白底终端用，颜色加深保可读",
        "colors": {
            "banner_border": "#8B6914",
            "banner_title": "#8B6914",
            "banner_text": "#333333",
            "ui_accent": "#A0522D",
            "response_border": "#A0522D",
            "prompt": "#007050",
            "status_bar_fg": "",
            "separator": "#AAAAAA",
            "placeholder": "#999999",
            "tool_line": "",
        },
        "branding": {
            "agent_name": "CodeAgent",
            "response_label": "☀ CodeAgent",
            "prompt_symbol": "❯",
            "goodbye": "再见！",
        },
        "tool_prefix": "┊",
    },
}


# =============================================================================
# 激活态（模块级单例——皮肤是全局的，跟 hermes 一样）
# =============================================================================

_active_skin: Optional[SkinConfig] = None
_active_skin_name: str = "default"


def _skins_dir() -> Path:
    """用户皮肤目录 ~/.codeAgent/skins/（没有就当空集，不报错）。"""
    try:
        from constants import get_codeagent_home
        return get_codeagent_home() / "skins"
    except Exception:
        return Path.home() / ".codeAgent" / "skins"


def _load_skin_from_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """读一个用户皮肤 YAML；坏文件返回 None（大声记日志但不炸）。"""
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("皮肤文件 %s 读取失败（跳过）: %s", path.name, e)
        return None


def _build_skin_config(data: Dict[str, Any]) -> SkinConfig:
    """把皮肤字典变成 SkinConfig——以 default 为底逐节覆盖（overlay）。

    大白话：用户 YAML 不用写全所有键，写哪节换哪节，剩下的继承 default。
    """
    base = _BUILTIN_SKINS["default"]
    merged: Dict[str, Any] = {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in base.items()
    }
    for key in ("colors", "branding"):
        overlay = data.get(key)
        if isinstance(overlay, dict):
            merged.setdefault(key, {}).update(overlay)
    for key in ("name", "description", "tool_prefix"):
        if data.get(key):
            merged[key] = data[key]
    frames = data.get("spinner_frames")
    return SkinConfig(
        name=str(merged.get("name", "default")),
        description=str(merged.get("description", "")),
        colors=merged.get("colors", {}),
        branding=merged.get("branding", {}),
        tool_prefix=str(merged.get("tool_prefix", "┊")),
        spinner_frames=list(frames) if isinstance(frames, list) else [],
    )


def list_skins() -> List[Dict[str, str]]:
    """列出全部可用皮肤（内置 + 用户 YAML），/skin list 的数据源。"""
    out = [{"name": k, "description": v.get("description", ""),
            "source": "内置"}
           for k, v in _BUILTIN_SKINS.items()]
    try:
        for p in sorted(_skins_dir().glob("*.yaml")):
            data = _load_skin_from_yaml(p)
            if data and data.get("name"):
                out.append({"name": str(data["name"]),
                            "description": str(data.get("description", "")),
                            "source": p.stem})
    except Exception:
        pass
    return out


def load_skin(name: str) -> SkinConfig:
    """按名字加载皮肤：内置 → 用户 YAML → 都没有就 default（宽松回退）。"""
    if name in _BUILTIN_SKINS:
        return _build_skin_config(_BUILTIN_SKINS[name])
    try:
        for p in sorted(_skins_dir().glob("*.yaml")):
            data = _load_skin_from_yaml(p)
            if data and str(data.get("name", "")) == name:
                return _build_skin_config(data)
    except Exception:
        pass
    logger.warning("皮肤 %r 不存在，回退 default", name)
    return _build_skin_config(_BUILTIN_SKINS["default"])


def get_active_skin() -> SkinConfig:
    """当前激活皮肤（没激活过就 default）。"""
    global _active_skin
    if _active_skin is None:
        _active_skin = _build_skin_config(_BUILTIN_SKINS["default"])
    return _active_skin


def set_active_skin(name: str) -> SkinConfig:
    """激活一套皮肤（只换内存态；持久化由调用方走 settings）。"""
    global _active_skin, _active_skin_name
    _active_skin = load_skin(name)
    _active_skin_name = _active_skin.name
    return _active_skin


def get_active_skin_name() -> str:
    """当前激活皮肤名。"""
    return _active_skin_name


def init_skin_from_config(config: dict) -> None:
    """启动时按 settings.json 的 display.skin 激活皮肤（失败回退 default）。"""
    try:
        display = (config or {}).get("display") or {}
        set_active_skin(str(display.get("skin", "default")))
    except Exception:
        set_active_skin("default")


def hex_to_truecolor_ansi(hex_color: str, *, bold: bool = False) -> str:
    """hex 颜色 → 24bit ANSI 转义（流式框用它给普通 print 上色）。

    非法/空颜色返回空串（mono 皮肤就这么走，一行 ANSI 都不发）。
    """
    h = (hex_color or "").strip()
    if not h.startswith("#") or len(h) != 7:
        return ""
    try:
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    except ValueError:
        return ""
    prefix = "\033[1;" if bold else "\033["
    return f"{prefix}38;2;{r};{g};{b}m"


# =============================================================================
# 与 prompt_toolkit 样式表的对接
# =============================================================================

def get_pt_style_overrides() -> Dict[str, str]:
    """皮肤 → pt 样式覆盖字典（cli_layout 的基础样式表拿它做 overlay）。

    空颜色键（mono）干脆不出现在覆盖里——底表里的 class 就没颜色。
    """
    skin = get_active_skin()
    out: Dict[str, str] = {}

    def _put(class_name: str, hex_color: str, extra: str = ""):
        if hex_color:
            out[class_name] = f"fg:{hex_color}" + (f" {extra}" if extra else "")

    _put("prompt", skin.get_color("prompt", ""), "bold")
    _put("separator", skin.get_color("separator", ""))
    _put("placeholder", skin.get_color("placeholder", ""))
    sb = skin.get_color("status_bar_fg", "")
    out["status-bar"] = (f"fg:{sb} " if sb else "") + "reverse"
    return out


def get_active_prompt_symbol(fallback: str = "❯") -> str:
    """当前皮肤的输入提示符（❯ / > 等）。"""
    return get_active_skin().get_branding("prompt_symbol", fallback) or fallback


def apply_skin_to_app(app) -> bool:
    """把当前皮肤热套到活着的 Application 上（/skin 切换用）。

    大白话：换皮肤不用重启——重新织一件样式表给操作台套上就行。
    app 是 None（理论上不会）或样式失败都返回 False。
    """
    if app is None:
        return False
    try:
        from prompt_toolkit.styles import Style
        base = {
            "status-bar": "reverse",
            "separator": "fg:#555555",
            "placeholder": "fg:#777777",
            "prompt": "bold fg:#00aa88",
        }
        base.update(get_pt_style_overrides())
        app.style = Style.from_dict(base)
        app.invalidate()
        return True
    except Exception as e:
        logger.warning("皮肤热套失败（下次重绘仍是旧皮肤）: %s", e)
        return False


# =============================================================================
# /skin 命令（import 本模块即自登记）
# =============================================================================

def _skin_candidates(text: str) -> List[str]:
    """参数补全：/skin <Tab> 出皮肤名。"""
    return [s["name"] for s in list_skins()]


@slash_command(
    name="/skin",
    aliases=["/skins"],
    category="配置",
    usage="/skin [list|<名字>]",
    summary="查看/切换界面皮肤",
    arg_completer=_skin_candidates,
)
def skin_cmd(args: str, rt) -> bool:
    """处理 /skin——列出或热切换界面皮肤。

    子命令：
        /skin            列出全部皮肤（标出当前款）
        /skin <名字>     热切换（内存即时生效；持久化写 settings）
    """
    parts = (args or "").strip().split()
    if not parts or parts[0].lower() == "list":
        current = get_active_skin_name()
        rows = []
        for s in list_skins():
            mark = "← 当前" if s["name"] == current else ""
            source = s.get("source", "")
            rows.append(
                f"  [cyan]{s['name']}[/cyan]"
                f"{('（' + source + '）') if source and source != '内置' else ''}"
                f"  {s['description']}  {mark}"
            )
        console.print("[bold]可用皮肤[/bold]（/skin <名字> 切换）\n" + "\n".join(rows))
        return True

    name = parts[0]
    if name == get_active_skin_name():
        console.print(f"[dim]已经在用 {name} 了[/dim]")
        return True
    skin = set_active_skin(name)
    if skin.name != name and name != "default":
        console.print(f"[yellow]皮肤 {name} 不存在（/skin 看清单）[/yellow]")
        return True
    # 热套到活着的操作台
    applied = apply_skin_to_app(getattr(rt, "prompt_session", None))
    # 持久化走 settings 唯一正道
    try:
        from agent.settings import save_settings
        save_settings({"display": {"skin": skin.name}})
        console.print(f"[green]皮肤已切换：{skin.name}"
                      + ("（操作台已热套）" if applied else "（下次重绘生效）")
                      + "[/green]")
    except Exception as e:
        console.print(f"[green]皮肤已切换：{skin.name}（本次会话生效；"
                      f"持久化失败: {e}）[/green]")
    return True
