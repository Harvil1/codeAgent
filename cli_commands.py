"""slash 命令注册表——命令元数据的唯一权威来源。

一份登记表喂三个消费者：Tab 补全（cli_layout）、/help 帮助、命令分发
（cli._handle_command 查表）。好处：三处永远同步，不会出现「补全提示有
但实际不存在」的错位。

用法（自登记模式，抄 tools/registry 的作业——命令模块被 import 就登记）：

    @slash_command(name="/sessions", aliases=["/ses"], category="会话",
                   usage="/sessions [list|resume|delete]",
                   summary="列出/恢复/删除历史会话")
    def sessions_cmd(args: str, rt) -> bool:
        ...  # 返回 True 表示已处理
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SlashCommand:
    """一条 slash 命令的全部元数据。

    字段：
        name: 主命令名（如 "/sessions"）
        aliases: 别名列表（代码写死，不是用户配置）
        category: /help 分组（会话/配置/诊断/技能…）
        usage: 用法行（如 "/sessions [list|resume|delete]"）
        summary: 一句话帮助
        handler: 处理函数 fn(args: str, rt) -> bool（True=已处理）
        arg_completer: 参数补全器 fn(text: str) -> list[str]（None=无参数补全）
    """
    name: str
    aliases: List[str] = field(default_factory=list)
    category: str = ""
    usage: str = ""
    summary: str = ""
    handler: Callable = None
    arg_completer: Optional[Callable] = None


# token（主名+别名）→ 命令对象
_registry: Dict[str, SlashCommand] = {}


def reset() -> None:
    """清空注册表（测试隔离专用，运行时没人调）。"""
    _registry.clear()


def register(*, name, aliases, category, usage, summary, handler,
             arg_completer=None) -> SlashCommand:
    """登记一条命令（名字和别名都进索引，指向同一对象）。"""
    cmd = SlashCommand(name=name, aliases=list(aliases or []),
                       category=category, usage=usage, summary=summary,
                       handler=handler, arg_completer=arg_completer)
    for key in [name, *cmd.aliases]:
        if key in _registry:
            logger.warning("slash 命令 %s 重复登记，后者覆盖前者", key)
        _registry[key] = cmd
    return cmd


def slash_command(*, name, category, usage, summary,
                  aliases=None, arg_completer=None):
    """装饰器版登记：被装饰函数就是 handler，登记完原样返回函数。"""
    def deco(fn):
        register(name=name, aliases=aliases, category=category,
                 usage=usage, summary=summary, handler=fn,
                 arg_completer=arg_completer)
        return fn
    return deco


def lookup(token: str) -> Optional[SlashCommand]:
    """按 token（主名或别名）查命令；没有返回 None。"""
    return _registry.get(token)


def all_commands() -> List[SlashCommand]:
    """全部命令（按主名去重，别名不单独计）。"""
    seen, out = set(), []
    for cmd in _registry.values():
        if cmd.name not in seen:
            seen.add(cmd.name)
            out.append(cmd)
    return out


def suggest(token: str, n: int = 3) -> List[str]:
    """给敲错的命令找最近似的候选（difflib 编辑距离）。"""
    import difflib
    names = [c.name for c in all_commands()]
    return difflib.get_close_matches(token, names, n=n, cutoff=0.6)


def dispatch(cmd_line: str, rt) -> Optional[bool]:
    """查表分发：cmd_line 拆成「命令 token + 剩余参数」转给 handler。

    返回：None=注册表里没这个命令（调用方走未知命令分支）；
          否则透传 handler 的返回值（True=已处理）。
    """
    token = cmd_line.split()[0].lower() if cmd_line.split() else ""
    entry = lookup(token)
    if entry is None:
        return None
    rest = cmd_line[len(token):].strip()
    try:
        return bool(entry.handler(rest, rt))
    except Exception:
        logger.exception("命令 %s 执行失败", token)
        return True


def arg_completer_map() -> Dict[str, Callable]:
    """token（含别名）→ 参数补全器 的映射（没有补全器的命令不进表）。"""
    out: Dict[str, Callable] = {}
    for token, cmd in _registry.items():
        if cmd.arg_completer is not None:
            out[token] = cmd.arg_completer
    return out


def all_tokens() -> List[str]:
    """全部可补全 token（主名+别名，去重）——补全器一级候选的数据源。"""
    return sorted(_registry.keys())


def help_renderable():
    """/help 的表格（按 category 分组，一行一命令：命令(别名)/用法/说明）。

    返回：rich Table（调用方负责 console.print）。注册表为空时返回提示文本。
    """
    from rich.table import Table
    from rich.text import Text

    cmds = all_commands()
    if not cmds:
        return Text("（注册表为空）", style="dim")
    table = Table(title="全部命令（/help）", show_lines=False)
    table.add_column("命令", style="cyan", no_wrap=True)
    table.add_column("用法", style="dim")
    table.add_column("说明")
    for cat in sorted({c.category or "其他" for c in cmds}):
        # rich 13.x 的 add_section() 不收参数（只画分隔线），类别名单独作一行标题
        table.add_section()
        table.add_row(Text(cat, style="bold green"))
        for c in sorted([x for x in cmds if (x.category or "其他") == cat],
                        key=lambda x: x.name):
            shown = c.name + (f" ({','.join(c.aliases)})" if c.aliases else "")
            table.add_row(shown, c.usage, c.summary)
    return table
