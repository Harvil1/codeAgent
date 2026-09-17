"""声明式 hook 的 if 条件过滤——让 hook 只在"特定工具/特定参数"被调用时才跑。

比如配置 hook 时写 if: "terminal(git *)"，就只有 agent 执行 git 开头的 terminal 命令时
这个 hook 才触发，其他命令直接跳过（不用白起一个子进程，省资源）。

写法（形如权限规则）：
  - "ToolName(arg_pattern)"，例如 "terminal(git *)" / "read_file(*.py)"
  - 只写 tool 名不带括号 = 只要工具名对得上就匹配，不管参数
  - 条件为空 / None = 无条件，hook 永远跑
  - 只在这 4 个事件上有效：PRE_TOOL_USE / POST_TOOL_USE / POST_TOOL_USE_FAILURE / PERMISSION_REQUEST
  - 不匹配 → 跳过这个 hook（不起子进程，省资源）

匹配规则：
  - tool 名要一字不差（区分大小写）
  - arg_pattern 用 fnmatch（文件名通配符那套 * 和 ?）对所有参数值做匹配，任何一个值命中就算匹配
  - 条件语法写错（括号不配对等）→ fail-open（放行返回 True，不让配置笔误把 hook 静默废掉）

设计取舍：tool 名精确匹配 + 对 args 的所有值统一做 fnmatch（不区分是哪个字段），
对用户更直观——"terminal(git *)" 能命中任何值里带 "git " 前缀的参数。
"""
import fnmatch
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# 用来拆解 "ToolName(pattern)" 写法的正则（pattern 里允许空格和特殊字符）
_PATTERN_RE = re.compile(r"^(\w+)\((.*)\)$")


def match_if_condition(
    tool_name: str,
    args: dict,
    condition: Optional[str],
) -> bool:
    """判断当前这次工具调用是否命中 hook 的 if 条件。

    hook 执行前先过这道筛子，不命中就不必起子进程，省资源。

    参数：
        tool_name：当前调用的工具名（如 "terminal"）
        args：工具入参 dict（如 {"command": "git status"}）
        condition：if 条件字符串（如 "terminal(git *)"），None/空 = 无条件匹配

    返回：
        True = 命中（该跑 hook）；False = 不命中（跳过 hook）

    fail-open（宁可多跑不可漏跑）策略：
        - condition 为空 → True
        - condition 语法非法 → True（配置写错不让 hook 被静默跳过）
        - tool 名不匹配 → False
        - 有 pattern 但参数全不匹配 → False
    """
    if not condition or not condition.strip():
        return True  # 没写条件 = 永远跑

    cond = condition.strip()

    # 先按 "ToolName(pattern)" 格式拆
    m = _PATTERN_RE.match(cond)
    if m:
        cond_tool = m.group(1)
        arg_pattern = m.group(2)
        # 工具名对不上就直接不命中
        if cond_tool != tool_name:
            return False
        # pattern 为空 = 只限工具名，不看参数
        if not arg_pattern:
            return True
        # 拿 pattern 对所有参数值做通配匹配
        return _match_any_arg(args, arg_pattern)

    # 不带括号 = 纯工具名匹配
    if re.match(r"^\w+$", cond):
        return cond == tool_name

    # 到这说明条件语法不合法 → fail-open 放行
    logger.warning(
        "if 条件 '%s' 语法无法解析（应为 'ToolName(pattern)' 或 'ToolName'），"
        "fail-open 返回 True", condition,
    )
    return True


def _match_any_arg(args: dict, pattern: str) -> bool:
    """拿通配符 pattern 逐个比对 args 的所有值，任何一个命中就返回 True。

    参数：
        args：工具入参 dict
        pattern：fnmatch 通配符模式（如 "git *"）

    返回：
        bool——任一值命中为 True，全不命中为 False。

    说明：不是字符串的值先转成字符串再比；嵌套的 dict/list 不往里拆（只看第一层的值）。
    """
    if not isinstance(args, dict):
        return False
    for v in args.values():
        if v is None:
            continue
        s = v if isinstance(v, str) else str(v)
        if fnmatch.fnmatch(s, pattern):
            return True
    return False


# 允许挂 if 条件的 4 个工具相关事件（其他事件挂了也会被忽略）
_IF_CONDITION_EVENTS = frozenset({
    "pre_tool_use",
    "post_tool_use",
    "post_tool_use_failure",
    "permission_request",
})


def is_if_condition_applicable(event_str: str) -> bool:
    """判断某个事件支不支持挂 if 条件。

    只有 4 个工具相关事件（工具调用前后、失败后、权限请求时）才有
    "参数"可过滤，别的事件挂了也白挂。

    参数：
        event_str：事件名字符串（如 "pre_tool_use"）

    返回：
        bool——在这 4 个事件集合内为 True。其余事件的 if 条件会被当作无条件处理。
    """
    return event_str in _IF_CONDITION_EVENTS
