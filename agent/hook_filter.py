"""P3.5: 声明式 hook 的 if 条件过滤（permission rule 语法）。

对齐 claude-code-main 的 prepareIfConditionMatcher：
  - 语法: "ToolName(arg_pattern)"，例如 "terminal(git *)" / "read_file(*.py)"
  - 仅 tool 名（无括号）= 只匹配 tool 名，不限参数
  - 空条件 / None = 无条件匹配（永远跑）
  - 适用事件: PRE_TOOL_USE / POST_TOOL_USE / POST_TOOL_USE_FAILURE / PERMISSION_REQUEST
  - 不匹配 → 跳过该 hook（不 spawn 子进程，省资源）

匹配语义：
  - tool 名精确匹配（大小写敏感）
  - arg_pattern 用 fnmatch 对所有 args 值做通配匹配，任一命中即视为匹配
  - 语法错（括号不配对等）→ fail-open（返回 True，不阻塞 hook）

设计权衡（OmniMate 简化）：
  - claude-code-main 用 permission rule 解析 + 分字段匹配（tool 名 + arg 字段）
  - OmniMate 简化为：tool 名 + 对 args 所有值做 fnmatch（不区分字段名）
  - 这样对用户更直观："terminal(git *)" 命中任何值含 "git " 前缀的 args
"""
import fnmatch
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# 解析 "ToolName(pattern)" 的正则（容许 pattern 含空格/特殊字符）
_PATTERN_RE = re.compile(r"^(\w+)\((.*)\)$")


def match_if_condition(
    tool_name: str,
    args: dict,
    condition: Optional[str],
) -> bool:
    """判断当前 tool 调用是否匹配 if 条件。

    Args:
        tool_name: 当前调用的工具名（如 "terminal"）
        args: 工具入参 dict（如 {"command": "git status"}）
        condition: if 条件字符串（如 "terminal(git *)"），None/空 = 无条件匹配

    Returns:
        True 表示匹配（应执行 hook），False 表示不匹配（跳过 hook）

    Fail-open 策略：
        - condition 为空 → True
        - condition 语法非法 → True（不阻塞 hook，避免 hook 因配置错被静默跳过）
        - tool 名不匹配 → False
        - 有 pattern 但参数全不匹配 → False
    """
    if not condition or not condition.strip():
        return True  # 无条件 = 永远跑

    cond = condition.strip()

    # 尝试解析 "ToolName(pattern)" 格式
    m = _PATTERN_RE.match(cond)
    if m:
        cond_tool = m.group(1)
        arg_pattern = m.group(2)
        # tool 名不匹配 → 直接 False
        if cond_tool != tool_name:
            return False
        # pattern 为空 = 只匹配 tool 名
        if not arg_pattern:
            return True
        # 对 args 所有字符串值做 fnmatch
        return _match_any_arg(args, arg_pattern)

    # 无括号 = 纯 tool 名匹配
    if re.match(r"^\w+$", cond):
        return cond == tool_name

    # 语法非法 → fail-open（True）
    logger.warning(
        "if 条件 '%s' 语法无法解析（应为 'ToolName(pattern)' 或 'ToolName'），"
        "fail-open 返回 True", condition,
    )
    return True


def _match_any_arg(args: dict, pattern: str) -> bool:
    """对 args 所有字符串值做 fnmatch，任一命中即 True。

    非 str 值转 str 后匹配。嵌套 dict/list 不展开（只看顶层值）。
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


# 适用 if 条件过滤的事件集合
_IF_CONDITION_EVENTS = frozenset({
    "pre_tool_use",
    "post_tool_use",
    "post_tool_use_failure",
    "permission_request",
})


def is_if_condition_applicable(event_str: str) -> bool:
    """该事件是否支持 if 条件过滤。

    对齐 claude-code-main：仅 4 个工具相关事件支持。
    其他事件 if 条件被忽略（无条件匹配）。
    """
    return event_str in _IF_CONDITION_EVENTS
