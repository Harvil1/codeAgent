"""5-field cron 表达式解析器。

字段顺序：minute hour day_of_month month day_of_week
每字段语法：* | N | */N | N-M | N,M,K | N-M/S
"""

import logging
from datetime import datetime
from typing import Set

logger = logging.getLogger(__name__)

FIELD_RANGES = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day_of_month": (1, 31),
    "month": (1, 12),
    "day_of_week": (0, 6),  # 0=Sunday, 6=Saturday
}


def parse_field(expr: str, min_val: int, max_val: int) -> Set[int]:
    """解析单字段为允许值的集合。

    支持：'*'、'N'、'*/N'、'N-M'、'N,M,K'、'N-M/S'
    Raises ValueError on invalid syntax or out-of-range values.
    """
    if not expr:
        raise ValueError("empty field")

    result: Set[int] = set()
    for part in expr.split(","):
        result.update(_parse_part(part.strip(), min_val, max_val))
    return result


def _parse_part(part: str, min_val: int, max_val: int) -> Set[int]:
    """解析逗号分隔的单段。"""
    # 处理 step：N-M/S 或 */S
    step = 1
    if "/" in part:
        range_part, step_str = part.split("/", 1)
        try:
            step = int(step_str)
        except ValueError:
            raise ValueError(f"invalid step '{step_str}' in part '{part}'")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        part = range_part

    if part == "*":
        start, end = min_val, max_val
    elif "-" in part:
        bounds = part.split("-", 1)
        try:
            start = int(bounds[0])
            end = int(bounds[1])
        except ValueError:
            raise ValueError(f"invalid range '{part}'")
    else:
        try:
            v = int(part)
        except ValueError:
            raise ValueError(f"invalid value '{part}'")
        if "/" in part or step > 1:
            # '5/N' 形式：从 5 到 max_val，步进 N
            start, end = v, max_val
        else:
            _check_range(v, min_val, max_val, part)
            return {v}

    _check_range(start, min_val, max_val, part)
    _check_range(end, min_val, max_val, part)
    if start > end:
        raise ValueError(f"range start > end in '{part}'")

    return set(range(start, end + 1, step))


def _check_range(val: int, min_val: int, max_val: int, raw: str):
    if val < min_val or val > max_val:
        raise ValueError(f"value {val} out of range [{min_val}, {max_val}] in '{raw}'")


def cron_match(cron_expr: str, dt: datetime) -> bool:
    """检查 dt 是否匹配 cron 表达式。

    Raises ValueError 如果表达式格式错误。
    """
    fields = cron_expr.split()
    if len(fields) != 5:
        raise ValueError(
            f"cron expression must have 5 fields, got {len(fields)}: '{cron_expr}'"
        )

    minute_set = parse_field(fields[0], *FIELD_RANGES["minute"])
    hour_set = parse_field(fields[1], *FIELD_RANGES["hour"])
    dom_set = parse_field(fields[2], *FIELD_RANGES["day_of_month"])
    month_set = parse_field(fields[3], *FIELD_RANGES["month"])
    dow_set = parse_field(fields[4], *FIELD_RANGES["day_of_week"])

    if dt.minute not in minute_set:
        return False
    if dt.hour not in hour_set:
        return False
    if dt.month not in month_set:
        return False

    # day_of_month 和 day_of_week 的特殊 OR 语义：
    # 当两者都不是通配（即解析后集合不等于完整范围）时，匹配任一即触发
    # 注意：不能仅做字符串 fields[i] == "*" 比较，因为 */1 等也等同于通配
    full_dom = set(range(FIELD_RANGES["day_of_month"][0], FIELD_RANGES["day_of_month"][1] + 1))
    full_dow = set(range(FIELD_RANGES["day_of_week"][0], FIELD_RANGES["day_of_week"][1] + 1))
    dom_is_star = dom_set == full_dom
    dow_is_star = dow_set == full_dow

    # Python weekday(): Monday=0 ... Sunday=6
    # cron day_of_week: Sunday=0 ... Saturday=6
    cron_dow = (dt.weekday() + 1) % 7

    dom_match = dt.day in dom_set
    dow_match = cron_dow in dow_set

    if not dom_is_star and not dow_is_star:
        return dom_match or dow_match
    elif not dom_is_star:
        return dom_match
    elif not dow_is_star:
        return dow_match
    else:
        return True  # 都是 '*'，必然匹配
