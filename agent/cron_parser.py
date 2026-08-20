"""cron（定时任务——到点自动执行，像闹钟）表达式的解析器。

在项目里的位置：最底层的小工具，被 agent/cron.py 的调度器调用，
负责回答"现在这个时间点，这条 cron 表达式该不该触发"。

cron 表达式由 5 个字段组成，用空格隔开，顺序固定：
minute（分）hour（时）day_of_month（日）month（月）day_of_week（星期）。
每个字段支持这些写法：
* （任意值）| N（单个数字）| */N（每隔 N）| N-M（区间）| N,M,K（列表）| N-M/S（区间内每隔 S）
"""

from datetime import datetime
from typing import Set

FIELD_RANGES = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day_of_month": (1, 31),
    "month": (1, 12),
    "day_of_week": (0, 6),  # 星期字段取值 0-6：0=周日，6=周六（cron 的老传统）
}


def parse_field(expr: str, min_val: int, max_val: int) -> Set[int]:
    """把一个字段的写法翻译成一个数字集合。

    背景：判断"现在该不该触发"最简单的办法，就是把字段展开成所有
    允许的数字（比如 "1-5" 变成 {1,2,3,4,5}），再看当前时间在不在里面。
    支持的写法：'*'、'N'、'*/N'、'N-M'、'N,M,K'、'N-M/S'。

    参数：
    - expr：字段原文，如 "*/5" 或 "1,15"
    - min_val：该字段允许的最小值（如分钟是 0）
    - max_val：该字段允许的最大值（如分钟是 59）

    返回：所有允许值的数字集合。
    写法不合法或数字超出范围时抛 ValueError。
    """
    if not expr:
        raise ValueError("empty field")

    result: Set[int] = set()
    for part in expr.split(","):
        result.update(_parse_part(part.strip(), min_val, max_val))
    return result


def _parse_part(part: str, min_val: int, max_val: int) -> Set[int]:
    """解析逗号切开后的一段（如 "1-5/2"），同样返回数字集合。

    参数：
    - part：单段原文（不含逗号）
    - min_val / max_val：该字段的合法范围

    返回：这一段展开后的数字集合；写法不合法抛 ValueError。
    """
    # 先处理斜杠步进：N-M/S 或 */S，S 就是"每隔几个取一个"
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
        if step > 1:
            # 'N/S' 这种省略写法：从 N 一直数到字段最大值，每隔 S 取一个
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
    """检查一个数字是否在字段合法范围内，超了就抛 ValueError。

    参数：
    - val：要检查的数字
    - min_val / max_val：合法范围
    - raw：原始字段文本，只用于拼进报错信息方便定位
    """
    if val < min_val or val > max_val:
        raise ValueError(f"value {val} out of range [{min_val}, {max_val}] in '{raw}'")


def cron_match(cron_expr: str, dt: datetime) -> bool:
    """判断"dt 这个时间点"是否命中 cron 表达式（该不该触发）。

    背景：调度器每分钟都会拿当前时间来问一次，这里返回 True 就表示
    到点了、该触发任务了。

    参数：
    - cron_expr：5 字段 cron 表达式原文
    - dt：要检查的时间点

    返回：True=到点该触发；False=不到点。
    表达式格式错误（字段数不是 5、写法不合法等）抛 ValueError。
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

    # cron 的老规矩：日(day_of_month) 和星期(day_of_week) 同时都限定时，
    # 满足任意一个就算命中（OR，不是 AND）。
    # 判断"是否限定"要看解析后的集合是否覆盖了整个范围——
    # 不能只比较原文是不是 "*"，因为 "*/1" 这种写法效果等同于 "*"
    full_dom = set(range(FIELD_RANGES["day_of_month"][0], FIELD_RANGES["day_of_month"][1] + 1))
    full_dow = set(range(FIELD_RANGES["day_of_week"][0], FIELD_RANGES["day_of_week"][1] + 1))
    dom_is_star = dom_set == full_dom
    dow_is_star = dow_set == full_dow

    # 两套星期编号不一样，得换算：
    # Python 的 weekday() 周一=0 ... 周日=6；
    # cron 的 day_of_week 周日=0 ... 周六=6
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
        return True  # 日和星期都是通配，剩下的条件前面已查过，必然命中
