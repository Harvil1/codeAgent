"""跨模块共享的杂项小工具。

定位：只放真正被多个模块反复用到的小函数，不放业务逻辑
（业务逻辑各有各的模块，别往这儿塞）。
"""
from datetime import datetime, timezone
from typing import Callable, Optional


def parse_iso(
    value,
    *,
    on_failure=None,
    failure_factory: Optional[Callable[[], object]] = None,
):
    """解析 ISO8601 格式的时间字符串，转成带时区的 datetime。

    项目里到处存时间戳（任务、handoff 包、轨迹……），来源五花八门——
    有的带 Z 后缀、有的带毫秒、有的没时区。这个函数统一兜住这些差异，
    解析失败也能按调用方给的兜底值返回，不抛异常打断主流程。

    参数：
        value：输入值，可以是字符串 / datetime / None。
            空值（None/空串）直接走失败分支。
        on_failure：解析失败时返回的固定值（老用法，向后兼容）。
        failure_factory：解析失败时调用的工厂函数，用它的返回值
            当结果（适合要"动态默认值"的场合，比如
            ``lambda: datetime.now(timezone.utc)``——失败就当"现在"）。
            优先级高于 on_failure，两个都传时听它的。

    返回：
        带时区的 datetime；解析失败时返回 on_failure 或
        failure_factory() 的结果。

    用法示例：
        # 失败返 None
        parse_iso(value, on_failure=None)

        # 失败返当前时间（handoff 移交包的用法：时间戳坏了就当刚生成）
        parse_iso(s, failure_factory=lambda: datetime.now(timezone.utc))
    """
    if not value:
        return _resolve_failure(on_failure, failure_factory)
    try:
        parsed = datetime.fromisoformat(str(value).rstrip("Z"))
        if parsed.tzinfo is None:
            # 没带时区的时间统一当作 UTC（naive datetime 直接比较会报错）
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return _resolve_failure(on_failure, failure_factory)


def _resolve_failure(on_failure, failure_factory):
    """统一决定解析失败时返回什么（failure_factory 优先）。

    参数：
        on_failure：固定兜底值。
        failure_factory：动态兜底值工厂；给了它就调它取结果。

    返回：
        failure_factory 不为 None 时返回 failure_factory()，
        否则返回 on_failure。
    """
    if failure_factory is not None:
        return failure_factory()
    return on_failure
