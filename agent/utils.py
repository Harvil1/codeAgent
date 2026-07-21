"""通用工具函数(跨模块共享)。

只放真正跨多个模块复用的小工具,不放业务逻辑。
"""
from datetime import datetime, timezone
from typing import Callable, Optional


def parse_iso(
    value,
    *,
    on_failure=None,
    failure_factory: Optional[Callable[[], object]] = None,
):
    """解析 ISO8601 时间戳(容错:处理带/不带 Z、毫秒、naive datetime)。

    参数:
        value: 输入值(字符串/datetime/None)。空值直接走失败分支。
        on_failure: 失败时返回的固定值(向后兼容用法)。
        failure_factory: 失败时调用的工厂函数,返回动态默认值
                         (如 ``lambda: datetime.now(timezone.utc)``)。
                         优先级高于 on_failure。

    返回:
        timezone-aware datetime,或失败值。

    用法:
        # 失败返 None
        parse_iso(value, on_failure=None)

        # 失败返当前时间(handoff 用法)
        parse_iso(s, failure_factory=lambda: datetime.now(timezone.utc))
    """
    if not value:
        return _resolve_failure(on_failure, failure_factory)
    try:
        parsed = datetime.fromisoformat(str(value).rstrip("Z"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError):
        return _resolve_failure(on_failure, failure_factory)


def _resolve_failure(on_failure, failure_factory):
    """统一处理失败值返回(failure_factory 优先)。"""
    if failure_factory is not None:
        return failure_factory()
    return on_failure
