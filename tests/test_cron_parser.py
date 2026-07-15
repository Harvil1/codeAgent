"""cron 表达式解析器测试。"""
from datetime import datetime

import pytest

from agent.cron_parser import parse_field, cron_match


# ---------------------------------------------------------------------------
# parse_field
# ---------------------------------------------------------------------------

def test_parse_star_returns_full_range():
    assert parse_field("*", 0, 5) == {0, 1, 2, 3, 4, 5}
    assert parse_field("*", 1, 3) == {1, 2, 3}


def test_parse_single_int():
    assert parse_field("5", 0, 59) == {5}


def test_parse_step():
    assert parse_field("*/15", 0, 59) == {0, 15, 30, 45}


def test_parse_range():
    assert parse_field("9-17", 0, 23) == {9, 10, 11, 12, 13, 14, 15, 16, 17}


def test_parse_list():
    assert parse_field("0,15,30,45", 0, 59) == {0, 15, 30, 45}


def test_parse_combined_range_list():
    assert parse_field("1-3,10-12", 0, 23) == {1, 2, 3, 10, 11, 12}


def test_parse_step_with_offset():
    """'2-10/2' = 从 2 到 10 步进 2。"""
    assert parse_field("2-10/2", 0, 59) == {2, 4, 6, 8, 10}


def test_parse_invalid_int_raises():
    with pytest.raises(ValueError):
        parse_field("abc", 0, 59)


def test_parse_out_of_range_raises():
    with pytest.raises(ValueError):
        parse_field("60", 0, 59)


# ---------------------------------------------------------------------------
# cron_match
# ---------------------------------------------------------------------------

def test_cron_match_every_minute():
    """'* * * * *' 匹配任意时间。"""
    assert cron_match("* * * * *", datetime(2026, 7, 12, 15, 30)) is True


def test_cron_match_specific_minute():
    """'30 * * * *' 只匹配 minute=30。"""
    assert cron_match("30 * * * *", datetime(2026, 7, 12, 15, 30)) is True
    assert cron_match("30 * * * *", datetime(2026, 7, 12, 15, 31)) is False


def test_cron_match_step_minute():
    """'*/15 * * * *' 匹配 0/15/30/45 分。"""
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 0)) is True
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 15)) is True
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 45)) is True
    assert cron_match("*/15 * * * *", datetime(2026, 7, 12, 15, 7)) is False


def test_cron_match_daily_at_9am():
    """'0 9 * * *' 匹配每天 9:00。"""
    assert cron_match("0 9 * * *", datetime(2026, 7, 12, 9, 0)) is True
    assert cron_match("0 9 * * *", datetime(2026, 7, 12, 10, 0)) is False
    assert cron_match("0 9 * * *", datetime(2026, 7, 12, 9, 30)) is False


def test_cron_match_dom_or_dow():
    """当 day_of_month 和 day_of_week 都不是 '*'，匹配任一即触发（OR 语义）。"""
    # 2026-07-12 是周日（weekday 6 in Python is Sunday=6 → 我们的 day_of_week 0=Sunday）
    # 在 cron_expr "0 0 1 * 0" 中：
    #   day_of_month=1 (每月 1 号)
    #   day_of_week=0 (每周日)
    # 2026-07-12 是周日 → day_of_week 匹配 → 整体匹配（即使 dom != 1）
    assert cron_match("0 0 1 * 0", datetime(2026, 7, 12, 0, 0)) is True
    # 2026-07-01 是周三，但是 dom=1 → 匹配
    assert cron_match("0 0 1 * 0", datetime(2026, 7, 1, 0, 0)) is True
    # 2026-07-15 是周三，dom != 1 且 dow != 0 → 不匹配
    assert cron_match("0 0 1 * 0", datetime(2026, 7, 15, 0, 0)) is False


def test_cron_match_dom_and_dow_when_one_is_star():
    """当 day_of_month='*' 而 day_of_week='0'，AND 语义（其他字段全 AND）。"""
    # 2026-07-12 是周日 → dow=0 匹配 → 整体匹配
    assert cron_match("0 0 * * 0", datetime(2026, 7, 12, 0, 0)) is True
    # 2026-07-13 是周一 → dow != 0 → 不匹配
    assert cron_match("0 0 * * 0", datetime(2026, 7, 13, 0, 0)) is False


def test_cron_match_too_few_fields_raises():
    with pytest.raises(ValueError):
        cron_match("* * * *", datetime.now())


def test_cron_match_invalid_expr_raises():
    with pytest.raises(ValueError):
        cron_match("60 * * * *", datetime.now())  # 60 越界
