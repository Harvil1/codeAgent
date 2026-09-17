"""审批前缀规则的保守派生。

大白话：用户每批准一条命令，系统会记一个"下次不再问"的白名单。如果只是
逐字记录（exact 匹配），用户每次跑 `uv run pytest tests/test_a.py`、
`uv run pytest tests/test_b.py` 都要重新点一遍批准，很烦。这个模块做的事是：
批准的命令如果以一张人工精选（curated）表里的"可泛化前缀"开头（比如
`uv run pytest`），就额外存一条前缀规则——下次任何以这个前缀开头的命令
都自动放行，不再打扰用户。

为什么不干脆"取命令第一个词来泛化"？因为太危险：用户批准了
``git push origin feature-x``，不代表该放行 ``git push --force origin master``。
所以只有明确无破坏性的迭代类命令（跑测试、静态检查）进了这张表；
破坏性命令永远保持逐字匹配（exact）——宁可多问一次，不可错放一次。

泛化范围收窄到这张人工维护的表，不是任意泛化。
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 可泛化前缀表（按词边界匹配；往表里追加条目立刻生效）
_PREFIXABLE = (
    "uv run pytest",
    "uv run python -m pytest",
    "python -m pytest",
    "python -m unittest",
    "pytest",
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "cargo test",
    "go test",
    "ruff check",
    "mypy",
)

# 命令里出现这些复合操作符/重定向就不泛化：后半段可能藏着破坏性命令，
# 重定向（> >>）还会写文件——前缀只覆盖开头，盖不住后面。
# 换行/\r 也要算（`pytest tests\nrm -rf build`
# 靠换行拼成的复合命令），还有 ${（参数展开）和 <（进程替换）、
# 单个 &（后台执行，`cmd & rm xxx` 借前半段免审）——
# 漏掉任何一个，都等于让前缀免审放行了整条复合命令。
_COMPOUND_RE = re.compile(r"&&|\|\||;|\||&|`|\$\(|\$\{|<\(|\n|\r|>|>>")


def derive_approved_prefix(command: str) -> Optional[str]:
    """从一条已批准的命令里提取可泛化的前缀；不能泛化就返回 None。

    干什么：看这条命令是否以 curated 表里某个前缀开头且后面还带了参数，
    是就返回那个前缀（供存成前缀规则）。

    为什么需要：跑测试这类高频命令每次都弹审批太烦，详见模块头说明。

    参数：
        command: 用户刚批准的完整命令字符串。

    返回：可泛化的前缀（如 "uv run pytest"）；以下情况返回 None——
      - 命令为空或含复合操作符/重定向（后半段可能藏危险动作）；
      - 命令和某个表项一模一样（没有多余参数）——逐字白名单已经覆盖它，
        再泛化没有额外好处。
    """
    cmd = (command or "").strip()
    if not cmd or _COMPOUND_RE.search(cmd):
        return None
    tokens = cmd.split()
    for entry in _PREFIXABLE:
        entry_tokens = entry.split()
        if len(tokens) <= len(entry_tokens):
            continue
        if tokens[: len(entry_tokens)] == entry_tokens:
            return entry
    return None


def is_prefix_match_safe(command: str) -> bool:
    """前缀规则匹配时的安全护栏。

    干什么：检查一条命令适不适合走"前缀免审"。

    为什么需要：派生入库那一刻能保证命令没有复合形态，但以后来的命令
    是新输入——比如 ``uv run pytest && rm xxx``、``pytest > ~/.bashrc``，
    前半段虽然匹配前缀，后半段却藏着危险动作。这种情况不能让整条命令
    免审批，必须回落到逐字匹配/正常权限闸门。

    参数：
        command: 待检查的命令字符串。

    返回：True 表示可以安全地走前缀匹配（命令里没有复合操作符/重定向）；
    False 表示不行，得走正常审批。
    """
    return not _COMPOUND_RE.search(command or "")
