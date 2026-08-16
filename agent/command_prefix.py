"""审批前缀规则的保守派生（R25 #2）。

用户批准一条命令后，如果它以 curated 表里的"可泛化前缀"开头，
额外存一条前缀规则——下次同前缀的命令不再询问。

为什么是 curated 表而不是"首 token 泛化"：
批准 ``git push origin feature-x`` 不该自动放行 ``git push --force origin master``。
只有明确无破坏性的迭代类命令（跑测试/静态检查）才值得泛化，
破坏性命令保持 exact 匹配（宁可多问一次）。

对齐 CCB PermissionUpdate 的"审批持久化为规则"，但泛化面收窄到本表。
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 可泛化前缀表（词边界匹配；追加条目即生效）
_PREFIXABLE = (
    "uv run pytest",
    "uv run python -m pytest",
    "python -m pytest",
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

# 复合操作符出现 → 不泛化（后半段可能藏破坏性命令）
_COMPOUND_RE = re.compile(r"&&|\|\||;|\||`|\$\(")


def derive_approved_prefix(command: str) -> Optional[str]:
    """从已批准命令派生可泛化前缀；不可泛化返回 None。

    规则：
      - 空/含复合操作符 → None
      - 与表项完全相同（无多余 token）→ None（exact 白名单已覆盖，泛化无增益）
      - 以某表项的 token 序列开头且其后还有 token → 返回该表项
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
