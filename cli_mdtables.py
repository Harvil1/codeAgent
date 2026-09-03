"""CJK/宽字符感知的 markdown 表格重排（从 hermes 的 markdown_tables 移植）。

大白话：模型排表格时假设「一个字符占一格」，但中文/emoji 在终端里占
两格——模型自己觉得对齐了的表，一到真终端每行就往右漂。这个模块按
「显示宽度」（不是字符数）重新补空格，让竖线在屏幕上真的对齐。

估宽用 unicodedata（东亚宽字符算 2 格、组合记号算 0 格），不引
wcwidth 依赖——精度对表格对齐足够。

保守策略：
- 只重写「| ... | 块 + 分隔行」都齐全的表格；
- 不像表格的行原样返回；
- 表太宽放不进可用宽度时，退成竖排「表头: 值」——终端软折行会把
  对齐彻底撕烂，竖排反而看得清。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

__all__ = [
    "is_table_divider",
    "looks_like_table_row",
    "realign_markdown_tables",
    "split_table_row",
]


_DIVIDER_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")
_MIN_COL_WIDTH = 3  # 与分隔行的最少横杠数对齐


def _disp_width(s: str) -> int:
    """字符串的显示宽度（东亚宽字符 2 格，组合记号 0 格）。"""
    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad_to_width(s: str, target: int) -> str:
    """补空格到目标显示宽度。"""
    return s + " " * max(0, target - _disp_width(s))


def split_table_row(row: str) -> List[str]:
    """把 ``| a | b | c |`` 拆成 ``["a", "b", "c"]``（两端竖线剥掉、格前后去空格）。"""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_table_divider(row: str) -> bool:
    """这行是不是表格分隔行（|---|---| 那种）。"""
    cells = split_table_row(row)
    return len(cells) > 1 and all(_DIVIDER_CELL_RE.match(c) for c in cells)


def looks_like_table_row(row: str) -> bool:
    """这行像不像表格行（流式调用方用来决定要不要先缓冲）。

    故意放宽——真正的重排只动带分隔行的块，这里误报最多让一行晚打
    一会儿，不致命。
    """
    if "|" not in row:
        return False
    stripped = row.strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return True
    return stripped.count("|") >= 2


def _render_block(rows: List[List[str]],
                  available_width: Optional[int] = None) -> List[str]:
    """按统一列宽渲染一个表格块（首行是表头，分隔行隐含）。

    给了 available_width 且横排放不下时，退成竖排。
    """
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    widths = [
        max(_MIN_COL_WIDTH, *(_disp_width(r[c]) for r in rows))
        for c in range(ncols)
    ]

    # 横排整行宽度：每列「| + 空格 + 格 + 空格」，最后多一根收尾竖线
    horizontal_width = sum(widths) + 3 * ncols + 1

    if available_width is not None and horizontal_width > max(available_width, 20):
        return _render_vertical(rows, ncols, available_width)

    def _row(cells: List[str]) -> str:
        return (
            "| "
            + " | ".join(_pad_to_width(c, widths[k]) for k, c in enumerate(cells))
            + " |"
        )

    out = [_row(rows[0])]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows[1:]:
        out.append(_row(r))
    return out


def _wrap_to_width(text: str, width: int) -> List[str]:
    """按词软折行到 width 显示格；单个超长词硬拆。空输入给 [""]。"""
    if width <= 0 or not text:
        return [text]

    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = ""
    current_w = 0

    def _hard_break(word: str, w: int) -> List[str]:
        out, buf, bw = [], "", 0
        for ch in word:
            cw = _disp_width(ch) or 1
            if bw + cw > w and buf:
                out.append(buf)
                buf, bw = ch, cw
            else:
                buf += ch
                bw += cw
        if buf:
            out.append(buf)
        return out

    for word in words:
        ww = _disp_width(word)
        if not current:
            if ww <= width:
                current, current_w = word, ww
            else:
                pieces = _hard_break(word, width)
                lines.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""
                current_w = _disp_width(current)
            continue
        if current_w + 1 + ww <= width:
            current += " " + word
            current_w += 1 + ww
        else:
            lines.append(current)
            if ww <= width:
                current, current_w = word, ww
            else:
                pieces = _hard_break(word, width)
                lines.extend(pieces[:-1])
                current = pieces[-1] if pieces else ""
                current_w = _disp_width(current)
    if current:
        lines.append(current)
    return lines or [""]


def _render_vertical(rows: List[List[str]], ncols: int,
                     available_width: int) -> List[str]:
    """放不下横排就竖排：每行数据变成一小块「表头: 值」。"""
    if not rows:
        return []

    headers = rows[0] + [""] * (ncols - len(rows[0]))
    body = rows[1:]
    labels = [h or f"第{i + 1}列" for i, h in enumerate(headers)]

    sep_width = max(20, min(40, available_width - 2)) if available_width else 30
    separator = "─" * sep_width
    indent = "  "
    indent_w = _disp_width(indent)

    out: List[str] = []
    for ri, row in enumerate(body):
        if ri > 0:
            out.append(separator)
        for ci in range(ncols):
            label = labels[ci]
            value = row[ci] if ci < len(row) else ""
            label_w = _disp_width(label)
            first_budget = max(10, available_width - label_w - 2)
            cont_budget = max(10, available_width - indent_w)
            if not value:
                out.append(f"{label}:")
                continue
            wrapped = _wrap_to_width(value, first_budget)
            out.append(f"{label}: {wrapped[0]}")
            if len(wrapped) > 1:
                cont_text = " ".join(wrapped[1:])
                for cl in _wrap_to_width(cont_text, cont_budget):
                    if cl.strip():
                        out.append(f"{indent}{cl}")
    return out


def realign_markdown_tables(text: str,
                            available_width: Optional[int] = None) -> str:
    """重排文本里所有「| ... | + 分隔行」块，其余行原样返回。

    参数：
        text: 任意模型文本（可安全作用于普通散文）
        available_width: 可用显示宽度；给了且表格超宽就退竖排
    """
    if "|" not in text:
        return text

    lines = text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # 表格 = 表头行 + 紧随的分隔行
        if "|" in line and i + 1 < n and is_table_divider(lines[i + 1]):
            header = split_table_row(line)
            body: List[List[str]] = []
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                if is_table_divider(lines[j]):
                    j += 1
                    continue
                body.append(split_table_row(lines[j]))
                j += 1
            if any(c for c in header) or body:
                out.extend(_render_block([header] + body, available_width))
                i = j
                continue
        out.append(line)
        i += 1

    return "\n".join(out)
