"""幻觉检测：核对子代理「自称干了什么」和「实际干了什么」是否对得上。

背景：子代理（LLM）会一本正经地说瞎话——比如回复「已创建 task_001 /
task_002 / task_003」，实际上只创建了 task_001。如果编排者（Orchestrator）
信了这句话、拿着 task_002 去跑下一步，就会当场崩溃。

做法（三步）：
  1. 从子代理返回文本里把声称的 task_id / 文件路径抠出来
  2. 去真实的任务库 / 文件系统里查证
  3. 查到「说有其实没有」就在文本后面追加警告，让编排者自己判断

设计取舍：只追加警告、不拦截流程（fail-open）——宁可放过也不误伤，
因为检测本身可能误报，拦了正常路径损失更大。
"""
import re
from pathlib import Path
from typing import List, Tuple


def extract_claimed_ids(text: str) -> Tuple[List[str], List[str]]:
    """从一段文本里抠出「声称的」任务 ID 和文件路径。

    参数：
        text：子代理的返回文本

    返回：(task_ids, file_paths) 两个列表
    - task_ids：形如 task_xxx / task-xxx 的 ID（去重、保持出现顺序）
    - file_paths：长得像文件路径的（带 .扩展名，排除了 URL）
    """
    # 任务 ID 长相：task_ 或 task- 后面至少跟 3 个字母/数字/下划线/短横
    task_ids_raw = re.findall(r"\btask[_-][a-zA-Z0-9_-]{3,}\b", text)
    # 文件路径长相：带 .扩展名；用「前一个字符不能是 : / 或字母数字」
    # 排除 URL 片段和单词中段，避免把网址、变量名误当路径
    file_paths_raw = re.findall(
        r"(?<![:/\w])[A-Za-z0-9_\-./\\]+\.[A-Za-z]{1,5}\b", text,
    )

    # 去重但保序（第一次出现的位置为准，方便人工对照原文）
    def _dedup(seq):
        seen: set = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    task_ids = _dedup(task_ids_raw)
    # 再过滤一遍：去掉 URL 和明显不是本地文件的杂项
    file_paths = []
    for f in _dedup(file_paths_raw):
        if "://" in f:
            continue
        if len(f) < 3:
            continue
        # 排除版本号（v1.2.3 这种）——长得像路径但不是
        if re.match(r"^v?\d+(\.\d+)+$", f):
            continue
        file_paths.append(f)
    return task_ids, file_paths


def verify_claims(
    text: str,
    *,
    task_store=None,
    fs_cwd: Path = None,
) -> dict:
    """拿着声称清单去现实里对账，看哪些是空头支票。

    参数：
        text：子代理的返回文本（声称来源）
        task_store：TaskStore 任务库实例；传 None 就跳过任务校验
        fs_cwd：工作目录，文件路径以它为基准查；传 None 跳过文件校验

    返回 dict，各键含义：
        claimed_tasks：文本里声称的任务 ID 列表
        claimed_files：文本里声称的文件路径列表
        missing_tasks：实际不存在的任务 ID（幻觉实锤）
        missing_files：实际不存在的文件路径（幻觉实锤）
        hallucination_detected：是否检测到幻觉（只要有一项缺失就是 True）
    """
    claimed_tasks, claimed_files = extract_claimed_ids(text)
    missing_tasks: List[str] = []
    missing_files: List[str] = []

    # 任务对账：拿任务库里全部 ID 做比对；库读不出来就当没查（不算缺失）
    if task_store and claimed_tasks:
        try:
            existing = {t.get("id") for t in task_store.list_all()}
        except Exception:
            existing = set()
        missing_tasks = [t for t in claimed_tasks if t not in existing]

    # 文件对账：挨个看文件在不在；路径非法（如夹着非法字符）直接跳过不算缺失
    if fs_cwd and claimed_files:
        cwd = Path(fs_cwd)
        for f in claimed_files:
            try:
                if not (cwd / f).exists():
                    missing_files.append(f)
            except (OSError, ValueError):
                continue

    return {
        "claimed_tasks": claimed_tasks,
        "claimed_files": claimed_files,
        "missing_tasks": missing_tasks,
        "missing_files": missing_files,
        "hallucination_detected": bool(missing_tasks or missing_files),
    }


def append_warning(child_text: str, verification: dict) -> str:
    """检测到幻觉时，在子代理文本后面贴一张警告条。

    参数：
        child_text：子代理的原始返回文本
        verification：verify_claims 的返回结果 dict

    返回：拼好警告的新文本；没有幻觉就原样返回，一个字不动。
    """
    if not verification.get("hallucination_detected"):
        return child_text

    missing_t = verification.get("missing_tasks", [])
    missing_f = verification.get("missing_files", [])
    parts = [
        "",
        "",
        "⚠️ 幻觉检测：以下 ID 在子代理返回中声称已创建/修改，但实际不存在：",
    ]
    if missing_t:
        parts.append(f"  任务: {missing_t}")
    if missing_f:
        parts.append(f"  文件: {missing_f}")
    parts.append("Orchestrator 请勿依赖这些 ID，需要时请重新创建或确认。")
    return child_text + "\n".join(parts)
