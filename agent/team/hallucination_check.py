"""幻觉检测：校验子代理返回里声称的 ID 是否真实存在。

子代理可能产生幻觉："已创建 task_001 / task_002 / task_003"，但实际只创建了 task_001。
Orchestrator 依赖 task_002/003 跑下一步会崩溃。

策略：
  - 从子代理返回文本提取声称的 task_id / 文件路径
  - 在真实 store / 文件系统里校验
  - 检测到幻觉时在文本后追加警告，由 Orchestrator 自行判断

不阻断流程，仅追加警告（fail-open，避免误报影响正常路径）。
"""
import re
from pathlib import Path
from typing import List, Tuple


def extract_claimed_ids(text: str) -> Tuple[List[str], List[str]]:
    """从子代理返回文本提取声称创建/修改的 ID。

    返回 (task_ids, file_paths)。
    - task_ids: 形如 task_xxx / task-xxx 的 ID（去重保序）
    - file_paths: 明显是文件路径的（含 .扩展名，非 URL）
    """
    # task_id 形态：task_ 或 task- 后跟 3+ 字符（字母数字下划线短横）
    task_ids_raw = re.findall(r"\btask[_-][a-zA-Z0-9_-]{3,}\b", text)
    # 文件路径：含 .扩展名（用 negative lookbehind 排除 URL 里的子串：
    # 前面不能是 : / 或字母数字，否则就是 URL 或单词中间）
    file_paths_raw = re.findall(
        r"(?<![:/\w])[A-Za-z0-9_\-./\\]+\.[A-Za-z]{1,5}\b", text,
    )

    # 去重保序
    def _dedup(seq):
        seen: set = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    task_ids = _dedup(task_ids_raw)
    # 文件路径过滤掉 URL 和明显非本地文件的
    file_paths = []
    for f in _dedup(file_paths_raw):
        if "://" in f:
            continue
        if len(f) < 3:
            continue
        # 排除常见的版本号（v1.2.3）等
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
    """校验声称的 ID 是否真实存在。

    参数：
        text: 子代理返回文本
        task_store: TaskStore 实例（None 时跳过 task 校验）
        fs_cwd: 工作目录（None 时跳过文件校验）

    返回 dict:
        claimed_tasks: 声称的 task id 列表
        claimed_files: 声称的文件路径列表
        missing_tasks: 不存在的 task id 列表
        missing_files: 不存在的文件路径列表
        hallucination_detected: 是否检测到幻觉（有缺失即 True）
    """
    claimed_tasks, claimed_files = extract_claimed_ids(text)
    missing_tasks: List[str] = []
    missing_files: List[str] = []

    # task 校验
    if task_store and claimed_tasks:
        try:
            existing = {t.get("id") for t in task_store.list_all()}
        except Exception:
            existing = set()
        missing_tasks = [t for t in claimed_tasks if t not in existing]

    # 文件校验
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
    """如果检测到幻觉，在子代理返回文本后追加警告。

    无幻觉时原样返回。
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
