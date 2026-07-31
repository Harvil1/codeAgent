"""Worker 任务归属强制（Task Binding）。

防止 spawned worker 被 prompt 注入后跨任务操作。

工作方式：
  Coordinator.spawn(task_id=X) 时把 OMNIMATE_KANBAN_TASK=X 注入子进程 env。
  task_tools 的写工具（task_update / task_complete）调用前用
  assert_owned(task_id) 校验：
    - env 未设（主 agent / legacy 调用）：直接 pass
    - env 与传入 task_id 匹配：pass
    - 不匹配：抛 TaskOwnershipError

注意：本模块是「身份来源」的唯一处。下一批 Kanban heartbeat 桥接
会复用 get_bound_task_id()。
"""
import os
from typing import Optional


ENV_VAR = "OMNIMATE_KANBAN_TASK"


class TaskOwnershipError(PermissionError):
    """Worker 试图操作未绑定的 task。"""


def get_bound_task_id() -> Optional[str]:
    """返回当前进程绑定的 task_id；未绑定（env 未设或空）返回 None。"""
    return os.environ.get(ENV_VAR) or None


def assert_owned(task_id: str) -> None:
    """断言当前进程有权操作 task_id。

    - 未绑定（主 agent / legacy 调用）：直接 pass
    - 绑定且匹配：pass
    - 绑定但不匹配：raise TaskOwnershipError
    """
    bound = get_bound_task_id()
    if bound is not None and bound != task_id:
        raise TaskOwnershipError(
            f"worker bound to task {bound!r}, cannot operate on {task_id!r}"
        )
