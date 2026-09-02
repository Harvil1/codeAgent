"""工人的「任务门禁」：保证一个工人只能动自己名下的任务。

打个比方：每个工人上岗时领一张工牌（环境变量 CODEAGENT_KANBAN_TASK，
写着「你是 task_001 的负责人」）。动任务前要刷工牌核对身份。

为什么需要：工人（spawned worker）是被 spawn 出来的独立进程，它的
任务文本可能被 prompt 注入（instructions 里夹带恶意指令），不设防的话
它可能去改别人任务的状态。

规则（task_update / task_complete 等写操作前用 assert_owned 核对）：
  - 没戴工牌（主 agent 或老式调用，env 未设）：放行
  - 工牌和要动的任务一致：放行
  - 工牌和要动的任务不一致：抛 TaskOwnershipError 拒绝

设计取舍：这个模块是「我是谁」的唯一权威来源（读 env 只在这一处），
后面的心跳桥接（auto_heartbeat）也复用 get_bound_task_id()，不另开一路。
"""
import os
from typing import Optional


ENV_VAR = "CODEAGENT_KANBAN_TASK"


class TaskOwnershipError(PermissionError):
    """工人试图操作不属于自己工牌的任务时抛的权限错误。"""


def get_bound_task_id() -> Optional[str]:
    """读工牌：返回当前进程绑定的 task_id。

    为什么存在：spawn 时通过环境变量把「你负责哪个任务」塞给子进程，
    这里是唯一的读取口。

    返回：绑定的 task_id 字符串；没戴工牌（env 未设或为空）返回 None。
    """
    return os.environ.get(ENV_VAR) or None


def assert_owned(task_id: str) -> None:
    """刷工牌：核对当前进程有权操作这个任务，没权就抛错。

    用在哪：task_tools 的写操作（改状态、完成任务）动手前先调这个。

    参数：
        task_id：想操作的任务 ID

    结果：
        - 没戴工牌（主 agent / 老式调用）：放行
        - 工牌与 task_id 一致：放行
        - 不一致：抛 TaskOwnershipError

    无返回值。
    """
    bound = get_bound_task_id()
    if bound is not None and bound != task_id:
        raise TaskOwnershipError(
            f"worker bound to task {bound!r}, cannot operate on {task_id!r}"
        )
