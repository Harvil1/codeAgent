"""subagent_kill 工具簇：从 delegate_tool 拆出（纯搬迁）。

本模块收拢 kill 三件套，全部从 tools/delegate_tool.py 平移：

  - SUBAGENT_KILL_SCHEMA     工具 schema（LLM 看的说明书）
  - _handle_subagent_kill    kill 处理器（按取消信号让后台子代理优雅退出）
  - _subagent_kill_check_fn  可见性开关（config.delegation.async_kill_enabled）

搬迁铁律（委托链拆分约定）：
  - 行为零变化——三成员与 delegate_tool 原文逐字节一致；
  - 唯一改写：_async_tasks 花名册本体留在主文件（spawn 侧写、kill 侧读），
    处理器在函数体内 from tools.delegate_tool import _async_tasks 运行时
    回读——主文件加载完成早于任何工具调用，不构成循环 import；
  - 模块顶层不调 registry.register()——注册仍由主文件底部注册块完成
    （delegate_tool 模块级 import 三符号，注册块引用零改动）。
"""

import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# subagent_kill 工具：中断后台跑的子代理
# ---------------------------------------------------------------------------

SUBAGENT_KILL_SCHEMA = {
    "name": "subagent_kill",
    "description": (
        "中断后台子代理（让子代理优雅退出 + 保留已完成部分）。"
        "适用场景：async 子代理跑偏、用户 ESC 想停、任务已完成想提前 kill。"
        "\n\n**注意**：\n"
        "- 只能 kill async 子代理（subagent(background=True) 返回的 delegation_id）\n"
        "- sync 子代理由父代理超时机制管理，不需要显式 kill\n"
        "- kill 是协作式的：set cancel_event，子代理在下次 LLM 调用前检查退出"
        "（不会真杀线程）\n"
        "- kill 后子代理仍有 sync_cancel_timeout_seconds 秒响应窗口"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": (
                    "要中断的子代理 task_id（即 subagent(background=True) "
                    "返回的 delegation_id）"
                ),
            },
        },
        "required": ["task_id"],
    },
}


def _handle_subagent_kill(args: dict, **kwargs) -> str:
    """中断后台子代理。

    做法：查 _async_tasks 花名册找到目标，按下它的取消信号——子代理的
    对话主循环每轮开头都查这个信号，一发现被按下就退出，并返回
    _extract_partial_result() 保留已完成的部分。

    参数：
      - args：LLM 传的参数（task_id，即 subagent(background=True) 返回的 delegation_id）
      - **kwargs：运行时上下文（config 等）

    返回：JSON 字符串（已通知退出 / 各类错误）。查询或操作出错不影响主流程。
    """
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({
            "error": "task_id 不能为空",
            "error_type": "invalid_argument",
        }, ensure_ascii=False)

    # 查 config 开关（config.delegation.async_kill_enabled）
    _cfg = kwargs.get("config") or {}
    _delegation_cfg = (_cfg.get("delegation") or {}) if isinstance(_cfg, dict) else {}
    if not _delegation_cfg.get("async_kill_enabled", True):
        return json.dumps({
            "error": "subagent_kill 工具已被 config.delegation.async_kill_enabled=False 禁用",
            "error_type": "disabled",
        }, ensure_ascii=False)

    # 花名册本体留在主文件（spawn 侧写、kill 侧读）——这里运行时回读：
    # 主文件早已加载完，函数内 import 不构成循环
    from tools.delegate_tool import _async_tasks

    info = _async_tasks.get(task_id)
    if info is None:
        return json.dumps({
            "error": f"任务 {task_id} 不存在（可能已完成或 task_id 错误）",
            "error_type": "not_found",
        }, ensure_ascii=False)

    try:
        cancel_event = info.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()

        # 给子代理留出优雅退出的时间（不等 thread.join() 完整跑完，只是软通知）
        # 注意：这里不阻塞主流程，线程什么时候真正退出由子代理主循环自己检查决定
        thread = info.get("thread")
        if thread is not None and thread.is_alive():
            # 不等 join——kill 工具本身得快速返回
            # 子代理在自己的线程里继续跑到取消信号检查生效为止
            logger.info(
                "Task K: subagent_kill task_id=%s，cancel_event 已 set",
                task_id,
            )

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "status": "killed",
            "message": (
                "已通知子代理退出（cancel_event 已 set），"
                "子代理将在下次 LLM 调用前退出并返回 partial result"
            ),
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("subagent_kill 异常（fail-open）")
        return json.dumps({
            "error": f"kill 操作异常: {e}",
            "error_type": "internal_error",
            "task_id": task_id,
        }, ensure_ascii=False)


def _subagent_kill_check_fn() -> bool:
    """可见性开关：按 config.delegation.async_kill_enabled 决定 subagent_kill 对 LLM 显不显示。

    工具注册（登记）和暴露给 LLM（可见）是两步，这个 check_fn 就是那个
    动态开关——True（默认）→ 工具对 LLM 可见；False → 隐藏。
    注意：registry._check_fn_cached 调用 fn() 时不传任何参数，必须用
    无参签名（跟其他 check_fn 保持一致）。

    返回：bool，是否可见；读配置出错时返 True（fail-open）。
    """
    try:
        from agent.settings import load_settings
        cfg = load_settings() or {}
        delegation = cfg.get("delegation") or {}
        return bool(delegation.get("async_kill_enabled", True))
    except Exception:
        return True  # fail-open：读不到配置就默认可见
