"""ask_user 工具:向用户提问(单选/多选),复刻 业界 的 AskUserQuestion。

LLM 调用 → 前端弹窗(CLI 模式:终端 readline)→ 用户选 → handler 拿到答案返回。

用于 plan 模式确认方向 / 澄清需求 / 头脑风暴。
设计:
  - handler 生成 qid,push 到 _pending,block 等 _events[qid]
  - 桥接层(GUI:HTTP /api/ask/*;CLI:readline)调 submit_answer 唤醒 handler
超时 fail-open(5 分钟),避免永久卡死 agent。
"""
import json
import threading
import uuid
from typing import Dict, List, Optional

from tools.registry import registry


# 全局状态(跨 handler / 请求,单用户 spike)
_pending: Dict[str, dict] = {}
_answers: Dict[str, list] = {}
_events: Dict[str, threading.Event] = {}
_lock = threading.Lock()

ASK_TIMEOUT_SECONDS = 300  # 5 分钟


def reset_pending() -> None:
    """清空所有 pending(新会话/取消时,唤醒所有等待的 handler)。"""
    with _lock:
        qids = list(_pending.keys())
        _pending.clear()
        _answers.clear()
    for qid in qids:
        ev = _events.pop(qid, None)
        if ev:
            ev.set()


def get_pending() -> Optional[dict]:
    """拿第一个 pending question(前端轮询/CLI 查询用)。返回拷贝或 None。"""
    with _lock:
        if not _pending:
            return None
        qid = next(iter(_pending))
        return dict(_pending[qid])


def submit_answer(qid: str, answers: List[str]) -> bool:
    """桥接层提交答案 → 唤醒阻塞的 handler。返回是否找到该问题。"""
    with _lock:
        if qid not in _pending:
            return False
        _answers[qid] = answers
        ev = _events.get(qid)
    if ev:
        ev.set()
    return True


ASK_USER_SCHEMA = {
    "name": "ask_user",
    "toolset": "core",
    "description": (
        "向用户提问(单选/多选)。用于 plan 模式确认方向、澄清需求、头脑风暴方案选择。"
        "不要用于简单 yes/no(那种直接在回复里问即可)。"
        "问题要具体,options 要互斥且覆盖主要可能(2-4 个)。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "要问用户的问题(具体、清晰、不要技术黑话)",
            },
            "options": {
                "type": "array",
                "description": "选项列表(2-4 个,互斥,覆盖主要可能)",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "选项标签(1-5 字简短)"},
                        "description": {"type": "string", "description": "选项说明(为什么选这个/影响)"},
                    },
                    "required": ["label"],
                },
            },
            "multi": {
                "type": "boolean",
                "description": "是否多选(默认 false 单选)",
                "default": False,
            },
        },
        "required": ["question", "options"],
    },
}


def _handle_ask_user(args: dict, **kwargs) -> str:
    """提问并等用户答。

    优先用注入的桥接层（agent.ask_user_bridge：CLI readline / GUI HTTP）。
    无桥接层时立即报错 fail-fast——绝不阻塞 5 分钟
    （历史 bug：CLI 没接桥接层，ask_user 一调就干等 300s，界面"没反应"）。
    """
    question = (args.get("question") or "").strip()
    options = args.get("options") or []
    multi = bool(args.get("multi", False))

    if not question:
        return json.dumps({"error": "question 不能为空"}, ensure_ascii=False)
    if not options or len(options) < 2:
        return json.dumps({"error": "options 至少要 2 个"}, ensure_ascii=False)

    agent_ref = kwargs.get("agent_ref")
    bridge = (
        getattr(agent_ref, "ask_user_bridge", None)
        if agent_ref is not None else None
    )
    if callable(bridge):
        qdata = {
            "id": uuid.uuid4().hex[:12],
            "question": question,
            "options": options,
            "multi": multi,
        }
        try:
            answers = bridge(qdata) or []
        except (EOFError, KeyboardInterrupt):
            return json.dumps({
                "error": "用户中断提问",
                "error_type": "user_interrupt",
                "question": question,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "error": f"ask_user 桥接层异常: {e}",
                "error_type": "bridge_error",
                "question": question,
            }, ensure_ascii=False)
        return json.dumps({
            "question": question,
            "answers": answers,
            "multi": multi,
        }, ensure_ascii=False)

    # 无桥接层：fail-fast，不阻塞
    return json.dumps({
        "error": "ask_user 无可用桥接层（CLI/GUI 未注入 ask_user_bridge），无法向用户提问",
        "error_type": "no_bridge",
        "question": question,
    }, ensure_ascii=False)


# 模块级注册(import 时自动)
registry.register(
    name="ask_user",
    toolset="core",
    schema=ASK_USER_SCHEMA,
    handler=_handle_ask_user,
    emoji="❓",
)
