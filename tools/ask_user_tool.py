"""「问用户」工具：让 AI 出选择题（单选/多选）等真人回答（复刻业界 AskUserQuestion）。

背景：模型遇到"方向二选一""需求没说清"时，与其猜不如直接问人。
整条链路是：模型调本工具 → 界面侧弹问题（终端模式下用命令行问答）→
用户选完 → 界面侧把答案交回来 → 本工具把答案作为结果返回给模型。
典型用在计划确认、澄清需求、头脑风暴选方案。

实现思路（一个"留字条 + 等回信"的小邮箱）：
  - 工具登记一个问题（qid 编号），塞进待答列表 _pending，
    然后原地等 _events[qid] 这盏"信号灯"亮
  - 界面侧（GUI 走 HTTP /api/ask/*；CLI 走命令行输入）收到用户答案后
    调 submit_answer 点亮信号灯，等待中的工具就被唤醒继续跑
本文件属于工具层（tools/），被 tools/registry.py 自动发现注册。
"""
import json
import threading
import uuid
from typing import Dict, List, Optional

from tools.registry import registry


# 三个全局小账本：待答问题、已收答案、"答案到了"信号灯。
# 因为工具 handler 和界面回调是两条线，得靠模块级共享才能传话（单用户产品，够用）
_pending: Dict[str, dict] = {}
_answers: Dict[str, list] = {}
_events: Dict[str, threading.Event] = {}
_lock = threading.Lock()

ASK_TIMEOUT_SECONDS = 300  # 最多等用户 5 分钟，防止永远卡住


def reset_pending() -> None:
    """把所有没答完的问题作废，并唤醒所有还在干等的提问。

    背景：开新会话或用户取消时，留着旧问题只会让等待中的调用
    白等超时，所以统一叫醒它们。
    """
    with _lock:
        qids = list(_pending.keys())
        _pending.clear()
        _answers.clear()
    for qid in qids:
        ev = _events.pop(qid, None)
        if ev:
            ev.set()


def get_pending() -> Optional[dict]:
    """取走第一个还没答的问题（界面侧轮询/查询用）。

    返回：问题的拷贝（防止外面改坏内部账本），没有待答问题时返回 None。
    """
    with _lock:
        if not _pending:
            return None
        qid = next(iter(_pending))
        return dict(_pending[qid])


def submit_answer(qid: str, answers: List[str]) -> bool:
    """界面侧把用户的答案存进账本，并叫醒还在等的那个提问。

    背景：这是整条问答链路的"回信"入口——GUI 或 CLI 收到用户选择后调它。

    参数：
    - qid：问题编号（提问时生成的那个 id）
    - answers：用户选中的选项 label 列表

    返回：是否找到了这个问题（False = 问题已被作废或不存在）。
    """
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
    """把问题转交给界面（桥接层）等用户作答，返回答案 JSON。

    背景：优先用注入的桥接层（agent.ask_user_bridge：CLI 是命令行问答，
    GUI 是 HTTP 接口）。没配桥接层就立刻报错返回，绝不傻等 5 分钟。

    参数：
    - args：工具参数字典。question 是问题文本；options 是选项列表
      （每项含 label 简短标签和可选 description 说明）；multi 控制单选/多选。
    - kwargs：运行时注入的命名上下文，本函数只用到 agent_ref
      （AIAgent 主实例，从它身上找 ask_user_bridge 桥接层）。

    返回：JSON 字符串，含用户选的 answers；用户中断/无桥接层/参数不合法
    时返回对应 error。

    注意：没接桥接层的话，这个工具一被调用就干等 300 秒，
    界面看起来像死机。所以没桥接层就秒回错误（fail-fast）。
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

    # 没有桥接层：立即报错（fail-fast），不能让调用白等 5 分钟
    return json.dumps({
        "error": "ask_user 无可用桥接层（CLI/GUI 未注入 ask_user_bridge），无法向用户提问",
        "error_type": "no_bridge",
        "question": question,
    }, ensure_ascii=False)


# import 本模块时顺手把工具登记进中央注册表（项目惯例：工具文件顶层自注册）
registry.register(
    name="ask_user",
    toolset="core",
    schema=ASK_USER_SCHEMA,
    handler=_handle_ask_user,
    emoji="❓",
    isConcurrencySafe=False,  # 会卡住等真人输入，两个同时问会在屏幕上串行错乱，只能排队
)
