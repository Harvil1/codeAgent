"""「问用户」工具：让 AI 出选择题（单选/多选）等真人回答。

模型遇到"方向二选一""需求没说清"时，与其猜不如直接问人。
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

    开新会话或用户取消时调用，免得等待中的调用白等超时。
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

    整条问答链路的"回信"入口——GUI 或 CLI 收到用户选择后调它。

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
        "向用户提问(单选/多选/自填)。相关的几个问题(1-4个)应一次调用问完,"
        "用户逐题作答后你拿到全部答案汇总。用于 plan 模式确认方向、澄清需求、"
        "头脑风暴方案选择。不要用于简单 yes/no(那种直接在回复里问即可)。"
        "每个问题要具体,options 要互斥且覆盖主要可能(2-4 个)。"
        "用户始终可以自填选项(Type something)或选择转对话(Chat about this)。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "description": "问题列表(1-4 个,相关的几个问题一次问完)",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "问题文本(具体、清晰、不要技术黑话)",
                        },
                        "header": {
                            "type": "string",
                            "description": (
                                "问题的短标题(最多12字,界面上当标签显示,"
                                "如'FAQ 位置');不给则界面自动截问题前12字"
                            ),
                        },
                        "options": {
                            "type": "array",
                            "description": "选项列表(2-4 个,互斥,覆盖主要可能)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string",
                                              "description": "选项标签(1-5 字简短)"},
                                    "description": {"type": "string",
                                                    "description": "选项说明(为什么选这个/影响)"},
                                },
                                "required": ["label"],
                            },
                        },
                        "multi": {
                            "type": "boolean",
                            "description": "是否多选(默认 false 单选)",
                        },
                    },
                    "required": ["question", "options"],
                },
            },
        },
        "required": ["questions"],
    },
}


def _handle_ask_user(args: dict, **kwargs) -> str:
    """批量问题经桥接层（agent.ask_user_bridge）问用户，返回汇总 JSON。

    大白话流程：把 1-4 个问题打包成 qdata 交给桥接层（CLI 是命令行
    面板），用户逐题作答（或自填/转对话/取消），桥接层交回汇总结果，
    本函数转成 JSON 给模型。老的单问字段（question/options/multi/header）
    也认——包成一项问题处理，协议升级不摔旧调用方。

    参数：
        args：{"questions": [...]} 或老格式 {"question", "options", ...}
        kwargs：运行时注入，本函数只用到 agent_ref（找 ask_user_bridge）

    返回：JSON 字符串。正常 {"answers": [{"question","answers","multi"}...]}
    （用户转对话时多 "chat" 文本键）；取消/无桥接层/参数不合法返回对应 error。
    """
    # ---- 参数归一化：questions 数组优先（空列表也算新格式，
    # 落进「1-4 个」检查），老单问字段包成一项 ----
    raw_qs = args.get("questions")
    if isinstance(raw_qs, list):
        questions = raw_qs
    else:
        questions = [{
            "question": args.get("question"),
            "header": args.get("header"),
            "options": args.get("options"),
            "multi": args.get("multi", False),
        }]
    if not (1 <= len(questions) <= 4):
        return json.dumps({"error": "questions 需要 1-4 个问题"},
                          ensure_ascii=False)
    norm = []
    for q in questions:
        q = q or {}
        question = (q.get("question") or "").strip()
        options = q.get("options") or []
        if not question:
            return json.dumps({"error": "question 不能为空"},
                              ensure_ascii=False)
        if len(options) < 2:
            return json.dumps({"error": "options 至少要 2 个"},
                              ensure_ascii=False)
        norm.append({
            "question": question,
            "header": (q.get("header") or "").strip()[:12],
            "options": options,
            "multi": bool(q.get("multi", False)),
        })

    agent_ref = kwargs.get("agent_ref")
    bridge = (
        getattr(agent_ref, "ask_user_bridge", None)
        if agent_ref is not None else None
    )
    if not callable(bridge):
        # 没有桥接层：立即报错（fail-fast），不能让调用白等
        return json.dumps({
            "error": "ask_user 无可用桥接层（CLI/GUI 未注入 ask_user_bridge），无法向用户提问",
            "error_type": "no_bridge",
        }, ensure_ascii=False)

    qdata = {"id": uuid.uuid4().hex[:12], "questions": norm}
    try:
        result = bridge(qdata)
    except (EOFError, KeyboardInterrupt):
        return json.dumps({
            "error": "用户中断提问",
            "error_type": "user_interrupt",
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": f"ask_user 桥接层异常: {e}",
            "error_type": "bridge_error",
        }, ensure_ascii=False)

    # ---- 返回值归一化：dict=新协议；list/str=老单问格式兼容 ----
    if isinstance(result, dict):
        if result.get("cancelled"):
            return json.dumps({
                "error": "用户中断提问",
                "error_type": "user_interrupt",
            }, ensure_ascii=False)
        payload = {"answers": []}
        for a in (result.get("answers") or []):
            # 类型护栏：桥接层给的非 dict 条目直接丢掉，别裸抛 TypeError
            if isinstance(a, dict):
                payload["answers"].append(dict(a))
        chat = result.get("chat")
        chat = chat.strip() if isinstance(chat, str) else ""
        if chat:
            payload["chat"] = chat
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(result, str):
        # 防呆：裸字符串按字符迭代会拆成单字列表，包一层
        result = [result]
    answers = [str(a) for a in (result or [])]
    return json.dumps({"answers": [{"question": norm[0]["question"],
                                    "answers": answers,
                                    "multi": norm[0]["multi"]}]},
                      ensure_ascii=False)


# import 本模块时顺手把工具登记进中央注册表（项目惯例：工具文件顶层自注册）
registry.register(
    name="ask_user",
    toolset="core",
    schema=ASK_USER_SCHEMA,
    handler=_handle_ask_user,
    emoji="❓",
    isConcurrencySafe=False,  # 会卡住等真人输入，两个同时问会在屏幕上串行错乱，只能排队
)
