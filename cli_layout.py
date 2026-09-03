"""常驻操作台布局——hermes 默认 CLI 骨架的复刻（块①）。

设计一句话（跟 hermes 学的）：**屏幕上下分工**——
- 底部几行是 prompt_toolkit Application 管的「操作台」：
  spinner 行 + 多行输入区 + 分隔线 + 状态栏；
- 上面滚走的对话区不归布局管，还是 console.print + patch_stdout。
- 老主循环搬进工作线程，靠 _input_q 队列跟 UI 线程传话：
  Enter 键位回调是队列的生产端，老主循环还是消费端（零改动）。

本文件分三层：
1. 纯函数层（状态栏段/spinner 文案/提交小函数）——可单测，不碰终端；
2. 组装层 build_application（布局+键位+样式）；
3. 运行层（spinner 线程、节流重绘、跨线程退出请求）。
"""

import logging
import math
import time

logger = logging.getLogger(__name__)

# spinner 动画帧（盲文点阵转圈，一圈 10 帧）
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# 状态栏三档宽度阈值（跟 hermes 学的：窄屏只留最要紧的）
_TIER_NARROW = 52    # < 52 列：模型 + 计时
_TIER_MEDIUM = 76    # < 76 列：加 目录/后台/正在跑的工具


def _fmt_elapsed(seconds) -> str:
    """秒数 → 状态栏计时文案（None/未开始返回空串）。"""
    if not seconds or seconds < 0:
        return ""
    return f"{int(seconds)}s"


def status_bar_segments(rt, width: int, elapsed_s=None) -> list:
    """状态栏内容段（纯函数）：按宽度三档过滤，返回段文本列表。

    大白话：状态栏像行李箱——宽屏全装上，中屏扔掉按键说明书，
    窄屏只留证件（模型）和手表（计时）。

    参数：
        rt: RuntimeContext（读 agent.model/workspace_cwd/bg_count/
            event_pending——全是现有黑板，零新状态）
        width: 终端列数
        elapsed_s: 回合已进行秒数（None=空闲）

    返回：str 列表，调用方用 " │ " 拼接。
    """
    segs = []
    try:
        model = getattr(getattr(rt, "agent", None), "model", "") or ""
        if model:
            segs.append(f"⚡{model}")
    except Exception:
        pass
    if width >= _TIER_NARROW:
        try:
            cwd = getattr(rt, "workspace_cwd", "") or ""
            if cwd:
                tail = str(cwd).replace("\\", "/").rstrip("/").split("/")[-1]
                segs.append(f"📂{tail}")
        except Exception:
            pass
        bg = getattr(rt, "bg_count", None)
        if bg:
            segs.append(f"☂{bg}个后台")
        pend = getattr(rt, "event_pending", None)
        if pend:
            segs.append(f"◐{pend[-1]}")   # 正在跑的工具（最晚出发的）
    if elapsed_s is not None:
        t = _fmt_elapsed(elapsed_s)
        if t:
            segs.append(t)
    if width >= _TIER_MEDIUM:
        segs.append("Enter发送 Alt+↵换行")
    return segs


def spinner_text(frame: int, rt, turn_started_at, now: float) -> str:
    """spinner 行文案（纯函数）：空闲返回空串（行隐藏）。

    参数：
        frame: 动画帧下标（spinner 线程递增，这里只取模）
        rt: 读 turn_active / event_pending
        turn_started_at: 回合开始时刻（time.monotonic 值，None=没在跑）
        now: 当前时刻（传进来而不是函数内取，方便测试）

    返回：如 "⠋ terminal 3s" / "⠋ 思考中… 3s" / ""（空闲）
    """
    try:
        if not getattr(rt, "turn_active", False):
            return ""
        elapsed = (now - turn_started_at) if turn_started_at else 0.0
        t = _fmt_elapsed(elapsed)
        pend = getattr(rt, "event_pending", None)
        mark = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
        if pend:
            return f"{mark} {pend[-1]} {t}".strip()
        return f"{mark} 思考中… {t}".strip()
    except Exception:
        return ""   # 纯视觉，任何异常都当「不显示」


def submit_input(buffer, input_queue) -> None:
    """提交小函数（键位回调只是薄壳，逻辑全在这——方便单测）。

    大白话：把输入框里的字塞给老主循环的队列，然后擦黑板。
    空白输入不入队（老语义：空行直接跳过，不烧一轮 LLM）。

    参数：
        buffer: prompt_toolkit 的 Buffer（TextArea 的肚子）
        input_queue: 老主循环消费的 _input_q
    """
    try:
        text = buffer.text
        if text.strip():
            input_queue.put(text)
        # validate_and_handle 会把非空文本记进历史再把框清空
        #（等价于老 PromptSession 按回车的动作）
        buffer.validate_and_handle()
    except Exception:
        logger.exception("提交输入失败（这行字丢了，但不许炸输入线程）")
