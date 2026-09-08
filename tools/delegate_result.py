"""委托结果后处理与进度播报：从 delegate_tool 拆出（纯搬迁）。

本模块收拢六个模块级函数，全部从 tools/delegate_tool.py 逐字节平移：

  - 结果后处理五件套：
      _review_handoff           交接复审（辅助 LLM 安检员，危险产出贴警告）
      _offload_child_result     原始结果落盘，返回 offload 占位 JSON
      _attach_full_result_pointer 摘要尾部附「完整结果已落盘」找回指针
      _summarize_child_result   用 LLM 把超长结果压成摘要
      _build_child_system_prompt 拼子代理的 system prompt
  - 进度播报：
      _start_progress_ticker    并行子代理的定时进度播报线程

搬迁铁律（委托链拆分约定）：
  - 行为零变化——函数体与 delegate_tool 原文逐字节一致；
  - 函数内延迟 import（agent.loop_host / agent.llm_retry /
    agent.output_offload 等）原样保留，不提升到模块顶层；
  - 模块顶层不调 registry.register()——这里是函数库，不是注册工具；
  - delegate_tool 侧带下划线原名 re-export，外部 import 与调用点解析零改动。
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _review_handoff(result: str, parent_agent) -> str:
    """交接复审：辅助 LLM 检查子代理要交回来的产出，发现危险内容就在前面贴警告。

    放权模式下子代理产出直接进父代理上下文，这一步让辅助 LLM 当安检员。
    判安全、辅助模型不可用或出错，都原样返回（fail-open，不拦路）。接口是
    同步的（_run_child 在线程里跑、不在宿主循环线程，内部交给进程级常驻
    循环宿主桥接 async 调用）。

    参数：
      - result：子代理的产出文本
      - parent_agent：父代理（从它身上取 aux_llm_router 辅助模型）

    返回：可能带 `[⚠ 交接复审警告]` 前缀的产出文本。
    """
    aux = getattr(parent_agent, "aux_llm_router", None)
    if aux is None or not result or not result.strip():
        return result
    if len(result) < 200:
        return result  # 短结果一眼能看完，省一次辅助模型调用

    prompt = (
        "复审以下子代理交给主代理的产出片段。判断它是否包含危险操作的证据：\n"
        "- 删除/覆盖大量文件或系统目录\n"
        "- 数据外发（上传/推送到外部、大段凭证或密钥输出）\n"
        "- 修改凭证/SSH/认证配置\n"
        "- 绕过或关闭安全机制\n"
        "只看产出本身，不推测意图。\n\n"
        f"产出：\n{result[:4000]}\n\n"
        '输出严格 JSON：{"dangerous": true/false, "warning": "<=40 字中文说明，仅 dangerous 时非空>"}'
    )
    try:
        from agent.loop_host import loop_host
        resp = loop_host.run_async(aux.chat_completions(
            [{"role": "user", "content": prompt}],
        ), exempt_from_fence=True)  # 后台线程长活，豁免回合栅栏（见 run_async docstring）
        import json as _json
        text = resp.choices[0].message.content or ""
        parsed = _json.loads(text.strip().strip("`"))
        if isinstance(parsed, dict) and parsed.get("dangerous"):
            warning = str(parsed.get("warning", ""))[:80] or "子代理产出含危险操作证据"
            logger.warning("交接复审命中: %s", warning)
            return f"[⚠ 交接复审警告] {warning}\n\n{result}"
    except Exception as e:
        logger.debug("交接复审 aux 调用失败（放行原文）: %s", e)
    return result


def _offload_child_result(result: str, kwargs: dict) -> Optional[str]:
    """把子代理的原始结果落盘，返回 offload 占位 JSON（失败返回 None）。

    摘要是有损压缩——父代理想看细节（精确 diff/路径/命令输出）时得能
    读回原文，对齐 maybe_offload 的「预览+指针」模式。走摘要就强制落盘
    （threshold=0），几百字的小结果也留底，指针成本可忽略。

    参数：
      - result：子代理的原始结果文本
      - kwargs：delegate handler 收到的上下文（取 tool_call_id/
        codeagent_home/task_id）
    返回：offload 占位 JSON 字符串；没传 home 或落盘失败返回 None。
    """
    try:
        home = kwargs.get("codeagent_home")
        if not home:
            return None
        from agent.output_offload import maybe_offload
        tcid = str(
            kwargs.get("tool_call_id")
            or f"delegate_{kwargs.get('task_id') or 'result'}",
        )
        off = maybe_offload(
            result, tool_call_id=tcid, agent_home=Path(home), threshold=0,
        )
        if isinstance(off, str) and '"full_at"' in off:
            return off
    except Exception as e:
        logger.warning("子代理结果落盘失败（fail-open 不摘要原文）: %s", e)
    return None


def _attach_full_result_pointer(summary: str, offloaded_json: str) -> str:
    """摘要尾部附「完整结果已落盘」的找回指针（占位解析失败就原样返回）。"""
    try:
        data = json.loads(offloaded_json)
        full_at = data.get("full_at", "")
        n = data.get("orig_chars", 0)
        if full_at:
            return (
                f"{summary}\n\n"
                f"[完整结果 {n} 字符已落盘] full_at: {full_at}"
                "（需要细节时用 read_file 读回）"
            )
    except Exception:
        pass
    return summary


def _summarize_child_result(
    result: str, client, model: str, max_chars: int = 300,
) -> str:
    """用 LLM 把子代理的结果压成指定字数以内的摘要（默认 300 字）。

    子代理动辄输出几千字，全文塞回主对话太费上下文——超长结果先摘要。
    摘要失败就返回原文（不能因为压缩失败把整个委托卡死）。

    参数：
      - result：子代理的原始结果文本
      - client：子代理的 LLM 客户端（child.llm_client）
      - model：模型名
      - max_chars：摘要字数上限（长任务的深度调研可调大，如 800/1500）

    返回：`[摘要] ...` 格式的压缩文本；失败时返回原文。

    接口说明：call_with_retry 是 async，本函数仍保持
    同步接口（调用方 _run_child 在独立线程里跑、不在宿主循环线程），
    内部交给进程级常驻循环宿主同步等结果（等价旧的 asyncio.run）。
    """
    # 输入材料随目标长度放宽（要写更长摘要就得多给原文），封顶 30000
    input_limit = min(30000, max(8000, max_chars * 10))
    prompt = (
        f"把以下子代理执行结果总结成 {max_chars} 字以内的摘要，保留：\n"
        "1. 核心结论\n"
        "2. 关键发现和数据\n"
        "3. 重要的文件路径、命令、错误信息\n"
        "4. 待办事项\n\n"
        f"子代理结果：\n{result[:input_limit]}"
    )
    try:
        from agent.llm_retry import call_with_retry
        from agent.loop_host import loop_host
        response = loop_host.run_async(call_with_retry(
            client,  # child.llm_client（LLM 客户端实例）
            [{"role": "user", "content": prompt}],
            background=True,  # 摘要属于后台活：遇 529 过载直接放弃不重试
        ), exempt_from_fence=True)  # 后台线程长活，豁免回合栅栏（见 run_async docstring）
        summary = response.choices[0].message.content
        return f"[摘要] {summary}\n\n[完整结果 {len(result)} 字符已省略]"
    except Exception as e:
        logger.debug("子代理结果摘要失败，返回原文: %s", e)
        return result


def _build_child_system_prompt(goal: str, context: str, role: str, override: str = None) -> str:
    """拼出子代理的 system prompt（开场设定词）。

    告诉子代理「你是谁、要干什么、守什么规矩」。

    参数：
      - goal：任务描述
      - context：父代理给的补充背景
      - role：leaf / orchestrator 角色
      - override：自定义子代理 .md 里写的 system_prompt；非空时以它为底，
        只在后面补上下文/约束/角色提示；None 时走默认模板

    返回：拼好的 system prompt 字符串。
    """
    if override:
        parts = [override]
    else:
        parts = [
            "你是一个子代理，由父代理派生执行独立任务。",
            f"\n你的角色: {role}",
            f"\n你的任务目标: {goal}",
        ]

    if context:
        parts.append(f"\n来自父代理的上下文:\n{context}")

    # 自定义 override 也要补一份通用约束（只是不再塞默认的 goal/role 那几句）
    if override:
        parts.append(
            "\n## 约束\n"
            "- 独立执行，不假设父代理历史\n"
            "- 结果要具体（文件路径、命令、数据）\n"
        )
    else:
        parts.append(
            "\n要求:\n"
            "- 专注完成任务，不要偏离目标\n"
            "- 完成后给出清晰的总结\n"
            "- 遇到不可解决的阻碍时，返回错误说明\n"
            "- 不要做任务范围外的事"
        )

    if role == "leaf":
        parts.append("\n你是 leaf 角色，不能再派生子代理。")

    return "\n".join(parts)


def _start_progress_ticker(
    children_state: dict,
    stop_event,
    *,
    aux,
    session_id: str,
    interval: float = 30.0,
) -> "threading.Thread":
    """多个子代理并行跑的时候，每隔 interval 秒写一条进度播报。

    让并行等待时进度可见：有辅助小模型（aux）就让它把状态归纳成 1-2 句人话；
    没有就直接机械拼一行状态。写进 scratchpad 涂鸦区的 progress.md（涂鸦区
    7 天自动清理）+ 打一条 logger.info。出错全吞（fail-open）：进度播报绝不
    能反过来影响子代理本身。

    参数：
      - children_state：各子代理的状态字典（名字 → {status, goal...}）
      - stop_event：叫停信号，set 了 ticker 线程就退出
      - aux：辅助 LLM 路由（可空；空则用机械拼接）
      - session_id：当前会话 ID（定位涂鸦区目录）
      - interval：播报间隔秒数（默认 30）

    返回：启动好的 ticker 线程对象（调用方负责 set stop_event 停掉它）。
    """
    import threading

    def _tick():
        while not stop_event.wait(interval):
            try:
                lines = [
                    f"{name}: {info.get('status', '?')}（{info.get('goal', '')[:40]}）"
                    for name, info in children_state.items()
                ]
                if not lines:
                    continue
                text = "\n".join(lines)
                if aux is not None:
                    try:
                        # aux.chat_completions 是 async 的，而本函数跑在 ticker
                        # 守护线程里（不在宿主循环线程）——交给进程级常驻循环
                        # 宿主同步等结果（等价旧的 asyncio.run，aux 缓存 client
                        # 绑定常驻循环不再每次换新循环漂移）
                        from agent.loop_host import loop_host
                        resp = loop_host.run_async(aux.chat_completions([
                            {"role": "user", "content":
                             f"把以下子代理状态摘要成 1-2 句中文进度：\n{text}"},
                        ]), exempt_from_fence=True)  # 后台线程长活，豁免回合栅栏（见 run_async docstring）
                        summarized = resp.choices[0].message.content or ""
                        if summarized.strip():
                            text = summarized.strip()
                    except Exception:
                        pass  # 摘要失败就退回机械拼接的原文
                from agent.scratchpad import scratchpad_dir
                d = scratchpad_dir(session_id)
                d.mkdir(parents=True, exist_ok=True)
                p = d / "progress.md"
                # 只留最近 20 条（防长时间运行把文件撑大；涂鸦区不是知识库）
                try:
                    old = p.read_text(encoding="utf-8").splitlines()
                except OSError:
                    old = []
                stamp = time.strftime("%H:%M:%S")
                new = old[-19:] + [f"[{stamp}] {text}"]
                p.write_text("\n".join(new) + "\n", encoding="utf-8")
                logger.info("[子代理进度] %s", text)
            except Exception as e:
                logger.debug("progress ticker fail-open: %s", e)

    t = threading.Thread(target=_tick, daemon=True, name="delegate-progress")
    t.start()
    return t
