"""流式调 LLM 的「心脏」：边生成边吐字的整套流水线（从 agent/__init__.py 平移）。

大白话：普通调用像等别人把整封信写完才给你看；流式是写信人每写几个字
就念给你听一句——用户不用干等，只读的安全工具还能趁模型还在吐字
先跑起来（预执行）。流式失败自动退回非流式重试，绝不耽误出结果。

本模块收两件（行为零变化搬迁：函数体逐字节平移、self 改名 agent 传入，
_stream_callback / _streaming_preset_results 等属性全留 AIAgent 实例）：
  - call_llm_streaming: 流式主流程（增量累积、工具预执行、max_tokens
    截断升级、失败退非流式重试）
  - discard_partial_stream_state: 流式失败后扔掉半截状态的「墓碑」清理

处于 agent/__init__.py（主循环）之下、llm_client.py（接线层）之上，
和 llm_retry（防摔垫）平级协作。
"""

import logging

# max_tokens 升级/用量加总/长退避心跳在 llm_retry（拆分二期块 A）。
# import 方向 llm_streaming → llm_retry 无回边（llm_retry 模块级只 import
# 标准库），故走模块级引入，函数体内裸名直调
from agent.llm_retry import (
    llm_retry_heartbeat,
    merge_usage_tokens,
    try_escalate_max_tokens,
)
# StreamingToolExecutor 顶部只 import 标准库、不回 import agent root——
# 方向 llm_streaming → streaming_executor 无回边，故模块级引入而非函数内
# 惰性（构造仍包在函数内 try/except 里，构造失败退正常路径不硬崩）
from agent.streaming_executor import StreamingToolExecutor

logger = logging.getLogger(__name__)


async def call_llm_streaming(agent, *, messages, tools):
    """流式调用 LLM：每收到一小段就调 stream_callback 报告一次。

    流式让用户边生成边看到字，不用干等。流式失败时自动退回
    非流式重试（带备用客户端）。返回值和非流式路径完全同构
    （用 SimpleNamespace 拼出 OpenAI 响应的形状），下游的用量记账 /
    hook / 工具调用处理代码一行都不用改。

    注意：这个方法是「async 函数返回 response 对象」，不是 async 生成器。
    流式过程通过 stream_callback 回调报告，最终结果用 return 返回。

    stream_callback 会收到的事件：
        {"type": "content", "delta": str, "accumulated": str}  # 文本增量
        {"type": "tool_call_start", "name": str, "id": str}    # 工具调用开始
        {"type": "done", "finish_reason": str}                  # 流结束

    参数：
        messages: 发给 LLM 的消息列表
        tools: 工具 schema 列表（没有工具传 None）

    返回：拼装好的 OpenAI 兼容响应对象（SimpleNamespace）。
    """
    from types import SimpleNamespace
    full_content = ""
    tool_call_buffers: dict[int, dict] = {}  # idx → {id, name, arguments}
    final_usage = None
    finish_reason = "stop"
    reasoning_content = None   # DeepSeek 的思考内容（下次带工具调用回传时要带上）
    thinking_signature = None

    # === 流式并发执行（只读的安全工具趁模型还在吐字先跑起来）===
    # 开关在 config 的 agent.streaming_tool_execution（默认关）。预执行
    # 结果存 _streaming_preset_results，工具分发阶段按调用 id 取用、跳过重复执行。
    _executor = None
    if (agent.config or {}).get("agent", {}).get(
        "streaming_tool_execution", False,
    ):
        try:
            _executor = StreamingToolExecutor(agent)
        except Exception as e:
            logger.debug("流式执行器构造失败（退正常路径）: %s", e)
    _last_seen_idx = None

    try:
        # 从 config 读 max_tokens（用户可在 settings.json 的 llm 块配
        # "max_tokens": 8192）。不配就不传，让 API 用默认值——换模型不用改代码
        _extra = {}
        _cfg_mt = (
            (agent.config or {}).get("model", {}).get("max_tokens")
            or (agent.config or {}).get("llm", {}).get("max_tokens")
        )
        if _cfg_mt:
            _extra["max_tokens"] = _cfg_mt
        async for delta in agent.llm_client.chat_completions_stream(
            messages, tools=tools, **_extra,
        ):
            # 内容流式
            delta_text = delta.get("content") or ""
            if delta_text:
                full_content += delta_text
                if agent._stream_callback is not None:
                    try:
                        agent._stream_callback({
                            "type": "content",
                            "delta": delta_text,
                            "accumulated": full_content,
                        })
                    except Exception as cb_err:
                        logger.warning(
                            "stream_callback(content) 异常（忽略）: %s", cb_err
                        )

            # 工具调用增量累积
            for tc in delta.get("tool_calls") or []:
                idx = getattr(tc, "index", 0)
                # 出现新的 index 说明上一个工具的参数已经拼完整了，
                # 安全工具立刻预执行（和模型继续吐字的时间重叠，省等待）
                if (
                    _executor is not None
                    and _last_seen_idx is not None
                    and idx != _last_seen_idx
                    and _last_seen_idx in tool_call_buffers
                ):
                    _executor.complete(
                        _last_seen_idx, tool_call_buffers[_last_seen_idx],
                    )
                _last_seen_idx = idx
                buf = tool_call_buffers.setdefault(
                    idx, {"id": "", "name": "", "arguments": ""}
                )
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    buf["id"] = tc_id
                func = getattr(tc, "function", None)
                if func is not None:
                    fname = getattr(func, "name", None)
                    if fname:
                        buf["name"] = fname
                    fargs = getattr(func, "arguments", None)
                    if fargs:
                        buf["arguments"] += fargs
                # 第一次拿到 name 时通知 callback
                if buf["name"] and not buf.get("_notified"):
                    buf["_notified"] = True
                    if agent._stream_callback is not None:
                        try:
                            agent._stream_callback({
                                "type": "tool_call_start",
                                "name": buf["name"],
                                "id": buf["id"],
                            })
                        except Exception as cb_err:
                            logger.warning(
                                "stream_callback(tool_call_start) 异常: %s",
                                cb_err,
                            )

            # 最后一个数据块里带 finish_reason / usage / 思考内容
            if delta.get("finish_reason"):
                finish_reason = delta["finish_reason"]
            if delta.get("usage"):
                final_usage = delta["usage"]
            # DeepSeek 思考内容提取（后续带工具调用的请求要回传）
            if delta.get("reasoning_content"):
                reasoning_content = delta["reasoning_content"]
                # 思考流也通知回调（CLI 画暗色思考框用）。加法式：
                # 没回调/回调不认识该类型时零行为变化。
                if agent._stream_callback is not None:
                    try:
                        agent._stream_callback({
                            "type": "reasoning",
                            "delta": delta["reasoning_content"],
                        })
                    except Exception as cb_err:
                        logger.warning(
                            "stream_callback(reasoning) 异常（忽略）: %s",
                            cb_err,
                        )
            if delta.get("thinking_signature"):
                thinking_signature = delta["thinking_signature"]
    except Exception as stream_err:
        # 流式失败：退回非流式重试（带备用客户端）
        logger.warning(
            "流式调用失败，fallback 到非流式重试: %s", stream_err
        )
        # 流出错 → 等预执行任务跑完但扔掉结果（防留僵尸任务）
        if _executor is not None:
            await _executor.drain()
        # 显式扔掉已累积的半截状态（防御性「墓碑」清理，出错也放行）
        try:
            discard_partial_stream_state(agent)
        except Exception:
            pass
        from agent.llm_retry import call_with_retry
        response = await call_with_retry(
            agent.llm_client,
            messages,
            tools=tools,
            fallback_llm_client=agent.fallback_llm_client,
            config=agent.config,
            heartbeat_cb=llm_retry_heartbeat,  # 长退避心跳
        )
        # 流式回调已经错过，但至少把完整内容回放给 callback
        choice_msg = response.choices[0].message
        if choice_msg.content and agent._stream_callback is not None:
            try:
                agent._stream_callback({
                    "type": "content",
                    "delta": choice_msg.content,
                    "accumulated": choice_msg.content,
                })
            except Exception:
                pass
        return response

    # 流正常结束 → 补完最后一个工具的完整化 + 收集预执行结果
    if _executor is not None:
        try:
            if _last_seen_idx is not None and _last_seen_idx in tool_call_buffers:
                _executor.complete(
                    _last_seen_idx, tool_call_buffers[_last_seen_idx],
                )
            agent._streaming_preset_results = await _executor.collect()
        except Exception as e:
            logger.debug("流式预执行 collect 失败（弃用）: %s", e)
            agent._streaming_preset_results = {}

    # 合成 tool_calls 列表（按 idx 排序，过滤掉没 name 的）
    tool_calls_out = []
    for idx in sorted(tool_call_buffers.keys()):
        buf = tool_call_buffers[idx]
        if not buf["name"]:
            continue
        tool_calls_out.append(SimpleNamespace(
            id=buf["id"],
            type="function",
            function=SimpleNamespace(
                name=buf["name"],
                arguments=buf["arguments"] or "{}",
            ),
        ))

    # === max_tokens 截断的「调大上限重试」 ===
    # finish_reason=length 说明输出被单次回复长度上限掐断了。
    # DeepSeek-reasoner 还有一种隐蔽截断：纯思考（正文空、思考有值）——
    # 思考把长度额度用光了，正文没地方写，但 finish_reason 可能还是 "stop"。
    # 策略：先调大 max_tokens 整个重试一次（走非流式，避免把半截正文重复发
    # 一遍）；调大后还是空才认输，交给主循环处理。
    is_pure_thinking = (
        not full_content and not tool_calls_out and bool(reasoning_content)
    )
    new_max = None
    if finish_reason == "length" or is_pure_thinking:
        trigger_reason = "纯 thinking（content 空）" if is_pure_thinking else "finish_reason=length"
        new_max = try_escalate_max_tokens(agent, trigger_reason)
    # 拿到新上限才重试；None = 不该升级/已升过级，沿用截断响应
    if new_max is not None:
        try:
            from agent.llm_retry import call_with_retry
            retried = await call_with_retry(
                agent.llm_client,
                messages,
                tools=tools,
                fallback_llm_client=agent.fallback_llm_client,
                max_tokens=new_max,
                config=agent.config,
                heartbeat_cb=llm_retry_heartbeat,  # 长退避心跳
            )
            retried_choice = retried.choices[0]
            retried_msg = retried_choice.message
            # 用重试结果整体覆盖（重试拿到的是完整响应）。
            # 不能只在「重试有 tool_calls」时才覆盖，
            # 重试结果没有工具调用时会把截断那次的半截 tool_calls 残留进
            # 最终响应——必须无条件清空。
            finish_reason = (
                getattr(retried_choice, "finish_reason", None) or "stop"
            )
            full_content = retried_msg.content or ""
            tool_calls_out = list(
                getattr(retried_msg, "tool_calls", None) or []
            )
            # 把重试结果回放给回调（和 fallback 路径同款做法）
            if retried_msg.content and agent._stream_callback is not None:
                try:
                    agent._stream_callback({
                        "type": "content",
                        "delta": retried_msg.content,
                        "accumulated": retried_msg.content,
                    })
                except Exception:
                    pass
            # 更新 usage：截断那次 + 升级重试这次都真实花过钱，两边加总
            # （直接覆盖会漏记截断那次的花费）
            final_usage = merge_usage_tokens(final_usage, retried)
        except Exception as esc_err:
            logger.warning(
                "max_tokens 升级重试失败（沿用截断响应）: %s", esc_err
            )

    # 通知回调：流结束了
    if agent._stream_callback is not None:
        try:
            agent._stream_callback({
                "type": "done",
                "finish_reason": finish_reason,
            })
        except Exception:
            pass

    # 拼一个 OpenAI 兼容的响应对象（让记账 / hook 等下游代码不用改）
    message = SimpleNamespace(
        content=full_content if full_content else None,
        tool_calls=tool_calls_out if tool_calls_out else None,
        reasoning_content=reasoning_content,
        thinking_signature=thinking_signature,
    )
    usage_ns = None
    if final_usage is not None:
        usage_ns = SimpleNamespace(
            prompt_tokens=final_usage.get("prompt_tokens", 0),
            completion_tokens=final_usage.get("completion_tokens", 0),
            prompt_cache_hit_tokens=final_usage.get("cache_read", 0),
            cache_read_input_tokens=final_usage.get("cache_read", 0),
            prompt_cache_miss_tokens=final_usage.get("cache_creation", 0),
            cache_creation_input_tokens=final_usage.get("cache_creation", 0),
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=message,
            finish_reason=finish_reason,
        )],
        usage=usage_ns,
    )


def discard_partial_stream_state(agent) -> None:
    """流式失败后显式扔掉半截累积状态（防御性的「墓碑」清理）。

    说明：半截增量只进 UI 回调、不进历史（那些变量是流式函数的
    局部变量，出作用域自己就没了），所以这个方法目前是保险带——万一
    以后有人把增量提前塞进历史或暂存区，这里负责清痕迹 + 打日志提醒，
    顺手清空流式预执行结果。

    参数：无。返回：无。
    """
    agent._streaming_preset_results = {}
    logger.debug("流式失败：半截增量已丢弃（不入 history）")
