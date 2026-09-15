"""LLM 调用的「翻译官」层：把两家画风不同的 API 统一成一副面孔。

市面上的大模型 API 主要分两种格式——OpenAI 兼容格式（DeepSeek、
OpenRouter、本地 Ollama 等都用这套）和 Anthropic 原生格式（Claude 官方）。
本文件提供两个 client（可以理解为「接线员」），都实现同一个方法
chat_completions()，返回的响应统一长成 OpenAI 的样子（从
response.choices[0].message 里取 content 和 tool_calls）——上层
AIAgent 主循环完全不用关心底层接的是哪家 API。

用哪个 client 由 model_config（模型配置字典）里的 "format" 字段决定：
  - "openai"    → OpenAICompatClient（DeepSeek/OpenAI/OpenRouter 等）
  - "anthropic" → AnthropicClient（Claude 原生 API）

在依赖链里位于 model_tools.py 之下，是所有 LLM 请求的最终出口；
aux_llm.py 的辅助路由器也通过本文件的工厂函数创建 client。
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

# AsyncOpenAI 必须在模块顶层 import，测试才能用
# patch("agent.llm_client.AsyncOpenAI") 把它换成假对象。
# （写在函数内部的局部 import，unittest.mock.patch 找不到、也换不了）
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _is_client_closed_error(exc: BaseException) -> bool:
    """顺异常链找「client has been closed」（httpx 连接池已被 close）。

    reset_client 换池/收尾关池和并发调用之间有个竞态窗口：拿到旧池
    引用的调用方会在 await 处撞上这个错。顺着 __cause__/__context__
    链找（openai SDK 会把 httpx 的 RuntimeError 包好几层）。
    """
    cur = exc
    hops = 0
    while cur is not None and hops < 8:
        if "client has been closed" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
        hops += 1
    return False

# 流式「看门狗」的默认空闲超时（90 秒）。
# 看门狗 = 盯着流式输出，太久没新数据就认为卡死并中止。
DEFAULT_STREAM_IDLE_TIMEOUT = 90.0


class LLMStreamIdleTimeout(Exception):
    """流式回答「卡住不动」的异常：看门狗发现 idle_timeout 秒内一个字都没来。

    上层有个「扣留-恢复」机制会接住这个异常，改用非流式方式再试一次，
    而不是直接向用户报错（详见 agent/__init__.py 的恢复逻辑）。
    """


async def _iterate_with_watchdog(
    stream: AsyncIterator,
    *,
    idle_timeout: float,
    describe: str = "",
):
    """给流式输出套一个「闹钟」：太久没下一段内容就中止并报超时。

    网络或服务器卡住时普通写法会永远干等；本函数在每次等下一段时设一个
    idle_timeout 秒的闹钟，超时还没等到就关掉流、抛 LLMStreamIdleTimeout。

    做法：Python 的 `async for` 语法没法直接加超时，所以改成手工调
    __anext__() 并用 asyncio.wait_for 包一层；超时时 wait_for 会顺手
    取消那个还在傻等的接收协程（安全，不会留尾巴）。

    设计取舍：只做超时中止，不做「半超时警告 + 停顿计数」遥测上报
    （本项目没有对应的遥测通道）。

    参数：
        stream：原始的 async 迭代器（流式响应）
        idle_timeout：闹钟秒数；小于等于 0 表示关掉看门狗（原样迭代，
                      兼容旧配置）
        describe：附加到报错信息里的说明文字（比如带上模型名，方便排查）

    产出（yield）：流里原本的每个 item；超时则抛 LLMStreamIdleTimeout。
    """
    if idle_timeout <= 0:
        async for item in stream:
            yield item
        return
    it = stream.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(it.__anext__(), timeout=idle_timeout)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            closer = getattr(stream, "close", None)
            if closer is not None:
                try:
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass  # 关流失败也无所谓：等数据的协程已被取消，不会泄漏
            raise LLMStreamIdleTimeout(
                f"流式空闲超时（{idle_timeout:.0f}s 无数据{describe}），已中止流"
            )
        yield item


# ---------------------------------------------------------------------------
# token 用量（usage）字段提取：两家 SDK 的字段名不一样，这里统一成
# 同一种字典格式，上层只用认一种
# ---------------------------------------------------------------------------

def _extract_openai_usage(usage_obj) -> Optional[dict]:
    """把 OpenAI 兼容 SDK 的「用量统计对象」翻译成统一格式的字典。

    DeepSeek 和 OpenAI 官方各有一套字段名（比如缓存命中，一个叫
    prompt_cache_hit_tokens，一个叫 cache_read_input_tokens），这里把
    两种叫法都兜住，输出统一的 5 个字段：prompt_tokens（输入）/
    completion_tokens（输出）/ total_tokens（总计）/ cache_read（命中
    缓存省下的）/ cache_creation（新写缓存的）。流式和非流式两条路共用。

    参数：
        usage_obj：SDK 响应里的 usage 对象（可能为 None）

    返回：统一格 dict；没有 usage 对象时返回 None。
    """
    if usage_obj is None:
        return None
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
        "cache_read": getattr(usage_obj, "prompt_cache_hit_tokens", 0)
            or getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(usage_obj, "prompt_cache_miss_tokens", 0)
            or getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
    }


def _extract_anthropic_usage(usage_obj) -> Optional[dict]:
    """把 Anthropic SDK 的用量统计翻译成上面 OpenAI 那套统一格式。

    Anthropic 把输入/输出叫做 input_tokens/output_tokens，而且
    不直接给总数——这里改名映射，总数自己加出来。

    参数：
        usage_obj：Anthropic 响应里的 usage 对象（可能为 None）

    返回：统一格式 dict；没有 usage 对象时返回 None。
    """
    if usage_obj is None:
        return None
    input_t = getattr(usage_obj, "input_tokens", 0) or 0
    output_t = getattr(usage_obj, "output_tokens", 0) or 0
    return {
        "prompt_tokens": input_t,
        "completion_tokens": output_t,
        "total_tokens": input_t + output_t,
        "cache_read": getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
    }


def _anthropic_finish_reason(stop_reason) -> str:
    """把 Anthropic 的 stop_reason 翻译成 OpenAI 语义的 finish_reason。

    两家方言不同：Anthropic 用 "max_tokens" 表示输出被截断（OpenAI 叫
    "length"）、"tool_use" 表示要调工具（OpenAI 叫 "tool_calls"）。
    不翻译的话，上层的 detect_length_finish / max_tokens 升级重试 /
    续写恢复整条截断救援链路都认不出来——截断回答会被当正常完成
    直接交付。
    """
    if stop_reason == "max_tokens":
        return "length"
    if stop_reason == "tool_use":
        return "tool_calls"
    return "stop"


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class LLMClient:
    """所有 LLM client 的「模板」基类：规定必须长什么样，本身不能直接用。"""

    async def chat_completions(
        self,
        messages: List[dict],
        *,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        """非流式地调一次 LLM，回答一口气全回来。

        约定：返回的对象必须能这样取值（也就是 OpenAI 的样子）——
            response.choices[0].message.content   （回答文本，str 或 None）
            response.choices[0].message.tool_calls（模型想调的工具列表，或 None）

        参数：
            messages：对话历史（OpenAI 格式的消息字典列表）
            tools：可用的工具清单（OpenAI 格式 schema，可省略）
            **kwargs：其余参数原样透传给底层 SDK

        返回：OpenAI 兼容格式的响应对象。
        """
        raise NotImplementedError

    async def chat_completions_stream(
        self,
        messages: List[dict],
        *,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        """流式地调 LLM：回答像打字机一样一小段一小段地往外吐。

        吐出来的每一小段（chunk）都是统一格式的 dict，不是 SDK 原始对象——
        这样上层不管底层接的是哪家 API，都只认这一种格式：
            {
                "content": str,        # 这一小段新增的文本（可能是空串）
                "tool_calls": list,    # 这一小段新增的工具调用（可能是空列表）
                "finish_reason": str?, # 结束原因，只有最后一个 chunk 才有
                "usage": dict?,        # token 用量，一般只有最后一个 chunk 有
            }

        基类的默认实现是个「假流式」：先走非流式拿完整回答，再一口气
        yield 出去——这样不支持流式的 client 也能套用同一个接口。
        真正的子类会重写成真流式。

        参数：
            messages：对话历史（OpenAI 格式消息列表）
            tools：可用工具清单（可省略）
            **kwargs：其余参数透传给底层 SDK
        """
        resp = await self.chat_completions(messages, tools=tools, **kwargs)
        choice = resp.choices[0]
        msg = choice.message
        usage_dict = _extract_openai_usage(getattr(resp, "usage", None))
        yield {
            "content": msg.content or "",
            "tool_calls": list(msg.tool_calls or []),
            "finish_reason": getattr(choice, "finish_reason", "stop"),
            "usage": usage_dict,
        }

    def reset_client(self) -> None:
        """把底下的 HTTP 连接池整个换新。

        连接被重置弄坏后，在坏连接池上重试大概率还是失败，所以重试
        逻辑会先调这个方法重建 client 再试。

        基类里什么都不做（no-op）；需要的子类自己重写。调用方是串行的
        重试路径，不存在两个线程同时重建的并发问题。
        """

    def close(self) -> None:
        """同步释放底层连接（进程退出/收尾打扫用）。

        基类提供无害默认；有连接池的子类必须重写——agent.cleanup() 是用
        getattr(client, "close") 逐个关客户端的，不实现就是静默 no-op、
        连接池泄漏。
        """


# ---------------------------------------------------------------------------
# OpenAI 兼容格式的 client（DeepSeek/OpenAI/OpenRouter/本地 Ollama 等都用这套）
# ---------------------------------------------------------------------------

class OpenAICompatClient(LLMClient):
    """接 OpenAI 兼容 API 的 client（异步），项目默认走这一类。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
        request_timeout: float = None,
    ):
        # request_timeout：单次 HTTP 请求的超时秒数（None = SDK 默认 600s）。
        # 看门狗只护流式，非流式挂死全靠这里兜底
        self._request_timeout = request_timeout
        # 旧关池 task 的强引用列表（reset 可反复发生；单槽会互相顶掉，
        # 被顶掉的正是注释自己警告的 GC 蒸发场景）——留最近 4 个足够
        self._old_close_tasks: list = []
        _client_kwargs = {"base_url": base_url, "api_key": api_key}
        if request_timeout is not None:
            _client_kwargs["timeout"] = request_timeout
        self.client = AsyncOpenAI(**_client_kwargs)
        self.model = model
        self.base_url = base_url
        # 把密钥自己存一份：SDK 不保证让你读回
        # 旧 client 的密钥，重建时没存就得不偿失。
        self._api_key = api_key
        # 流式看门狗的空闲秒数；小于等于 0 表示关掉看门狗
        self.stream_idle_timeout = stream_idle_timeout

    def reset_client(self) -> None:
        """扔掉可能坏掉的连接池，重新造一个 AsyncOpenAI。"""
        try:
            # 尽力关掉旧 client；它已经坏了也无所谓，反正要扔
            import asyncio as _aio
            try:
                loop = _aio.get_running_loop()
                if loop is not None:
                    # 强引用必须自己存：asyncio 对 task 只持弱引用，
                    # 没人引用的 fire-and-forget task 可能被 GC 中途蒸发
                    # （关一半的池子状态更诡异）；列表容纳连续多次 reset
                    self._old_close_tasks.append(loop.create_task(self.client.close()))
                    del self._old_close_tasks[:-4]
            except RuntimeError:
                pass
        except Exception:
            pass
        _client_kwargs = {"base_url": self.base_url, "api_key": self._api_key}
        if self._request_timeout is not None:
            _client_kwargs["timeout"] = self._request_timeout
        self.client = AsyncOpenAI(**_client_kwargs)
        # 换池也留栈：幻影关池排查线索（谁在什么路径上 reset 的）
        logger.info("reset_client 已换池（%s）", self.model, stack_info=True)

    def close(self) -> None:
        """尽力释放底层 SDK 的 HTTP 连接池（同步版，agent.cleanup() 调）。

        底层的 close 是协程：调用线程不定（主线程收尾、后台线程兜底
        都有），按「线程归属」分流——就在宿主循环上就排个任务异步关；
        其余情况（没在循环里 / 在别的循环线程上）交给宿主循环跑完
        （run_async 阻塞等结果，等于原来的 asyncio.run，但协程落在
        常驻循环上，连接池不会绑到用完即弃的临时循环）。关不上也不
        报错——打扫失败不该影响正事。

        带调用栈的 INFO 日志：close/reset 期间曾出现「幻影关池者」
        （有人把在用的池子关了，受害者只看到 client has been closed），
        栈日志让下一次发生时能直接从 codeagent.log 认出是谁关的。
        """
        try:
            coro = self.client.close()
            if asyncio.iscoroutine(coro):
                logger.info("关闭 LLM 连接池（%s）", self.model,
                            stack_info=True)
                from agent.loop_host import loop_host
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                try:
                    if loop is not None and loop is loop_host.loop:
                        task = asyncio.create_task(coro)
                        self._close_task = task   # 强引用防 GC 蒸发
                    else:
                        # 有界关池：卡死 30 秒后放弃，daemon 退出兜底仍在
                        loop_host.run_async(coro, timeout=30)
                except Exception:
                    # 退出窗口 loop 已停、协程排不进去——coro.close() 消毒，
                    # 不然解释器收尾要甩「coroutine never awaited」警告
                    coro.close()
        except Exception:
            pass

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        """非流式调用：直接转交给 AsyncOpenAI SDK，原样返回它的响应对象。

        参数：
            messages：对话历史（OpenAI 格式消息列表）
            tools：可用工具清单（可省略）
            **kwargs：其余参数透传给 SDK
        """
        # 防御性处理：上层几个模块（记忆管理/反思/检索）习惯在 kwargs 里
        # 带上 model=xxx。但模型名已经是本 client 的属性，这里再收一次
        # 同名参数，SDK 会报「model 参数传了两遍」。所以先扔掉外来的。
        kwargs.pop("model", None)
        try:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                **kwargs,
            )
        except Exception as e:
            if not _is_client_closed_error(e):
                raise
            # 连接池已被关闭（reset_client/收尾的竞态窗口正好撞上）——
            # 重建一次重试，别让一次竞态炸掉整轮调用（反思/主对话都遭过）
            logger.warning("LLM client 已关闭（竞态窗口），重建后重试一次")
            self.reset_client()
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                **kwargs,
            )

    async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
        """真流式调用：开 stream=True，一段一段收 chunk。

        每个 chunk 都转成统一格式的 dict（格式见基类
        chat_completions_stream 的说明）。工具调用的增量部分保留 SDK
        原始对象（里面有 index/id/function 等字段），由上层按序号拼装
        成完整调用。

        参数：
            messages：对话历史（OpenAI 格式消息列表）
            tools：可用工具清单（可省略）
            **kwargs：其余参数透传给 SDK
        """
        # 同上：扔掉外来的 model，防止同名参数传两遍
        kwargs.pop("model", None)
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
        except Exception as e:
            if not _is_client_closed_error(e):
                raise
            logger.warning("LLM client 已关闭（竞态窗口），重建后重试一次")
            self.reset_client()
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
        # 流式看门狗：90 秒没等到新 chunk 就中止流
        async for chunk in _iterate_with_watchdog(
            stream,
            idle_timeout=self.stream_idle_timeout,
            describe=f"，model={self.model}",
        ):
            usage_dict = _extract_openai_usage(getattr(chunk, "usage", None))
            if not chunk.choices:
                # 收尾的那个 chunk 可能不带内容、只带用量统计
                if usage_dict:
                    yield {
                        "content": "",
                        "tool_calls": [],
                        "finish_reason": None,
                        "usage": usage_dict,
                    }
                continue
            delta = chunk.choices[0].delta
            yield {
                "content": delta.content or "",
                "tool_calls": list(delta.tool_calls or []),
                "finish_reason": chunk.choices[0].finish_reason,
                "usage": usage_dict,
                # DeepSeek 思考内容的流式分片（SDK extra 字段，标准 OpenAI
                # 模型没有此属性——getattr 兜 None）。不透传的话上层
                # llm_streaming 永远拿不到思考内容：UI 思考框、纯思考
                # 截断检测、思考兜底回复全失效
                "reasoning_content": getattr(delta, "reasoning_content", None),
            }


# ---------------------------------------------------------------------------
# Anthropic 原生格式的 client（Claude 系列）
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """接 Anthropic 原生 API 的 client（异步），对外仍装成 OpenAI 的样子。

    Claude 的 API 和 OpenAI 格式有几处根本性的「说方言」差异：
    - 系统提示词（system）是顶层参数，不放在消息列表里
    - 工具定义的字段名不同（input_schema vs parameters）
    - 回答的 content 是「积木块」数组（文本块/工具调用块混着排），
      不是一整根字符串

    所以本类做双向翻译：请求发出去前转成 Anthropic 方言，回答拿回来后
    再转回 OpenAI 样子。底层用 AsyncAnthropic SDK，全部异步调用。
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "",
        base_url: str = None,
        *,
        auth_token: str = None,
        effort_level: str = None,
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
        request_timeout: float = None,
    ):
        """创建 Anthropic 异步 client。

        认证方式二选一（就像门禁卡有两种刷法）：
        - api_key：走 x-api-key 请求头（Anthropic 官方用这种）
        - auth_token：走 Authorization: Bearer 请求头（DeepSeek 的
          Anthropic 兼容端点用这种）

        参数：
            api_key：Anthropic 官方密钥
            model：模型名
            base_url：API 地址（换端点时用）
            auth_token：Bearer 认证令牌（与 api_key 二选一，优先用它）
            effort_level：思考强度（不思考就让它直接回答）：
                - "max"：85% 的输出额度给思考（推理最强）
                - "high"：50% 给思考
                - "medium"：25% 给思考
                - "low" / None：不思考，直接回答
            stream_idle_timeout：流式看门狗秒数（<=0 关闭）
        """
        from anthropic import AsyncAnthropic
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        # auth_token 优先（DeepSeek 等 Anthropic 兼容端点用 Bearer 认证）
        if auth_token:
            kwargs["auth_token"] = auth_token
        elif api_key:
            kwargs["api_key"] = api_key
        else:
            raise ValueError("AnthropicClient 需要 api_key 或 auth_token")
        # 非流式请求超时（None = SDK 默认；看门狗只护流式）
        self._request_timeout = request_timeout
        if request_timeout is not None:
            kwargs["timeout"] = request_timeout
        self.client = AsyncAnthropic(**kwargs)
        self.model = model
        self.effort_level = (effort_level or "").lower() or None
        # 把密钥/地址各存一份，reset_client 重建时直接用
        # （SDK 不保证让你读回旧值）
        self._api_key = api_key
        self._auth_token = auth_token
        self._base_url = base_url
        # 流式看门狗秒数；<=0 表示关闭
        self.stream_idle_timeout = stream_idle_timeout

    def reset_client(self) -> None:
        """扔掉可能坏掉的连接池，按同一份配置重建 AsyncAnthropic。"""
        try:
            # 尽力关掉旧 client；它已经坏了也无所谓，反正要扔
            import asyncio as _aio
            try:
                loop = _aio.get_running_loop()
                if loop is not None:
                    # 强引用必须自己存（与 OpenAI 版同款）：asyncio 对 task
                    # 只持弱引用，没人引用的 fire-and-forget task 可能被
                    # GC 中途蒸发——半关的池子状态更诡异
                    if not hasattr(self, "_old_close_tasks"):
                        self._old_close_tasks: list = []
                    self._old_close_tasks.append(loop.create_task(self.client.close()))
                    del self._old_close_tasks[:-4]
            except RuntimeError:
                pass
        except Exception:
            pass
        from anthropic import AsyncAnthropic
        kwargs = {}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        # auth_token 优先（与 __init__ 同序），兜底用 api_key
        if self._auth_token:
            kwargs["auth_token"] = self._auth_token
        elif self._api_key:
            kwargs["api_key"] = self._api_key
        if self._request_timeout is not None:
            kwargs["timeout"] = self._request_timeout
        self.client = AsyncAnthropic(**kwargs)

    def close(self) -> None:
        """尽力释放底层 SDK 的 HTTP 连接池（同步版，agent.cleanup() 调）。

        底层的 close 是协程：调用线程不定（主线程收尾、后台线程兜底
        都有），按「线程归属」分流——就在宿主循环上就排个任务异步关；
        其余情况（没在循环里 / 在别的循环线程上）交给宿主循环跑完
        （run_async 阻塞等结果，等于原来的 asyncio.run，但协程落在
        常驻循环上，连接池不会绑到用完即弃的临时循环）。关不上也不
        报错——打扫失败不该影响正事。

        带调用栈的 INFO 日志：close/reset 期间曾出现「幻影关池者」
        （有人把在用的池子关了，受害者只看到 client has been closed），
        栈日志让下一次发生时能直接从 codeagent.log 认出是谁关的。
        """
        try:
            coro = self.client.close()
            if asyncio.iscoroutine(coro):
                logger.info("关闭 LLM 连接池（%s）", self.model,
                            stack_info=True)
                from agent.loop_host import loop_host
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                try:
                    if loop is not None and loop is loop_host.loop:
                        task = asyncio.create_task(coro)
                        self._close_task = task   # 强引用防 GC 蒸发
                    else:
                        # 有界关池：卡死 30 秒后放弃，daemon 退出兜底仍在
                        loop_host.run_async(coro, timeout=30)
                except Exception:
                    # 退出窗口 loop 已停、协程排不进去——coro.close() 消毒，
                    # 不然解释器收尾要甩「coroutine never awaited」警告
                    coro.close()
        except Exception:
            pass

    # effort_level（思考强度）换算成 DeepSeek 的思考参数。
    # 参考: https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
    # 设计取舍：DeepSeek 的 Anthropic 端点不用 Anthropic 原生的
    # budget_tokens，而是用 output_config.effort 控制思考强度。

    def _build_thinking_config(self):
        """返回思考模式的开关参数。

        DeepSeek 默认开着思考模式；effort_level=low（或未设）时必须
        **显式关**——发 None 只是"不表态"，服务商默认开着的话用户拿到
        的还是思考模式。{"type": "disabled"} 是 Anthropic SDK 认可的
        显式关法，其余档位显式开。
        """
        if not self.effort_level or self.effort_level == "low":
            return {"type": "disabled"}
        return {"type": "enabled"}

    def _build_output_config(self):
        """返回 DeepSeek 的思考强度参数（output_config.effort）。

        DeepSeek 的 Anthropic 格式是 output_config={"effort": "max"/"high"}；
        medium 档映射成 high（DeepSeek 只认这两档，就近往上靠）。
        """
        if not self.effort_level or self.effort_level == "low":
            return None
        effort_map = {"max": "max", "high": "high", "medium": "high"}
        return {"effort": effort_map.get(self.effort_level, "high")}

    # OpenAI 风格参数名 → Anthropic 参数名（透传白名单，不认识的静默忽略）
    _KWARG_ALIASES = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop": "stop_sequences",
    }

    def _build_anthropic_kwargs(
        self,
        messages: list,
        tools: Optional[List[dict]],
        max_tokens: int,
        extra_kwargs: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """拼装一次 Anthropic 请求的全部参数（流式和非流式共用）。

        翻译活包括：拼系统提示词、转消息格式、转工具格式、配思考模式
        和强度；抽成公共函数免得两个方法各抄一遍。

        参数：
            messages：OpenAI 格式的消息列表
            tools：OpenAI 格式的工具清单（可为 None）
            max_tokens：本次回答的输出上限
            extra_kwargs：调用方透传的采样参数（temperature/top_p 等，
                按白名单映射后并入；不认识的键静默忽略——对照 OpenAI
                路径的 **kwargs 全透传，这里至少不让常用参数无声蒸发）

        返回：可直接拆开传给 Anthropic SDK 的参数字典。
        """
        system_parts = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system" and m.get("content")
        ]
        system = "\n\n".join(system_parts) if system_parts else None
        conversation = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools(tools) if tools else None

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": conversation,
            "max_tokens": max_tokens,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        # 采样参数透传（白名单映射；值非 None 才带上）
        if extra_kwargs:
            for src, dst in self._KWARG_ALIASES.items():
                val = extra_kwargs.get(src)
                if val is not None:
                    kwargs[dst] = val

        # 思考模式：DeepSeek 格式 = thinking 开关 + output_config 强度，两个参数
        thinking = self._build_thinking_config()
        if thinking:
            kwargs["thinking"] = thinking
        # output_config 是 DeepSeek 的私有扩展，Anthropic SDK 不认识，
        # 只能塞在 extra_body 里捎过去
        output_config = self._build_output_config()
        if output_config:
            kwargs["extra_body"] = {"output_config": output_config}

        return kwargs

    async def chat_completions(self, messages, *, tools=None, **kwargs):
        """非流式调用：调 Anthropic 的 messages.create，再翻译回 OpenAI 样子。

        参数：
            messages：OpenAI 格式消息列表
            tools：工具清单（可省略）
            **kwargs：可含 max_tokens（输出上限，默认 4096）
        """
        max_tokens = kwargs.get("max_tokens", 4096)
        create_kwargs = self._build_anthropic_kwargs(
            messages, tools, max_tokens, extra_kwargs=kwargs,
        )
        response = await self.client.messages.create(**create_kwargs)
        return self._wrap_response(response)

    async def chat_completions_stream(self, messages, *, tools=None, **kwargs):
        """Anthropic 原生流式：用 messages.stream 一边收一边翻译。

        参数：
            messages：OpenAI 格式消息列表
            tools：工具清单（可省略）
            **kwargs：可含 max_tokens（默认 4096）
        """
        max_tokens = kwargs.get("max_tokens", 4096)
        stream_kwargs = self._build_anthropic_kwargs(
            messages, tools, max_tokens, extra_kwargs=kwargs,
        )

        # Anthropic 的流式把一次工具调用拆成很多碎片发过来，
        # 得准备几个「篮子」把每个调用的 id/名字/参数碎片攒齐
        tool_buffers: Dict[int, Dict[str, Any]] = {}
        current_tool_idx: Optional[int] = None
        # 思考内容是否已按分片实时吐过——吐过的话收尾那次汇总 yield
        # 就不能再带（上层是累加语义，带了会全文翻倍）
        streamed_thinking = False

        async with self.client.messages.stream(**stream_kwargs) as stream:
            # 看门狗：90 秒没等到新事件就中止流（async with 保证善后清理）
            async for event in _iterate_with_watchdog(
                stream,
                idle_timeout=self.stream_idle_timeout,
                describe=f"，model={self.model}",
            ):
                evt_type = getattr(event, "type", "")
                if evt_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block is not None and getattr(block, "type", "") == "tool_use":
                        # 一个新的工具调用块开张，登记个新篮子
                        idx = len(tool_buffers)
                        tool_buffers[idx] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            # 参数碎片先攒篮子（list），收尾 join 一次成串
                            "input_json": [],
                        }
                        current_tool_idx = idx
                elif evt_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", "")
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "") or ""
                        if text:
                            yield {
                                "content": text,
                                "tool_calls": [],
                                "finish_reason": None,
                                "usage": None,
                            }
                    elif delta_type == "input_json_delta":
                        # 工具参数的 JSON 被切片发来，一片片往篮子里攒
                        partial = getattr(delta, "partial_json", "") or ""
                        if current_tool_idx is not None and partial:
                            tool_buffers[current_tool_idx]["input_json"].append(partial)
                    elif delta_type == "thinking_delta":
                        # 思考分片实时透传（UI 思考框逐段出，不用等收尾）
                        t = getattr(delta, "thinking", "") or ""
                        if t:
                            streamed_thinking = True
                            yield {
                                "content": "",
                                "tool_calls": [],
                                "finish_reason": None,
                                "usage": None,
                                "reasoning_content": t,
                            }
                elif evt_type == "content_block_stop":
                    current_tool_idx = None

            # 流结束：把攒在篮子里的工具调用拼完整，一次性 yield 出去
            final_message = await stream.get_final_message()
            tool_calls_out = []
            for idx in sorted(tool_buffers.keys()):
                buf = tool_buffers[idx]
                tool_calls_out.append(SimpleNamespace(
                    id=buf["id"],
                    index=idx,
                    type="function",
                    function=SimpleNamespace(
                        name=buf["name"],
                        # 结账：攒的碎片 join 成完整参数串（空篮子 join 出
                        # 空串，照旧走 or "{}" 兜底）
                        arguments="".join(buf["input_json"]) or "{}",
                    ),
                ))
            usage_dict = _extract_anthropic_usage(getattr(final_message, "usage", None))
            # 从最终消息里抠出思考内容（DeepSeek 要求下轮带工具调用时回传它）
            thinking_text = ""
            thinking_sig = ""
            for block in getattr(final_message, "content", []):
                if getattr(block, "type", "") == "thinking":
                    thinking_text += getattr(block, "thinking", "")
                    thinking_sig = getattr(block, "signature", "") or thinking_sig
            # stop_reason 要翻译成 OpenAI 语义再上报（max_tokens→length）：
            # 硬编码 "stop" 会把截断当正常完成，上层升级重试/续写全认不出
            _finish = _anthropic_finish_reason(
                getattr(final_message, "stop_reason", None))
            if tool_calls_out and _finish == "stop":
                _finish = "tool_calls"  # 有工具调用但 stop_reason 缺失时的兜底
            yield {
                "content": "",
                "tool_calls": tool_calls_out,
                "finish_reason": _finish,
                "usage": usage_dict,
                # 分片已实时吐过就不再汇总重复（上层累加语义，重复=翻倍）；
                # 没吐过（服务商不分片/老协议）才在收尾一次性给全文
                "reasoning_content": None if streamed_thinking else (thinking_text or None),
                "thinking_signature": thinking_sig or None,
            }

    def _convert_messages_to_anthropic(self, messages: list) -> list:
        """把 OpenAI 格式的整段对话历史翻译成 Anthropic 格式。

        最关键的一步：把连续多条工具结果合并成一条 user 消息。
        因为 Anthropic 协议规定所有工具结果必须装在同一个 user 消息里，
        不允许两条 user 消息挨着；而 OpenAI 那边每个工具结果各自成条。
        （另外 system 消息在这里被抽走——它要走顶层参数，见上层拼装。）

        参数：
            messages：OpenAI 格式的消息列表

        返回：Anthropic 格式的消息列表。
        """
        conversation = []
        i = 0
        while i < len(messages):
            m = messages[i]
            role = m.get("role")

            if role == "system":
                i += 1
                continue

            if role == "tool":
                # 把连续的工具结果合并成一条 user 消息（Anthropic 的规矩）
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": messages[i].get("tool_call_id", ""),
                        "content": messages[i].get("content") or "",
                    })
                    i += 1
                conversation.append({"role": "user", "content": tool_results})
            else:
                conversation.append(self._convert_message(m))
                i += 1

        return conversation

    def _convert_message(self, msg: dict) -> dict:
        """翻译单条 OpenAI 消息为 Anthropic 消息。

        特别注意：助手消息里带工具调用时，必须把上一轮的思考内容
        （thinking 块）一起回传——这是 DeepSeek 思考模式的硬性要求，
        不带会报错。

        参数：
            msg：一条 OpenAI 格式的消息字典

        返回：一条 Anthropic 格式的消息字典。
        """
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            blocks = []
            # 带工具调用时必须回传思考内容（DeepSeek 思考模式的硬性要求）
            rc = msg.get("reasoning_content")
            sig = msg.get("thinking_signature")
            if rc and sig:
                blocks.append({"type": "thinking", "thinking": rc, "signature": sig})

            has_tool_calls = bool(msg.get("tool_calls"))
            if has_tool_calls or blocks:
                # 有工具调用或思考内容 → 得用「积木块」格式组装
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in (msg.get("tool_calls") or []):
                    try:
                        args = tc.get("function", {}).get("arguments", "{}")
                        if isinstance(args, str):
                            args = json.loads(args)
                    except (json.JSONDecodeError, KeyError):
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": args,
                    })
                return {"role": "assistant", "content": blocks}

        if role == "tool":
            # 工具结果：Anthropic 的说法是 user 角色 + tool_result 积木块
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content or "",
                }],
            }

        # 普通的 user/assistant 文本消息，直接照搬
        return {"role": role, "content": content or ""}

    def _convert_tools(self, openai_tools: List[dict]) -> List[dict]:
        """把 OpenAI 格式的工具说明书翻译成 Anthropic 格式。

        参数：
            openai_tools：OpenAI 格式的工具 schema 列表

        返回：Anthropic 格式（name/description/input_schema）的工具列表。
        """
        anthropic_tools = []
        for t in openai_tools:
            if t.get("type") == "function":
                func = t["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    # 本项目内部 schema 用的键是 input_schema，
                    # 跟 OpenAI 的 parameters 不是同一个名字。两种都认：
                    # 先试 parameters，再试 input_schema，都没有就给个空壳
                    "input_schema": func.get("parameters") or func.get("input_schema") or {
                        "type": "object",
                        "properties": {},
                    },
                })
        return anthropic_tools

    def _wrap_response(self, anthropic_response):
        """把 Anthropic 的回答重新包装成 OpenAI 的样子。

        用 SimpleNamespace（一个能随意挂属性的轻量对象）手工搭出
        choices[0].message 那套结构。思考块的正文和签名也要留出来——
        DeepSeek 要求下一轮带工具调用时回传它们。

        参数：
            anthropic_response：Anthropic SDK 的原始响应对象

        返回：长得像 OpenAI 响应的 SimpleNamespace。
        """
        content_text = ""
        tool_calls = None
        reasoning_content = ""
        thinking_signature = ""

        for block in anthropic_response.content:
            if block.type == "thinking":
                # DeepSeek 思考模式：思考块里装着思维链正文 + 防伪签名
                reasoning_content += getattr(block, "thinking", "")
                thinking_signature = getattr(block, "signature", "") or thinking_signature
            elif block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=block.name,
                        arguments=json.dumps(block.input, ensure_ascii=False),
                    ),
                ))

        message = SimpleNamespace(
            content=content_text if content_text else None,
            tool_calls=tool_calls,
            # 下轮带工具调用时必须把思考内容回传（DeepSeek 的硬性要求）
            reasoning_content=reasoning_content or None,
            thinking_signature=thinking_signature or None,
        )

        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=message,
                # finish_reason 必须翻译带上——上层 detect_length_finish
                # 靠 getattr(choice, "finish_reason") 判截断，缺这个字段
                # Anthropic 路径的截断救援（升级/续写）整条失效
                finish_reason=_anthropic_finish_reason(
                    getattr(anthropic_response, "stop_reason", None)),
            )],
            usage=SimpleNamespace(
                prompt_tokens=getattr(anthropic_response.usage, "input_tokens", 0),
                completion_tokens=getattr(anthropic_response.usage, "output_tokens", 0),
                # 缓存字段用 Anthropic 原名透传——_record_llm_usage 的锚点
                # 口径按字段名判语义（有这俩名字→三项相加），丢了它们
                # Anthropic 路径的缓存记账就失真、锚点退化成单值
                cache_read_input_tokens=getattr(
                    anthropic_response.usage, "cache_read_input_tokens", 0),
                cache_creation_input_tokens=getattr(
                    anthropic_response.usage, "cache_creation_input_tokens", 0),
            ) if hasattr(anthropic_response, "usage") else None,
        )


# ---------------------------------------------------------------------------
# 工厂函数（「按订单造 client」的入口）
# ---------------------------------------------------------------------------

def create_llm_client(model_config: Dict[str, Any]) -> LLMClient:
    """按配置字典造一个对应的 LLM client（项目的统一入口）。

    参数：
        model_config：模型配置字典，长这样——
            {
                "format": "openai" 或 "anthropic"（默认 openai），
                "base_url": "API 地址"，
                "api_key": "密钥"，
                "model": "模型名"，
                另可选 "auth_token"（Bearer 认证）、"effort_level"
                （思考强度）、"stream_idle_timeout"（看门狗秒数）
            }

    返回：OpenAICompatClient 或 AnthropicClient 实例。
    """
    fmt = (model_config.get("format") or "openai").lower()
    api_key = model_config.get("api_key") or ""
    model = model_config.get("model") or ""
    base_url = model_config.get("base_url")
    # 看门狗秒数从配置读；读不到或读出来不是数字就用默认 90 秒（<=0 关闭）
    try:
        idle_timeout = float(model_config.get("stream_idle_timeout", DEFAULT_STREAM_IDLE_TIMEOUT))
    except (TypeError, ValueError):
        idle_timeout = DEFAULT_STREAM_IDLE_TIMEOUT
    # 非流式单请求超时（None = SDK 默认 600s；看门狗只护流式，这里补非流式）
    try:
        request_timeout = float(model_config.get("request_timeout")) \
            if model_config.get("request_timeout") is not None else None
    except (TypeError, ValueError):
        request_timeout = None

    if fmt == "anthropic":
        # auth_token 给 DeepSeek 等兼容端点用（Bearer 认证那种）
        auth_token = model_config.get("auth_token") or ""
        effort_level = model_config.get("effort_level") or ""
        return AnthropicClient(
            api_key=api_key or None,
            auth_token=auth_token or None,
            model=model,
            base_url=base_url,
            effort_level=effort_level or None,
            stream_idle_timeout=idle_timeout,
            request_timeout=request_timeout,
        )

    # 没特别说明就走 OpenAI 兼容格式
    return OpenAICompatClient(
        base_url=base_url, api_key=api_key, model=model,
        stream_idle_timeout=idle_timeout,
        request_timeout=request_timeout,
    )


async def aclose_llm_client(client) -> None:
    """尽力关掉 client 底下的 HTTP 连接（收尾打扫用，关不上也不报错）。

    两种 SDK 的内部对象都有 close 方法（协程），调一下就行；
    出任何异常都吞掉——打扫失败不该影响正事。

    参数：
        client：上面任意一种 LLMClient 实例
    """
    try:
        inner = getattr(client, "client", None)
        if inner is not None and hasattr(inner, "close"):
            result = inner.close()
            if asyncio.iscoroutine(result):
                await result
    except Exception:
        pass
