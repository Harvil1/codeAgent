"""流式并发执行器。

模型流式输出时，一个工具调用的参数其实早早就传完整了，但默认流程
要等整个响应全部输出完才开始跑工具——白白干等。这个模块让"已经收齐
参数的工具调用"提前开跑：只读类工具（read_file、搜索这类，改名为
safe 组）一边模型继续说话一边并发执行；流结束后主循环只需补跑剩下
的 unsafe 部分（串行）。收益是"跑工具的时间"和"模型继续输出的时间"
重叠（DeepSeek 一次发多个工具调用时，第一个不用等全批输出完）。

实现上的关键取舍：
- OpenAI 的增量（delta）格式没有内容块边界（Anthropic 的块格式有明确
  边界），只能靠经验规则猜：**新工具的序号出现 = 前一个
  工具的参数已经收齐**（服务商按序号顺序输出）+ 参数 JSON 能完整解析，
  双重确认；JSON 解析失败就放回正常路径处理
- **只预执行 safe 组**：写文件/跑命令这类 unsafe 工具保持主循环串行
  （副作用有先后顺序，不能因为流式就乱序）；terminal 的只读命令
  动态放宽、也允许预执行
- 中断安全：流出异常（转非流式重试）时，先把所有跑了一半的任务等完
  再丢弃（防僵尸任务）；预执行结果按调用 ID 匹配，没预执行的照常执行

开关：config 的 agent.streaming_tool_execution（默认关——还在灰度）。
整条链路 fail-open：执行器任何异常都退回正常的工具分发路径。
"""
import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _tc_from_buf(buf: dict) -> SimpleNamespace:
    """把流式累积的 buffer 字典包成 tool_call 对象（字段形状和主循环
    分发用的完全一致，这样后续代码可以复用同一套处理逻辑）。

    参数：
        buf: 流式过程攒下来的 {id, name, arguments} 字典

    返回：带 id / type / function.name / function.arguments 属性的对象。
    """
    return SimpleNamespace(
        id=buf.get("id", ""),
        type="function",
        function=SimpleNamespace(
            name=buf.get("name", ""),
            arguments=buf.get("arguments", "") or "{}",
        ),
    )


def _is_preset_safe(tc: SimpleNamespace) -> bool:
    """判断这个工具调用能不能预执行（和主循环的分组逻辑同源）。

    判定规则：工具注册表里标了"可并发安全"的才算；terminal 命令
    动态放宽——只读命令也算安全。注册表里查不到的工具一律按不安全
    处理（宁可慢也不能乱跑）。

    参数：
        tc: tool_call 对象

    返回：True 可以预执行；False 不行（走正常串行路径）。
    """
    try:
        from tools.registry import registry
        entry = registry.get(tc.function.name)
        is_safe = bool(entry.isConcurrencySafe) if entry else False
        if not is_safe and tc.function.name == "terminal":
            args = json.loads(tc.function.arguments or "{}")
            from agent.permission import is_readonly_command
            if is_readonly_command(str(args.get("command", ""))):
                is_safe = True
        return is_safe
    except Exception:
        return False


class StreamingToolExecutor:
    """模型还在流式输出时，抢先执行已收齐参数的工具调用的执行器。

    用法（在 agent/llm_streaming.py 的流式调用里）：
        ex = StreamingToolExecutor(agent) if enabled else None
        # 流循环里：新序号出现 → ex.complete(idx, prev_buf)（预执行）
        # 流正常结束 → results = await ex.collect() → agent 暂存结果
        # 流异常 → await ex.drain()（收尾并丢弃）

    主循环消费（_dispatch_tool_calls）：
        preset = self._pop_streaming_preset()  # {调用ID: 结果文本}
        已预执行过的调用跳过执行，结果直接并入汇总。
    """

    def __init__(self, agent):
        self._agent = agent
        self._tasks: list = []       # 已启动的 [(tool_call, 异步任务)]
        self._results: Dict[str, str] = {}  # 调用ID → 执行结果文本

    @property
    def has_pending(self) -> bool:
        return bool(self._tasks)

    def complete(self, idx: int, buf: dict) -> None:
        """报告"某个工具调用的参数已收齐"（新序号出现或流结束时调）。

        只预执行"safe 且参数 JSON 能完整解析"的调用，其余留给正常路径。
        任何异常都静默吞掉（不影响主流程）。

        参数：
            idx: 这个工具在流里的序号（只用于日志）
            buf: 攒好的 {id, name, arguments} 字典
        """
        try:
            tc = _tc_from_buf(buf)
            if not tc.id or not tc.function.name:
                return
            # 参数 JSON 再校验一遍（"新序号=前序号收齐"是经验规则，服务商
            # 不一定守约——坏 JSON 就放回正常路径处理）
            try:
                json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                logger.debug(
                    "流式预执行跳过（arguments JSON 未完整）: %s idx=%s",
                    tc.function.name, idx,
                )
                return
            if not _is_preset_safe(tc):
                return  # 不安全的工具留给主循环串行跑
            # 工具前置回调（和并发安全组的处理顺序保持一致）
            self._agent._run_tool_pre_callbacks(tc)
            task = asyncio.create_task(self._run(tc))
            self._tasks.append((tc, task))
            logger.info("流式预执行启动: %s（idx=%s）", tc.function.name, idx)
        except Exception as e:
            logger.warning("流式预执行启动失败（fail-open）: %s", e)

    async def _run(self, tc) -> str:
        """实际预执行一个 safe 工具。

        参数、上下文（会话/记忆/hook 等）全按正常路径那套传——保证预执行
        的行为和主循环执行完全一样，不会因为走捷径而缺东西。

        参数：
            tc: tool_call 对象

        返回：工具产出的结果文本（JSON 字符串）。
        """
        from model_tools import handle_function_call
        tool_args = json.loads(tc.function.arguments or "{}")
        agent = self._agent
        return await handle_function_call(
            tc.function.name, tool_args,
            session_id=agent.session_id,
            memory_store=agent.memory_store,
            session_store=agent.session_store,
            codeagent_home=agent.codeAgent_home,
            tool_call_id=tc.id,
            config=agent.config,
            hooks_registry=agent.hooks_registry,
            bg_manager=agent.bg_manager,
            team_bus=agent.team_bus,
            team_coordinator=agent.team_coordinator,
            team_name=agent.team_name,
            agent_ref=agent,
        )

    async def collect(self) -> Dict[str, str]:
        """等所有跑了一半的预执行任务全部完成。

        返回：{调用ID: 结果文本}；某个任务抛了异常就转成 JSON 错误文本
        放进结果里（保证每个 ID 都有条目，不炸整体）。
        """
        if not self._tasks:
            return self._results
        try:
            outcomes = await asyncio.gather(
                *[t for _, t in self._tasks], return_exceptions=True,
            )
            for (tc, _t), outcome in zip(self._tasks, outcomes):
                if isinstance(outcome, Exception):
                    self._results[tc.id] = json.dumps({
                        "error": f"streaming preset failed: {outcome}",
                        "error_type": "streaming_preset_error",
                    }, ensure_ascii=False)
                else:
                    self._results[tc.id] = outcome
        except Exception as e:
            # 结果整体不可用（调用方会丢弃）——read 去重账要撤，
            # 否则重试重发同样的 read 只会拿到「文件未变」空提示
            self._undo_read_dedup()
            logger.warning("流式预执行 collect 失败: %s", e)
        finally:
            self._tasks.clear()
        return self._results

    async def drain(self) -> None:
        """流出异常时的收尾：等跑了一半的任务结束，但结果全扔掉。

        直接弃置会留僵尸任务在后台漂着，必须等完再扔。这条路径的结果
        不会被使用（要走非流式重试了）。
        """
        try:
            if self._tasks:
                await asyncio.gather(
                    *[t for _, t in self._tasks], return_exceptions=True,
                )
        except Exception:
            logger.warning("异常被吞(fail-open)", exc_info=True)
        finally:
            # 结果要被扔掉了——预执行 read 记的去重账一并撤
            # （模型从没见过内容，重试重读必须给全文）
            self._undo_read_dedup()
            self._tasks.clear()
            self._results = {}

    def _undo_read_dedup(self) -> None:
        """撤销本次预执行 read_file 的「读过去重」记账（结果被丢弃时用）。

        预执行成功会把 (路径, 范围) 记进 file_operations._READ_SEEN；
        如果之后流异常 drain / collect 失败，结果被扔掉而模型从没见过
        内容——账不撤的话，重试重发同样的 read 只会拿到「文件未变，
        不再返回全文」的省 token 提示，模型被误导以为已经看过。
        正常 collect 保留不动（结果会被模型看到，后续重读给
        unchanged 提示才是对的）。fail-open 全吞。
        """
        try:
            from tools.file_operations import _read_seen_invalidate
            for tc, _t in self._tasks:
                try:
                    if tc.function.name != "read_file":
                        continue
                    args = json.loads(tc.function.arguments or "{}")
                    path = args.get("path", "")
                    if path:
                        _read_seen_invalidate(path)
                except Exception:
                    continue
        except Exception as e:
            logger.warning("read 去重撤销失败（fail-open）: %s", e)
