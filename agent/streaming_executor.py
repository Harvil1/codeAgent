"""流式并发执行（R23 #7，对齐 CC StreamingToolExecutor）。

模型流式输出期间，**已完整到达的 tool_call 立即执行**——safe 工具
（read_file/search 等只读）create_task 并发预执行，流结束后主循环只补跑
unsafe 串行部分。收益：工具执行延迟与模型继续输出的时间重叠
（DeepSeek 多 tool_call 场景首 call 不必等全批输出完）。

与 CC 的差异（如实记录）：
- CC 是 Anthropic 格式（content_block_stop 显式边界）；OpenAI delta 格式
  无块边界，用启发式：**新 index 出现 = 前 index 参数完整**（provider 按
  index 顺序输出）+ arguments JSON 可解析双重确认，解析失败交正常路径
- **只预执行 safe 组**：unsafe（write/terminal 等）保持主循环串行语义
  （顺序副作用不因流式乱序）；terminal 只读命令按 T7 动态放宽进预执行
- 中断安全：流异常（fallback 非流式）时 drain 全部 in-flight task 再弃用
  （防僵尸任务）；预执行结果按 tc.id 匹配，未预执行的照常执行

门控：config agent.streaming_tool_execution（默认 False——灰度）。
整链 fail-open：executor 任何异常退回正常 dispatch 路径。
"""
import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _tc_from_buf(buf: dict) -> SimpleNamespace:
    """累积 buffer → tool_call 对象（与 _dispatch_tool_calls 的形态一致）。"""
    return SimpleNamespace(
        id=buf.get("id", ""),
        type="function",
        function=SimpleNamespace(
            name=buf.get("name", ""),
            arguments=buf.get("arguments", "") or "{}",
        ),
    )


def _is_preset_safe(tc: SimpleNamespace) -> bool:
    """预执行安全性判定（与 _dispatch_tool_calls 分组同源）。

    registry.isConcurrencySafe + terminal 只读命令动态放宽（T7）。
    查不到 registry 的工具按 unsafe（fail-closed）。
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
    """流式期间的 tool_call 预执行器。

    用法（_call_llm_streaming 内）：
        ex = StreamingToolExecutor(agent) if enabled else None
        # 流循环里：新 index 出现 → ex.complete(idx, prev_buf)（预执行）
        # 流正常结束 → results = await ex.collect()  → agent 侧暂存
        # 流异常 → await ex.drain()（收尾丢弃）

    主循环消费（_dispatch_tool_calls）：
        preset = self._pop_streaming_preset()  # {tc_id: content}
        已预执行的 call 跳过执行，结果直接进 merge。
    """

    def __init__(self, agent):
        self._agent = agent
        self._tasks: list = []       # [(tc, asyncio.Task)]
        self._results: Dict[str, str] = {}  # tc_id → content

    @property
    def has_pending(self) -> bool:
        return bool(self._tasks)

    def complete(self, idx: int, buf: dict) -> None:
        """一个 tool_call 的 arguments 已完整（新 index 出现/流结束）。

        只预执行 safe + JSON 可解析的 call；其余留正常路径。fail-open。
        """
        try:
            tc = _tc_from_buf(buf)
            if not tc.id or not tc.function.name:
                return
            # JSON 双重确认（启发式边界可能被 provider 违反——坏 JSON 交正常路径）
            try:
                json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                logger.debug(
                    "流式预执行跳过（arguments JSON 未完整）: %s idx=%s",
                    tc.function.name, idx,
                )
                return
            if not _is_preset_safe(tc):
                return  # unsafe 保持主循环串行
            # pre-callback（与 _run_safe_group_concurrently 顺序版同款）
            self._agent._run_tool_pre_callbacks(tc)
            task = asyncio.create_task(self._run(tc))
            self._tasks.append((tc, task))
            logger.info("流式预执行启动: %s（idx=%s）", tc.function.name, idx)
        except Exception as e:
            logger.debug("流式预执行启动失败（fail-open）: %s", e)

    async def _run(self, tc) -> str:
        """预执行单个 safe 工具（handle_function_call 全 ctx 与正常路径一致）。"""
        from model_tools import handle_function_call
        tool_args = json.loads(tc.function.arguments or "{}")
        agent = self._agent
        return await handle_function_call(
            tc.function.name, tool_args,
            session_id=agent.session_id,
            memory_store=agent.memory_store,
            session_store=agent.session_store,
            omnimate_home=agent.omnimate_home,
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
        """等待全部 in-flight 完成。返回 {tc_id: content}（异常转 JSON error）。"""
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
            logger.warning("流式预执行 collect 失败: %s", e)
        finally:
            self._tasks.clear()
        return self._results

    async def drain(self) -> None:
        """流异常路径收尾：等 in-flight 完成但丢弃结果（防僵尸任务）。"""
        try:
            if self._tasks:
                await asyncio.gather(
                    *[t for _, t in self._tasks], return_exceptions=True,
                )
        except Exception:
            pass
        finally:
            self._tasks.clear()
            self._results = {}
