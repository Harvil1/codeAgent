"""记忆系统的总调度员（编排器）。

记忆（AI 对用户/项目沉淀下来的事实条目，跨会话保留）由两路来源提供：
内置的文件记忆库（MemoryStore）和可选的外部记忆服务（provider）。
本文件站在两者之上做统一调度，供 agent 主循环使用。

它管这几件事（打个比方：像餐厅经理，不管做菜但决定谁什么时候上）：
1. 手下管着 MemoryStore（自家文件记忆库）
2. 手下管着至多一个外部 provider（外包记忆服务，如有配置）
3. 在合适的时机调用各自的接口（会话开始、每轮前后、会话结束）
4. 写入类操作全部扔到后台线程做，主对话循环不等它
5. 上下文压缩前自动让 LLM 顺手提炼几条稳定事实存起来

关键限制：同时只激活一个外部 provider——多了会在工具层面和记忆后端层面
打架（两边都写、内容冲突）。
"""

import contextvars
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from agent.memory_store import MemoryStore

logger = logging.getLogger(__name__)


# 提取 prompt：压缩前让 LLM 从对话里挑值得长期记的事实
_EXTRACTION_PROMPT = """从以下对话中提取值得长期记住的稳定事实（用户偏好、环境细节、项目约定）。忽略临时任务进度和一次性操作。输出 JSON 数组 [{{type, name, description, body}}]，最多 5 条。无则输出 []。

type 必须是以下之一: user, feedback, project, reference, other
name: 简短标识（≤30 字符）
description: 一句话描述
body: 完整内容

对话内容：

<conversation>
{conversation_text}
</conversation>
"""


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """后台"守护"线程池：进程退出时不会被它卡住。

    只放一个工作线程是有意的——保证写入按顺序执行
    （第 N 轮的写入必须先于第 N+1 轮完成，否则后写的可能被先写的覆盖顺序打乱）。
    """

    def __init__(self, max_workers: int = 1):
        super().__init__(max_workers=max_workers)


class MemoryManager:
    """记忆系统总调度员：统一管理内置记忆库和外部 provider 的调用时机。"""

    def __init__(
        self,
        memory_store: MemoryStore,
        external_provider: Optional[MemoryProvider] = None,
        *,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
    ):
        """组装调度员（主循环只面对这一个对象，不关心底下是内置库还是外部服务）。

        参数：
        - memory_store：内置文件记忆库（必填，自家数据）
        - external_provider：外部记忆服务（可选，没配就传 None）
        - llm_client：LLM 客户端，供压缩前自动提取用（可选，不传就不做提取）
        - llm_model：提取时用的模型名（可选，None 走客户端默认）
        """
        self.memory_store = memory_store
        self.external_provider = external_provider
        # on_pre_compress（压缩前钩子）做 LLM 提取要用，可选
        self._llm_client = llm_client
        self._llm_model = llm_model

        # 后台写入执行器：单线程是有意的，保证写入顺序不乱
        self._sync_executor = DaemonThreadPoolExecutor(max_workers=1)
        self._initialized = False
        self._session_id: Optional[str] = None

    def initialize(self, session_id: str, **kwargs) -> None:
        """会话开始时初始化外部 provider（建连接、准备资源）。

        参数：
        - session_id：本次会话的 ID，传给 provider 做会话隔离
        - **kwargs：透传给 provider 的额外初始化参数

        返回：无。初始化失败只记日志不抛错（外部服务挂了不该连累主对话）。
        """
        self._session_id = session_id
        if self.external_provider and self.external_provider.is_available():
            try:
                self.external_provider.initialize(session_id, **kwargs)
                self._initialized = True
                logger.info("外部记忆 provider 已激活: %s", self.external_provider.name)
            except Exception as e:
                logger.warning("外部记忆 provider 初始化失败: %s", e)

    def on_pre_compress(self, snapshot_path, messages: list) -> None:
        """钩子：上下文压缩前被调用，让 LLM 从对话里提炼稳定事实存档（压缩会把旧对话摘要掉，长期信息不先捞出来就丢了）。

        做法：
        - 消息少于 6 条（太短没东西可提炼）直接返回
        - 把 messages 交给 LLM 挑出值得长期记的事实
        - 逐条存进 memory_store
        - 整条链路 try/except 包住（fail-open：提取失败绝不影响压缩本身）
        - 提交到后台线程池跑，主循环不等待

        参数：
        - snapshot_path：压缩快照路径（本方法不使用，钩子签名带上的）
        - messages：当前完整对话历史

        返回：无。

        注意（项目铁律）：记忆写完立即落盘，但要到下次会话才会注入给 LLM
        ——中途改 system prompt 会让 prompt 缓存失效、成本翻倍。
        """
        if self._llm_client is None or self.memory_store is None:
            return
        if not messages or len(messages) < 6:
            return
        # 扔后台线程跑，主循环继续做压缩不被拖住。ThreadPoolExecutor
        # 不拷贝 contextvars（不像 asyncio.to_thread 有自动复制）——
        # 手动 copy_context 快照 + ctx.run 包线程入口（对齐 reflection.py
        # 后台线程的姿势），否则会话内切过工作目录后，后台提取写项目
        # 记忆会落错项目区
        _ctx = contextvars.copy_context()
        self._sync_executor.submit(
            _ctx.run, self._safe_extract_and_save, messages)

    def _safe_extract_and_save(self, messages: list) -> None:
        """后台线程里实际干活的入口：提取 + 保存，所有异常吞掉。

        参数：
        - messages：对话历史

        返回：无（失败只记日志）。
        """
        try:
            self._extract_and_save(messages)
        except Exception as e:
            logger.warning("on_pre_compress 提取失败（fail-open）: %s", e)

    def _extract_and_save(self, messages: list) -> None:
        """同步执行：调 LLM 提取事实并逐条保存。

        参数：
        - messages：对话历史（会被序列化成纯文本喂给 LLM）

        返回：无。
        """
        # 第 1 步：把消息列表摊平成纯文本
        conversation_text = self._serialize_messages(messages)
        if not conversation_text.strip():
            return

        # 第 2 步：调 LLM 做提取
        prompt = _EXTRACTION_PROMPT.format(
            # 保头截尾——头部正是 L4 即将摘要掉的早期内容，窗口对准将被
            # 压缩的部分（R11 快照语义后此方向变关键）
            conversation_text=conversation_text[:8000],
        )
        try:
            # chat_completions 是 async 的，而本函数跑在后台线程池（不在
            # 宿主循环线程）——同步调它只会拿到 coroutine，随后
            # response.choices 直接 AttributeError，提取每次必失败。
            # 交给进程级常驻循环宿主同步等结果（照抄 reflection.py 的
            # run_reflection 姿势）；后台线程长活，豁免回合栅栏
            from agent.loop_host import loop_host
            response = loop_host.run_async(
                self._llm_client.chat_completions(
                    [{"role": "user", "content": prompt}],
                    model=self._llm_model,
                ),
                exempt_from_fence=True,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("on_pre_compress LLM 调用失败: %s", e)
            return

        # 第 3 步：解析 JSON（模型常在 JSON 外多说话，要容错）
        facts = self._parse_facts(content)
        if not facts:
            return

        # 第 4 步：逐条保存（字段缺失/类型非法的单条跳过，不连累其他）
        valid_types = {"user", "feedback", "project", "reference", "other"}
        saved = 0
        for fact in facts[:5]:  # 上限 5 条，防止一次灌太多
            try:
                ftype = fact.get("type", "other")
                if ftype not in valid_types:
                    ftype = "other"
                name = fact.get("name", "").strip()
                description = fact.get("description", "").strip()
                body = fact.get("body", "").strip()
                if not name or not description:
                    continue
                self.memory_store.save(
                    name=name[:60],
                    description=description[:200],
                    type=ftype,
                    body=body,
                    source_session_id=self._session_id or "",
                    source="self",  # 模型自提取，未经用户确认（索引不戴 ⭐ 不置顶）
                )
                saved += 1
            except Exception as e:
                logger.warning("on_pre_compress 保存单条失败: %s", e)
        if saved > 0:
            logger.info("on_pre_compress 提取并保存了 %d 条记忆", saved)

    @staticmethod
    def _serialize_messages(messages: list) -> str:
        """把消息列表摊平成"[角色] 内容"的纯文本，供 LLM 阅读（复杂结构只取文字部分）。

        参数：
        - messages：对话历史（OpenAI 消息格式）

        返回：每条一行的纯文本；空内容的消息跳过。
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            if isinstance(content, list):
                # 结构化 content（如多段文本块）：只拼出里面的文字
                content = " ".join(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b)
                    for b in content
                )
            content = str(content).strip()
            if not content:
                continue
            parts.append(f"[{role}] {content[:500]}")
        return "\n".join(parts)

    @staticmethod
    def _parse_facts(content: str) -> List[Dict[str, Any]]:
        """把 LLM 回复解析成事实列表：先试整体 JSON 解析，失败再从文本里捞 [ ... ] 片段（容忍 JSON 外的多余文字）。

        参数：
        - content：LLM 的原始回复文本

        返回：事实字典列表；解析不出返回空列表。
        """
        # 先试整段直接解析
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        # 退而求其次：正则抠出第一个 [ ... ] 再解析
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group(0))
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    def build_system_prompt(self) -> str:
        """拼出要塞进 system prompt 的记忆部分——只含外部 provider 的静态块（会话开始时确定、之后不变，不伤 prompt 缓存；内置记忆不进 prompt）。

        返回：拼接后的文本；没有可用内容时返回空串。
        """
        parts = []

        # 外部 provider 的静态说明块
        if self.external_provider and self._initialized:
            ext_block = self.external_provider.system_prompt_block()
            if ext_block:
                parts.append(ext_block)

        return "\n\n".join(parts)

    def sync_all(self, user_content: str, assistant_content: str) -> None:
        """每轮对话结束后调用：把这一轮写进外部 provider（后台异步）。

        参数：
        - user_content：本轮用户说的话
        - assistant_content：本轮 AI 的回复

        返回：无（写入扔后台线程，主循环不等）。
        """
        if not self.external_provider or not self._initialized:
            return
        # 后台线程写，不拖主循环
        self._sync_executor.submit(self._safe_sync_turn, user_content, assistant_content)

    def _safe_sync_turn(self, user_content: str, assistant_content: str) -> None:
        """后台线程里安全地同步一轮到外部 provider（异常全捕获）。

        参数：
        - user_content：本轮用户输入
        - assistant_content：本轮 AI 回复

        返回：无（失败只记日志）。
        """
        try:
            self.external_provider.sync_turn(
                user_content, assistant_content,
                session_id=self._session_id or "",
            )
        except Exception as e:
            logger.warning("sync_turn 失败: %s", e)

    def shutdown(self) -> None:
        """会话结束时清理：通知 provider 关门、关掉后台线程池。

        返回：无（清理失败也静默，退出路径不能再抛错）。
        """
        if self.external_provider and self._initialized:
            try:
                self.external_provider.shutdown()
            except Exception:
                pass
                logger.warning("异常被吞(fail-open)", exc_info=True)
        self._sync_executor.shutdown(wait=False)
