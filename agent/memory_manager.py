"""记忆系统编排器。

职责：
1. 管理 MemoryStore（内置文件记忆）
2. 管理单个外部 provider（如有配置）
3. 在正确的时机调用各方法
4. 后台执行写入，不阻塞主循环
5. 压缩前自动提取稳定事实（batch2-T1）

关键限制：同时只激活一个外部 provider（防止工具冲突和记忆后端冲突）。
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from agent.memory_store import MemoryStore

logger = logging.getLogger(__name__)


# batch2-T1: 压缩前自动记忆提取的 prompt
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
    """守护线程池：进程退出时不阻塞。

    单工作线程保证写入顺序（turn N 必须在 turn N+1 前完成）。
    """

    def __init__(self, max_workers: int = 1):
        super().__init__(max_workers=max_workers)


class MemoryManager:
    """记忆系统编排器。"""

    def __init__(
        self,
        memory_store: MemoryStore,
        external_provider: Optional[MemoryProvider] = None,
        *,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
    ):
        self.memory_store = memory_store
        self.external_provider = external_provider
        # batch2-T1: 用于 on_pre_compress 的 LLM 调用（可选）
        self._llm_client = llm_client
        self._llm_model = llm_model

        # 后台执行器：单线程，保证写入顺序
        self._sync_executor = DaemonThreadPoolExecutor(max_workers=1)
        self._initialized = False
        self._session_id: Optional[str] = None

    def initialize(self, session_id: str, **kwargs) -> None:
        """会话开始时初始化。"""
        self._session_id = session_id
        if self.external_provider and self.external_provider.is_available():
            try:
                self.external_provider.initialize(session_id, **kwargs)
                self._initialized = True
                logger.info("外部记忆 provider 已激活: %s", self.external_provider.name)
            except Exception as e:
                logger.warning("外部记忆 provider 初始化失败: %s", e)

    def on_pre_compress(self, snapshot_path, messages: list) -> None:
        """钩子：压缩前调用。batch2-T1 实现 LLM 后台评审提取稳定事实。

        策略：
        - messages 少于 6 条（太短不值得），return
        - 用 LLM 分析 messages，提取稳定事实
        - 对每条调 memory_store.save(...) 保存
        - 完全 try/except 包（fail-open，失败不影响压缩）
        - 异步执行（提交到 _sync_executor 后台线程池）

        注意：记忆写入立即落盘，但下次会话才注入（保护 prompt cache）。
        """
        if self._llm_client is None or self.memory_store is None:
            return
        if not messages or len(messages) < 6:
            return
        # 提交到后台线程，不阻塞主循环
        self._sync_executor.submit(self._safe_extract_and_save, messages)

    def _safe_extract_and_save(self, messages: list) -> None:
        """后台执行：LLM 提取事实 + 保存（所有异常被吞）。"""
        try:
            self._extract_and_save(messages)
        except Exception as e:
            logger.warning("on_pre_compress 提取失败（fail-open）: %s", e)

    def _extract_and_save(self, messages: list) -> None:
        """同步执行 LLM 提取 + 保存。"""
        # 1. 把 messages 序列化为文本
        conversation_text = self._serialize_messages(messages)
        if not conversation_text.strip():
            return

        # 2. 调 LLM 提取
        prompt = _EXTRACTION_PROMPT.format(
            conversation_text=conversation_text[:8000],  # 防止过长
        )
        try:
            response = self._llm_client.chat_completions(
                [{"role": "user", "content": prompt}],
                model=self._llm_model,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("on_pre_compress LLM 调用失败: %s", e)
            return

        # 3. 解析 JSON 数组（容忍模型输出多余文本）
        facts = self._parse_facts(content)
        if not facts:
            return

        # 4. 保存每条事实
        valid_types = {"user", "feedback", "project", "reference", "other"}
        saved = 0
        for fact in facts[:5]:  # 最多 5 条
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
                )
                saved += 1
            except Exception as e:
                logger.debug("on_pre_compress 保存单条失败: %s", e)
        if saved > 0:
            logger.info("on_pre_compress 提取并保存了 %d 条记忆", saved)

    @staticmethod
    def _serialize_messages(messages: list) -> str:
        """把 messages 列表序列化为纯文本（截取 role + content）。"""
        parts = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            if isinstance(content, list):
                # tool_calls 等复杂 content，取文本部分
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
        """解析 LLM 输出为事实列表。容忍多余文本。"""
        # 尝试直接 parse
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        # 尝试提取 [ ... ] 子串
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group(0))
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    def build_system_prompt(self) -> str:
        """组装要注入 system prompt 的记忆部分。"""
        parts = []

        # 内置记忆（frozen snapshot）
        if self.memory_store:
            snapshot = self.memory_store.snapshot_for_prompt()
            if snapshot:
                parts.append(snapshot)

        # 外部 provider 的静态块
        if self.external_provider and self._initialized:
            ext_block = self.external_provider.system_prompt_block()
            if ext_block:
                parts.append(ext_block)

        return "\n\n".join(parts)

    def prefetch_all(self, query: str) -> str:
        """每轮 API 调用前调用，返回召回的上下文。"""
        if not self.external_provider or not self._initialized:
            return ""
        try:
            return self.external_provider.prefetch(query, session_id=self._session_id or "")
        except Exception as e:
            logger.debug("预取失败: %s", e)
            return ""

    def sync_all(self, user_content: str, assistant_content: str) -> None:
        """每轮结束后调用。后台异步写入外部 provider。"""
        if not self.external_provider or not self._initialized:
            return
        # 提交到后台线程，不阻塞主循环
        self._sync_executor.submit(self._safe_sync_turn, user_content, assistant_content)

    def _safe_sync_turn(self, user_content: str, assistant_content: str) -> None:
        """安全地同步一轮（捕获所有异常）。"""
        try:
            self.external_provider.sync_turn(
                user_content, assistant_content,
                session_id=self._session_id or "",
            )
        except Exception as e:
            logger.warning("sync_turn 失败: %s", e)

    def queue_prefetch_all(self, query: str) -> None:
        """为下一轮排队后台预取。"""
        if not self.external_provider or not self._initialized:
            return
        try:
            self.external_provider.queue_prefetch(query, session_id=self._session_id or "")
        except Exception as e:
            logger.debug("queue_prefetch 失败: %s", e)

    def shutdown(self) -> None:
        """会话结束时清理。"""
        if self.external_provider and self._initialized:
            try:
                self.external_provider.shutdown()
            except Exception:
                pass
        self._sync_executor.shutdown(wait=False)
