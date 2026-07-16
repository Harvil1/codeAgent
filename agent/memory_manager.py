"""记忆系统编排器。

职责：
1. 管理 MemoryStore（内置文件记忆）
2. 管理单个外部 provider（如有配置）
3. 在正确的时机调用各方法
4. 后台执行写入，不阻塞主循环

关键限制：同时只激活一个外部 provider（防止工具冲突和记忆后端冲突）。
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from agent.memory_provider import MemoryProvider
from agent.memory_store import MemoryStore

logger = logging.getLogger(__name__)


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
    ):
        self.memory_store = memory_store
        self.external_provider = external_provider

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
        """钩子：压缩前调用。Phase 1 留空（no-op），未来扩展用。

        设计原因：HarvilAgent 当前记忆模型是主动式（LLM 通过 memory_tool 自己写），
        强行加 LLM 被动抽取会和现有模型冲突。Phase 5（或独立 Phase 1.5）实现。
        """
        pass

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
