"""外部记忆 provider 的抽象基类。

外部 provider（Honcho/Mem0/Supermemory 等）实现这个接口，
通过 MemoryManager 编排。

生命周期（由 MemoryManager 调用）：
  initialize()          - 连接后端，创建资源
  system_prompt_block() - 返回注入 system prompt 的静态文本
  prefetch(query)       - 每轮 API 调用前召回上下文
  sync_turn(user, asst) - 每轮结束后异步写入
  get_tool_schemas()    - 暴露给 LLM 的工具
  handle_tool_call()    - 处理工具调用
  shutdown()            - 清理退出

可选钩子：
  on_turn_start()       - 每轮开始
  on_session_end()      - 会话结束时的总结提取
  on_session_switch()   - session_id 切换（/resume, /reset）
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryProvider(ABC):
    """记忆 provider 抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 标识（如 'builtin', 'honcho'）。"""

    @abstractmethod
    def is_available(self) -> bool:
        """是否配置就绪（不发网络请求，只检查配置）。"""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """会话开始时初始化（建立连接、创建资源）。"""

    def system_prompt_block(self) -> str:
        """返回注入 system prompt 的静态文本。默认空。"""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """每轮 API 调用前召回相关上下文。应快速返回缓存结果。

        实际的后端查询应该后台进行，这里返回上次缓存的结果。
        """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """为下一轮排队后台预取。默认无操作。"""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """持久化一个完成的对话轮次。应非阻塞。"""

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回暴露给 LLM 的工具 schema 列表。"""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理工具调用，返回 JSON 字符串。"""
        raise NotImplementedError(f"Provider {self.name} 不处理工具 {tool_name}")

    def shutdown(self) -> None:
        """清理退出。"""

    # 可选钩子
    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        pass
