"""外部记忆 provider（记忆服务后端）的抽象基类。

想接第三方的记忆服务（Honcho / Mem0 / Supermemory 等），就实现这个
接口，然后交给 MemoryManager（agent/memory_manager.py）统一编排。
打个比方：这是"记忆服务供应商"的岗位说明书，谁来干这活都得会这几样。

MemoryManager 按下面的生命周期顺序调用（谁在什么时候被叫）：
  initialize()          - 连接后端，创建资源
  system_prompt_block() - 返回注入 system prompt 的静态文本
  prefetch(query)       - 每轮 API 调用前召回上下文
  sync_turn(user, asst) - 每轮结束后异步写入
  get_tool_schemas()    - 暴露给 LLM 的工具
  handle_tool_call()    - 处理工具调用
  shutdown()            - 清理退出

可选钩子（不实现也有默认空实现）：
  on_turn_start()       - 每轮开始
  on_session_end()      - 会话结束时的总结提取
  on_session_switch()   - session_id 切换（/resume, /reset 时）
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryProvider(ABC):
    """记忆服务后端的抽象接口：子类实现具体的外部记忆服务对接。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """provider 的标识名（如 'builtin'、'honcho'），日志和调试用。"""

    @abstractmethod
    def is_available(self) -> bool:
        """配置是否就绪。只查本地配置，不发网络请求（要快、要便宜）。"""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """会话开始时初始化：建立连接、创建所需资源。

        参数：
        - session_id：本次会话 ID（provider 用它做会话隔离）
        - **kwargs：额外初始化参数
        """

    def system_prompt_block(self) -> str:
        """返回要注入 system prompt 的静态文本。默认空（不注入）。

        注意"静态"：这段文本会话中途不能变，否则击穿 prompt 缓存。
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """每轮 API 调用前召回相关上下文，返回上次缓存好的结果。

        设计要求：这个方法必须立刻返回，不能卡住主循环——真正的后端
        查询应该放到后台做（配合 queue_prefetch），这里只交出上一轮
        预取好的缓存。

        参数：
        - query：当前用户输入
        - session_id：会话 ID

        返回：召回的上下文文本；默认空。
        """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """为下一轮排队后台预取（先去查，下一轮 prefetch 来取）。

        参数：
        - query：当前用户输入（预判下一轮需要什么）
        - session_id：会话 ID

        返回：无。默认无操作。
        """

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """持久化一个刚完成的对话轮次。实现应当非阻塞。

        参数：
        - user_content：本轮用户输入
        - assistant_content：本轮 AI 回复
        - session_id：会话 ID
        - messages：完整消息列表（可选，给需要上下文的 provider 用）
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回暴露给 LLM 的工具 schema 列表（OpenAI function 格式）。"""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理 LLM 发来的工具调用。

        参数：
        - tool_name：工具名（来自 get_tool_schemas 里暴露的那些）
        - args：工具参数
        - **kwargs：额外上下文

        返回：JSON 字符串（项目工具契约）。默认抛 NotImplementedError
        （不提供工具的 provider 不会被调到）。
        """
        raise NotImplementedError(f"Provider {self.name} 不处理工具 {tool_name}")

    def shutdown(self) -> None:
        """清理退出（断连接、释放资源）。默认无操作。"""

    # 以下都是可选钩子：不实现就是空操作
    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """每轮开始时的钩子。参数：turn_number 轮次号；message 本轮消息。"""
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """会话结束钩子（可做总结提取）。参数：messages 完整消息列表。"""
        pass

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        """会话切换钩子（/resume、/reset 时）。参数：new_session_id 新会话 ID。"""
        pass
