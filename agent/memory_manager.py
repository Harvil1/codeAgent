"""记忆系统的总调度员（编排器）。

记忆（AI 对用户/项目沉淀下来的事实条目，跨会话保留）由两路来源提供：
内置的文件记忆库（MemoryStore）和可选的外部记忆服务（provider）。
本文件站在两者之上做统一调度，供 agent 主循环使用。

它管这几件事（打个比方：像餐厅经理，不管做菜但决定谁什么时候上）：
1. 手下管着 MemoryStore（自家文件记忆库）
2. 手下管着至多一个外部 provider（外包记忆服务，如有配置）
3. 在合适的时机调用各自的接口（会话开始、每轮前后、会话结束）
4. 写入类操作全部扔到后台线程做，主对话循环不等它
5. 上下文压缩前自动让 LLM 顺手提炼几条稳定事实存起来（batch2-T1 特性）

关键限制：同时只激活一个外部 provider——多了会在工具层面和记忆后端层面
打架（两边都写、内容冲突）。
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from agent.memory_store import MemoryStore

logger = logging.getLogger(__name__)


# batch2-T1 特性用的提取 prompt：压缩前让 LLM 从对话里挑值得长期记的事实
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
        """组装调度员。

        背景：主循环只需要面对这一个对象，不必关心底下是内置库还是外部服务。

        参数：
        - memory_store：内置文件记忆库（必填，自家数据）
        - external_provider：外部记忆服务（可选，没配就传 None）
        - llm_client：LLM 客户端，供压缩前自动提取用（可选，不传就不做提取）
        - llm_model：提取时用的模型名（可选，None 走客户端默认）
        """
        self.memory_store = memory_store
        self.external_provider = external_provider
        # batch2-T1：on_pre_compress（压缩前钩子）做 LLM 提取要用，可选
        self._llm_client = llm_client
        self._llm_model = llm_model

        # 后台写入执行器：单线程是有意的，保证写入顺序不乱
        self._sync_executor = DaemonThreadPoolExecutor(max_workers=1)
        self._initialized = False
        self._session_id: Optional[str] = None

    def initialize(self, session_id: str, **kwargs) -> None:
        """会话开始时初始化外部 provider。

        背景：外部服务需要先建立连接、准备资源才能用。

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
        """钩子：上下文压缩前被调用，让 LLM 从对话里提炼稳定事实存档（batch2-T1）。

        背景：压缩会把旧对话摘要掉，里面的长期信息（用户偏好、项目约定）
        如果不先捞出来就永久丢了。所以在压缩前抢救一次。

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
        # 扔后台线程跑，主循环继续做压缩不被拖住
        self._sync_executor.submit(self._safe_extract_and_save, messages)

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
            conversation_text=conversation_text[:8000],  # 掐头防 prompt 过长
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
                )
                saved += 1
            except Exception as e:
                logger.debug("on_pre_compress 保存单条失败: %s", e)
        if saved > 0:
            logger.info("on_pre_compress 提取并保存了 %d 条记忆", saved)

    @staticmethod
    def _serialize_messages(messages: list) -> str:
        """把消息列表摊平成"[角色] 内容"的纯文本，供 LLM 阅读。

        背景：LLM 只需要文本；工具调用等复杂结构取其中的文字部分即可。

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
        """把 LLM 回复解析成事实列表，容忍 JSON 外的多余文字。

        背景：模型经常在 JSON 前后加解释，先试整体解析，不行再从文本里
        捞出 [ ... ] 片段解析。

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
        """拼出要塞进 system prompt 的记忆部分。

        背景（CCAR10 Task 2 的历史决策）：内置记忆的固定快照已经从
        system prompt 里退役——中途改 prompt 会击穿缓存。现在只保留
        外部 provider 的静态块（这个块在会话开始时确定，之后不变，
        不伤缓存）。

        返回：拼接后的文本；没有可用内容时返回空串。
        """
        parts = []

        # 外部 provider 的静态说明块
        if self.external_provider and self._initialized:
            ext_block = self.external_provider.system_prompt_block()
            if ext_block:
                parts.append(ext_block)

        return "\n\n".join(parts)

    def prefetch_all(self, query: str) -> str:
        """每轮调 LLM 前调用：向外部 provider 要一份召回的上下文。

        参数：
        - query：当前用户输入（用于判断召回什么）

        返回：召回的上下文文本；无 provider 或失败返回空串（不影响主流程）。
        """
        if not self.external_provider or not self._initialized:
            return ""
        try:
            return self.external_provider.prefetch(query, session_id=self._session_id or "")
        except Exception as e:
            logger.debug("预取失败: %s", e)
            return ""

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

    def queue_prefetch_all(self, query: str) -> None:
        """为下一轮排队后台预取（外部 provider 自己异步去查）。

        参数：
        - query：当前用户输入，provider 拿它预判下一轮可能要什么

        返回：无；失败静默（预取只是优化，丢了不影响正确性）。
        """
        if not self.external_provider or not self._initialized:
            return
        try:
            self.external_provider.queue_prefetch(query, session_id=self._session_id or "")
        except Exception as e:
            logger.debug("queue_prefetch 失败: %s", e)

    def shutdown(self) -> None:
        """会话结束时清理：通知 provider 关门、关掉后台线程池。

        返回：无（清理失败也静默，退出路径不能再抛错）。
        """
        if self.external_provider and self._initialized:
            try:
                self.external_provider.shutdown()
            except Exception:
                pass
        self._sync_executor.shutdown(wait=False)
