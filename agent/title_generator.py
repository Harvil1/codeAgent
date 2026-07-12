"""自动生成会话标题。

简化版：截取第一条用户消息的前 40 字符。
完整版可调用轻量 LLM 生成更精炼的标题。
"""

from typing import Optional


def generate_title(
    first_user_message: str,
    *,
    llm_client=None,
    model: Optional[str] = None,
) -> Optional[str]:
    """根据第一条用户消息生成简短标题。

    简化版：截取前 40 字符。完整版用轻量模型总结。
    """
    if not first_user_message:
        return None

    # 取第一行
    title = first_user_message.strip().split("\n")[0]
    # 去掉 slash 命令前缀
    if title.startswith("/"):
        title = title.split(None, 1)[-1] if " " in title else title

    if len(title) > 40:
        title = title[:37] + "..."
    return title or None


def maybe_set_title(
    session_store,
    session_id: str,
    user_message: str,
    current_title: Optional[str],
    *,
    llm_client=None,
    model: Optional[str] = None,
) -> None:
    """如果是第一条消息且无标题，自动生成。"""
    if current_title:
        return  # 已有标题

    title = generate_title(
        user_message, llm_client=llm_client, model=model,
    )
    if title:
        session_store.set_title(session_id, title)
