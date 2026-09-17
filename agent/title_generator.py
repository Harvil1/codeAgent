"""自动给会话起标题（会话列表里显示的那行名字）。

现在是简化版：直接拿第一条用户消息的开头 40 个字符当标题——便宜、够用。
完整版可以接一个轻量 LLM 把长问题总结成精炼标题（接口参数已预留，
llm_client 传进来即可），目前没启用。
"""

from typing import Optional


def generate_title(
    first_user_message: str,
    *,
    llm_client=None,
    model: Optional[str] = None,
) -> Optional[str]:
    """根据用户的开场白生成一个简短标题。

    参数：
        first_user_message：会话里第一条用户消息。
        llm_client：预留的 LLM 客户端参数（完整版用来总结标题，目前不用）。
        model：预留的模型名参数（同上）。

    返回：
        标题字符串；消息为空或生成不出有效标题时返回 None。
    简化版做法：只取第一行、超 40 字截断；完整版换轻量模型总结（未启用）。
    """
    if not first_user_message:
        return None

    # 多行消息只要第一行——标题一行就够
    title = first_user_message.strip().split("\n")[0]
    # 开头是 / 命令（如 "/compact 清理一下"）时把命令名扔掉，只留人话部分
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
    """会话还没标题的话就自动起一个并写进会话库（已有标题则不动）。

    典型场景：会话第一条消息进来时调它，列表里就有名字可显示了。

    参数：
        session_store：会话存储（用来写入标题）。
        session_id：哪个会话。
        user_message：用户消息原文（起标题的素材）。
        current_title：现有标题；非空说明已经起过，直接返回。
        llm_client / model：透传给 generate_title 的预留参数。
    """
    if current_title:
        return  # 已经有标题就不重复劳动

    title = generate_title(
        user_message, llm_client=llm_client, model=model,
    )
    if title:
        session_store.set_title(session_id, title)
