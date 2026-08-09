"""用户画像归纳器:从记忆库自动归纳用户画像。

每 N 次反思后触发,读全部记忆 → aux_llm 归纳 → 存 USER_PROFILE.md → 注入 system prompt。
让 agent 从"知道用户 10 条碎片习惯"变成"理解用户是什么样的人"。
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_UPDATE_INTERVAL = 5  # 每 5 次反思后更新一次
_reflection_count = 0

PROFILE_PROMPT = """你是用户画像分析助手。从以下记忆条目中归纳用户的整体画像。

记忆列表:
{memories}

请归纳用户画像,用中文输出,分类:
- **技术背景**: 熟悉的语言/框架/工具链
- **沟通风格**: 偏好简洁/详细,语言
- **工作习惯**: 依赖管理/测试/部署偏好
- **项目信息**: 当前项目/技术栈

格式: 简洁的要点,不超过 200 字。直接输出画像文本,不要 JSON,不要多余解释。"""


def build_and_save_profile(memory_store, aux_llm, agent_home) -> bool:
    """读全部记忆 → aux_llm 归纳 → 存 USER_PROFILE.md。

    返回 True 表示画像已更新,False 表示跳过(记忆太少/调用失败)。
    """
    try:
        entries = memory_store.list_all()
    except Exception:
        return False

    if len(entries) < 3:
        return False  # 记忆太少不值得归纳

    memories = "\n".join(
        f"- [{e.type}] {e.name}: {e.description}"
        + (f" | {e.summary}" if e.summary else "")
        for e in entries[:50]  # 最多 50 条
    )

    prompt = PROFILE_PROMPT.format(memories=memories[:5000])

    try:
        # Task D4 fix: aux_llm.chat_completions 已改 async。
        # build_and_save_profile 由 _bg() daemon thread 调用（无事件循环），
        # 用 asyncio.run 驱动。
        import asyncio
        response = asyncio.run(aux_llm.chat_completions(
            [{"role": "user", "content": prompt}],
        ))
        profile = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("用户画像归纳失败(fail-open): %s", e)
        return False

    if not profile or len(profile) < 20:
        return False

    # 存 USER_PROFILE.md
    from agent.atomic_io import atomic_write_text
    profile_path = Path(agent_home) / "USER_PROFILE.md"
    content = f"# 用户画像(自动归纳)\n\n{profile}\n"
    atomic_write_text(profile_path, content)
    logger.info("用户画像已更新: %s (%d 字)", profile_path, len(profile))
    return True


def should_update_profile() -> bool:
    """每 N 次反思后返回 True。"""
    global _reflection_count
    _reflection_count += 1
    return _reflection_count % PROFILE_UPDATE_INTERVAL == 0
