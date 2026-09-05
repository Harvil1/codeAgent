"""用户画像归纳器：定期把记忆库里关于用户的零散记忆，总结成一份"用户画像"。

工作流程：每 N 次"反思"（reflection，从对话中沉淀经验的过程）之后触发一次，
把全部记忆交给辅助 LLM（aux_llm，干杂活的小模型）归纳 → 写进
USER_PROFILE.md → 之后注入 system prompt（系统提示词）。

目的：让 agent 从"知道用户有 10 条碎片习惯"升级成"理解用户是个什么样的
人"——好比从一沓便签纸变成一封人物小传。
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_UPDATE_INTERVAL = 5  # 攒够 5 次反思才更新一次画像：太频繁既浪费又没必要
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
    """干一次完整的归纳：读全部记忆 → 辅助 LLM 总结 → 存成 USER_PROFILE.md。

    参数：
        memory_store：记忆库（读用户相关记忆的来源）。
        aux_llm：辅助 LLM 客户端（负责归纳总结）。
        agent_home：agent 数据根目录（一般 ~/.codeAgent，画像存这里）。

    返回：
        True = 画像已更新；False = 这次跳过了（记忆太少 / 调用失败等）。
    """
    try:
        entries = memory_store.list_all()
    except Exception:
        return False

    if len(entries) < 3:
        return False  # 记忆还不到 3 条，硬归纳只会瞎编，不值得做

    memories = "\n".join(
        f"- [{e.type}] {e.name}: {e.description}"
        + (f" | {e.summary}" if e.summary else "")
        for e in entries[:50]  # 最多喂 50 条：再多 prompt 也装不下，归纳质量反而下降
    )

    prompt = PROFILE_PROMPT.format(memories=memories[:5000])

    try:
        # aux_llm.chat_completions 是 async 的，而本函数是被 _bg() 后台线程
        # 调用的（不在宿主循环线程）——交给进程级常驻循环宿主同步等结果
        # （等价旧的 asyncio.run，但 aux 缓存 client 绑定常驻循环不再
        # 每次换新循环漂移）。
        from agent.loop_host import loop_host
        response = loop_host.run_async(aux_llm.chat_completions(
            [{"role": "user", "content": prompt}],
        ))
        profile = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("用户画像归纳失败(fail-open): %s", e)
        return False

    if not profile or len(profile) < 20:
        return False

    # 归纳结果落盘成 USER_PROFILE.md（原子写入，写一半断电也不会留半个文件）
    from agent.atomic_io import atomic_write_text
    profile_path = Path(agent_home) / "USER_PROFILE.md"
    content = f"# 用户画像(自动归纳)\n\n{profile}\n"
    atomic_write_text(profile_path, content)
    logger.info("用户画像已更新: %s (%d 字)", profile_path, len(profile))
    return True


def should_update_profile() -> bool:
    """反思次数计数器：每被调用一次加一，攒满 N 次（默认 5）才返回 True。

    返回：
        True = 到了该更新画像的节点；False = 还没到，继续攒。

    用法：反思流程每次结束时调一下，返回 True 才去跑
    build_and_save_profile。
    """
    global _reflection_count
    _reflection_count += 1
    return _reflection_count % PROFILE_UPDATE_INTERVAL == 0
