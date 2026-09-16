"""批间摘要 + 条件技能激活的自由函数区（从 agent/__init__.py 平移而来）。

大白话：这里的五个函数原本是 AIAgent 的五个方法，拆出来只为给
agent/__init__.py 减负——行为零变化，搬的是同一份代码：

- 条件技能三件（收集 → 批量冲刷 → 逐路径激活）：工具碰到匹配文件时
  先收集路径去重，等组装消息时统一激活，激活通知走临时消息队列。
- 批间摘要二件（启动 + 生成）：每批工具跑完后，后台让辅助小模型写
  一句话总结，下一轮以临时消息注入。

本模块三条铁律（跟拆分一期约定一致）：
1. 属性全留 AIAgent——函数不自己存状态，一律写 ``agent._xxx``，
   第一参固定收 agent 实例（原 ``self``）。
2. 禁止模块级 import agent root（防循环导入）——``_spawn_detached``
   用函数内延迟导入 ``from agent import _spawn_detached``。
3. 函数体与原方法逐字节平移，唯一改写是 ``self`` → ``agent``；
   ``_is_long_task`` 等仍留在 AIAgent 上的方法经 ``agent._xxx(...)`` 调。
"""

import logging

logger = logging.getLogger(__name__)


def queue_skill_activation(agent, path: str) -> None:
    """工具触发的条件技能激活，走「先收集、后批量执行」。

    回调里同步扫技能目录（磁盘 glob）太浪费——一轮跑多个工具
    就扫多次。所以回调里只收集路径（去重），
    等组装消息时统一处理——时机正好赶在临时消息队列消费之前，激活
    结果当轮可见。

    参数：agent: AIAgent 实例（属性读宿主、写经 agent. 前缀）；
    path: 工具触碰到的文件路径。返回：无。
    """
    if agent._pending_skill_paths is None:
        agent._pending_skill_paths = []
    if path and path not in agent._pending_skill_paths:
        agent._pending_skill_paths.append(path)


def flush_skill_activations(agent) -> None:
    """把收集到的条件技能激活批量处理掉（出错放行）。

    这里保持同步是因为调用方（组装消息）本身是同步方法，主循环
    每轮 LLM 调用前来一次；扫描有「修改时间+文件大小」双因子缓存兜底，
    又已从「每工具一次」去重到「每轮一次」——不值得为此改成 async。

    参数：agent: AIAgent 实例（属性读宿主、写经 agent. 前缀）。返回：无。
    """
    paths, agent._pending_skill_paths = (agent._pending_skill_paths or []), []
    for p in paths:
        try:
            activate_conditional_skills(agent, p)
        except Exception as e:
            logger.warning("条件技能激活失败（fail-open）: %s", e)


def activate_conditional_skills(agent, path) -> None:
    """碰到匹配的文件 → 动态激活「条件技能」。

    技能的 frontmatter（文件头元数据）里写了 paths 的，默认不进
    静态索引（省索引空间）；等读/写/替换工具真的碰到匹配文件时才激活
    ——发一条临时消息告诉模型这个技能可用了（用 load_skill 取正文），
    会话内只激活一次。出错放行。

    参数：agent: AIAgent 实例（属性读宿主、写经 agent. 前缀）；
    path: 被触碰的文件路径。返回：无。
    """
    try:
        from agent.skill_commands import find_conditional_skill_matches
        hits = find_conditional_skill_matches(str(path))
        new_hits = [
            h for h in hits
            if h["name"] not in agent._activated_conditional_skills
        ]
        if not new_hits:
            return
        for h in new_hits:
            agent._activated_conditional_skills.add(h["name"])
            agent._record_recent("skill", h["name"])
        lines = "\n".join(
            f"- {h['name']}: {h['description']}" for h in new_hits
        )
        agent._pending_ephemeral_messages.append({
            "role": "user",
            "content": (
                "<conditional_skills_ready>你刚触碰了匹配的文件，"
                "以下技能现已激活（用 load_skill(name) 获取完整正文）：\n"
                f"{lines}\n</conditional_skills_ready>"
            ),
            "_ephemeral": True,
        })
        logger.info(
            "条件技能激活（触碰 %s）：%s",
            path, ", ".join(h["name"] for h in new_hits),
        )
    except Exception as e:
        logger.warning("条件技能激活失败（fail-open）: %s", e)


def start_tool_batch_summary(agent, tool_calls, safe_processed, unsafe_processed) -> None:
    """启动「批间摘要」后台任务。

    每批工具跑完后，后台让辅助小模型写一句话总结，下一轮以临时
    消息注入（生成延迟藏在模型流式输出期间，感觉不到）。
    门控：config 的 context.tool_batch_summary_enabled 开着（默认关）+
    有辅助模型 + 只有主代理做（防止子代理也来一套白花钱）。
    整条链失败放行。

    参数：
        agent: AIAgent 实例（属性读宿主、写经 agent. 前缀）
        tool_calls: 本批工具调用列表
        safe_processed: safe 组的 (调用, 结果) 列表
        unsafe_processed: unsafe 组的 (调用, 结果) 列表

    返回：无。
    """
    try:
        # 延迟导入 agent root（防循环导入——本模块被 root 在模块级引入）
        from agent import _spawn_detached
        ctx_cfg = (agent.config or {}).get("context", {})
        flag = ctx_cfg.get("tool_batch_summary_enabled", False)
        # flag 显式开照旧；flag 关但长任务信号也开
        # （auto 是独立 kill switch，只关自动开、不影响手动开关语义）
        auto = ctx_cfg.get("tool_batch_summary_auto_long_task", True)
        if not flag and not (auto and agent._is_long_task()):
            return
        if agent.aux_llm_router is None or agent.spawn_depth > 0:
            return
        if not tool_calls:
            return
        # 汇总 (工具名, 结果前 300 字)
        items = []
        for tc, content in list(safe_processed) + list(unsafe_processed):
            items.append((tc.function.name, str(content)[:300]))
        if not items:
            return
        # 上一批的任务还没跑完 → 直接覆盖（新摘要取代旧的，符合「看最新」语义）
        # 用 _spawn_detached 跑：submit 到常驻宿主循环（进回合栅栏
        # 豁免名单），不会被回合收尾当遗留清掉
        agent._tool_summary_task = _spawn_detached(
            generate_tool_batch_summary(agent, items), "tool-batch-summary",
        )
    except Exception as e:
        logger.warning("批间摘要启动失败（fail-open）: %s", e)


async def generate_tool_batch_summary(agent, items) -> None:
    """让辅助小模型用一句话总结这批工具干了什么（出错全吞不炸）。

    参数：agent: AIAgent 实例（属性读宿主、写经 agent. 前缀）；
    items: (工具名, 结果片段) 列表。返回：无（结果存到
    agent._pending_tool_batch_summary，下轮注入）。
    """
    try:
        lines = "\n".join(
            f"- {name}: {content}" for name, content in items[:20]
        )
        prompt = (
            "用一句中文（不超过 80 字）总结这批工具调用做了什么，"
            "突出关键产出（文件/命令/结论），直接输出句子：\n" + lines
        )
        resp = await agent.aux_llm_router.chat_completions(
            [{"role": "user", "content": prompt}],
        )
        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            text = resp if isinstance(resp, str) else ""
        text = str(text).strip()
        if text:
            # 只留最新一批的摘要（后到的覆盖先到的）
            agent._pending_tool_batch_summary = text[:200]
            logger.debug("批间摘要已生成（%d 字）", len(text))
    except Exception as e:
        logger.warning("批间摘要生成失败（fail-open）: %s", e)
