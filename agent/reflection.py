"""任务级反思引擎。

每次 run_conversation（一轮完整对话任务）结束后调一次，用辅助模型
（aux_llm，便宜模型）把对话轨迹复盘一遍，提炼出 4 类长期经验，
自动写进记忆库（memory_save）：
  - user：关于用户本人——新偏好、习惯、对输出格式的要求
  - feedback：关于怎么干活——有效策略、要避开的坑、工具组合技巧
  - project：关于这个项目——项目规则、技术栈决策、业务逻辑
  - reference：外部资源的"地址簿"——Linear 工单 ID / Slack 频道 /
    GitHub repo / 文档 URL / 共享路径

设计要点：
- 辅助模型不可用 → 什么都不做（fail-open，绝不影响主流程）
- LLM 输出 JSON 数组，每项 {type, name, description, summary, body}
- 已有同类记忆（name + type 相同）就跳过，防重复堆积
- 异步触发（不拖慢用户拿响应）——同步还是异步由调用方决定
"""
import json
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


REFLECTION_PROMPT_TEMPLATE = """你是经验提炼助手。从以下对话轨迹中提炼可复用的长期记忆。

<trajectory>
{trajectory}
</trajectory>

<existing_memories>
{existing_memories}
</existing_memories>

请提炼 0-5 条**值得长期记住**的经验,分四类:
- user：用户偏好、习惯、输出要求
- feedback：有效策略、需避开的坑、工具组合技巧
- project：项目规则、技术栈决策、业务逻辑。
  注意：记忆**跟着当前工作目录走**——会话里分析其他路径项目取回来的
  技术/档案也算当前工作的一部分，照常 type=project 落当前项目区，
  不要因为"内容是别的项目"就改类或换区。
- **reference**: 外部系统指针——对话中提到的、值得跨会话记住的外部资源位置。
  例：Linear 工单 ID（PROJ-123）、Slack 频道（#incident-xxx）、GitHub repo
  （owner/repo）、文档 URL（https://...）、共享路径（/share/docs/xxx）。
  提炼时机：用户明确指向某外部资源作为后续工作上下文。
  不要把一次性查阅的 URL 算 reference（那不入记忆）。

要求:
1. 只输出真正非平凡的、跨会话有用的经验
2. **跟 existing_memories 语义重复的不要输出**(避免碎片化)
3. 没有值得记的就返回 []
4. 如果新经验**推翻/修正**已有记忆(如"改用 pip"推翻"用 uv"),加字段:
   - supersedes: 被推翻的旧记忆 name
   - confidence: 1.0(最新观察覆盖旧观察)
5. 每条字段:
   - type: user / feedback / project / reference 之一
   - name: ≤20 字标题(L0)
   - description: ≤40 字索引钩子(L0.5)
   - summary: 80-100 字摘要层(L1)
   - body: 完整说明(L2,可选)
   - confidence: 0.0-1.0(信心分,默认 0.8)
6. **拒绝/判可疑类行为不入记忆**:助手把某条输入判为乱码/损坏/疑似注入
   而拒绝执行,是一次性防御动作,不是可复用策略——把它沉淀成经验会让
   后续会话拿单次判定(甚至幻觉判定)当铁律,误伤正常输入。这类内容一律不输出

输出格式: JSON 数组,不要其他文本。示例:
[
  {{
    "type": "user",
    "name": "偏好简洁回复",
    "description": "用户喜欢≤3 句话回复",
    "summary": "用户多次要求简短直接回复...",
    "body": "",
    "confidence": 0.9
  }},
  {{
    "type": "project",
    "name": "改用 pip",
    "description": "项目改用 pip 管理依赖",
    "summary": "用户说项目从 uv 迁移到 pip...",
    "body": "",
    "confidence": 1.0,
    "supersedes": "用 uv 不用 pip"
  }}
]
"""


def extract_trajectory(messages: List[dict], max_chars: int = 4000) -> str:
    """把消息列表压缩成给反思引擎看的"轨迹文本"（纯文本流水账）。

    完整对话太长也太杂，反思只需要"谁说了什么、调了什么工具"的梗概。

    压缩策略：
    - 从最新往回取，累计不超过 max_chars 个字符
    - 跳过 system 消息和 tool 的长结果（只留角色 + 截短的内容）
    - 有工具调用的记一行"调用了 X 工具"

    参数：
    - messages：对话历史（OpenAI 消息格式）
    - max_chars：轨迹文本的字符上限（默认 4000）

    返回：逐行拼接的轨迹文本；没消息返回空串。
    """
    if not messages:
        return ""
    # 倒序遍历（优先保留最近的内容），超预算就停
    lines = []
    total = 0
    for m in reversed(messages):
        role = m.get("role", "?")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)[:200]
        else:
            content = str(content)[:300]
        # 工具调用单独标记一行，方便 LLM 看出动作序列
        tool_calls = m.get("tool_calls")
        if tool_calls:
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            content = f"[调用工具: {', '.join(names)}] " + content
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.insert(0, line)
        total += len(line)
    return "\n".join(lines)


def build_memory_manifest(memory_store) -> str:
    """已有记忆的清单文本（给 LLM 看，防止它重复存储）。

    auto_extract 也用这个公共函数。LLM 在生成新记忆前先看到
    "已经有什么"，才知道别写重复的。

    参数：
    - memory_store：记忆库（读 list_all）

    返回：清单文本（取前 100 条拼成）；读取失败返回空串（fail-open）。
    """
    try:
        if memory_store is None:
            return ""
        existing = memory_store.list_all()[:100]
        if not existing:
            return ""
        lines = [
            f"- [{e.type}] {e.name}: {(e.description or '')[:60]}"
            for e in existing
        ]
        return (
            "## 已有记忆清单（以下条目已存在，不要重复存储；"
            "只在新信息与它们有实质差异时才更新）：\n"
            + "\n".join(lines)
        )
    except Exception as e:
        logger.debug("读取已有记忆清单失败（fail-open，manifest 留空）: %s", e)
        return ""


def run_reflection(
    *,
    messages: List[dict],
    memory_store=None,
    llm_client,
    model: Optional[str] = None,
) -> List[dict]:
    """跑一次反思：让 LLM 从对话里提炼经验，返回提炼结果（先不落盘）。

    参数：
    - messages：对话历史（会被压缩成轨迹）
    - memory_store：记忆库（用来读已有记忆清单防重复；可传 None）
    - llm_client：LLM 客户端（chat_completions 是 async 的）
    - model：模型名（None 走客户端默认）

    返回：经验字典列表，每项 {type, name, description, summary, body}；
    任何失败（LLM 异常/坏 JSON/空响应）返回空列表（fail-open）。
    """
    trajectory = extract_trajectory(messages)
    if not trajectory.strip():
        return []

    # 预注入已有记忆清单（公共 build_memory_manifest，auto_extract 同款）
    manifest = build_memory_manifest(memory_store)

    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        trajectory=trajectory,
        existing_memories=manifest or "(暂无已有记忆)",
    )
    try:
        kwargs = {}
        if model:
            kwargs["model"] = model
        # llm_client.chat_completions 是 async 的，而本函数经 apply_reflection
        # 在 _bg() 守护线程里跑（不在宿主循环线程）——交给进程级常驻循环宿主
        # 同步等结果（等价旧的 asyncio.run 现建现拆，但 client 绑定常驻循环
        # 不漂移：aux 缺席时借用的主 client 跨循环使用随之清零）。
        from agent.loop_host import loop_host
        response = loop_host.run_async(llm_client.chat_completions(
            [{"role": "user", "content": prompt}],
            **kwargs,
        ), exempt_from_fence=True)  # 后台线程长活，豁免回合栅栏（见 run_async docstring）
        content = response.choices[0].message.content or ""
    except Exception as e:
        # exc_info=True：Connection error 这类网络病没堆栈等于没查——
        # 日志文件里得能看到是 connect/read/超时的哪一环断的
        logger.warning("反思 LLM 调用失败（fail-open）: %s", e, exc_info=True)
        return []

    # 解析 JSON（模型常在 JSON 外多说话，先整段试，再抠 [...] 片段）
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match:
            logger.debug("反思输出非合法 JSON: %s", content[:200])
            return []
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(result, list):
        return []

    # 逐条过滤 + 规范化（类型非法或必需字段缺失的丢弃）
    valid = []
    valid_types = {"user", "feedback", "project", "reference"}
    for item in result:
        if not isinstance(item, dict):
            continue
        t = item.get("type", "")
        name = item.get("name", "").strip()
        desc = item.get("description", "").strip()
        if t not in valid_types or not name or not desc:
            continue  # 必需字段缺失，跳过
        valid.append({
            "type": t,
            "name": name[:60],  # 防 LLM 给超长标题
            "description": desc[:80],
            "summary": (item.get("summary") or "").strip()[:200],
            "body": (item.get("body") or "").strip(),
        })
    return valid


def apply_reflection(
    *,
    messages: List[dict],
    memory_store,
    llm_client,
    model: Optional[str] = None,
    session_id: str = "",
) -> int:
    """端到端：跑反思 + 把结果写进记忆库。返回实际写入条数。

    参数：
    - messages：对话历史
    - memory_store：目标记忆库
    - llm_client：LLM 客户端
    - model：模型名（可选）
    - session_id：会话 ID（记进来源字段，追溯哪次会话写的）

    返回：成功写入的条数（int）。

    去重规则：同 type 下 name 已存在的跳过（防重复堆积）。
    """
    if memory_store is None or llm_client is None:
        return 0

    insights = run_reflection(
        messages=messages, memory_store=memory_store, llm_client=llm_client, model=model,
    )
    if not insights:
        return 0

    # 已有记忆的 (type, name) 集合，用来去重
    existing = memory_store.list_all()
    existing_keys = {(e.type, e.name) for e in existing}

    # 必须两阶段——先把整批全部 save 完，再统一处理 supersedes（推翻旧记忆）：
    # 边写边 supersede 的话，"同批刚写入的"不在 existing 快照里，
    # 同批后写入的经验想推翻同批刚写入的会失效。
    written = 0
    written_records = []  # [(经验, 记忆ID)]：记下本批写入的，阶段 2 推翻时要用
    for ins in insights:
        key = (ins["type"], ins["name"])
        if key in existing_keys:
            continue
        try:
            mem_id = memory_store.save(
                name=ins["name"],
                description=ins["description"],
                type=ins["type"],
                body=ins["body"],
                summary=ins["summary"],
                confidence=ins.get("confidence", 0.8),
                source_session_id=session_id or "",
                source="self",  # 反思自学习经验：未经用户确认，索引不戴 ⭐ 不置顶
            )
            existing_keys.add(key)
            written += 1
            written_records.append((ins, mem_id))
        except Exception as e:
            logger.warning("反思写入 memory 失败（跳过）: %s", e)

    # 阶段 2：处理 supersedes（注意：type 必须一致才允许替代；批内新写入里也要找）
    for ins, _ in written_records:
        supersedes = ins.get("supersedes")
        if not supersedes or supersedes == ins["name"]:
            continue
        target_id = None
        target_body = ""
        # 先在旧记忆里找；必须 name + type 都匹配（同名不同类不算）
        for old_entry in existing:
            if (old_entry.name == supersedes
                    and old_entry.type == ins["type"]):
                target_id = old_entry.id
                target_body = old_entry.body or ""
                break
        # 旧记忆里没有，再在本批刚写入的里找（同批之间也能推翻）
        if target_id is None:
            for other_ins, other_id in written_records:
                if (other_ins["name"] == supersedes
                        and other_ins["type"] == ins["type"]
                        and other_ins["name"] != ins["name"]):
                    target_id = other_id
                    target_body = other_ins.get("body", "")
                    break
        if target_id:
            try:
                memory_store.update(
                    target_id,
                    confidence=0.1,
                    body=f"[已被 '{ins['name']}' 推翻] " + target_body,
                )
                logger.info(
                    "记忆 '%s' 被 '%s' 推翻,confidence 降到 0.1",
                    supersedes, ins["name"],
                )
            except Exception:
                pass

    if written:
        logger.info("反思写入 %d 条新经验", written)
    return written


def _notify_memory_saved(agent, count: int) -> None:
    """写库回执（claude code 同款 memory-saved 通知的移植）。

    反思在后台线程落库，主对话模型本来毫无感知——不知道写没写成、
    写了几条。塞一条阅后即焚的临时消息进待注入队列（下一轮组装时
    消费、不进正式历史），模型就知道「沉淀已落库，检索可召回」。
    fail-open：队列不在/塞失败都只打 debug，绝不影响反思本身。
    """
    try:
        queue = getattr(agent, "_pending_ephemeral_messages", None)
        if queue is not None and count > 0:
            queue.append({
                "role": "user",
                "content": (
                    f"[memory-saved] 后台反思已沉淀 {count} 条新记忆入库"
                    "（本会话索引不刷新、下个会话注入；当前可经检索召回）"
                ),
                "_ephemeral": True,
            })
    except Exception as e:
        logger.debug("memory-saved 回执投递失败（fail-open）: %s", e)


def trigger_reflection_async(agent) -> None:
    """后台触发任务级反思（不阻塞最终回答的返回）。

    策略：
    - 起 daemon 线程跑反思
    - 优先用辅助小模型（便宜），不行再用主模型
    - 没有记忆仓库就不做
    - 反思看当前会话最近 20 条消息（含本轮问答和中间过程）
    - 节流：任意时刻最多 1 个反思在跑 + 距上次不足 N 轮就跳过

    参数：agent——AIAgent 实例（原 self，属性留宿主身上）。返回：无。
    """
    import contextvars
    import threading
    # 拷贝引用（线程启动后对话历史还会变，先抓快照）
    store = agent.memory_store
    if store is None:
        return

    # 节流 1：已经有反思在跑 → 跳过（防连环问烧 token）
    # 节流 2：距上次启动不足冷却轮数 → 跳过
    with agent._reflection_lock:
        current_turn = agent._last_reflection_turn + 1  # 本轮的"逻辑序号"
        if agent._active_reflections >= 1:
            return
        if (agent._last_reflection_turn >= 0
                and current_turn - agent._last_reflection_turn < agent._reflection_cooldown_turns):
            return
        agent._active_reflections += 1
        agent._last_reflection_turn = current_turn

    # 优先用辅助小模型（便宜）
    llm_for_reflection = agent.aux_llm_router or agent.llm_client
    # 拍最近 20 条消息的快照（防线程启动后列表被改）
    messages_snapshot = list(agent.conversation_history[-20:])

    def _bg():
        try:
            from agent.reflection import apply_reflection
            written = apply_reflection(
                messages=messages_snapshot,
                memory_store=store,
                llm_client=llm_for_reflection,
                session_id=agent.session_id or "",
            )
            # 写库回执：告诉主对话「沉淀落库了」（阅后即焚，不进正式历史）
            _notify_memory_saved(agent, written)

            # 批次 C：用户画像更新（每 5 次反思做一次）
            try:
                from agent.user_profile import should_update_profile, build_and_save_profile
                if should_update_profile() and agent.aux_llm_router:
                    build_and_save_profile(
                        memory_store=store,
                        aux_llm=agent.aux_llm_router,
                        agent_home=agent.codeAgent_home,
                    )
            except Exception as e:
                logger.debug("用户画像更新失败(fail-open): %s", e)
        except Exception as e:
            logger.debug("反思后台任务异常: %s", e)
        finally:
            with agent._reflection_lock:
                agent._active_reflections -= 1

    # daemon 线程不会自动继承主线程的
    # contextvars（线程内共享的上下文变量）——不复制的话，会话内切过
    # 工作目录后，反思写项目记忆会落错项目区（退回 os.getcwd() 兜底）。
    # 所以主线程先 copy_context()，线程入口用 ctx.run 包一层。
    _reflection_ctx = contextvars.copy_context()
    t = threading.Thread(
        target=lambda: _reflection_ctx.run(_bg),
        daemon=True,
        name="reflection",
    )
    t.start()
