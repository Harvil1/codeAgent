# agent/goal.py
"""Goal（目标）驱动系统：给 agent 一个目标后，它自己一轮接一轮地干，直到目标完成。

打个比方：普通对话是"你一句我一句"；goal 模式像给员工下了个任务书，
他自己反复干活、自己检查进度，干完或出问题才回来找你。

核心设计：
- 同步阻塞：输入 /goal 后 CLI 卡在那等，agent 自动多轮跑，直到暂停(pause)或完成(complete)
- 有 pause/resume/continue/clear 四个子命令控制节奏
- 网络断开、token 预算（花钱额度）超限时会自动暂停，防止失控烧钱
- 每一轮用"用完即弃"的临时 user 消息驱动下一步，绝不动 system prompt
  （system prompt 一变，之前的缓存全作废、费用翻倍——这是项目的铁律）
- 状态存到 ~/.OmniMate/.goal/current.json，程序崩了重启也能接上

本文件在项目里的位置：只放"状态机 + 存档/读档"这两块底层零件；
和主循环的集成（怎么在对话循环里推进 goal）在 agent/__init__.py 里。
"""
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GoalState:
    """一个目标的全部状态数据（现在跑到第几轮、花了多少 token、是暂停还是进行中……）。

    全局同时只允许一个进行中的目标（单例语义），新开目标会先把旧的暂停。
    各字段用大白话说：
    - objective：目标描述文本（任务书）
    - goal_id：自动生成的唯一编号
    - status：目标处于哪个阶段——active（跑着）/ paused（暂停）/ completed（完成）/ failed（失败）/ cancelled（取消）
    - created_at：创建时间戳
    - iteration_count：已经自动跑了几轮
    - token_budget：已花掉的 token 累计数
    - token_budget_limit：花钱上限；None=不设限
    - pause_reason：上次为什么暂停（网络/预算/手动/完成）
    - task_ids：目标拆出的子任务编号列表
    - last_progress：最近一次进展的描述
    - notes：流水账（暂停/恢复/完成各记一笔）
    """

    objective: str
    goal_id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}")
    status: str = "active"  # 取值：active(跑着)/paused(暂停)/completed(完成)/failed(失败)/cancelled(取消)
    created_at: float = field(default_factory=time.time)
    iteration_count: int = 0
    token_budget: int = 0
    token_budget_limit: Optional[int] = None  # 花钱上限；None=不设限
    pause_reason: Optional[str] = None  # 暂停原因：network(断网)/budget(超预算)/manual(手动)/completed(完成)
    task_ids: List[str] = field(default_factory=list)
    last_progress: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # ---- 状态转换 ----

    def pause(self, reason: str = "manual") -> None:
        """把目标暂停。

        背景：网络断、预算花超、用户手动叫停，最后都走这一个入口，
        顺便在这里统一发桌面通知（历史踩坑：通知逻辑以前散在各处，统一收口是 CCAR13 B5 的修复）。

        参数：
        - reason：暂停原因，manual（手动）/ network（断网）/ budget_exceeded（超预算）

        返回：无（直接改自身状态）。
        """
        self.status = "paused"
        self.pause_reason = reason
        self.notes.append(f"paused: reason={reason}, iteration={self.iteration_count}")
        # 设计取舍：发通知失败（或通知模块本身出问题）绝不炸状态机——
        # 这个文件被大量单元测试直接调用，不能因为通知挂了就全崩。
        # import 放函数内（lazy）：模块加载时不连带拉起 notifier/config。
        try:
            from agent.notifier import notify
            notify("Goal 已暂停", f"原因: {reason}")
        except Exception as e:
            logger.debug("pause notify fail-open: %s", e)

    def resume(self) -> None:
        """把暂停的目标恢复成进行中，并清掉上次的暂停原因。

        参数：无。返回：无。
        """
        self.status = "active"
        self.pause_reason = None
        self.notes.append(f"resumed: iteration={self.iteration_count}")

    def complete(self) -> None:
        """标记目标已完成，并记一笔流水账。参数无，返回无。"""
        self.status = "completed"
        self.pause_reason = "completed"
        self.notes.append(f"completed: iteration={self.iteration_count}")

    def cancel(self) -> None:
        """用户主动放弃目标（区别于失败）。参数无，返回无。"""
        self.status = "cancelled"
        self.pause_reason = "cancelled"
        self.notes.append(f"cancelled: iteration={self.iteration_count}")

    def fail(self, reason: str = "") -> None:
        """标记目标失败。

        参数：
        - reason：失败原因描述，会记进流水账

        返回：无。
        """
        self.status = "failed"
        self.pause_reason = f"failed:{reason}" if reason else "failed"
        self.notes.append(f"failed: {reason}, iteration={self.iteration_count}")

    # ---- 每轮评估 ----

    def evaluate_after_turn(
        self,
        tokens_used: int = 0,
        all_tasks_done: bool = False,
    ) -> str:
        """每跑完一轮就调一次，由状态机自己判断"接下来干嘛"。

        背景：goal 模式不需要人盯，靠这个方法在每轮结束后自动做裁判。

        参数：
        - tokens_used：这一轮实际花掉的 token 数（累加进 token_budget）
        - all_tasks_done：拆出的子任务是否已全部完成

        返回：决策字符串——"complete"（全干完了，收工）/
        "pause"（预算超限，暂停）/ "continue"（没事，接着干）。
        """
        self.iteration_count += 1
        self.token_budget += tokens_used

        if all_tasks_done:
            self.complete()
            return "complete"

        if (
            self.token_budget_limit is not None
            and self.token_budget >= self.token_budget_limit
        ):
            self.pause(reason="budget_exceeded")
            return "pause"

        return "continue"

    def should_nudge(self, recent_tool_success: bool) -> bool:
        """判断要不要"踢一脚"（nudge）让 agent 继续干而不是提前收工。

        背景（R26 #9 引入）：典型场景——goal 要求修 20 个文件，agent 修了
        14 个就宣布"完成了"。如果预算还剩 10% 以上、且最近一轮还有成功的
        工具调用（说明没在空转），就值得注入一条催促消息让它接着干，
        而不是等用户重新下命令。对齐 CCB "预算未满 + 无收益递减 → nudge"
        的思路。

        参数：
        - recent_tool_success：最近一轮是否有成功的工具调用

        返回：True=该踢一脚继续；False=不踢。
        """
        if self.status != "active" or self.token_budget_limit is None:
            return False
        if self.iteration_count < 2:
            return False
        if self.token_budget >= 0.9 * self.token_budget_limit:
            return False
        return bool(recent_tool_success)

    # ---- 持久化 ----

    def save(self, path: Path) -> None:
        """把目标状态存到磁盘文件（先写临时文件再改名，防止写一半崩了留下坏文件）。

        设计取舍：存档失败只记一条日志、不抛错——不能因为存档失败
        把正在跑的目标搞崩。

        参数：
        - path：存档文件路径（一般是 ~/.OmniMate/.goal/current.json）

        返回：无。
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            data = asdict(self)
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("goal save 失败（fail-open）: %s", e)

    @classmethod
    def load(cls, path: Path) -> Optional["GoalState"]:
        """从磁盘读回目标状态。

        参数：
        - path：存档文件路径

        返回：GoalState 对象；文件不存在（从没存过）返回 None；
        文件损坏（JSON 解析失败等）也返回 None，但额外记一条 warning。
        """
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except Exception as e:
            logger.warning("goal load 失败: %s", e)
            return None


# =============================================================================
# 共享启动函数（CCAR12 Task 4 抽出）：CLI 的 /goal 命令和 LLM 的 goal_start
# 工具走同一份代码，避免两处各写一套以后改不齐。
# =============================================================================


def goal_persist_path(agent) -> Path:
    """算出 agent 的 goal 存档文件路径（一般是 ~/.OmniMate/.goal/current.json）。

    背景：调用方传来的 agent 对象能力不一（可能是真 AIAgent，也可能是
    测试替身），所以按三档优先级依次试：

    1. agent._goal_state_path() —— AIAgent 实例方法，最准
    2. agent.omnimate_home —— AIAgent 字段，次选
    3. get_omnimate_home() —— 全局默认，测试可用 OMNIMATE_HOME 环境变量覆盖

    参数：
    - agent：AIAgent 实例或测试替身

    返回：存档文件路径。
    """
    fn = getattr(agent, "_goal_state_path", None)
    if callable(fn):
        try:
            return Path(fn())
        except Exception:
            pass
    home = getattr(agent, "omnimate_home", None)
    if home:
        return Path(home) / ".goal" / "current.json"
    from constants import get_omnimate_home
    return get_omnimate_home() / ".goal" / "current.json"


def start_goal_agent(
    agent,
    objective: str,
    token_budget: int = 200_000,
    persist_path=None,
) -> GoalState:
    """启动一个新 goal——CLI 的 /goal 命令和 LLM 的 goal_start 工具共用的核心。

    做两件事（逻辑从 cli.py 的 _start_new_goal 原样搬来，界面输出留在 CLI 层）：
    1. 如果旧 goal 还在跑，先把它暂停（原因记为 superseded_by_new_goal）并落盘
    2. 建一个新的 GoalState，落盘，挂到 agent 身上

    ⚠️ 历史踩坑提醒：本函数故意不碰 conversation_history（对话消息列表）。
    CLI 在会话循环外追加 `[goal_start]` user 消息没问题；但工具路径是在
    assistant(tool_calls) 之后、tool 结果还没回填的节骨眼上，这时插一条
    user 消息会破坏"工具调用和结果必须严格交替"的规矩，直接 API 400。
    工具路径不需要手动塞消息——主循环看到 goal 在跑就会自动多轮推进。

    参数：
    - agent：AIAgent（或测试 mock，需支持 _goal_state / set_goal_state / 路径解析）
    - objective：目标描述文本
    - token_budget：token 花钱上限（默认 20 万）
    - persist_path：显式指定存档路径（CLI 传自己 home 下的路径；None=自动算）

    返回：新建的 GoalState（已经挂到 agent 上）。
    """
    if persist_path is None:
        persist_path = goal_persist_path(agent)
    persist_path = Path(persist_path)

    # 1. 只暂停还在跑的旧 goal；已经暂停/完成的不重复动
    old_gs = getattr(agent, "_goal_state", None)
    if old_gs is not None and old_gs.status == "active":
        old_gs.pause(reason="superseded_by_new_goal")
        old_gs.save(persist_path)

    # 2. 建新 GoalState：先落盘再挂到 agent（优先用正式的 setter，没有就直接赋值）
    gs = GoalState(objective=objective, token_budget_limit=token_budget)
    gs.save(persist_path)
    setter = getattr(agent, "set_goal_state", None)
    if callable(setter):
        setter(gs)
    else:
        agent._goal_state = gs
    return gs


# =============================================================================
# TaskStore（持久化任务库）集成（CCAR8 Task 12 新增）：
# 把目标拆成子任务 + 检查子任务是否全干完。
# =============================================================================


async def decompose_with_llm(
    goal_state: "GoalState",
    objective: str,
    aux_llm_router,
) -> List[str]:
    """用辅助小模型（aux_llm——干杂活用的便宜模型）把目标拆成几条子任务，存进任务库。

    背景：大目标直接干容易乱，先让小模型拆成 1-5 条可独立执行的小任务，
    goal 跑的时候就能逐条对账"还剩几个没干完"。

    设计取舍（一切求稳，拆解失败不影响 goal 本身）：
    - 辅助模型不可用 → 返回空列表（goal 照跑，只是没有子任务追踪）
    - 模型输出的不是合法 JSON / 空数组 → 也返回空列表
    - 拆解成功后，每条子任务建进 TaskStore，并打上属于哪个 goal 的标记

    参数：
    - goal_state：目标状态对象（函数会直接把拆出的任务编号填进它的 task_ids）
    - objective：目标描述文本（送给模型去拆）
    - aux_llm_router：辅助模型路由器（或任何带 async chat_completions 方法的对象）

    返回：子任务编号列表（拆解失败时可能是空列表）。
    """
    if aux_llm_router is None:
        logger.info("decompose_with_llm: 无 aux_llm_router，跳过拆解（goal 仍可跑）")
        return []

    from agent.task_store import get_task_store
    store = get_task_store()

    prompt = (
        f"目标：{objective}\n\n"
        "请把这个目标拆解为 1-5 个具体的、可独立执行的子任务。"
        "每个子任务包含 subject（简短标题）和 description（详细描述）。\n"
        "只输出 JSON 数组，不要其他文字：\n"
        '[{"subject": "...", "description": "..."}]'
    )
    messages = [
        {"role": "system", "content": "你是任务拆解助手。只输出 JSON。"},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await aux_llm_router.chat_completions(
            messages, max_tokens=800, temperature=0.3,
        )
    except Exception as e:
        logger.warning("decompose_with_llm: aux_llm 调用失败（fail-open）: %s", e)
        return []

    # 从响应里抠出文本（不同客户端返回对象或 dict，两种都试）
    raw_text = ""
    try:
        choices = getattr(response, "choices", None) or []
        if choices:
            raw_text = choices[0].message.content or ""
        # 某些 client 返回 dict
        if not raw_text and isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                raw_text = msg.get("content", "")
    except Exception as e:
        logger.warning("decompose_with_llm: 解析响应失败: %s", e)
        return []

    # 模型爱把 JSON 包在 ```json ... ``` 代码块里，剥掉这层包装
    text = raw_text.strip()
    if text.startswith("```"):
        # 去掉首行的 ```json 标记和末尾的 ```
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(
            "decompose_with_llm: LLM 输出非合法 JSON（fail-open 返回 []）: %s; raw=%r",
            e, text[:200],
        )
        return []

    if not isinstance(items, list) or not items:
        return []

    task_ids: List[str] = []
    for item in items[:5]:  # 上限 5 条，防模型拆出一大串
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not subject:
            continue
        try:
            task = store.create(subject=subject, description=desc)
            tid = task["id"]
            # 给任务打上"属于哪个 goal"的标记（TaskStore.update 接受任意附加字段）
            store.update(tid, metadata={"goal_id": goal_state.goal_id})
            task_ids.append(tid)
        except Exception as e:
            logger.warning("decompose_with_llm: 创建 task 失败（跳过）: %s", e)
            continue

    goal_state.task_ids = task_ids
    logger.info(
        "decompose_with_llm: 为 goal %s 拆出 %d 个子任务",
        goal_state.goal_id, len(task_ids),
    )
    return task_ids


def check_all_tasks_done(goal_state: "GoalState") -> bool:
    """检查这个 goal 名下的子任务是不是全部干完了。

    判断规则（宁慢勿错）：
    - 一条子任务都没有 → 返回 False（不能凭空宣布 goal 完成）
    - 有任何一条没完成 → False
    - 全部完成 → True

    参数：
    - goal_state：要检查的目标状态对象

    返回：True=全干完了，可以收工；False=还没。
    """
    if not goal_state.task_ids:
        return False

    from agent.task_store import get_task_store
    store = get_task_store()
    for tid in goal_state.task_ids:
        task = store.get(tid)
        if task is None:
            # 子任务被删了（可能是用户手动删的）→ 当没这条继续查，别卡死 goal
            continue
        if task.get("status") != "completed":
            return False
    # 能走到这里说明：至少有一条子任务，且没有查到未完成的
    return True
