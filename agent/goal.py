# agent/goal.py
"""Goal 驱动系统：跨多轮 LLM 调用自动持续追目标。

设计：
- 同步阻塞（/goal 后 CLI 阻塞，agent 自动多轮跑直到 pause/complete）
- pause/resume/continue/clear 子命令
- 网络断开 / budget 超限自动 pause
- 通过 ephemeral user 消息驱动下一轮（不动 system prompt，保护 cache）
- 持久化到 ~/.OmniMate/.goal/current.json（断电恢复）

本文件只含状态机 + 持久化。主循环集成在 Task 11。
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
    """目标状态。单例（同时只有一个 active goal）。"""

    objective: str
    goal_id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}")
    status: str = "active"  # active / paused / completed / failed / cancelled
    created_at: float = field(default_factory=time.time)
    iteration_count: int = 0
    token_budget: int = 0
    token_budget_limit: Optional[int] = None  # None=不限
    pause_reason: Optional[str] = None  # network / budget / manual / completed
    task_ids: List[str] = field(default_factory=list)
    last_progress: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # ---- 状态转换 ----

    def pause(self, reason: str = "manual") -> None:
        """暂停。reason: manual / network / budget_exceeded。"""
        self.status = "paused"
        self.pause_reason = reason
        self.notes.append(f"paused: reason={reason}, iteration={self.iteration_count}")

    def resume(self) -> None:
        """恢复（清除 pause_reason）。"""
        self.status = "active"
        self.pause_reason = None
        self.notes.append(f"resumed: iteration={self.iteration_count}")

    def complete(self) -> None:
        self.status = "completed"
        self.pause_reason = "completed"
        self.notes.append(f"completed: iteration={self.iteration_count}")

    def cancel(self) -> None:
        self.status = "cancelled"
        self.pause_reason = "cancelled"
        self.notes.append(f"cancelled: iteration={self.iteration_count}")

    def fail(self, reason: str = "") -> None:
        self.status = "failed"
        self.pause_reason = f"failed:{reason}" if reason else "failed"
        self.notes.append(f"failed: {reason}, iteration={self.iteration_count}")

    # ---- 每轮评估 ----

    def evaluate_after_turn(
        self,
        tokens_used: int = 0,
        all_tasks_done: bool = False,
    ) -> str:
        """每轮 LLM 调用后评估。返回决策：continue / pause / complete。

        - all_tasks_done=True → complete
        - 超 token_budget_limit → pause(budget_exceeded)
        - 否则 → continue
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

    # ---- 持久化 ----

    def save(self, path: Path) -> None:
        """原子写（tmp + rename）。fail-open：失败只 log。"""
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
        """加载。文件不存在 → None；损坏 → None + warning。"""
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
# CCAR8 Task 12 NEW: TaskStore 集成（decompose + check_done）
# =============================================================================


async def decompose_with_llm(
    goal_state: "GoalState",
    objective: str,
    aux_llm_router,
) -> List[str]:
    """用 aux_llm 把目标拆解为子 task，写入 TaskStore。返回 task_id 列表。

    设计：
    - aux_llm 不可用 → 返回 []（fail-open，goal 仍可跑，只是没有子任务追踪）
    - LLM 输出非合法 JSON / 空数组 → 返回 []
    - 成功解析后，每个 item 在 TaskStore.create + update metadata.goal_id

    Args:
        goal_state: 要填充 task_ids 的 GoalState（本函数会原地修改 goal_state.task_ids）
        objective: 目标文本（送给 LLM 拆解）
        aux_llm_router: AuxLLMRouter 实例（或任何有 async chat_completions 的对象）

    Returns:
        task_id 列表（可能为空）
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

    # 提取文本
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

    # 容忍 LLM 可能加 ```json ... ``` 包装
    text = raw_text.strip()
    if text.startswith("```"):
        # 去掉首行 ```json 和末尾 ```
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
    for item in items[:5]:  # 最多 5 个
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not subject:
            continue
        try:
            task = store.create(subject=subject, description=desc)
            tid = task["id"]
            # metadata 字段标记归属 goal（TaskStore.update 接受任意字段）
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
    """检查所有关联 task 是否全部 completed。

    - 无 task_ids → False（保守，不自动 complete goal）
    - task_ids 中有任一非 completed → False
    - 全部 completed → True
    """
    if not goal_state.task_ids:
        return False

    from agent.task_store import get_task_store
    store = get_task_store()
    for tid in goal_state.task_ids:
        task = store.get(tid)
        if task is None:
            # task 被删了 → 视为未完成（可能用户手动删的，让 goal 继续）
            continue
        if task.get("status") != "completed":
            return False
    # 至少存在一个 task 且全部 completed
    return True
