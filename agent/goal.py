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
