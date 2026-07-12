"""迭代预算（工具调用次数上限）。

每次 LLM API 调用消耗 1 个预算。
预算耗尽后通过 grace_call 给模型最后一次说话的机会。
支持 refund()，用于 execute_code 等程序化调用退款。
"""

import threading


class IterationBudget:
    """线程安全的迭代预算计数器。"""

    def __init__(self, total: int):
        self._total = max(0, int(total))
        self._consumed = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self._total

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self._total - self._consumed)

    def consume(self) -> bool:
        """尝试消耗一次迭代。返回是否允许。"""
        with self._lock:
            if self._consumed < self._total:
                self._consumed += 1
                return True
            return False

    def refund(self) -> None:
        """退还一次迭代（用于程序化工具调用，如 execute_code）。"""
        with self._lock:
            if self._consumed > 0:
                self._consumed -= 1

    def reset(self, total: int = None) -> None:
        """重置预算（开始新会话时调用）。"""
        with self._lock:
            self._total = max(0, int(total)) if total is not None else self._total
            self._consumed = 0

    def __repr__(self) -> str:
        return f"IterationBudget(remaining={self.remaining}/{self._total})"
