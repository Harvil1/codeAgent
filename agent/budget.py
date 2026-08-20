"""迭代预算：给一次对话设「最多调多少次 LLM」的上限，防止失控烧钱。

是什么：一个线程安全的计数器。每调一次 LLM API 就扣 1 个名额；
名额用完之前，通过 grace_call（宽限发言）给模型最后一次机会把话说完
收尾，而不是戛然而止。

在项目里的位置：由 AIAgent 主循环消费——每轮先 consume() 问「还有名额吗」，
没有就收尾。程序化的工具调用（如 execute_code）不算模型主动决策，
可以用 refund() 把名额退回来。
"""

import threading


class IterationBudget:
    """线程安全的迭代预算计数器（加锁是因为主循环和后台线程都会碰它）。"""

    def __init__(self, total: int):
        """初始化预算。

        参数：
            total —— 总名额数（负数会被压到 0，即「一步都不许走」）
        """
        self._total = max(0, int(total))
        self._consumed = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        """总名额数。"""
        return self._total

    @property
    def consumed(self) -> int:
        """已经用掉的名额数。"""
        with self._lock:
            return self._consumed

    @property
    def remaining(self) -> int:
        """还剩多少名额（不会小于 0）。"""
        with self._lock:
            return max(0, self._total - self._consumed)

    def consume(self) -> bool:
        """尝试扣 1 个名额。

        返回：True = 扣成功，可以继续；False = 名额已用完，该收尾了。
        """
        with self._lock:
            if self._consumed < self._total:
                self._consumed += 1
                return True
            return False

    def refund(self) -> None:
        """退还 1 个名额（最多退到没超支的程度）。

        背景：程序化工具调用（如 execute_code）花的这轮不该算在模型
        头上，用完退回来。
        """
        with self._lock:
            if self._consumed > 0:
                self._consumed -= 1

    def reset(self, total: int = None) -> None:
        """清零重新计数（开始新会话时调用）。

        参数：
            total —— 新的总名额；不传则沿用原来的总数
        """
        with self._lock:
            self._total = max(0, int(total)) if total is not None else self._total
            self._consumed = 0

    def __repr__(self) -> str:
        return f"IterationBudget(remaining={self.remaining}/{self._total})"
