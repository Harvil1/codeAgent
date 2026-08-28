"""唤醒预算：给"空闲自动唤醒"记一本账，防自激励循环。

背景（大白话）：后台任务完成会自动唤醒主循环，但被唤醒的这轮对话
可能又启动新的后台任务/异步子代理——它们的完成通知又会再唤醒主循环。
没人管的话这个循环能自己转一整晚，白烧 token。

解法（借鉴 dsh 的 self-exciting wake budget）：
- 连续自动唤醒超过上限（默认 3 次）→ 不再自动唤醒，通知留在队列里，
  等用户下次真实输入时照常消费（不丢通知，只是不主动跑）；
- 只有用户真实输入才把预算回血——插件/后台自己触发的唤醒不许回填。

边界：纯内存计数，不落盘；与"哨兵被吞=通知已被消费"的自愈语义
互不影响（预算只决定"要不要为通知跑一轮"，不管哨兵本身）。
"""


class WakeBudget:
    """连续唤醒的记账本。用法：主循环里收到唤醒哨兵时 consume()，
    收到用户真实输入时 reset()。"""

    def __init__(self, max_wakes: int = 3):
        """建账本。

        参数：
            max_wakes：最多连续自动唤醒几次；0=完全禁用自动唤醒
        """
        self._max = max(0, int(max_wakes))
        self._used = 0

    def consume(self) -> bool:
        """尝试占一个唤醒名额。

        返回：True=还有名额，可以唤醒；False=额度用完（或已禁用），
        调用方应跳过本轮唤醒，通知留给用户下次输入时消费。
        """
        if self._max <= 0:
            return False
        if self._used >= self._max:
            return False
        self._used += 1
        return True

    def reset(self) -> None:
        """用户真实输入到达时回血预算（清零重计）。"""
        self._used = 0

    @property
    def used(self) -> int:
        """已连续消耗的名额数。"""
        return self._used

    @property
    def exhausted(self) -> bool:
        """额度是否已用完（禁用状态恒 False，由 consume 直接拒绝）。"""
        return self._max > 0 and self._used >= self._max
