"""第 7 轮 bug 修复测试：S8 _budget_grace_call 死代码。"""
import inspect


def test_budget_grace_call_is_set_after_dispatch_when_budget_exhausted():
    """S8 fix: run_conversation 源码在 dispatch_tool_calls 后，
    如果预算耗尽应置 _budget_grace_call = True。

    bug：之前 _budget_grace_call 只在 __init__ 设 False，全代码无 True 赋值，
    while 条件 `or self._budget_grace_call` 永远 False 分支死。
    """
    from agent import AIAgent
    src = inspect.getsource(AIAgent.run_conversation)
    # 应含 _budget_grace_call = True 的赋值（不是 ==，不是 !=）
    assert "_budget_grace_call = True" in src, (
        f"run_conversation 应在 dispatch 后置 _budget_grace_call = True，"
        f"实际源码不含该赋值。\n相关片段:\n{src[src.find('tool_calls')-50:src.find('tool_calls')+500]}"
    )


def test_budget_grace_call_cleared_after_use():
    """S8 fix: grace call 跑完后应清标志（防无限循环）。

    bug：之前 grace 跑完不清，会导致 while 无限循环（grace 永远 True）。
    """
    from agent import AIAgent
    src = inspect.getsource(AIAgent.run_conversation)
    # 应含 grace 跑完后置 False 的逻辑（在 else 分支或类似）
    # 关键：源码应含 _budget_grace_call = False（不只是 __init__ 的初始值）
    # 用源码不含 __init__ 的方式检查（只看 run_conversation）
    false_assigns = []
    idx = 0
    while True:
        p = src.find("_budget_grace_call = False", idx)
        if p == -1:
            break
        false_assigns.append(p)
        idx = p + 1
    # run_conversation 内应至少有 1 处 False 赋值（清标志）
    # __init__ 的赋值不在 run_conversation 里，所以这里找到的都是"清"逻辑
    assert len(false_assigns) >= 1, (
        f"run_conversation 应在 grace 跑完后置 _budget_grace_call = False（防无限循环），"
        f"实际未找到。"
    )


def test_budget_grace_call_uses_iteration_budget_remaining_check():
    """S8 fix: 触发条件应基于 iteration_budget.remaining（不是 max_iterations）。"""
    from agent import AIAgent
    src = inspect.getsource(AIAgent.run_conversation)
    # 应含 iteration_budget.remaining 检查 + grace 赋值的关联
    # 简化：源码应含 "iteration_budget.remaining" + "_budget_grace_call = True"
    assert "iteration_budget.remaining" in src, "应用 iteration_budget.remaining 判断"
    assert "_budget_grace_call = True" in src, "应置 grace"


def test_budget_grace_call_still_in_while_condition():
    """回归：while 条件仍含 _budget_grace_call（保证 grace 能让循环多跑一轮）。"""
    from agent import AIAgent
    src = inspect.getsource(AIAgent.run_conversation)
    # while 条件应含 _budget_grace_call
    assert "self._budget_grace_call" in src, (
        "while 条件应保留 _budget_grace_call（grace 机制入口）"
    )
