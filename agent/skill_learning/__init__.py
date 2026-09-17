# -*- coding: utf-8 -*-
"""skill_learning 子包：instinct 行为记忆。

instinct（本能/条件反射）= 从对话轨迹里提炼出的「遇到什么情况 → 该怎么
做」的习惯，比如「用户要求跑测试 → 先跑 pytest 再汇报」。每重复观察到
一次，这条习惯的可信度就涨一点；攒够了之后，演化层会把它变成正式技能。

四个零件：observer（观察器，从轨迹里找习惯）、llm_observer（用辅助 LLM
做同样的观察，更聪明但有成本）、store（习惯的存取与累积）、evolver
（攒够的习惯升级成技能 MD）。
"""
from agent.skill_learning.store import Instinct, InstinctStore
from agent.skill_learning.observer import observe_turn
from agent.skill_learning.evolver import maybe_evolve
from agent.skill_learning.llm_observer import (
    observe_turn_llm,
    reset_llm_observer_state,
)

__all__ = [
    "Instinct", "InstinctStore", "observe_turn", "maybe_evolve",
    "observe_turn_llm", "reset_llm_observer_state",
]
