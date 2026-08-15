# -*- coding: utf-8 -*-
"""skill_learning 子包：instinct 行为记忆（CCAR15）。

从对话轨迹中提炼 "trigger → action" 的条件反射式习惯，
置信度随重复观察累积，供技能生成/推荐层消费。
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
