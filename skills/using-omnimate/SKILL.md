---
name: using-omnimate
description: "技能总纲。任何任务(尤其创造性工作/调试/写计划/执行计划)开始前先扫技能索引,判断有无适用流程技能。| Meta-skill: check the skill index and invoke the matching process skill before any task."
---

# Using OmniMate Process Skills

This is the entry skill for OmniMate's process-skill library (ported from the
superpowers methodology). **Invoke relevant skills BEFORE any response or
action** — including clarifying questions, exploring the codebase, or checking
files. If a skill turns out wrong for the situation, you don't have to use it.

## Skill Priority

Process skills come first — they set the approach, then domain/implementation
work carries it out:

- "Let's build X" → `brainstorming` first, then `writing-plans`
- "Fix this bug" → `systematic-debugging` first
- "Execute this plan" → `subagent-driven-development` (same session) or `executing-plans` (parallel session)
- "Review this work" → `requesting-code-review`
- "Done, wrap up" → `verification-before-completion` then `finishing-a-development-branch`

## Red Flags (stop and load the matching skill)

| Thought | Reality |
|---|---|
| "This is just a simple question" | Questions are tasks. Check the skill index. |
| "I need more context first" | Skill check comes BEFORE clarifying questions. |
| "Let me explore the codebase first" | Skills tell you HOW to explore. Check first. |
| "This doesn't need a formal skill" | If a skill exists, use it. |

## Core Rule

Every feature goes through brainstorming → writing-plans → execution. Do not
skip the design step because a task seems simple — unexamined assumptions
waste the most time.

## Asking the User

When a process skill (e.g. `brainstorming`) says to ask the user clarifying
questions, ask through the `ask_user` tool — one question at a time,
2-4 mutually exclusive options (multiple choice preferred). `ask_user` blocks
for the answer and works in both CLI and GUI. Do not fall back to asking
inline in your reply unless the question is genuinely open-ended.
