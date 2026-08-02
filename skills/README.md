# 内置技能目录

本目录放**项目自带的技能**(跟 git 走,装哪台机器都一样)。

## 内置 vs 用户技能

| 类型 | 位置 | 谁创建 | 跨机器 |
|---|---|---|---|
| **内置技能** | `项目/skills/`(本目录) | 开发者 | 跟代码走(git) |
| **用户技能** | `~/.OmniMate/skills/` | 用户 / agent | 跟用户数据走(需 rsync) |

同名时**用户目录优先**(用户能覆盖内置)。

## 怎么加内置技能

```bash
mkdir skills/<技能名>
# 写 SKILL.md(frontmatter: name + description + 正文)
```

启动 agent 自动扫到,LLM 看到 `/<技能名>` 索引。

## 内置流程技能（superpowers 方法论移植）

| 技能 | 用途 | 触发场景 |
|---|---|---|
| `/using-omnimate` | 技能总纲(入口) | 任何任务开始前,不确定用哪个流程技能时 |
| `/brainstorming` | 设计先行(HARD-GATE:未获批不得写码) | "做一个功能/改行为" |
| `/writing-plans` | 写实现计划 | 设计获批后 |
| `/executing-plans` | 分会话执行计划(带检查点) | 有书面计划,跨会话执行 |
| `/subagent-driven-development` | 同会话逐任务派子代理执行计划 | 计划任务相互独立,本会话内执行 |
| `/systematic-debugging` | 系统化调试(先找根因再修) | "排查 bug/测试失败" |
| `/test-driven-development` | TDD(先写失败测试) | 写功能/修 bug 前 |
| `/verification-before-completion` | 验证再收尾(证据先于断言) | 声称完成/修好前 |
| `/requesting-code-review` | 请求评审(派评审子代理) | 完成任务/合并前 |
| `/receiving-code-review` | 接受评审(技术严谨,不盲从) | 收到评审意见时 |
| `/dispatching-parallel-agents` | 并行派子代理 | 2+ 个独立任务 |
| `/finishing-a-development-branch` | 分支收尾(合并/PR/清理) | 功能完成测试通过后 |
| `/using-git-worktrees` | worktree 隔离工作区 | 需要隔离工作区避免冲突 |
| `/writing-skills` | 写/维护技能 | 沉淀方法/改技能 |

subagent-driven-development 附 3 个 Python 辅助脚本(`scripts/`):
`task-brief.py` / `review-package.py` / `sdd-workspace.py`。
