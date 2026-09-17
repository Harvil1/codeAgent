# CodeAgent

> 自学习 AI Agent 命令行工具——交互体验对标 claude code，Windows 优先，开箱即用。

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%E4%BC%98%E5%85%88%20%7C%20Linux%20%7C%20macOS-yellowgreen)]()

[English](./README.en.md) | 简体中文

CodeAgent 是一个跑在终端里的 AI 编程助手：你用自然语言下指令，它读代码、改文件、跑命令、查资料，把活干完。它有自己的一套「长本事」机制——从对话轨迹里沉淀可复用的技能，越用越顺手。

## ✨ 特性

**终端体验（对标 claude code）**
- 流式回答圆角框、思考流暗色框、工具事件行（`● Bash(...)` + `⎿ 结果`）
- 危险命令 / 白名单外写文件的**审批面板**：允许一次 / 总是允许 / 拒绝三选，面板答完自擦不污染上下文
- Ctrl+C 语义分明：回合中一键中断、双击强退；审批面板上 Ctrl+C 一击取消并中断整轮
- 皮肤系统、页脚状态栏、命令补全、输入历史、粘贴大段文本自动落盘引用

**模型与工具**
- 多模型接入：OpenAI 兼容协议（默认 DeepSeek）与 Anthropic 协议均可配；主模型 + 辅助小模型（压缩/检索/记忆提取等杂活走便宜模型）
- 工具自注册体系：`tools/` 下模块 import 即登记，按「套餐」（toolset）控制发给模型的工具集
- MCP 外部工具：项目级 `.mcp.json` 声明即接入，首次连接需审批
- 内置工具覆盖文件读写、终端执行、搜索、代码检索、Web 抓取、后台任务、子代理派发等

**安全防线**
- 破坏性命令审批 + 跨会话白名单（按命令 / 按前缀规则记忆）
- 注入面检测（`$()`、进程替换等「所见非所执行」形态升审批）、删除路径红线、自我保护
- 秘钥扫描（防 API key 写入文件/记忆）、SSRF 防护、Web 内容注入隔离
- Linux 下可选 bwrap 沙箱执行

**记忆与自学习**
- 长期记忆：`MEMORY.md` 索引 + JSONL 存储 + 检索结果临时注入（不污染正式历史）
- 技能系统：用户技能 + 插件市场技能（`/plugin` 管理），支持按触碰文件路径条件激活
- 自学习管线：观察对话轨迹 → 沉淀新技能（启发式 / 辅助模型两种观察方式）

**协作与自动化**
- 子代理：同步 / 后台派发，worktree 隔离，后台代理无人审批时自动 fail-closed
- 团队模式：coordinator / worker / 消息总线 / 邮箱
- 工作流引擎：确定性编排、日志断点续跑、花费封顶
- 定时任务（cron）、声明式 hooks、任务清单、会话 checkpoint 回溯（`/rewind`）

**上下文工程**
- 分级压缩（L1 轻裁剪 → L4 深度摘要）、批间工具摘要、时间基线旧结果清理
- 前缀缓存友好：临时消息注入与剥离时机讲究，不破坏缓存前缀

## 🖥 界面一览

<!-- 截图占位：运行 uv run python main.py 后截两张图（主界面对话 + 审批面板），
     放到 screenshots/ 目录，然后把下面注释解开即可
![主界面](screenshots/main-ui.png)
![审批面板](screenshots/approval-panel.png)
-->

📷 截图待补：主界面对话、审批面板各一张，放 `screenshots/` 后解开上方注释。

## 🚀 快速开始

前提：[Python 3.11+](https://www.python.org/) 和 [uv](https://docs.astral.sh/uv/)（Python 包管理器）。

```bash
# 1) 安装 uv（已有可跳过）
#    Windows (PowerShell):  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    macOS / Linux:         curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 拉代码 + 装依赖
git clone https://github.com/Harvil1/codeAgent.git
cd codeAgent
uv sync

# 3) 配置 API key（二选一）
#    a) 环境变量——按所用厂商设置其一即可，程序会自动识别：
#       DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY /
#       OPENROUTER_API_KEY / ZHIPUAI_API_KEY（BIGMODEL_API_KEY 亦可）
#    b) 写入 ~/.codeAgent/settings.json 的 models.<名字>.api_key
#       （首次运行会自动生成默认配置文件，默认模型为 DeepSeek）

# 4) 跑起来
uv run python main.py              # 交互模式
uv run python main.py -c           # 自动恢复最近一次会话
```

> 提示：换模型 / 换厂商在会话里输 `/model` 按引导配置，或直接编辑 `~/.codeAgent/settings.json`。

## ⌨️ 常用操作

会话内 40+ 个 slash 命令，常用如下：

| 命令 | 作用 |
|---|---|
| `/help` | 命令帮助 |
| `/model` | 切换 / 配置模型（按引导填厂商、key、端点） |
| `/new` `/resume` `/sessions` | 新会话 / 恢复 / 会话列表 |
| `/rewind` | 回溯到某个 checkpoint 重来 |
| `/skills` `/plugin` | 管理技能 / 插件市场 |
| `/memory` | 查看维护长期记忆 |
| `/permission` `/sandbox` `/approved` | 权限模式 / 沙箱开关 / 已批准命令白名单 |
| `/doctor` | 自诊断（配置、网络、组件健康检查） |
| `/plan` | 计划模式：先出方案批准再动手 |
| `/stats` `/usage` `/trace` | 花费统计 / 用量 / 调用链追踪 |

## ⚙️ 配置与数据

运行时数据统一放在 `~/.codeAgent/`（环境变量 `CODEAGENT_HOME` 可覆盖，测试 / 多配置隔离用）：

| 路径 | 内容 |
|---|---|
| `settings.json` | 全部配置（默认值单一源头在代码里，落盘深合并，手改安全） |
| `sessions.db/` | 会话持久化（JSONL） |
| `MEMORY.md` + 记忆库 | 长期记忆索引与正文 |
| `skills/` | 用户技能 |
| `plugins/` | 插件（含官方市场 `claude-plugins-official`） |
| `logs/` | 运行日志（滚动，5MB × 3） |

## 🏗 架构一览

调用链一句话：`main.py`（点火：控制台编码、终端自举）→ `cli.main` → `run_interactive`（REPL + slash 命令 + RuntimeContext 装配）→ `AIAgent.run_conversation`（主循环心脏）。

主循环每轮：组装消息（注入后台 / 定时 / 团队 / 计划等临时消息）→ 上下文压缩 → 刷新工具集 → 调 LLM → 分发工具调用，模型不再要工具即收尾。

| 子系统 | 模块 |
|---|---|
| LLM | `llm_client.py`（OpenAI 兼容 / Anthropic 协议）、`aux_llm.py`（辅助小模型路由） |
| 记忆 | `memory_store.py`、`memory_retriever.py` |
| 上下文 | `context_compressor.py`（分级压缩）、`context_pipeline.py` |
| 子代理 / 团队 | `tools/delegate_tool.py`、`agent/team/` |
| 工作流 | `workflow_engine.py` + `workflow_journal.py` |
| 技能 | `~/.codeAgent/skills/` + `agent/skill_learning/`（自学习） |
| 安全 | `bash_injection.py`、`ssrf_guard.py`、`secret_scanner.py`、`sandbox_runner.py`、`permission.py` |
| 持久化 | `session_store.py`、`task_store.py`、`hooks.py` |

## 🧪 开发

```bash
uv run python scripts/verify.py   # 60+ 项回归检查（改完代码先跑这个）
uv run ruff check .               # lint
```

## ❓ FAQ

- **支持哪些终端？** cmd / Windows Terminal / PowerShell 直接可用；git-bash（mintty）下会自动用 winpty 重新拉起，无需手动处理。
- **支持 Linux / macOS 吗？** Windows 优先开发与验证；Linux 可运行（含 bwrap 沙箱支持），macOS 理论可用、未充分验证。
- **模型必须用 DeepSeek 吗？** 不。任何 OpenAI 兼容端点或 Anthropic 协议端点都可以，`/model` 或改 `settings.json` 即可，key 支持环境变量与配置文件两种方式。
- **会话数据在哪？** 全部在 `~/.codeAgent/`，删掉目录即完全重置。

## 📄 许可证

[MIT](./LICENSE) © 2026 Harvil1
