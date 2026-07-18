# HarvilAgent Hooks 预制库

4 个开箱即用的声明式 hook 脚本。每个脚本独立、跨平台（纯 Python 标准库）。

## 安装

把 `hooks/example-settings.json` 的内容合并到你的 `~/.agent/.hooks/settings.json`：

```bash
# 备份现有配置
cp ~/.agent/.hooks/settings.json ~/.agent/.hooks/settings.json.bak

# 合并示例配置（手工编辑，把需要的 hook 段拷过去）
# 或者直接全量替换（如果还没配过任何 hook）
cp hooks/example-settings.json ~/.agent/.hooks/settings.json
```

重启 HarvilAgent 即生效。

## 4 个 hook

### 1. `secrets_redactor.py`（POST_TOOL_USE）

**作用**：扫描工具输出，把 API key / Bearer token / PEM 私钥等替换为 `[REDACTED:类型]`。
**收益**：防止 accidentally 把密钥暴露给 LLM（被存进 conversation history，可能再被工具调用传出）。
**性能**：~5ms/次（正则扫描）。

### 2. `audit_log.py`（USER_PROMPT_SUBMIT + POST_TOOL_USE）

**作用**：把用户 prompt 和工具结果写到 `~/.agent/.audit.log`（每行一条，含时间戳+会话 ID）。
**收益**：审计 / 调试 / 复盘。可配 `AUDIT_LOG_PATH` 环境变量改路径。
**截断**：prompt 截 500 字符，tool result 截 200 字符。

### 3. `python_syntax_check.py`（PRE_TOOL_USE）

**作用**：`write_file` 写 `.py` 文件时先做 `ast.parse` 语法检查，失败则拒绝写入。
**收益**：防 agent 写坏代码到磁盘（语法错误文件被其他工具 import 会崩）。
**只管 write_file**：read_file / terminal 等工具不拦截。

### 4. `block_large_writes.py`（PRE_TOOL_USE）

**作用**：`write_file` 写超大文件（默认 5MB）时拒绝。
**收益**：防 agent 失控写爆磁盘（如循环写日志）。
**可调**：`MAX_WRITE_BYTES` 环境变量改阈值。

## IPC 协议

每个 hook 是独立子进程：
- stdin 收 JSON payload（含 event / session_id / 事件字段）
- stdout 输出 JSON 响应（空 = 不修改，原数据继续）
- 非零退出 / 解析失败 / 超时 → fail-open（视为不修改）

详细 payload 字段见 `agent/hooks.py` 对应 event 的 `_invoke_declarative_*` 方法。

## 关闭某个 hook

编辑 `~/.agent/.hooks/settings.json`，删掉对应条目重启即可。

## 写自己的 hook

复制任意一个现成 hook 改逻辑。关键：
1. `json.load(sys.stdin)` 读 payload
2. 处理后 `print(json.dumps({...}))` 输出（或不输出 = allow）
3. 任何异常都应 fail-open（不要让 agent 主循环因 hook 崩）
