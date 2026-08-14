# CCAR11 Task 3 报告：/status + /doctor + /diff 三命令

**Status**: DONE
**Date**: 2026-08-14

## 实现内容

### cli.py（修改）

1. **`_handle_status_cli(args, rt)`** — Rich Table「Status 一览」，6 段每段独立 try（`_status_row` helper，段内异常 → 显示「读取失败：…」不影响其他段）：
   - 主模型（`rt.config["model"]` name + provider）
   - aux LLM（`rt.agent.aux_llm_router` 有无 → 已配置/未配置）
   - goal（`rt.agent._goal_state`：objective 前 30 字 + status + iteration_count；无 → 「无 active goal」）
   - 项目记忆键（`rt._statusline_project_key`，空 → 提示）
   - MCP：`agent.mcp_client.get_mcp_manager()` 遍历 `_clients`，逐 server 打 is_connected（已连接/断开，带颜色）；空 → 「无已注册 server」
   - 工具总数（`tools.registry.registry.list_all()` 长度）

2. **`_handle_doctor_cli(args, rt)`** — 6 项检查，逐项独立 try/except，输出 `[green]✓[/green]` / `[red]✗[/red]` + 详情 + 汇总「N/6 项通过」：
   1. `load_config()` 成功
   2. API key env（`config["model"]["api_key_env"]`，缺省 `DEEPSEEK_API_KEY`，查 `os.environ`）
   3. agent home 可写（`.doctor_probe` tmp 文件写删；home 取 `rt.home` 回退 `get_omnimate_home()`）
   4. `.mcp.json` 解析（存在才查，`json.loads`；不存在 → 「未配置（跳过）」记 ✓）
   5. 关键依赖 import（rich / httpx / openai 逐个 `importlib.import_module`）
   6. sessions / skills 目录（`mkdir(parents=True, exist_ok=True)` 自动创建也算 ✓）

3. **`_handle_diff_cli(args, rt)`** — 基于 `agent/checkpoint.py` 实际能力（读源码确认，未新建基建）：
   - `rt.checkpoint_mgr` 为 None → 「本会话无 checkpoint 记录」
   - `tracked_files()` 空 → 「本会话无文件改动记录」（附说明：write_file/str_replace 修改过的文件会出现在这里）
   - 有记录 → 列改动文件（`M <path>`）+ `list_snapshots()` 快照数 + `/rewind` 提示

4. **命令链接入**：`_handle_command` 三条 dispatch（紧跟 Task 2 的 /compact /context 之后）+ `/help` 帮助文本 3 行。

## 依赖接口确认（先读再写，未猜）

| 接口 | 实际 API |
|---|---|
| MCP manager | `agent/mcp_client.py:get_mcp_manager()` → `MCPManager._clients: Dict[str, MCPClient]`，client 有 `is_connected` 属性 |
| checkpoint | `CheckpointManager.tracked_files()`（本会话编辑过的文件，去重排序）+ `list_snapshots()`；**无单文件操作类型记录**（只存 path），故 /diff 只标 `M`（modified），不虚构 add/delete |
| API key env | `config.py:DEFAULT_CONFIG["model"]["api_key_env"] = "DEEPSEEK_API_KEY"` |

## 测试

`tests/test_cli_commands.py` 追加 9 个测试（照 `_FakeRT` 模式，git add -f）：

- `test_status_basic` — 全字段断言（模型/goal 前 30 字/status/iter/项目键/MCP 双 server/工具段）
- `test_status_no_goal_no_mcp` — 空 goal / 空 MCP 的提示路径
- `test_status_section_failure_fail_open` — `_statusline_project_key` monkeypatch 成抛异常 property，断言其余段仍输出
- `test_doctor_all_pass` — DEEPSEEK_API_KEY setenv → 「6/6」+「✓」
- `test_doctor_missing_api_key` — delenv → 「✗」且非 6/6
- `test_doctor_config_broken` — monkeypatch `config.load_config` 抛异常 → 第 1 项 ✗ 不崩
- `test_diff_no_checkpoint` — 无 mgr → 提示
- `test_diff_tracked_files` — fake mgr → 列文件 + 快照数
- `test_help_contains_three_commands` — /help 含三命令

全套回归：**2241 passed, 1 skipped**（无回归）。

## 踩坑记录

- rich 会把 `[status=active, iter=3]` 当 markup tag 吞掉（测试断言「active」失败发现）——值里的括号改用全角括号 `（status=…, iter=…）`。
- `monkeypatch.setattr(_FakeRT, "_statusline_project_key", property(...))` 需 `raising=False`（_FakeRT 类本身无该属性，只有实例属性）。

## Concerns

1. `/doctor` 第 1 项 `load_config()` 读的是**真实磁盘配置**（rt.config 是 RuntimeContext 已加载的），测试用 monkeypatch 注入失败——生产语义是「重新加载一次确认配置文件没被改坏」，可接受。
2. `/diff` 的操作类型只有 `M`（checkpoint 只存 path，无 add/delete 语义）——brief 明确「API 不满足列文件清单就够」，未新建基建。
3. `/status` 的 MCP 段直接读 `mgr._clients`（带 lock 的 `servers` property 只给名字）——用 `dict(...)` 拷贝快照，避免迭代期间并发变更；`tools/mcp_tool.py` 内部同样直接访问 `_clients`，属既有惯例。
