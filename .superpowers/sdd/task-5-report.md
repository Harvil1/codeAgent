# CCAR12 Task 5 报告：MCP Resources 协议

## 状态：完成

## 改动清单

| 文件 | 改动 |
|---|---|
| `agent/mcp_client.py` | `MCPTransport` 基类加 `list_resources()` / `read_resource(uri)` 默认实现（复用子类通用 `send_request`，fail-open 返 None，4 种具体 transport 均无需 override）；`MCPClient` 加两方法透传；`MCPManager` 加 `list_resources(server)` / `read_resource(server, uri)`（错误风格与 `call()` 对齐：统一返 JSON 可序列化 dict） |
| `tools/mcp_tool.py` | `register_mcp_tools` 末尾调 `_register_resource_tools(manager)`：每个连接中的 server 注册 `mcp__<server>__list_resources`（无参）+ `mcp__<server>__read_resource`（uri 参数） |
| `tests/test_mcp_resources.py` | 21 个测试（新增，`git add -f`） |

## 工具注册形态的实际选择（关键决策）

**没走 brief 里的静态名 `mcp_list_resources(server)`，改成动态 per-server 命名 `mcp__<server>__list_resources` / `mcp__<server>__read_resource`**，原因：

1. **静态名对 LLM 不可见**：`TOOLSETS["mcp"]["tools"]` 是空列表（`toolsets.py:86`），mcp 工具集全靠 `model_tools.py:get_tool_definitions` 按 `mcp__` 前缀动态发现。静态名 `mcp_list_resources`（单下划线）不匹配前缀，注册了也发不出去——除非改 toolsets.py 发明新机制，违背"按项目现有模式定，别发明新机制"。
2. **动态命名空间天然继承全部既有机制**：catalog 精简条目（省 token）+ `tool_search` 按需取详细 schema + `mcp_server_filter`（子代理限定 server）+ per-server `check_fn` 门控（server 断开自动隐藏），零额外代码。
3. brief 的功能语义（给定 server 列/读 resources）完整保留——server 编码进工具名，`uri` 仍是 read 工具的参数。

## 行为细节

- **协议方法**：`resources/list`（params `{}`，解析 `resp["resources"]`）/ `resources/read`（params `{"uri": ...}`，返回含 `contents`）
- **fail-open 分层**：transport 层任何异常（含 server 返回 JSON-RPC error = 不支持）→ None；manager 层把 None 翻译成友好错误 `{"error_type": "mcp_resources_unsupported", "error": "...不支持 resources 协议"}`
- **server 不存在/断开**：`{"error_type": "mcp_server_not_connected" / "mcp_server_disconnected"}`（对齐 `manager.call` 的错误风格）
- **schema 键**：`"parameters"`（OpenAI 格式，CCAR11 教训，测试显式断言）
- **isConcurrencySafe=False**：与其他 MCP 工具一致保守（外部进程/网络调用走串行路径）
- **不支持 resources 的 server 工具仍注册**：调用时返友好错误而非启动时探测隐藏（能力探测留 follow-up）

## 测试

- 新增 `tests/test_mcp_resources.py`：21 个
  - transport 协议方法名/参数断言 5（含 fail-open、空响应、4 transport 未 override 走基类）
  - MCPClient 透传 2
  - MCPManager 错误/成功 7（server 缺失/断开/不支持/成功/缺 uri）
  - 注册 + 端到端 dispatch + check_fn 门控 5
  - 契约 2：handler 签名 `(args, **kwargs)`（CCAR8 教训）+ schema 用 `"parameters"` 键 + mcp__ 前缀不受分类测试 expected_safe_count 影响
- 全套 `uv run pytest tests/`：**2351 passed, 1 skipped**（无回归）
- `uv run python scripts/verify.py`：**22/22 PASS**
- 现有 `test_mcp.py::test_register_mcp_tools` 不受影响（MagicMock `_clients` 迭代为空 → 不注册 resources 工具，count 仍为 1）

## Concerns / Follow-up

1. **register_mcp_tools 返回计数语义变化**：resources 工具计入返回值（每 server +2）。`initialize_mcp` 只用它打日志，无调用方依赖具体数字，但如有外部脚本断言精确数需注意。
2. **工具名冲突**：若某 server 自己暴露叫 `list_resources` 的普通工具，会与本注册同名互相覆盖（registry 同 toolset 覆盖静默）——极端场景，未特判。
3. **能力探测未做**：不支持的 server 也注册两工具，LLM 调用才拿到"不支持"错误。可后续在 connect 时探测 capabilities 决定是否注册。
4. brief 原文的静态命名若 reviewer 坚持要，需同步改 `toolsets.py` + `model_tools.py` 的 built-in/mcp 拆分逻辑，成本高于收益，建议维持现状。

（注：本文件原是 CCAR10 Task 5 的 /resumable 报告，按本轮任务指定路径覆盖；CCAR10 内容已在 git 历史 `1d9374c3` 中可查。）
