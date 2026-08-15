# Task 7 报告：ConfigTool（LLM 安全改配置）

**状态**：完成
**Commit**：`8e05798a`
**测试**：`tests/test_config_tool.py` 38 个新断言全过；全套 2408 passed / 1 skipped；verify 22/22 PASS

## 交付物

| 文件 | 变更 |
|---|---|
| `tools/config_tool.py` | 新建：`config_get`（SAFE）+ `config_set`（UNSAFE） |
| `toolsets.py` | `_CORE_TOOLS` +2（config_get / config_set） |
| `tests/test_config_tool.py` | 新建：38 测试（白名单/get/set/hook/契约） |
| `tests/test_tool_concurrency_classification.py` | config_get 入 SAFE（16→17）、config_set 入 UNSAFE |

## 实现要点

1. **白名单精确匹配**：模块级 `frozenset` 7 键，`key not in _CONFIG_WHITELIST` 直接 `permission_denied`（错误信息带白名单清单，LLM 可自行纠正到合法键）。前缀相似键（`notifications`、`notifications.enabledx`）测试验证拒收。
2. **config_set 三步走**：
   - 读-改-写 settings.json（`load_settings`/`save_settings`，CCAR11 轨道，深拷贝防 DEFAULT_SETTINGS 污染）
   - runtime 生效：`_set_nested`（setdefault 逐层建）同步 `agent_ref.config` 同键——sync handler 只改 dict 无 contextvar，Task 6 教训不适用
   - CONFIG_CHANGE hook（`run_config_change`，payload 带 changed_keys/old_value/new_value/session_id，fail-open：hook 异常不影响配置写入）
3. **类型强转**：`_coerce_value` 按现有值类型——bool 先判（bool 是 int 子类，顺序不能反）；int 键传 "abc" → `invalid_args` 不落盘。类型推断优先 settings.json 现有值，磁盘无值再参考 runtime config。
4. **config_get 优先 runtime**：`agent_ref.config` 有值先回（反映本会话实际生效值，磁盘可能落后），无 agent_ref 或 runtime 无值时读 settings.json；键从未设置 → `value: null` 不报错。
5. **契约**：handler 签名 `(args, **dispatch_kwargs)`（inspect VAR_KEYWORD 验证）；schema 用 OpenAI `"parameters"` 键（CCAR11 第 5 例 silent-dead-code 教训防回归）；utf-8 编码 + 中文注释。

## TDD 过程

先写测试（红：`ModuleNotFoundError: tools.config_tool`）→ 实现 → 38/38 绿。中途 1 个测试自身断言错误（`"llm" not in load_settings()`——DEFAULT_SETTINGS 本来就含 llm 段），修正为验证 `auth_token` 未被写脏值。

## Concerns

- `memory.curator.enabled` / `context.reactive_compact_enabled` 写入 settings.json 后，**下次启动** RuntimeContext 是否从 settings 同名路径读取未逐一核对（runtime 生效已兜底本会话）；若启动装配用不同键名，持久化值会静默不生效——follow-up 可在启动装配处核对 7 键映射。
- config_get 对 runtime 与磁盘不同的键只回 runtime 值；如需对比视图可后续加 `source` 字段。

## FIX（review：白名单 2 个 dead key + trace.enabled 语义修正）

Reviewer 抓到第 6 次 silent-dead-code 预演：白名单里 2 个键全仓无读取点，写进去是黑洞还假报生效。

### 修了什么

1. **白名单换 2 键**（`tools/config_tool.py:_CONFIG_WHITELIST`）：
   - 移除 `trace.retention_days` —— config.py:364 有默认值但 `agent/trace.py` 无任何 retention/cleanup 逻辑，全仓无读取点（写它无效）
   - 移除 `context.reactive_compact_enabled` —— 真实开关是 `features.reactive_compact.enabled`（`agent/__init__.py:1650` 走 `is_feature_enabled`）
   - 换入 `context.reactive_compact_cooldown_seconds` + `context.reactive_compact_max_per_session` —— grep 验证 `agent/__init__.py:1660-1663` reactive_compact 分支有真实读取点（`ctx_cfg.get(...)`）
2. **poor_mode dead write 顺手修**（`agent/poor_mode.py:POOR_PRESET`）：`context.reactive_compact_enabled: False` → `features.reactive_compact.enabled: False`（真实开关；reactive 默认 OFF 与 poor "全关"语义一致——即使 features 被用户开过也压回 False）
3. **trace.enabled 语义修正**（`tools/config_tool.py:_NEXT_SESSION_KEYS`）：TraceSink 在 cli initialize 一次性装配（cli.py:435），会话中改 config 不重接线——`config_set` 对该键返回 `runtime_applied: "next_session"`（值照常同步+落盘，下次会话生效），不再假报 True
4. **测试同步**（+5 测试）：白名单含新 2 键 / dead 2 键拒收回白名单（负向）/ `config_set` dead 键 permission_denied / trace.enabled 返回 next_session / 其他键仍 runtime_applied=True；poor_mode 期望键与 flips 测试更新

### 验证

- `uv run pytest tests/test_config_tool.py tests/test_poor_mode.py -q` → 42 passed
- 全套 → **2413 passed / 1 skipped / 0 failed**（baseline 2408，净增 5）

### Commit

`<见 git log：fix(config_tool): 白名单换掉 2 个 dead key + trace.enabled 语义修正>`
