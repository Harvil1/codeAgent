# CCAR11 Task 4 Report: /add-dir 运行时白名单 + 持久化

> 注：本文件路径之前承载过 CCAR8 TraceSink / CCAR9 /init / CCAR10 subagent_resume 等报告，本份覆盖为 CCAR11 Task 4。

**Commit:** `955776d6`
**Branch:** cli-dev
**Date:** 2026-08-14
**Status:** 完成

## 任务范围

`/add-dir <path>` 命令：运行时追加 safe_path 写白名单 + 持久化到用户 config（`security.extra_allowed_roots`），下次启动自动加载。

## 实施步骤

### Step 1：读现状（brief 点名要先查的三件事）

1. **allowed_roots 的真实存取方式**：`safe_path` 的 `allowed_roots` 是**函数参数**（不是 PermissionChecker 实例字段），缺省时 fallback 到 `default_allowed_roots()`（workspace cwd + `~/.OmniMate`）。`PermissionChecker.check_path`（write_file 实际走的路径）根本不做白名单检查（闸门 3 全通过）。→ 结论：运行时追加要生效，必须挂在 `default_allowed_roots()` 上，做成 permission.py 模块级注册表。
2. **config 文件定位**：`constants.config_path()` → `~/.OmniMate/config.yaml`（`OMNIMATE_HOME` env 可覆盖）。注意 `load_config()` 默认优先读 settings.json（新 JSON 配置），config.yaml 是旧路径但仍是用户可手编的文件——持久化按 brief 走 yaml 读-改-写。
3. **cli 命令接入模式**：`_handle_command` 的 `if name == "/xxx"` 链 + Task 3 的 `_handle_status_cli` 系列 + help Panel + `tests/test_cli_commands.py` 的 `_FakeRT` 模式。

### Step 2：TDD——先写 7 个失败测试

追加到 `tests/test_cli_commands.py`：列白名单 / 拒绝不存在的目录 / **添加后 safe_path 放行 + config 出现该路径**（核心断言）/ 幂等 / 持久化不破坏其他字段 / 启动加载 / help 含 /add-dir。首跑 7 failed 确认红。

### Step 3：实现

1. **`agent/permission.py`**：`_EXTRA_ALLOWED_ROOTS` 模块级注册表 + `add_extra_allowed_root`（resolve + 去重幂等，返回是否新增）/ `list_extra_allowed_roots`（拷贝）/ `clear_extra_allowed_roots`（测试用）；`default_allowed_roots()` 末尾 `roots.extend(_EXTRA_ALLOWED_ROOTS)`。
2. **`cli.py`**：
   - `_persist_extra_root(root, config_file=None)`：yaml.safe_load 已有 config → `security.extra_allowed_roots` append（去重）→ `yaml.safe_dump(allow_unicode=True, sort_keys=False)` 写回；**代码注释注明注释会丢**；读失败按空配置处理（fail-open），返回是否新写入。
   - `_load_persisted_extra_roots(config)`：启动把 config 里的列表灌进运行时注册表（单条失败跳过）。
   - `_handle_add_dir_cli(args, rt)`：无参数列白名单（默认根 + 标注 /add-dir 追加项）；带参数 resolve + is_dir 校验 → 运行时追加 → 持久化（持久化失败仅黄字警告，运行时仍生效）→ 按新增/已存在分别提示。
   - `RuntimeContext.__init__` 开头调 `_load_persisted_extra_roots`（try/except 包裹）。
   - `_handle_command` 加 `/add-dir` 路由 + `_show_help` 加一行。
3. **`config.py`**：`DEFAULT_CONFIG["security"]` 加 `"extra_allowed_roots": []`（可发现性）。

### Step 4：修一处测试措辞

`test_add_dir_idempotent` 断言文案与实现输出对齐（"已在白名单"）。

## 测试结果

- 新增 7 个测试全过；`tests/test_cli_commands.py` 42 passed
- 全套 `uv run pytest tests/`：**2248 passed, 1 skipped**
- `uv run python scripts/verify.py`：**22/22 ALL PASS**

## 关键设计点 / concerns

1. **白名单不能绕过硬底线**：`safe_path` 里受保护路径（~/.ssh / /etc / C:\Windows）和项目代码写保护**先于**白名单检查，`/add-dir` 加任意目录都绕不过（代码注释已写明）。
2. **`check_path` 不查白名单**：`write_file` 走的是 `PermissionChecker.check_path`（闸门 3 全通过），所以 `/add-dir` 的运行时效果只作用于 `safe_path` 默认路径的调用方（offload/transcript 等显式传 allowed_roots 的不受影响）。与 brief 要求一致（测试核心断言就是 `safe_path(target/"x.txt", write=True)` 通过），但语义上 /add-dir 对 write_file 工具没有收紧或放宽效果——留 follow-up 如需让 write_file 也感知。
3. **yaml 注释会丢**：`yaml.safe_dump` 重写整个文件，用户 config.yaml 里手写的注释无法保留（yaml 格式限制），函数 docstring + 代码注释均注明。
4. **settings.json vs config.yaml 双轨**：`load_config()` 默认读 settings.json，`/add-dir` 持久化写的是 config.yaml——两轨并存时用户在 settings.json 手写的 `security.extra_allowed_roots` 也会被启动加载（`_load_persisted_extra_roots` 读的是 load_config 合并结果），但 `/add-dir` 只写 yaml 侧。
5. **注册表是进程级全局**：`_EXTRA_ALLOWED_ROOTS` 所有线程共享（与 default_allowed_roots 语义一致）；`list_extra_allowed_roots` 返回拷贝防外部篡改。
