# Vision/Image 工具集设计文档

- **日期**：2026-07-17
- **批次**：B1（中等项）
- **范围**：4 Task
- **状态**：设计阶段

---

## 1. 背景与目标

### 1.1 用户场景

让 agent 能处理用户提供的本地图片：
- 截图分析（报错截图、UI 截图）
- 文档 OCR（扫描件、图表文字提取）
- 视觉问答（"图里有几只猫？"）

### 1.2 现状

- `browser_vision` 工具存在但**默认配置下永远报错**——它读 `agent._browser_vision_client`，而该字段从未被赋值
- 没有处理本地图片文件的工具
- LLM client 已支持 OpenAI 兼容 vision API（参考 `tools/browser_tool.py:468-513` 实现）

### 1.3 目标

1. 加 2 个新工具：`image_analyze`（通用 vision）、`image_ocr`（文字提取）
2. 把 `_browser_vision_client` 升级为通用 `_vision_client`，新旧工具共用
3. 修 `browser_vision` 历史遗留 bug（默认配置可用）

### 1.4 非目标（YAGNI）

- ❌ `image_compare`（agent 自己分两步调用即可）
- ❌ URL 输入（用户先 download 或用 browser_navigate+browser_vision）
- ❌ 剪贴板粘贴（CLI 不支持）
- ❌ 图片缓存
- ❌ 新 toolset（加入 `core`，check_fn 检查 vision_client）

---

## 2. 配置设计

`config.py:DEFAULT_CONFIG` 加可选 `vision` 段：

```yaml
vision:
  enabled: true        # 默认 true
  provider: deepseek   # 可选，默认用主 provider
  model: deepseek-chat # 可选，默认用主 model
  max_bytes: 20971520  # 20MB（单图上限）
```

如果 `vision.model` 未配置，**回退到主模型**（假设主模型支持 vision）。

---

## 3. 模块结构

| 文件 | 操作 | 责任 |
|---|---|---|
| `agent/__init__.py` | Modify | `AIAgent.__init__` 加 `self.vision_client = None`（默认；外部注入或 cli.py 初始化） |
| `cli.py` | Modify | `RuntimeContext._create_agent` 根据 `vision` config 创建 `_vision_client` 并注入 |
| `tools/image_tool.py` | Create | 2 工具 handler + schema + 注册 + safe_path/大小/格式检查 |
| `tools/browser_tool.py` | Modify | `browser_vision` 改读 `agent._vision_client`，回退 `_browser_vision_client` |
| `tests/test_image_tool.py` | Create | 12 个单元 + 集成测试 |

`agent/llm_client.py` 不动（复用现有 `create_llm_client`）。

---

## 4. 工具 schema

### 4.1 `image_analyze`

```json
{
  "name": "image_analyze",
  "description": "用 vision LLM 分析本地图片。可描述内容、识别物体、回答视觉问题。",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {"type": "string", "description": "本地图片路径"},
      "query": {"type": "string", "description": "问 LLM 什么（如 '描述这张图' / '图里有几只猫'）"}
    },
    "required": ["image_path", "query"]
  }
}
```

### 4.2 `image_ocr`

```json
{
  "name": "image_ocr",
  "description": "从图片提取文字（OCR）。适合截图/扫描文档/图表。",
  "parameters": {
    "type": "object",
    "properties": {
      "image_path": {"type": "string", "description": "本地图片路径"},
      "hint": {"type": "string", "description": "可选：语言或上下文提示，如 '中文' / '代码' / '表格'"}
    },
    "required": ["image_path"]
  }
}
```

`image_ocr` 内部等价于 `image_analyze(query="请提取图中所有可见文字，保持原始结构和换行。" + hint)`，是便捷 shortcut。

---

## 5. 数据流

### 5.1 image_analyze 主流程

```
[LLM 调用 image_analyze(image_path, query)]
  ↓
1. 读 image_path 参数（必填校验）
2. 后缀白名单检查：png/jpeg/jpg/webp/gif（大小写不敏感）
   ├─ 不在白名单 → unsupported_format
3. safe_path(image_path, write=False)
   ├─ 受保护路径（~/.ssh、/etc 等）→ permission_denied
4. 文件存在检查
   ├─ 不存在 → file_not_found
5. 文件大小检查（≤ max_bytes，默认 20MB）
   ├─ 超限 → file_too_large
6. 读文件 → base64 编码 → MIME 推断（基于后缀）
7. 取 vision_client（agent.vision_client，回退 agent.llm_client）
   ├─ 都为 None → vision_unavailable
8. 调 OpenAI 兼容 vision API：
   messages = [{"role":"user", "content": [
       {"type":"text", "text": query},
       {"type":"image_url", "image_url": {"url": "data:{mime};base64,{b64}"}}
   ]}]
   max_tokens = 2000
9. 返回 {"success": True, "description": ..., "image_path": ..., "query": ...}
```

### 5.2 image_ocr 等价流程

```
[image_ocr(image_path, hint=None)]
  ↓
拼 prompt = "请提取图中所有可见文字，保持原始结构和换行。"
if hint:
    prompt += f"\n上下文提示：{hint}"
  ↓
复用 image_analyze 的全部检查 + 调用流程
  ↓
返回 {"success": True, "text": ..., "image_path": ...}
```

---

## 6. 错误矩阵

| 场景 | error_type | 说明 |
|---|---|---|
| `image_path` 或 `query` 空字符串 | （无 error_type） | 直接返 `{"error": "..."}` |
| 文件不存在 | `file_not_found` | |
| 受保护路径（`~/.ssh` 等） | `permission_denied` | `safe_path` 拒绝 |
| 后缀不在白名单 | `unsupported_format` | 含允许列表提示 |
| 文件 > `max_bytes` | `file_too_large` | 显示实际/限制大小 |
| `vision_client` 与 `llm_client` 都为 None | `vision_unavailable` | |
| LLM API 调用失败 | `vision_error` | 含原始异常信息 |

所有错误统一格式：`{"error": "...", "error_type": "...", "image_path": "..."}`。

---

## 7. 安全设计

### 7.1 路径白名单（复用现有 `agent/permission.py:safe_path`）

```python
from agent.permission import safe_path
perm = safe_path(image_path, write=False)
if not perm.allowed:
    return {"error": f"路径拒绝: {perm.reason}", "error_type": "permission_denied"}
```

走读模式检查，拒绝 `~/.ssh`、`/etc`、`C:\Windows` 等受保护路径。

### 7.2 格式白名单

```python
SUPPORTED_FORMATS = {
    ".png":  "image/png",
    ".jpeg": "image/jpeg",
    ".jpg":  "image/jpeg",
    ".webp": "image/webp",
    ".gif":  "image/gif",
}
```

后缀大小写不敏感。不在表里 → 拒绝。

### 7.3 大小上限

```python
DEFAULT_MAX_BYTES = 20 * 1024 * 1024  # 20MB
```

超过 → 拒绝。理由：避免把超大文件塞进 LLM API（成本+超时风险）。配置项 `vision.max_bytes` 可调。

### 7.4 base64 编码（不存盘）

图片读为 bytes → base64 编码进 `data:` URL → 直接传给 LLM。**不写中间文件**，不在日志泄露 base64 内容。

---

## 8. vision_client 初始化

### 8.1 AIAgent 字段

```python
# agent/__init__.py
class AIAgent:
    def __init__(self, ...):
        ...
        self._vision_client = None  # 默认 None；外部注入或 cli.py 初始化
```

### 8.2 RuntimeContext 初始化

```python
# cli.py RuntimeContext._create_agent
vision_cfg = self.config.get("vision", {})
if vision_cfg.get("enabled", True):
    try:
        from agent.llm_client import create_llm_client
        vision_provider = vision_cfg.get("provider") or model_cfg.get("provider")
        vision_model = vision_cfg.get("model") or model_cfg.get("name")
        # 复用主 client 的 base_url + api_key（除非 config 显式覆盖）
        agent._vision_client = create_llm_client({
            "format": model_cfg.get("format", "openai"),
            "base_url": model_cfg.get("base_url"),
            "api_key": api_key,
            "model": vision_model,
        })
    except Exception as e:
        logger.warning("vision_client 初始化失败: %s", e)
        agent._vision_client = None
```

如果 `vision.model` 未配置，**不创建独立 client**，工具回退到 `agent.llm_client`（主模型）。

### 8.3 browser_vision 兼容

```python
# tools/browser_tool.py:_handle_browser_vision
client = None
if agent:
    client = getattr(agent, "_vision_client", None) \
          or getattr(agent, "_browser_vision_client", None) \
          or getattr(agent, "llm_client", None)
if client is None:
    return _err("vision LLM client 未配置", "vision_unavailable")
```

三层回退保证：
1. 新的 `_vision_client`（推荐）
2. 旧的 `_browser_vision_client`（测试已注入）
3. 主 `llm_client`（最兜底）

---

## 9. 测试策略

`tests/test_image_tool.py` 新增 12 个测试：

### 9.1 Happy path

1. `test_image_analyze_happy_path` — mock client，验证 base64 + messages 结构正确
2. `test_image_ocr_uses_ocr_prompt` — 验证 OCR 自动拼 prompt（含"提取图中所有可见文字"）
3. `test_image_ocr_with_hint` — 带 hint 时拼到 prompt

### 9.2 错误路径

4. `test_unsupported_format_bmp` — `.bmp` 后缀拒绝
5. `test_unsupported_format_tiff` — `.tiff` 后缀拒绝
6. `test_file_not_found` — 不存在路径
7. `test_protected_path_rejected` — `~/.ssh/foo.png` 拒绝（mock safe_path 或用 monkeypatch）
8. `test_file_too_large` — mock Path.stat 返回 25MB
9. `test_vision_client_fallback_to_main_llm` — `_vision_client=None` 时用 `llm_client`
10. `test_vision_client_unavailable_all_none` — 两者都 None → `vision_unavailable`
11. `test_vision_api_failure_returns_vision_error` — mock client 抛异常

### 9.3 辅助

12. `test_mime_inference_all_formats` — png/jpeg/jpg/webp/gif 各自正确 MIME

### 9.4 测试基础设施

- 所有 LLM 调用 mock（不调真实 API）
- 用 `tmp_path` 写真实的小 PNG 文件（不用 mock 文件系统）
- FakeAgent 类含 `_vision_client` / `_browser_vision_client` / `llm_client` 三字段（按需设置）

---

## 10. 与现有架构的契合

| CLAUDE.md 设计原则 | 本设计如何遵守 |
|---|---|
| 核心是窄腰（#1） | 加 2 个工具到 `core`，schema 简洁；config 可关 |
| 完全可逆（#3） | 工具不修改任何持久化状态（只读图片 + 调 LLM） |
| 安全默认（#6） | safe_path + 格式白名单 + 大小上限 + base64 不落盘 |
| 同步阻塞 | 全部同步调用，不引入 asyncio |
| JSON 字符串契约 | 所有 handler 返 JSON 字符串，错误用 `error_type` |
| UTF-8 强制 | 不涉及文本文件 I/O（图片二进制读 bytes） |

---

## 11. 留给未来的接口

- `image_compare` 可在后续加（如果用户频繁对比图）
- URL 输入支持（先 download 到 tmp_path 再走本地路径）
- 多图输入（messages 含多个 image_url）
- 缓存（同图同 query 复用结果，省成本）

本期都不做。

---

## 12. 实施任务分解

1. **Task 1**: `tools/image_tool.py` — 2 工具 handler + schema + 注册 + check_fn + safe_path/大小/格式检查 + MIME 推断
2. **Task 2**: `agent/__init__.py` + `cli.py` — AIAgent 加 `_vision_client` 字段；RuntimeContext 根据 config 初始化
3. **Task 3**: `tools/browser_tool.py` — browser_vision 改三层回退读 client；跑测试确认旧测试不挂
4. **Task 4**: `tests/test_image_tool.py` — 12 测试 + 跑全量 + commit + push

---

## 13. 验收标准

- [ ] `tools/image_tool.py` 实现 `image_analyze` + `image_ocr`
- [ ] 工具在 `core` toolset 注册，`check_fn` 在无 client 时隐藏
- [ ] `agent._vision_client` 字段存在
- [ ] `RuntimeContext._create_agent` 根据 `vision` config 创建 `_vision_client`
- [ ] `browser_vision` 三层回退（`_vision_client` → `_browser_vision_client` → `llm_client`）
- [ ] 12 新测试全过
- [ ] 全量 ≥837（825 + 12）
- [ ] `verify.py` 22/22 不回归
- [ ] commit + push 到 origin/master
