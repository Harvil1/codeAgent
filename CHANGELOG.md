# CHANGELOG

## v0.11.0 - 2026-07-12

### 新增
- 上下文压缩分层管线（L1 snip / L2 micro / L4 llm + reactive 紧急通道）
- 大输出落盘（output_offload，工具结果 > 30KB 时写文件留预览，信息无损）
- 压缩前快照存档（transcript，L4 有损摘要前把完整 messages 落到 `.transcripts/`）
- 紧急上下文压缩（reactive_compact，API 报 `prompt_too_long` 时兜底，每会话一次）
- 200 轮端到端集成测试，覆盖 offload + transcript + L4 触发 + reactive 不崩溃

### 变更（BREAKING）
- `config.context.use_new_pipeline` 默认值：`False` → **`True`**
  - 所有新建 AIAgent 实例默认走分层压缩管线（L1→L2→L4 + reactive）
  - 旧的单层 LLM 摘要路径（`agent/context_compressor.maybe_compress`）标记为废弃
  - `agent/context_compressor.maybe_compress` 将在下个 minor 版本完全移除

### 回滚方式

如需恢复 Phase 1 之前的单层 LLM 摘要行为，在 `config.yaml`（或 `~/.agent/config.yaml`）中加：

```yaml
context:
  use_new_pipeline: false
```

或在代码中显式传 `config={"context": {"use_new_pipeline": False}}` 给 `AIAgent` 构造函数。

### 模块清单

| 模块 | 职责 |
|---|---|
| `agent/output_offload.py` | L3 旁路：大工具结果落盘 |
| `agent/transcript.py` | 压缩前快照存档（JSONL） |
| `agent/context_pipeline.py` | L1 snip / L2 micro / L4 llm / reactive 编排 |
| `agent/context_compressor.py` | 旧路径（废弃，保留 utility 函数） |
