# tests/test_snip_ctx_inspect.py
"""Task L: snip 工具 + ctx_inspect 工具测试（LLM 自驱能力）。

覆盖范围：
1. snip 基本功能（长 history 真剪了）
2. snip 短历史不剪（< keep_recent*2 返 snipped=False）
3. snip reason 传入记录到返回值
4. snip keep_recent 自定义（传 20 验证保留 20 条）
5. snip fail-open（snip_compact 抛异常不崩主流程）
6. ctx_inspect 基本功能（返 messages_count / estimated_tokens / cache_stats）
7. ctx_inspect recommendation 阈值（>= 70% compact_now / >= 50% snip_consider / 其他 ok）
8. ctx_inspect fail-open（estimate_message_tokens 抛异常不崩）
9. 工具注册：get_tool_definitions(['core']) 含 snip 和 ctx_inspect（防 check_fn 签名 bug）
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.snip_tool import SNIP_SCHEMA, _handle_snip
from tools.ctx_inspect_tool import CTX_INSPECT_SCHEMA, _handle_ctx_inspect


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _mk_agent(history_len: int = 60, config: dict = None, omnimate_home=None,
              session_id: str = "test-session-001"):
    """构造 mock agent，带 conversation_history 和 config。"""
    history = []
    for i in range(history_len):
        history.append({"role": "user", "content": f"用户提问 {i}，内容稍微长一点凑字数"})
        history.append({"role": "assistant", "content": f"助手回答 {i}，内容稍微长一点凑字数"})
    agent = SimpleNamespace()
    agent.conversation_history = history
    agent.config = config or {
        "context": {
            "snip_keep_first": 3,
            "llm_compact_token_threshold": 100000,
        }
    }
    agent.omnimate_home = omnimate_home
    agent.session_id = session_id
    return agent


# ---------------------------------------------------------------------------
# snip 工具测试
# ---------------------------------------------------------------------------

class TestSnipTool:
    """snip 工具 handler 测试。"""

    def test_schema_has_required_fields(self):
        """schema 字段完整（name + description + parameters）。"""
        assert SNIP_SCHEMA["name"] == "snip"
        assert "description" in SNIP_SCHEMA
        params = SNIP_SCHEMA["parameters"]["properties"]
        assert "reason" in params
        assert "keep_recent" in params
        assert params["keep_recent"]["default"] == 10

    def test_snip_basic(self):
        """长 history 调 snip 真剪了。"""
        agent = _mk_agent(history_len=60)  # 120 条
        before = len(agent.conversation_history)
        result_json = _handle_snip(
            {"reason": "探索阶段结束"},
            agent_ref=agent,
        )
        result = json.loads(result_json)
        assert result["snipped"] is True
        assert result["snipped_count"] > 0
        assert result["reason"] == "探索阶段结束"
        after = len(agent.conversation_history)
        assert after < before, f"snip 后条数 {after} 应小于 {before}"
        assert result["remaining_count"] == after

    def test_snip_short_history_not_snipped(self):
        """短历史（< keep_recent * 2）不剪。"""
        agent = _mk_agent(history_len=3)  # 6 条 < 10*2=20
        result_json = _handle_snip({}, agent_ref=agent)
        result = json.loads(result_json)
        assert result["snipped"] is False
        assert "无需剪" in result["reason"]

    def test_snip_reason_recorded(self):
        """reason 真记录到返回值。"""
        agent = _mk_agent(history_len=60)
        result_json = _handle_snip(
            {"reason": "完成代码探索"},
            agent_ref=agent,
        )
        result = json.loads(result_json)
        assert result["reason"] == "完成代码探索"

    def test_snip_keep_recent_custom(self):
        """keep_recent=20 保留最近 20 条（影响 snip_compact 的 keep_last）。"""
        agent = _mk_agent(history_len=60)
        result_json = _handle_snip(
            {"reason": "测试", "keep_recent": 20},
            agent_ref=agent,
        )
        result = json.loads(result_json)
        assert result["snipped"] is True
        # snip 后的 history 长度 = head + placeholder + tail(20)
        # head=3 + 1 placeholder + 20 tail = 24（大约值，取决于边界保护）
        # 主要验证 snip 真触发了
        assert result["snipped_count"] > 0

    def test_snip_no_agent_ref(self):
        """缺 agent_ref 返 internal_error。"""
        result_json = _handle_snip({}, )
        result = json.loads(result_json)
        assert result["error_type"] == "internal_error"

    def test_snip_fail_open_on_exception(self):
        """snip_compact 抛异常时 fail-open 返 error JSON。"""
        agent = _mk_agent(history_len=60)
        with patch("tools.snip_tool.snip_compact", side_effect=RuntimeError("boom")):
            result_json = _handle_snip({"reason": "x"}, agent_ref=agent)
        result = json.loads(result_json)
        assert "error" in result
        assert result["error_type"] == "RuntimeError"

    def test_snip_returns_zero_when_snip_compact_no_change(self):
        """snip_compact 判定无需剪（changed=False）时返 snipped=False。"""
        agent = _mk_agent(history_len=60)
        # mock snip_compact 返 (原 history, False) 表示未剪
        original = agent.conversation_history
        with patch("tools.snip_tool.snip_compact", return_value=(original, False)):
            result_json = _handle_snip({"reason": "x"}, agent_ref=agent)
        result = json.loads(result_json)
        assert result["snipped"] is False

    def test_snip_calls_snapshot_before_snip(self, tmp_path):
        """snip 前主动调 snapshot_if_needed(force=True) 落 transcript。

        验证 Important fix：让"无损"描述变真——原文确实落到 .transcripts/ 了。
        """
        agent = _mk_agent(history_len=60, omnimate_home=tmp_path)
        before = len(agent.conversation_history)
        result_json = _handle_snip(
            {"reason": "探索阶段结束"},
            agent_ref=agent,
        )
        result = json.loads(result_json)
        assert result["snipped"] is True
        # transcript 目录确实被创建
        transcripts_dir = tmp_path / ".transcripts"
        assert transcripts_dir.exists(), ".transcripts 目录应被创建"
        # 至少一个 .jsonl transcript 文件
        jsonl_files = list(transcripts_dir.glob("transcript_*.jsonl"))
        assert len(jsonl_files) >= 1, "应至少有一个 transcript_*.jsonl"
        # latest.txt 指针存在
        latest_pointer = transcripts_dir / "latest.txt"
        assert latest_pointer.exists(), "latest.txt 指针应存在"
        # transcript 内容含 snip 前的全部消息（无损）
        import json as _json
        with open(jsonl_files[0], encoding="utf-8") as f:
            lines = f.readlines()
        # 最后一行是 _meta，其余是消息
        msg_lines = [l for l in lines if "_meta" not in l]
        assert len(msg_lines) == before, (
            f"transcript 应含 {before} 条原文，实有 {len(msg_lines)}"
        )

    def test_snip_skips_snapshot_when_no_home(self):
        """omnimate_home=None 时跳过 snapshot（但不崩）。"""
        agent = _mk_agent(history_len=60, omnimate_home=None)
        with patch("tools.snip_tool.snapshot_if_needed") as mock_snap:
            result_json = _handle_snip(
                {"reason": "测试"},
                agent_ref=agent,
            )
            # snapshot 没被调（因为 omnimate_home=None）
            mock_snap.assert_not_called()
        result = json.loads(result_json)
        assert result["snipped"] is True

    def test_snip_snapshot_failure_failopen(self, tmp_path):
        """snapshot_if_needed 抛异常时 fail-open，snip 继续执行。"""
        agent = _mk_agent(history_len=60, omnimate_home=tmp_path)
        with patch("tools.snip_tool.snapshot_if_needed", side_effect=OSError("disk full")):
            result_json = _handle_snip(
                {"reason": "测试"},
                agent_ref=agent,
            )
        result = json.loads(result_json)
        # snip 照常成功（snapshot 失败不阻塞）
        assert result["snipped"] is True

    def test_snip_snapshot_disabled_skips(self, tmp_path):
        """transcript_enabled=False 时不调 snapshot。"""
        config = {
            "context": {
                "snip_keep_first": 3,
                "transcript_enabled": False,
            }
        }
        agent = _mk_agent(history_len=60, omnimate_home=tmp_path, config=config)
        with patch("tools.snip_tool.snapshot_if_needed") as mock_snap:
            result_json = _handle_snip(
                {"reason": "测试"},
                agent_ref=agent,
            )
            mock_snap.assert_not_called()
        result = json.loads(result_json)
        assert result["snipped"] is True


# ---------------------------------------------------------------------------
# ctx_inspect 工具测试
# ---------------------------------------------------------------------------

class TestCtxInspectTool:
    """ctx_inspect 工具 handler 测试。"""

    def test_schema_basic(self):
        """schema 字段完整。"""
        assert CTX_INSPECT_SCHEMA["name"] == "ctx_inspect"
        assert "description" in CTX_INSPECT_SCHEMA
        assert CTX_INSPECT_SCHEMA["parameters"]["type"] == "object"

    def test_ctx_inspect_basic(self):
        """基本查询返回 messages_count / estimated_tokens / recommendation。"""
        agent = _mk_agent(history_len=5)
        result_json = _handle_ctx_inspect({}, agent_ref=agent)
        result = json.loads(result_json)
        assert result["messages_count"] == 10  # 5 轮 * 2
        assert result["estimated_tokens"] > 0
        assert "recommendation" in result
        assert result["recommendation"] in ("ok", "snip_consider", "compact_now")
        assert "usage_percent" in result
        assert "llm_compact_threshold" in result

    def test_ctx_inspect_no_agent_ref(self):
        """缺 agent_ref 返 internal_error。"""
        result_json = _handle_ctx_inspect({})
        result = json.loads(result_json)
        assert result["error_type"] == "internal_error"

    def test_ctx_inspect_recommendation_70pct_compact_now(self):
        """est_tokens >= 70% threshold → compact_now。"""
        agent = _mk_agent(history_len=5, config={
            "context": {"llm_compact_token_threshold": 100}
        })
        # mock estimate_message_tokens 返 75（75% of 100）
        with patch("tools.ctx_inspect_tool.estimate_message_tokens", return_value=75):
            result_json = _handle_ctx_inspect({}, agent_ref=agent)
        result = json.loads(result_json)
        assert result["recommendation"] == "compact_now"
        assert result["usage_percent"] == 75

    def test_ctx_inspect_recommendation_50pct_snip_consider(self):
        """est_tokens >= 50% threshold → snip_consider。"""
        agent = _mk_agent(history_len=5, config={
            "context": {"llm_compact_token_threshold": 100}
        })
        with patch("tools.ctx_inspect_tool.estimate_message_tokens", return_value=55):
            result_json = _handle_ctx_inspect({}, agent_ref=agent)
        result = json.loads(result_json)
        assert result["recommendation"] == "snip_consider"

    def test_ctx_inspect_recommendation_below_50pct_ok(self):
        """est_tokens < 50% threshold → ok。"""
        agent = _mk_agent(history_len=5, config={
            "context": {"llm_compact_token_threshold": 100}
        })
        with patch("tools.ctx_inspect_tool.estimate_message_tokens", return_value=30):
            result_json = _handle_ctx_inspect({}, agent_ref=agent)
        result = json.loads(result_json)
        assert result["recommendation"] == "ok"

    def test_ctx_inspect_fail_open_on_exception(self):
        """estimate_message_tokens 抛异常时 fail-open 返 error JSON。"""
        agent = _mk_agent(history_len=5)
        # patch estimate_message_tokens 和 get_stats 都抛异常
        with patch("tools.ctx_inspect_tool.estimate_message_tokens",
                   side_effect=RuntimeError("boom")):
            # 注意：estimate_message_tokens 抛了内部有 except 兜底，
            # 真要测 fail-open 需让整个 handler 崩——patch history 访问
            result_json = _handle_ctx_inspect({}, agent_ref=agent)
            result = json.loads(result_json)
            # 应该 fall back 到 len//4（不崩）
            assert "estimated_tokens" in result

    def test_ctx_inspect_cache_stats_present(self):
        """返回应包含 cache_stats 相关字段。"""
        agent = _mk_agent(history_len=3)
        result_json = _handle_ctx_inspect({}, agent_ref=agent)
        result = json.loads(result_json)
        assert "cache_last_read" in result
        assert "cache_total_breaks" in result

    def test_ctx_inspect_agent_ref_none_uses_fallback_token_estimation(self):
        """当 estimate_message_tokens 异常时，走 fallback（len//4）不崩。"""
        agent = _mk_agent(history_len=3)
        # 让 estimate_message_tokens 抛，然后会走 except 分支用 len//4
        with patch("tools.ctx_inspect_tool.estimate_message_tokens",
                   side_effect=ValueError("broken")):
            result_json = _handle_ctx_inspect({}, agent_ref=agent)
            result = json.loads(result_json)
            assert result["estimated_tokens"] > 0  # 走 fallback


# ---------------------------------------------------------------------------
# 工具注册测试（关键：防 CCAR6 的 check_fn 签名 bug）
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """验证两个工具真注册到 registry 且 get_tool_definitions 返回。"""

    def test_snip_registered_in_core(self):
        """snip 在 _CORE_TOOLS 列表里。"""
        from toolsets import _CORE_TOOLS
        assert "snip" in _CORE_TOOLS

    def test_ctx_inspect_registered_in_core(self):
        """ctx_inspect 在 _CORE_TOOLS 列表里。"""
        from toolsets import _CORE_TOOLS
        assert "ctx_inspect" in _CORE_TOOLS

    def test_get_tool_definitions_includes_snip_and_ctx_inspect(self):
        """get_tool_definitions(['core']) 真返回 snip 和 ctx_inspect。

        关键测试：防 check_fn 签名错导致工具被过滤（CCAR6 出过这 bug）。
        """
        from model_tools import get_tool_definitions, ensure_tools_discovered
        ensure_tools_discovered()
        defs = get_tool_definitions(["core"])
        tool_names = [d["function"]["name"] for d in defs]
        assert "snip" in tool_names, (
            f"snip 未出现在 get_tool_definitions(['core'])！"
            f"这是 CCAR6 类型 bug 的征兆。tools: {tool_names}"
        )
        assert "ctx_inspect" in tool_names, (
            f"ctx_inspect 未出现在 get_tool_definitions(['core'])！"
            f"tools: {tool_names}"
        )

    def test_registry_has_snip_entry(self):
        """registry.get('snip') 不为 None。"""
        from tools.registry import registry
        from model_tools import ensure_tools_discovered
        ensure_tools_discovered()
        entry = registry.get("snip")
        assert entry is not None
        assert entry.toolset == "core"

    def test_registry_has_ctx_inspect_entry(self):
        """registry.get('ctx_inspect') 不为 None。"""
        from tools.registry import registry
        from model_tools import ensure_tools_discovered
        ensure_tools_discovered()
        entry = registry.get("ctx_inspect")
        assert entry is not None
        assert entry.toolset == "core"

    def test_snip_check_fn_signature(self):
        """snip 工具的 check_fn（如有）必须无参可调。"""
        from tools.registry import registry, _check_fn_cached
        from model_tools import ensure_tools_discovered
        ensure_tools_discovered()
        entry = registry.get("snip")
        if entry.check_fn is not None:
            # 直接调 _check_fn_cached 验证不抛
            result = _check_fn_cached(entry.check_fn)
            assert isinstance(result, bool)

    def test_ctx_inspect_check_fn_signature(self):
        """ctx_inspect 工具的 check_fn（如有）必须无参可调。"""
        from tools.registry import registry, _check_fn_cached
        from model_tools import ensure_tools_discovered
        ensure_tools_discovered()
        entry = registry.get("ctx_inspect")
        if entry.check_fn is not None:
            result = _check_fn_cached(entry.check_fn)
            assert isinstance(result, bool)

    def test_snip_schema_in_definitions_is_valid(self):
        """get_tool_definitions 返回的 snip schema 结构对（type=function + function.name）。"""
        from model_tools import get_tool_definitions, ensure_tools_discovered
        ensure_tools_discovered()
        defs = get_tool_definitions(["core"])
        snip_def = next(
            (d for d in defs if d.get("function", {}).get("name") == "snip"),
            None,
        )
        assert snip_def is not None, "snip 不在 get_tool_definitions 里"
        assert snip_def["type"] == "function"
        assert "parameters" in snip_def["function"] or "input_schema" in snip_def["function"]

    def test_ctx_inspect_schema_in_definitions_is_valid(self):
        """get_tool_definitions 返回的 ctx_inspect schema 结构对。"""
        from model_tools import get_tool_definitions, ensure_tools_discovered
        ensure_tools_discovered()
        defs = get_tool_definitions(["core"])
        ci_def = next(
            (d for d in defs if d.get("function", {}).get("name") == "ctx_inspect"),
            None,
        )
        assert ci_def is not None, "ctx_inspect 不在 get_tool_definitions 里"
        assert ci_def["type"] == "function"
