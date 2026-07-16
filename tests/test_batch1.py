"""batch1 四项改进的测试：
1. MCP include/exclude 过滤
2. Prompt Caching 命中率记账
3. 技能束 Bundles
4. 中断传播到子 agent
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# 1. MCP include/exclude 过滤
# ===========================================================================

class TestMCPFilter:
    """MCP include/exlude 工具过滤。"""

    def test_filter_tool_name_no_filter(self):
        """无 include/exclude 时保留所有。"""
        from agent.mcp_client import filter_tool_name
        assert filter_tool_name("tool1") is True
        assert filter_tool_name("anything") is True

    def test_filter_tool_name_include_only(self):
        """include 只保留列出的。"""
        from agent.mcp_client import filter_tool_name
        include = ["read", "write"]
        assert filter_tool_name("read", include=include) is True
        assert filter_tool_name("write", include=include) is True
        assert filter_tool_name("delete", include=include) is False

    def test_filter_tool_name_exclude_only(self):
        """exclude 跳过列出的。"""
        from agent.mcp_client import filter_tool_name
        exclude = ["dangerous"]
        assert filter_tool_name("read", exclude=exclude) is True
        assert filter_tool_name("dangerous", exclude=exclude) is False

    def test_filter_tool_name_include_overrides_exclude(self):
        """include 优先于 exclude。"""
        from agent.mcp_client import filter_tool_name
        include = ["read"]
        exclude = ["read"]  # 即使 include 和 exclude 冲突
        # include 优先
        assert filter_tool_name("read", include=include, exclude=exclude) is True

    def test_client_stores_include_exclude(self):
        """MCPClient 存储 include/exclude 配置。"""
        from agent.mcp_client import MCPClient
        client = MCPClient(
            name="test", command="cmd",
            include=["a", "b"], exclude=["c"],
        )
        assert client.include == ["a", "b"]
        assert client.exclude == ["c"]

    def test_client_defaults_none(self):
        """MCPClient 默认 include/exclude 为 None。"""
        from agent.mcp_client import MCPClient
        client = MCPClient(name="test", command="cmd")
        assert client.include is None
        assert client.exclude is None

    def test_manager_connect_all_passes_include_exclude(self):
        """connect_all 把 include/exclude 传给 MCPClient。"""
        from agent.mcp_client import MCPManager
        manager = MCPManager()

        # 用真实 MCPClient 实例（不 connect）
        captured_kwargs = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                self.include = kwargs.get("include")
                self.exclude = kwargs.get("exclude")
                self.connected = True
            def connect(self):
                pass

        with patch("agent.mcp_client.MCPClient", FakeClient):
            manager.connect_all({
                "server1": {
                    "command": "cmd",
                    "args": [],
                    "include": ["tool_a"],
                    "exclude": ["tool_b"],
                },
            })

        assert captured_kwargs.get("include") == ["tool_a"]
        assert captured_kwargs.get("exclude") == ["tool_b"]

    def test_manager_get_all_tools_with_include(self):
        """get_all_tools 应用 include 过滤。"""
        from agent.mcp_client import MCPManager
        manager = MCPManager()

        fake_client = MagicMock()
        fake_client.connected = True
        fake_client.include = ["read"]  # 只保留 read
        fake_client.exclude = None
        fake_client.list_tools.return_value = [
            {"name": "read", "description": "读", "inputSchema": {"type": "object"}},
            {"name": "write", "description": "写", "inputSchema": {"type": "object"}},
            {"name": "delete", "description": "删", "inputSchema": {"type": "object"}},
        ]
        manager._clients["fs"] = fake_client

        tools = manager.get_all_tools()
        assert len(tools) == 1
        assert tools[0]["original_name"] == "read"

    def test_manager_get_all_tools_with_exclude(self):
        """get_all_tools 应用 exclude 过滤。"""
        from agent.mcp_client import MCPManager
        manager = MCPManager()

        fake_client = MagicMock()
        fake_client.connected = True
        fake_client.include = None
        fake_client.exclude = ["delete"]
        fake_client.list_tools.return_value = [
            {"name": "read", "description": "读", "inputSchema": {"type": "object"}},
            {"name": "write", "description": "写", "inputSchema": {"type": "object"}},
            {"name": "delete", "description": "删", "inputSchema": {"type": "object"}},
        ]
        manager._clients["fs"] = fake_client

        tools = manager.get_all_tools()
        assert len(tools) == 2
        names = [t["original_name"] for t in tools]
        assert "read" in names
        assert "write" in names
        assert "delete" not in names

    def test_manager_get_all_tools_include_overrides_exclude(self):
        """include 优先于 exclude。"""
        from agent.mcp_client import MCPManager
        manager = MCPManager()

        fake_client = MagicMock()
        fake_client.connected = True
        fake_client.include = ["read"]
        fake_client.exclude = ["read", "write"]  # read 在两者都有
        fake_client.list_tools.return_value = [
            {"name": "read", "description": "读", "inputSchema": {"type": "object"}},
            {"name": "write", "description": "写", "inputSchema": {"type": "object"}},
        ]
        manager._clients["fs"] = fake_client

        tools = manager.get_all_tools()
        # include 优先，只保留 read
        assert len(tools) == 1
        assert tools[0]["original_name"] == "read"

    def test_load_mcp_config_preserves_include_exclude(self, tmp_path):
        """load_mcp_config 保留 include/exclude 字段。"""
        from agent.mcp_client import load_mcp_config
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["server"],
                    "include": ["create_issue"],
                    "exclude": [],
                }
            }
        }), encoding="utf-8")

        servers = load_mcp_config(cfg)
        assert servers["github"]["include"] == ["create_issue"]
        assert servers["github"]["exclude"] == []


# ===========================================================================
# 2. Prompt Caching 命中率记账
# ===========================================================================

class TestLLMUsageStats:
    """LLM 用量统计。"""

    def _make_agent(self):
        """创建一个带 mock llm_client 的 AIAgent。"""
        from agent import AIAgent
        with patch("agent.llm_client.create_llm_client"):
            agent = AIAgent(
                base_url="http://fake",
                api_key="fake",
                model="fake-model",
            )
        return agent

    def test_usage_stats_initialized(self):
        """agent 初始化时有空统计。"""
        agent = self._make_agent()
        assert agent.llm_usage_stats["total_calls"] == 0
        assert agent.llm_usage_stats["total_prompt_tokens"] == 0
        assert agent.llm_usage_stats["total_completion_tokens"] == 0
        assert agent.llm_usage_stats["total_cache_read_tokens"] == 0
        assert agent.llm_usage_stats["total_cache_creation_tokens"] == 0

    def test_usage_stats_returns_copy(self):
        """llm_usage_stats 返回副本（修改不影响内部）。"""
        agent = self._make_agent()
        stats = agent.llm_usage_stats
        stats["total_calls"] = 999
        assert agent._llm_usage_stats["total_calls"] == 0

    def test_record_llm_usage_basic(self):
        """记录基本 token 用量。"""
        agent = self._make_agent()
        fake_response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
            )
        )
        agent._record_llm_usage(fake_response)
        assert agent._llm_usage_stats["total_calls"] == 1
        assert agent._llm_usage_stats["total_prompt_tokens"] == 100
        assert agent._llm_usage_stats["total_completion_tokens"] == 50

    def test_record_llm_usage_accumulates(self):
        """多次记录累加。"""
        agent = self._make_agent()
        resp1 = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))
        resp2 = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=200, completion_tokens=100))
        agent._record_llm_usage(resp1)
        agent._record_llm_usage(resp2)
        assert agent._llm_usage_stats["total_calls"] == 2
        assert agent._llm_usage_stats["total_prompt_tokens"] == 300
        assert agent._llm_usage_stats["total_completion_tokens"] == 150

    def test_record_llm_usage_no_usage_attr(self):
        """response 无 usage 属性时不报错。"""
        agent = self._make_agent()
        fake_response = SimpleNamespace(usage=None)
        agent._record_llm_usage(fake_response)
        assert agent._llm_usage_stats["total_calls"] == 1
        assert agent._llm_usage_stats["total_prompt_tokens"] == 0

    def test_record_llm_usage_cache_hit(self):
        """记录 cache 命中 tokens。"""
        agent = self._make_agent()
        fake_response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=50,
                prompt_cache_hit_tokens=800,
                prompt_cache_miss_tokens=200,
            )
        )
        agent._record_llm_usage(fake_response)
        assert agent._llm_usage_stats["total_cache_read_tokens"] == 800
        assert agent._llm_usage_stats["total_cache_creation_tokens"] == 200

    def test_record_llm_usage_cache_alt_names(self):
        """Anthropic 风格的 cache 字段名。"""
        agent = self._make_agent()
        fake_response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=50,
                cache_read_input_tokens=500,
                cache_creation_input_tokens=500,
            )
        )
        agent._record_llm_usage(fake_response)
        assert agent._llm_usage_stats["total_cache_read_tokens"] == 500
        assert agent._llm_usage_stats["total_cache_creation_tokens"] == 500


# ===========================================================================
# 3. 技能束 Bundles
# ===========================================================================

class TestSkillBundles:
    """技能束加载。"""

    def test_load_bundles_config_nonexistent(self, tmp_path):
        """文件不存在返回空。"""
        from agent.skill_bundle import load_bundles_config
        result = load_bundles_config(tmp_path / "nope.json")
        assert result == {}

    def test_load_bundles_config_valid(self, tmp_path):
        """有效配置加载。"""
        from agent.skill_bundle import load_bundles_config
        cfg = tmp_path / ".skill-bundles.json"
        cfg.write_text(json.dumps({
            "bundles": {
                "python-dev": {
                    "skills": ["pytest", "venv"],
                    "description": "Python 开发",
                }
            }
        }), encoding="utf-8")

        result = load_bundles_config(cfg)
        assert "python-dev" in result
        assert result["python-dev"]["skills"] == ["pytest", "venv"]

    def test_load_bundles_config_invalid(self, tmp_path):
        """无效 JSON 返回空。"""
        from agent.skill_bundle import load_bundles_config
        cfg = tmp_path / ".skill-bundles.json"
        cfg.write_text("not json", encoding="utf-8")
        assert load_bundles_config(cfg) == {}

    def test_load_bundle_success(self, tmp_path):
        """成功加载技能束。"""
        from agent.skill_bundle import load_bundle

        # 创建技能目录
        skills_dir = tmp_path / "skills"
        for sname in ["pytest", "venv"]:
            sd = skills_dir / sname
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text(
                f"---\ndescription: {sname}\n---\n{sname} 正文\n",
                encoding="utf-8",
            )

        cfg = tmp_path / ".skill-bundles.json"
        cfg.write_text(json.dumps({
            "bundles": {
                "py": {"skills": ["pytest", "venv"], "description": "PY"}
            }
        }), encoding="utf-8")

        result = load_bundle("py", skills_dir, cfg)
        assert result["bundle"] == "py"
        assert result["description"] == "PY"
        assert set(result["skills_loaded"]) == {"pytest", "venv"}
        assert result["skills_missing"] == []
        assert "pytest 正文" in result["body"]
        assert "venv 正文" in result["body"]

    def test_load_bundle_not_found(self, tmp_path):
        """技能束不存在返回 error + available 列表。"""
        from agent.skill_bundle import load_bundle
        cfg = tmp_path / ".skill-bundles.json"
        cfg.write_text(json.dumps({
            "bundles": {"existing": {"skills": ["a"]}}
        }), encoding="utf-8")

        result = load_bundle("nonexistent", tmp_path / "skills", cfg)
        assert "error" in result
        assert "existing" in result["available"]

    def test_load_bundle_partial_missing(self, tmp_path):
        """部分技能不存在时记录 missing。"""
        from agent.skill_bundle import load_bundle

        skills_dir = tmp_path / "skills"
        sd = skills_dir / "pytest"
        sd.mkdir(parents=True)
        (sd / "SKILL.md").write_text(
            "---\ndescription: t\n---\npytest 正文\n",
            encoding="utf-8",
        )

        cfg = tmp_path / ".skill-bundles.json"
        cfg.write_text(json.dumps({
            "bundles": {"py": {"skills": ["pytest", "nonexistent"]}}
        }), encoding="utf-8")

        result = load_bundle("py", skills_dir, cfg)
        assert "pytest" in result["skills_loaded"]
        assert "nonexistent" in result["skills_missing"]

    def test_list_bundles(self, tmp_path):
        """list_bundles 返回元信息。"""
        from agent.skill_bundle import list_bundles
        cfg = tmp_path / ".skill-bundles.json"
        cfg.write_text(json.dumps({
            "bundles": {
                "a": {"skills": ["x", "y"], "description": "A"},
                "b": {"skills": ["z"], "description": "B"},
            }
        }), encoding="utf-8")

        result = list_bundles(cfg)
        assert len(result) == 2
        assert result["a"]["skill_count"] == 2
        assert result["b"]["skill_count"] == 1

    def test_load_skill_handler_with_bundle(self, tmp_path):
        """load_skill 工具 handler 支持 bundle: 前缀。"""
        from tools.skill_tools import _handle_load_skill

        skills_dir = tmp_path / "skills"
        for sname in ["pytest", "venv"]:
            sd = skills_dir / sname
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text(
                f"---\ndescription: {sname}\n---\n{sname} 正文\n",
                encoding="utf-8",
            )

        cfg_file = tmp_path / ".skill-bundles.json"
        cfg_file.write_text(json.dumps({
            "bundles": {"py": {"skills": ["pytest", "venv"]}}
        }), encoding="utf-8")

        # 临时设置 bundles_config_path 返回我们的测试文件
        with patch("agent.skill_bundle.bundles_config_path", return_value=cfg_file):
            result = json.loads(_handle_load_skill(
                {"name": "bundle:py"},
                harvil_home=tmp_path,
            ))

        assert result["bundle"] == "py"
        assert set(result["skills_loaded"]) == {"pytest", "venv"}
        assert "pytest 正文" in result["body"]

    def test_load_skill_handler_bundle_not_found(self, tmp_path):
        """bundle 不存在时返回 error。"""
        from tools.skill_tools import _handle_load_skill

        cfg_file = tmp_path / ".skill-bundles.json"
        cfg_file.write_text(json.dumps({"bundles": {}}), encoding="utf-8")

        with patch("agent.skill_bundle.bundles_config_path", return_value=cfg_file):
            result = json.loads(_handle_load_skill(
                {"name": "bundle:nonexistent"},
                harvil_home=tmp_path,
            ))

        assert "error" in result

    def test_scan_bundle_commands(self, tmp_path):
        """scan_bundle_commands 正确生成 /bundle:<name> 命令。"""
        from agent.skill_commands import scan_bundle_commands

        cfg_file = tmp_path / ".skill-bundles.json"
        cfg_file.write_text(json.dumps({
            "bundles": {
                "python-dev": {"skills": ["pytest"], "description": "PY"},
            }
        }), encoding="utf-8")

        with patch("agent.skill_bundle.bundles_config_path", return_value=cfg_file):
            commands = scan_bundle_commands(tmp_path / "skills")

        assert "/bundle:python-dev" in commands
        assert commands["/bundle:python-dev"]["is_bundle"] is True
        assert commands["/bundle:python-dev"]["skills"] == ["pytest"]

    def test_execute_bundle_success(self, tmp_path):
        """execute_bundle 合并技能正文返回。"""
        from agent.skill_commands import execute_bundle

        skills_dir = tmp_path / "skills"
        for sname in ["pytest", "venv"]:
            sd = skills_dir / sname
            sd.mkdir(parents=True)
            (sd / "SKILL.md").write_text(
                f"---\ndescription: {sname}\n---\n{sname} 正文\n",
                encoding="utf-8",
            )

        cfg_file = tmp_path / ".skill-bundles.json"
        cfg_file.write_text(json.dumps({
            "bundles": {"py": {"skills": ["pytest", "venv"]}}
        }), encoding="utf-8")

        with patch("agent.skill_bundle.bundles_config_path", return_value=cfg_file):
            result = execute_bundle("py", "测试消息", skills_dir)

        assert "技能束已加载" in result
        assert "pytest 正文" in result
        assert "venv 正文" in result
        assert "测试消息" in result

    def test_execute_bundle_not_found(self, tmp_path):
        """execute_bundle 技能束不存在时返回提示。"""
        from agent.skill_commands import execute_bundle

        cfg_file = tmp_path / ".skill-bundles.json"
        cfg_file.write_text(json.dumps({"bundles": {}}), encoding="utf-8")

        with patch("agent.skill_bundle.bundles_config_path", return_value=cfg_file):
            result = execute_bundle("nonexistent", "msg", tmp_path / "skills")

        assert "失败" in result
        assert "msg" in result


# ===========================================================================
# 4. 中断传播到子 agent
# ===========================================================================

class TestInterruptPropagation:
    """中断传播到子 agent。"""

    def _make_agent(self):
        from agent import AIAgent
        with patch("agent.llm_client.create_llm_client"):
            agent = AIAgent(
                base_url="http://fake",
                api_key="fake",
                model="fake-model",
            )
        return agent

    def test_children_list_initialized(self):
        """agent 初始化时有空 children 列表。"""
        agent = self._make_agent()
        assert agent._children == []

    def test_interrupt_sets_flag(self):
        """interrupt 设置中断标志。"""
        agent = self._make_agent()
        agent.interrupt()
        assert agent._interrupt_requested is True

    def test_interrupt_propagates_to_children(self):
        """interrupt 传播到子 agent。"""
        parent = self._make_agent()
        child1 = self._make_agent()
        child2 = self._make_agent()
        parent._children.append(child1)
        parent._children.append(child2)

        parent.interrupt()
        assert parent._interrupt_requested is True
        assert child1._interrupt_requested is True
        assert child2._interrupt_requested is True

    def test_interrupt_no_children_no_error(self):
        """interrupt 无子 agent 时不报错。"""
        agent = self._make_agent()
        agent.interrupt()  # 不应该抛异常
        assert agent._interrupt_requested is True

    def test_interrupt_child_failure_does_not_crash(self):
        """子 agent interrupt 抛异常不影响父 agent。"""
        parent = self._make_agent()
        bad_child = MagicMock()
        bad_child.interrupt.side_effect = RuntimeError("boom")
        parent._children.append(bad_child)

        parent.interrupt()
        assert parent._interrupt_requested is True

    def test_nested_interrupt_propagation(self):
        """中断传播到多层嵌套子 agent。"""
        grandparent = self._make_agent()
        parent = self._make_agent()
        child = self._make_agent()

        grandparent._children.append(parent)
        parent._children.append(child)

        grandparent.interrupt()
        assert grandparent._interrupt_requested is True
        assert parent._interrupt_requested is True
        assert child._interrupt_requested is True

    def test_run_child_registers_to_parent_children(self):
        """_run_child 注册子 agent 到父 agent._children（运行期间）。"""
        from tools.delegate_tool import _run_child
        from agent import AIAgent

        parent_agent = self._make_agent()
        registered_during_run = []

        # Mock AIAgent 构造
        mock_child = MagicMock()
        mock_child.llm_client = MagicMock()
        mock_child.model = "fake"

        def chat_side_effect(msg):
            # 在 chat 执行期间，子 agent 应该已注册到父 agent
            registered_during_run.append(len(parent_agent._children))
            return "子代理结果"
        mock_child.chat.side_effect = chat_side_effect

        with patch("agent.AIAgent", return_value=mock_child):
            with patch.dict("os.environ", {
                "DEEPSEEK_API_KEY": "fake",
                "_SPAWN_DEPTH": "0",
            }):
                with patch("config.load_config", return_value={
                    "model": {
                        "name": "fake-model",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "base_url": "http://fake",
                    }
                }):
                    result = _run_child(
                        "test goal", "", "leaf",
                        agent_ref=parent_agent,
                    )

        # 运行期间子 agent 被注册了
        assert registered_during_run == [1]  # chat 时有 1 个子 agent
        # 完成后从 _children 移除
        assert len(parent_agent._children) == 0

    def test_run_child_interrupts_child_on_parent_interrupt(self):
        """父 agent 中断时正在运行的子 agent 也被中断。

        模拟：父 agent 有一个子 agent 在跑，父被中断后子也被中断。
        """
        parent = self._make_agent()
        child = self._make_agent()
        parent._children.append(child)

        # 子 agent 正在 "跑"（中断标志是 False）
        assert child._interrupt_requested is False

        # 父 agent 被中断
        parent.interrupt()

        # 子 agent 也收到中断
        assert child._interrupt_requested is True
