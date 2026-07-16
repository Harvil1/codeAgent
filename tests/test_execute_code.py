"""execute_code 工具测试。

覆盖：
  - 正常执行（print → stdout）
  - 返回 JSON 格式正确
  - 超时
  - 语法错误 → stderr + exit_code != 0
  - 空 code 被拒
  - check_fn（config.execute_code.enabled=False 时工具隐藏）
  - 权限审批（require_approval=True，无 callback → 拒绝）
  - 权限审批缓存（批准后同指纹不再问）
"""
import json
import sys
import time

import pytest

from tools.execute_code_tool import (
    _handle_execute_code,
    _check_execute_code_enabled,
    _approved_code_fingerprints,
    EXECUTE_CODE_SCHEMA,
)


# ---------------------------------------------------------------------------
# 基础执行
# ---------------------------------------------------------------------------

class TestBasicExecution:
    """正常执行路径。"""

    def test_simple_print(self):
        """print 输出到 stdout。"""
        result = _handle_execute_code(
            {"code": "print('hello world')"},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["success"] is True
        assert data["exit_code"] == 0
        assert "hello world" in data["stdout"]

    def test_return_value_json(self):
        """返回值是合法 JSON 字符串，包含所有必需字段。"""
        result = _handle_execute_code(
            {"code": "x = 1 + 2; print(x)"},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        # 必需字段
        assert "success" in data
        assert "exit_code" in data
        assert "stdout" in data
        assert "stderr" in data
        assert data["stdout"].strip() == "3"

    def test_computation(self):
        """复杂计算（循环）。"""
        code = "total = sum(range(100)); print(total)"
        result = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert "4950" in data["stdout"]

    def test_stderr_captured(self):
        """stderr 被正确捕获。"""
        code = "import sys; sys.stderr.write('warn msg\\n')"
        result = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert "warn msg" in data["stderr"]

    def test_exit_code_nonzero_on_error(self):
        """代码抛异常时 exit_code != 0。"""
        code = "raise ValueError('boom')"
        result = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["success"] is False
        assert data["exit_code"] != 0
        assert "ValueError" in data["stderr"]
        assert "boom" in data["stderr"]


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

class TestArgumentValidation:
    def test_empty_code_rejected(self):
        """空 code 返回 invalid_args 错误。"""
        result = _handle_execute_code(
            {"code": ""},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"

    def test_whitespace_only_code_rejected(self):
        """纯空白 code 也被拒。"""
        result = _handle_execute_code(
            {"code": "   \n\t  "},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"

    def test_missing_code_key(self):
        """缺少 code 键。"""
        result = _handle_execute_code(
            {},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["error_type"] == "invalid_args"


# ---------------------------------------------------------------------------
# 超时
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_returns_error(self):
        """超时返回 execute_timeout 错误。"""
        code = "import time; time.sleep(10)"
        result = _handle_execute_code(
            {"code": code, "timeout": 1},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["error_type"] == "execute_timeout"
        assert data["timeout"] == 1.0

    def test_timeout_from_config(self):
        """config.default_timeout 生效（不传 args.timeout）。"""
        code = "import time; time.sleep(10)"
        result = _handle_execute_code(
            {"code": code},
            config={
                "execute_code": {
                    "require_approval": False,
                    "default_timeout": 1,
                }
            },
        )
        data = json.loads(result)
        assert data["error_type"] == "execute_timeout"


# ---------------------------------------------------------------------------
# 语法错误
# ---------------------------------------------------------------------------

class TestSyntaxError:
    def test_syntax_error_returns_stderr(self):
        """语法错误在 stderr，exit_code != 0。"""
        code = "def broken(:"
        result = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["exit_code"] != 0
        assert "SyntaxError" in data["stderr"]


# ---------------------------------------------------------------------------
# 输出截断
# ---------------------------------------------------------------------------

class TestOutputTruncation:
    def test_stdout_truncated(self):
        """超过 5000 字符的 stdout 被截断。"""
        code = "print('A' * 10000)"
        result = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["stdout_truncated"] is True
        # 截断后的 stdout 不超过 5000 + 续写提示
        assert len(data["stdout"]) < 6000
        assert "截断" in data["stdout"]

    def test_short_output_not_truncated(self):
        """短输出不截断。"""
        code = "print('short')"
        result = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": False}},
        )
        data = json.loads(result)
        assert data["stdout_truncated"] is False


# ---------------------------------------------------------------------------
# check_fn
# ---------------------------------------------------------------------------

class TestCheckFn:
    def test_enabled_by_default(self):
        """默认 enabled=True。"""
        assert _check_execute_code_enabled(config={}) is True

    def test_disabled_via_config(self):
        """config.execute_code.enabled=False 时隐藏。"""
        config = {"execute_code": {"enabled": False}}
        assert _check_execute_code_enabled(config=config) is False

    def test_enabled_explicit(self):
        """显式 enabled=True。"""
        config = {"execute_code": {"enabled": True}}
        assert _check_execute_code_enabled(config=config) is True


# ---------------------------------------------------------------------------
# 权限审批
# ---------------------------------------------------------------------------

class TestApprovalGate:
    """权限审批逻辑测试。"""

    def setup_method(self):
        """每个测试前清空会话缓存。"""
        _approved_code_fingerprints.clear()

    def test_require_approval_no_callback_denies(self):
        """require_approval=True 且无 callback → 拒绝。"""
        from agent.permission import PermissionChecker
        checker = PermissionChecker()  # 无 callback

        result = _handle_execute_code(
            {"code": "print('test')"},
            config={"execute_code": {"require_approval": True}},
            permission_checker=checker,
        )
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"

    def test_require_approval_false_skips(self):
        """require_approval=False 跳过审批。"""
        from agent.permission import PermissionChecker
        checker = PermissionChecker()  # 无 callback，但应被跳过

        result = _handle_execute_code(
            {"code": "print('test')"},
            config={"execute_code": {"require_approval": False}},
            permission_checker=checker,
        )
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert "test" in data["stdout"]

    def test_approval_callback_approves(self):
        """callback 返回 True → 批准执行。"""
        from agent.permission import PermissionChecker
        checker = PermissionChecker(approval_callback=lambda cmd: True)

        result = _handle_execute_code(
            {"code": "print('approved')"},
            config={"execute_code": {"require_approval": True}},
            permission_checker=checker,
        )
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert "approved" in data["stdout"]

    def test_approval_callback_denies(self):
        """callback 返回 False → 拒绝。"""
        from agent.permission import PermissionChecker
        checker = PermissionChecker(approval_callback=lambda cmd: False)

        result = _handle_execute_code(
            {"code": "print('denied')"},
            config={"execute_code": {"require_approval": True}},
            permission_checker=checker,
        )
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"

    def test_session_cache_skips_second_ask(self):
        """同指纹代码批准后，第二次不再调 callback。"""
        call_count = [0]

        def callback(cmd):
            call_count[0] += 1
            return True

        from agent.permission import PermissionChecker
        checker = PermissionChecker(approval_callback=callback)
        code = "print('cached')"

        # 第一次：调 callback
        r1 = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": True}},
            permission_checker=checker,
        )
        assert json.loads(r1)["exit_code"] == 0
        assert call_count[0] == 1

        # 第二次：缓存命中，不调 callback
        r2 = _handle_execute_code(
            {"code": code},
            config={"execute_code": {"require_approval": True}},
            permission_checker=checker,
        )
        assert json.loads(r2)["exit_code"] == 0
        assert call_count[0] == 1  # 没有增加

    def test_callback_exception_denies(self):
        """callback 抛异常 → 安全默认拒绝。"""
        from agent.permission import PermissionChecker

        def bad_callback(cmd):
            raise RuntimeError("boom")

        checker = PermissionChecker(approval_callback=bad_callback)

        result = _handle_execute_code(
            {"code": "print('test')"},
            config={"execute_code": {"require_approval": True}},
            permission_checker=checker,
        )
        data = json.loads(result)
        assert data["error_type"] == "permission_denied"


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_has_required_fields(self):
        """schema 包含 name / parameters / description。"""
        assert EXECUTE_CODE_SCHEMA["name"] == "execute_code"
        assert "code" in EXECUTE_CODE_SCHEMA["parameters"]["properties"]
        assert "code" in EXECUTE_CODE_SCHEMA["parameters"]["required"]

    def test_schema_has_timeout_optional(self):
        """timeout 是可选参数。"""
        props = EXECUTE_CODE_SCHEMA["parameters"]["properties"]
        assert "timeout" in props
        assert "timeout" not in EXECUTE_CODE_SCHEMA["parameters"].get("required", [])
