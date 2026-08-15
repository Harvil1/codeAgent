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
