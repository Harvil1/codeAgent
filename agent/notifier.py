"""桌面通知（CCAR11 Task 6，Windows toast，零依赖）。

设计原则：
- 零第三方依赖（用 PowerShell + WinRT ToastNotificationManager）
- fail-open（任何异常返回 False，不抛）
- 非 Windows no-op（返回 False）
- 30 秒同标题节流（防 bg 任务批量完成时刷屏）
- config.notifications.enabled 控制（默认 True）

使用：
    from agent.notifier import notify
    notify("标题", "内容")

触发点（CCAR11 Task 6 接线）：
1. bg 完成：agent/__init__.py:_drain_injected_messages drain 后过滤 status
2. 权限审批：agent/permission.py 审批 callback 调用前
3. goal pause：agent/__init__.py network 异常 pause 分支
"""
import logging
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

# 节流状态：title -> monotonic ts
_last_notify: dict = {}
_THROTTLE_SECONDS = 30.0


def _notifications_enabled() -> bool:
    """读 config.notifications.enabled。fail-open：读不到返回 True（保守通知）。"""
    try:
        from config import load_config
        return bool(load_config().get("notifications", {}).get("enabled", True))
    except Exception:
        return True  # fail-open：配置读不到默认开


def _reset_throttle_for_test() -> None:
    """测试专用：清空节流状态。生产代码勿调。"""
    global _last_notify
    _last_notify = {}


def notify(title: str, message: str) -> bool:
    """发 Windows toast。

    返回 True 表示成功投递 PowerShell；False 表示未投递（非 Windows / 被节流 /
    配置关 / 投递失败）。任何异常都 fail-open 不抛出。

    Args:
        title: 通知标题（建议 < 30 字符）
        message: 通知正文（>200 字符自动截断）

    Returns:
        bool: 是否成功投递
    """
    # 闸门 1：非 Windows no-op（用户在 Linux/macOS 不会看到 toast）
    if sys.platform != "win32":
        return False
    # 闸门 2：config 开关
    if not _notifications_enabled():
        return False
    # 闸门 3：30s 同标题节流（防刷屏）
    now = time.monotonic()
    if now - _last_notify.get(title, 0.0) < _THROTTLE_SECONDS:
        return False
    _last_notify[title] = now
    # 实际投递
    try:
        # XML 转义防注入（用户内容不能逃出 XML 节点）
        esc_t = (title or "").replace("&", "&amp;").replace("<", "&lt;")
        esc_m = (message or "")[:200].replace("&", "&amp;").replace("<", "&lt;")
        xml = (
            "<toast><visual><binding template='ToastGeneric'>"
            f"<text>{esc_t}</text><text>{esc_m}</text>"
            "</binding></visual></toast>"
        )
        # PowerShell + WinRT：零第三方依赖
        ps = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI."
            "Notifications, ContentType = WindowsRuntime] | Out-Null;"
            "$x = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom."
            "XmlDocument, ContentType = WindowsRuntime]::new();"
            f"$x.LoadXml('{xml}');"
            "$t = [Windows.UI.Notifications.ToastNotification]::new($x);"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier"
            "('OmniMate').Show($t)"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            timeout=10,
        )
        return True
    except Exception as e:
        logger.debug("notify fail-open: %s", e)
        return False
