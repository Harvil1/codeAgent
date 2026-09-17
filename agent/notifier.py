"""Windows 桌面弹窗通知（toast——屏幕右下角那种滑出来的小气泡）。

给谁用：主循环和权限模块在"需要用户抬头看一眼"的时机弹通知，比如后台
任务跑完了、有命令等你审批、目标被暂停了。用户切去干别的事也能被叫回来。

设计原则（大白话版）：
- 不装任何第三方包——直接调 Windows 自带的 PowerShell，让它用系统里的
  WinRT（Windows 运行时组件）弹 toast
- 出了任何问题都不报错、不炸主流程，只是安静地返回 False（fail-open：
  通知弹不出来不该影响正常干活）
- 不是 Windows（Linux/macOS）就什么都不做，返回 False
- 同一个标题 30 秒内只弹一次（防止后台任务扎堆完成时连环轰炸屏幕）
- 总开关在配置里：notifications.enabled，默认开

用法：
    from agent.notifier import notify
    notify("标题", "内容")

目前三个触发点（谁在调它）：
1. 后台任务完成：agent/__init__.py 的 _drain_injected_messages 在收完
   后台消息后过滤出状态类消息弹通知
2. 权限审批：agent/permission.py 在弹出审批询问前通知用户
3. goal 暂停：agent/goal.py 的 GoalState.pause() 集中接——不管因为
   断网/预算耗尽/手动暂停，统一在这一处弹
"""
import logging
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

# 节流记录：标题 → 上次弹通知的时间戳（monotonic 单调时钟，不受改系统时间影响）
_last_notify: dict = {}
_THROTTLE_SECONDS = 30.0


def _notifications_enabled() -> bool:
    """读配置里的通知总开关（notifications.enabled）。

    fail-open 取向：配置读不出来时默认当"开着"处理——宁可多弹一次通知，
    也不因为配置文件出问题就悄悄静音。返回 True/False 表示通知是否启用。
    """
    try:
        from config import load_config
        return bool(load_config().get("notifications", {}).get("enabled", True))
    except Exception:
        return True  # fail-open：配置读不到默认开


def _reset_throttle_for_test() -> None:
    """测试专用：把节流记录清空，让下条通知一定能弹。生产代码不要调。"""
    global _last_notify
    _last_notify = {}


def notify(title: str, message: str) -> bool:
    """弹一个 Windows toast 桌面通知（重要事件"跳到眼前"的提醒）。

    注意"成功投递"只代表命令交给 PowerShell 了，不保证用户真的看到气泡
    （Windows 通知设置可能关了）。

    参数：
        title: 通知标题（建议 30 字以内，太长显示不全）
        message: 通知正文（超过 200 字符会被截断）

    返回：True = 已成功交给 PowerShell 投递；False = 没投（不是 Windows /
        被节流 / 配置关了 / 投递过程出错）。任何异常都吞掉不往外抛。
    """
    # 闸门 1：非 Windows 直接放弃（Linux/macOS 用户根本看不到 toast）
    if sys.platform != "win32":
        return False
    # 闸门 2：配置总开关关了就不弹
    if not _notifications_enabled():
        return False
    # 闸门 3：同一标题 30 秒内只弹一次（防批量事件刷屏）
    now = time.monotonic()
    if now - _last_notify.get(title, 0.0) < _THROTTLE_SECONDS:
        return False
    _last_notify[title] = now
    # 节流表容量兜底：标题按 task_id 变化，长期运行会无限涨；
    # 超过 64 条整体清空（节流窗口只有 30 秒，清空代价可忽略）
    if len(_last_notify) > 64:
        _last_notify.clear()
    # 三道闸门都过了，真正去弹
    try:
        # 先把标题/正文里的 XML 特殊字符转义（toast 是 XML 格式）——
        # 不转义的话，内容里的 < & 之类会破坏 XML 结构，甚至夹带私货
        esc_t = (title or "").replace("&", "&amp;").replace("<", "&lt;")
        esc_m = (message or "")[:200].replace("&", "&amp;").replace("<", "&lt;")
        xml = (
            "<toast><visual><binding template='ToastGeneric'>"
            f"<text>{esc_t}</text><text>{esc_m}</text>"
            "</binding></visual></toast>"
        )
        # 用系统自带 PowerShell 调 WinRT 弹 toast——不用装任何第三方库
        ps = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI."
            "Notifications, ContentType = WindowsRuntime] | Out-Null;"
            "$x = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom."
            "XmlDocument, ContentType = WindowsRuntime]::new();"
            f"$x.LoadXml('{xml}');"
            "$t = [Windows.UI.Notifications.ToastNotification]::new($x);"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier"
            "('CodeAgent').Show($t)"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            timeout=10,
        )
        return True
    except Exception as e:
        logger.warning("notify fail-open: %s", e)
        return False
