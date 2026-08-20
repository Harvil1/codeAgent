"""团队协作子包。

这里面装的是「多个 agent 组队干活」的全套零件：协调员（coordinator，
管成员名单和进程生死）、消息总线（bus）、异步信箱（mailbox）、工人进程
入口（worker）、自主生命周期（lifecycle）等。上层由 AIAgent / 工具层调用。
"""
