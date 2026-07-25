"""HarvilAgent Flet 桌面 GUI。

后端代码(RuntimeContext / AIAgent / 工具 / 记忆)完全复用 cli.py,
只替换 I/O 层:console.input → TextField,console.print → Flet 组件。

用法:
    python main.py --gui     # 启动 GUI
    python main.py            # 启动 CLI(原终端界面)
"""
import json
import logging
import os
import threading
from pathlib import Path

import flet as ft

logger = logging.getLogger(__name__)

# ── 主题色 ──
_PRIMARY = ft.Colors.BLUE
_PRIMARY_LIGHT = ft.Colors.BLUE_50
_PRIMARY_DARK = ft.Colors.BLUE_700
_BG = ft.Colors.WHITE
_SURFACE = ft.Colors.GREY_50
_TEXT = ft.Colors.GREY_900
_TEXT_HINT = ft.Colors.GREY_500
_DIVIDER = ft.Colors.GREY_300


def run_gui():
    """启动 Flet GUI 应用。"""
    ft.app(target=_main, view=ft.AppView.FLET_APP)


def _build_theme():
    """构建蓝色亮色主题。"""
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=_PRIMARY,
            on_primary=ft.Colors.WHITE,
            primary_container=_PRIMARY_LIGHT,
            surface=_BG,
            on_surface=_TEXT,
            background=_BG,
        ),
    )


def _main(page: ft.Page):
    """Flet 主页面(每个窗口一个实例)。"""
    # ── 页面配置 ──
    page.title = "HarvilAgent"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = _build_theme()
    page.bgcolor = _BG
    page.width = 900
    page.height = 700
    page.padding = 0

    # ── 初始化 RuntimeContext(复用 cli.py) ──
    page.add(
        ft.Container(
            content=ft.Column(
                [
                    ft.ProgressRing(width=40, height=40, color=_PRIMARY),
                    ft.Text("正在初始化 HarvilAgent...", size=16, color=_TEXT),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            alignment=ft.Alignment(0, 0),
            expand=True,
            bgcolor=_BG,
        )
    )
    page.update()

    from cli import RuntimeContext
    rt = RuntimeContext()
    try:
        rt.initialize()
    except SystemExit:
        page.controls.clear()
        page.add(
            ft.Container(
                content=ft.Text(
                    "初始化失败:请在 settings.json 配置 API key",
                    color=ft.Colors.RED,
                    size=16,
                ),
                alignment=ft.Alignment(0, 0),
                expand=True,
            )
        )
        page.update()
        return

    # 初始化完成,清空加载页
    page.controls.clear()

    # ── 状态变量 ──
    state = {
        "busy": False,
        "current_md": None,
        "rt": rt,
    }

    # ── UI 组件 ──
    chat_list = ft.ListView(
        expand=True,
        spacing=10,
        padding=ft.Padding.all(20),
        auto_scroll=True,
    )

    input_field = ft.TextField(
        hint_text="输入消息,/help 查看命令...",
        hint_style=ft.TextStyle(color=_TEXT_HINT),
        expand=True,
        border_radius=10,
        border_color=_DIVIDER,
        focused_border_color=_PRIMARY,
        max_lines=5,
        min_lines=1,
        shift_enter=True,
        on_submit=lambda e: _on_send(e, page, chat_list, input_field, send_btn, status_text, state),
    )

    send_btn = ft.Button(
        "发送",
        icon=ft.Icons.SEND,
        on_click=lambda e: _on_send(e, page, chat_list, input_field, send_btn, status_text, state),
        style=ft.ButtonStyle(bgcolor=_PRIMARY, color=ft.Colors.WHITE),
    )

    status_text = ft.Text(
        size=12,
        color=_TEXT_HINT,
    )

    # ── 模型显示 + 配置入口 ──
    model_name = rt.config.get("model", {}).get("name", "?")
    effort = rt.config.get("model", {}).get("effort_level", "")

    model_display = ft.Text(
        f"模型: {model_name}" + (f" ({effort})" if effort else ""),
        size=12,
        color=_TEXT_HINT,
    )

    # ── AppBar ──
    app_bar = ft.AppBar(
        leading=ft.Icon(ft.Icons.SMART_TOY, color=_PRIMARY),
        leading_width=40,
        title=ft.Text("HarvilAgent", weight=ft.FontWeight.BOLD, color=_PRIMARY),
        actions=[
            model_display,
            ft.PopupMenuButton(
                icon=ft.Icons.SETTINGS,
                items=[
                    ft.PopupMenuItem(
                        content=ft.Text("切换模型", color=_TEXT),
                        icon=ft.Icons.SWAP_HORIZ,
                        on_click=lambda e: _show_model_config(page, chat_list, state, model_display),
                    ),
                    ft.PopupMenuItem(
                        content=ft.Text("思考强度", color=_TEXT),
                        icon=ft.Icons.PSYCHOLOGY,
                        on_click=lambda e: _show_effort_config(page, chat_list, state, model_display),
                    ),
                    ft.PopupMenuItem(),  # 分割线
                    ft.PopupMenuItem(
                        content=ft.Text("新对话", color=_TEXT),
                        icon=ft.Icons.ADD_COMMENT,
                        on_click=lambda e: _new_session(page, chat_list, status_text, state),
                    ),
                    ft.PopupMenuItem(
                        content=ft.Text("历史会话", color=_TEXT),
                        icon=ft.Icons.HISTORY,
                        on_click=lambda e: _show_sessions(page, chat_list, state),
                    ),
                    ft.PopupMenuItem(
                        content=ft.Text("记忆", color=_TEXT),
                        icon=ft.Icons.BOOK,
                        on_click=lambda e: _show_memory(page, chat_list, state),
                    ),
                    ft.PopupMenuItem(
                        content=ft.Text("技能", color=_TEXT),
                        icon=ft.Icons.BUILD,
                        on_click=lambda e: _show_skills(page, chat_list, state),
                    ),
                    ft.PopupMenuItem(
                        content=ft.Text("用量统计", color=_TEXT),
                        icon=ft.Icons.ANALYTICS,
                        on_click=lambda e: _show_usage(page, chat_list, state),
                    ),
                ]
            ),
        ],
        bgcolor=_SURFACE,
    )

    # ── 快捷命令按钮 ──
    quick_cmds = ft.Row(
        [
            ft.TextButton("/help", on_click=lambda e: _quick_cmd("/help", page, chat_list, input_field, send_btn, status_text, state)),
            ft.TextButton("/skills", on_click=lambda e: _quick_cmd("/skills", page, chat_list, input_field, send_btn, status_text, state)),
            ft.TextButton("/memory", on_click=lambda e: _quick_cmd("/memory", page, chat_list, input_field, send_btn, status_text, state)),
            ft.TextButton("/sessions", on_click=lambda e: _quick_cmd("/sessions", page, chat_list, input_field, send_btn, status_text, state)),
        ],
        spacing=5,
    )

    # ── 组装页面 ──
    page.add(
        ft.Column(
            [
                app_bar,
                # 聊天区
                ft.Container(
                    content=chat_list,
                    expand=True,
                    bgcolor=_BG,
                ),
                # 分割线
                ft.Divider(height=1, color=_DIVIDER),
                # 快捷命令
                ft.Container(
                    content=quick_cmds,
                    padding=ft.Padding.symmetric(horizontal=10, vertical=2),
                    bgcolor=_BG,
                ),
                # 输入区
                ft.Container(
                    content=ft.Row(
                        [input_field, send_btn],
                        spacing=10,
                    ),
                    padding=ft.Padding.all(10),
                    bgcolor=_BG,
                ),
                # 状态栏
                ft.Container(
                    content=status_text,
                    padding=ft.Padding.symmetric(horizontal=15, vertical=5),
                    bgcolor=_SURFACE,
                ),
            ],
            spacing=0,
        )
    )

    # 欢迎消息
    _add_system_message(chat_list, page, "HarvilAgent 已就绪。输入消息开始对话,/help 查看命令。")

    # 更新状态栏
    _update_status(status_text, page, state)

    # 设置 stream_callback(让 agent 流式输出到 UI)
    def _stream_cb(event: dict):
        _on_stream_event(event, chat_list, page, state)

    rt.agent._stream_callback = _stream_cb

    # 设置 on_tool_call callback
    def _tool_cb(name: str, args: dict):
        short_args = {}
        for k, v in (args or {}).items():
            s = str(v)
            short_args[k] = s if len(s) <= 80 else s[:77] + "..."
        _add_tool_progress(chat_list, page, name, short_args)

    rt.agent.on_tool_call = _tool_cb


# ---------------------------------------------------------------------------
# UI 辅助函数
# ---------------------------------------------------------------------------

def _add_user_bubble(chat_list: ft.ListView, page: ft.Page, text: str):
    """添加用户消息气泡(右对齐,蓝色背景)。"""
    chat_list.controls.append(
        ft.Row(
            [
                ft.Container(
                    content=ft.Text(text, color=ft.Colors.WHITE),
                    bgcolor=_PRIMARY_DARK,
                    border_radius=12,
                    padding=ft.Padding.all(12),
                )
            ],
            alignment=ft.MainAxisAlignment.END,
        )
    )
    page.update()


def _add_ai_bubble(chat_list: ft.ListView, page: ft.Page, text: str = ""):
    """添加 AI 消息气泡(左对齐,浅灰背景,Markdown 渲染)。"""
    md = ft.Markdown(
        value=text,
        selectable=True,
        extension_set="github",
    )
    container = ft.Container(
        content=md,
        bgcolor=_SURFACE,
        border_radius=12,
        padding=ft.Padding.all(12),
        border=ft.Border.all(1, _DIVIDER),
    )
    chat_list.controls.append(
        ft.Row(
            [container],
            alignment=ft.MainAxisAlignment.START,
        )
    )
    page.update()
    return md


def _add_tool_progress(chat_list: ft.ListView, page: ft.Page, name: str, args: dict):
    """添加工具调用进度(蓝色小字)。"""
    args_str = json.dumps(args, ensure_ascii=False)[:100]
    chat_list.controls.append(
        ft.Container(
            content=ft.Text(
                f"⟳ {name} {args_str}",
                size=12,
                color=_PRIMARY,
                selectable=True,
            ),
            padding=ft.Padding.only(left=20),
        )
    )
    page.update()


def _add_system_message(chat_list: ft.ListView, page: ft.Page, text: str):
    """添加系统消息(居中,灰色斜体)。"""
    chat_list.controls.append(
        ft.Container(
            content=ft.Text(text, size=13, color=_TEXT_HINT, italic=True),
            alignment=ft.Alignment(0, 0),
            padding=ft.Padding.all(10),
        )
    )
    page.update()


def _on_stream_event(event: dict, chat_list: ft.ListView, page: ft.Page, state: dict):
    """处理 agent 流式事件 → 更新 UI。"""
    etype = event.get("type")

    if etype == "content":
        delta = event.get("delta", "")
        if delta:
            if state["current_md"] is None:
                state["current_md"] = _add_ai_bubble(chat_list, page, delta)
            else:
                state["current_md"].value += delta
                page.update()

    elif etype == "tool_call_start":
        state["current_md"] = None

    elif etype == "done":
        state["current_md"] = None


def _update_status(status_text: ft.Text, page: ft.Page, state: dict):
    """更新底部状态栏。"""
    rt = state["rt"]
    try:
        msg_count = len(rt.agent.conversation_history) if rt.agent else 0
        mem_count = len(rt.memory_store.list_all()) if rt.memory_store else 0
        from agent.skill_commands import scan_skill_commands
        from constants import all_skills_dirs
        skill_count = len(scan_skill_commands(all_skills_dirs()))
    except Exception:
        msg_count = 0
        mem_count = 0
        skill_count = 0

    status_text.value = (
        f"  {msg_count} 条消息 | {mem_count} 条记忆 | {skill_count} 个技能"
        + (" | 思考中..." if state["busy"] else "")
    )
    page.update()


# ---------------------------------------------------------------------------
# 模型配置界面
# ---------------------------------------------------------------------------

def _show_model_config(page: ft.Page, chat_list: ft.ListView, state: dict, model_display: ft.Text):
    """显示模型切换对话框。"""
    rt = state["rt"]
    try:
        from agent.settings import list_models, get_current_model_config
        models = list_models()
        current = get_current_model_config().get("name", "")

        if not models:
            _add_system_message(chat_list, page, "未配置任何模型。在 settings.json 的 llm 段添加。")
            return

        # 构建选项文本
        lines = ["## 模型配置\n\n当前可用模型:\n"]
        for name, cfg in models.items():
            mark = "← **当前**" if name == current else ""
            model_id = cfg.get("model", "?")
            has_key = "✓" if (cfg.get("api_key") or cfg.get("auth_token")) else "❌ 无key"
            lines.append(f"- **{name}** ({model_id}) {has_key} {mark}")

        lines.append("\n\n输入 `/model <name>` 切换模型(如 `/model opus`)")

        _add_ai_bubble(chat_list, page, "\n".join(lines))
    except Exception as e:
        _add_system_message(chat_list, page, f"读取模型配置失败: {e}")


def _show_effort_config(page: ft.Page, chat_list: ft.ListView, state: dict, model_display: ft.Text):
    """显示思考强度配置。"""
    rt = state["rt"]
    current_effort = rt.config.get("model", {}).get("effort_level", "")

    lines = [
        "## 思考强度配置\n",
        f"当前: **{current_effort or '(未设置,默认 high)'}**\n",
        "可选值:\n",
        "- **max** — 最强推理(慢,10-30秒/轮)",
        "- **high** — 标准推理(推荐,3-8秒/轮)",
        "- **medium** — 轻量思考(1-3秒/轮)",
        "- **low** — 不思考(最快)",
        "\n\n修改方法: 编辑 `~/.agent/settings.json` 的 `llm.effort_level` 字段",
        f"\n\n当前文件路径: `{rt.home}/settings.json`",
    ]

    _add_ai_bubble(chat_list, page, "\n".join(lines))


# ---------------------------------------------------------------------------
# 用户操作处理
# ---------------------------------------------------------------------------

def _on_send(e, page: ft.Page, chat_list: ft.ListView,
             input_field: ft.TextField, send_btn: ft.Button,
             status_text: ft.Text, state: dict):
    """发送消息。"""
    if state["busy"]:
        return

    user_input = (input_field.value or "").strip()
    if not user_input:
        return

    input_field.value = ""
    page.update()

    rt = state["rt"]

    # 斜杠命令
    if user_input.startswith("/"):
        _handle_slash_command(user_input, page, chat_list, status_text, state)
        return

    # 技能命令
    cmd_name = user_input.split()[0] if user_input.split() else ""
    if cmd_name in rt.skill_commands:
        from agent.skill_commands import execute_skill
        skill_info = rt.skill_commands[cmd_name]
        user_input = execute_skill(skill_info["skill_md_path"], user_input)
    elif cmd_name in getattr(rt, "bundle_commands", {}):
        from agent.skill_commands import execute_bundle
        bundle_info = rt.bundle_commands[cmd_name]
        user_input = execute_bundle(cmd_name, user_input, rt.skills_dir if hasattr(rt, 'skills_dir') else None)

    # 显示用户消息
    _add_user_bubble(chat_list, page, user_input[:500])

    # 状态:忙
    state["busy"] = True
    send_btn.disabled = True
    _update_status(status_text, page, state)

    # 保存用户消息到会话
    if rt.session_store and rt.session_id:
        rt.session_store.append_message(rt.session_id, "user", user_input)

    # 后台线程运行 run_conversation(不阻塞 UI)
    def _run():
        try:
            response = rt.agent.run_conversation(user_input)
            if rt.session_store and rt.session_id:
                rt.session_store.append_message(rt.session_id, "assistant", response)
        except Exception as ex:
            logger.exception("agent 运行错误")
            _add_system_message(chat_list, page, f"[错误] {ex}")
        finally:
            state["busy"] = False
            send_btn.disabled = False
            _update_status(status_text, page, state)

    t = threading.Thread(target=_run, daemon=True, name="agent_run")
    t.start()


def _handle_slash_command(cmd: str, page: ft.Page, chat_list: ft.ListView,
                          status_text: ft.Text, state: dict):
    """处理斜杠命令(复用 cli.py 的 _handle_command)。"""
    rt = state["rt"]
    _add_system_message(chat_list, page, f"$ {cmd}")

    try:
        from cli import _handle_command
        handled = _handle_command(cmd, rt)
        if not handled and cmd.strip() in ("/quit", "/exit"):
            page.exit()
    except Exception as e:
        _add_system_message(chat_list, page, f"命令错误: {e}")

    _update_status(status_text, page, state)


def _quick_cmd(cmd: str, page: ft.Page, chat_list: ft.ListView,
               input_field: ft.TextField, send_btn: ft.Button,
               status_text: ft.Text, state: dict):
    """快捷命令按钮触发。"""
    _handle_slash_command(cmd, page, chat_list, status_text, state)


def _new_session(page: ft.Page, chat_list: ft.ListView, status_text: ft.Text, state: dict):
    """新对话。"""
    rt = state["rt"]
    rt.new_session()
    chat_list.controls.clear()
    _add_system_message(chat_list, page, "新对话已开始。")
    _update_status(status_text, page, state)


def _show_sessions(page: ft.Page, chat_list: ft.ListView, state: dict):
    """显示历史会话。"""
    rt = state["rt"]
    try:
        sessions = rt.session_store.list_sessions(limit=20)
        if not sessions:
            _add_system_message(chat_list, page, "暂无历史会话。")
            return

        lines = ["## 历史会话\n"]
        for i, s in enumerate(sessions):
            title = s.get("title") or "(无标题)"
            count = s.get("message_count", 0)
            lines.append(f"{i}. {title}({count} 条)")

        _add_ai_bubble(chat_list, page, "\n".join(lines))
    except Exception as e:
        _add_system_message(chat_list, page, f"读取会话失败: {e}")


def _show_memory(page: ft.Page, chat_list: ft.ListView, state: dict):
    """显示记忆。"""
    rt = state["rt"]
    try:
        snapshot = rt.memory_store.snapshot_for_prompt()
        if snapshot:
            _add_ai_bubble(chat_list, page, f"## 记忆索引\n\n{snapshot}")
        else:
            _add_system_message(chat_list, page, "暂无记忆。")
    except Exception as e:
        _add_system_message(chat_list, page, f"读取记忆失败: {e}")


def _show_skills(page: ft.Page, chat_list: ft.ListView, state: dict):
    """显示技能。"""
    rt = state["rt"]
    try:
        from agent.skill_commands import scan_skill_commands
        from constants import all_skills_dirs
        cmds = scan_skill_commands(all_skills_dirs())
        if cmds:
            lines = ["## 可用技能\n"]
            for name, info in sorted(cmds.items()):
                lines.append(f"- **{name}**: {info.get('description', '')}")
            _add_ai_bubble(chat_list, page, "\n".join(lines))
        else:
            _add_system_message(chat_list, page, "暂无技能。")
    except Exception as e:
        _add_system_message(chat_list, page, f"读取技能失败: {e}")


def _show_usage(page: ft.Page, chat_list: ft.ListView, state: dict):
    """显示用量统计。"""
    rt = state["rt"]
    try:
        stats = rt.agent._llm_usage_stats
        lines = [
            "## LLM 用量统计\n",
            f"- 总调用次数: {stats.get('total_calls', 0)}",
            f"- Prompt tokens: {stats.get('total_prompt_tokens', 0):,}",
            f"- Completion tokens: {stats.get('total_completion_tokens', 0):,}",
            f"- Cache read: {stats.get('total_cache_read_tokens', 0):,}",
            f"- Cache creation: {stats.get('total_cache_creation_tokens', 0):,}",
        ]
        _add_ai_bubble(chat_list, page, "\n".join(lines))
    except Exception as e:
        _add_system_message(chat_list, page, f"读取用量失败: {e}")
