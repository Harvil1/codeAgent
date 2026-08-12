"""工具集分组。

每个场景选一组工具暴露给 LLM。
为什么不全自动暴露？每个工具 schema 都消耗 token，需要人为控制可见性。
"""

from typing import Dict, List


# 核心工具：默认全部可见（项目做好的工具都暴露给 LLM；浏览器走 MCP，不进 core）
_CORE_TOOLS = [
    # 文件与命令
    "terminal",        # 执行 shell 命令
    "read_file",       # 读文件
    "write_file",      # 写文件
    "search_files",    # 搜索文件内容（grep）
    "str_replace",     # 文件定点替换编辑
    # 技能相关（05 实现）
    "skills_list",
    "skill_view",
    "skill_manage",
    "load_skill",  # P1：LLM 主动加载技能正文
    # 记忆（04 实现）
    "memory",
    # 会话搜索（07 实现）
    "session_search",
    # 委托（09 实现）
    "subagent",  # 子代理（对齐 Claude Code Agent；delegate_task 为兼容别名）
    "subagent_kill",  # Task K: 中断 async 子代理（set cancel_event）
    # 澄清（AskUserQuestion 复刻）
    "ask_user",
    # 持久化任务系统（P3 Task System，带 DAG 依赖）
    "task_create", "task_update", "task_complete", "task_list",
    # 任务系统扩展（Kanban）
    "task_block", "task_unblock", "task_link", "task_comment",
    "task_heartbeat", "task_artifacts",
    # Vision/Image 分析（B1：本地图片分析 + OCR）
    "image_analyze", "image_ocr",
    # 主动上下文压缩(借鉴 learn-claude-code s08,LLM 自己管理 context)
    "compact",
    # Task L: LLM 主动 snip（剪早期历史）+ ctx_inspect（查上下文状态）
    "snip",
    "ctx_inspect",
    # 网页抓取（对齐 Claude Code WebFetch）
    "web_fetch",
    # 网络搜索（对齐 Claude Code WebSearch，Tavily 后端，check_fn 门控）
    "web_search",
    # MCP 工具 schema 按需加载（对齐 Claude Code ToolSearch）
    "tool_search",
    # 后台任务（Phase 2b，check_fn 门控：无 bg 组件时自动隐藏）
    "bg_start", "bg_status", "bg_result", "bg_list", "bg_stop",
    # Team 多 agent 协作（Phase 4a）
    "team_send", "team_inbox", "team_members",
    "team_spawn", "team_shutdown", "idle",
]

TOOLSETS: Dict[str, dict] = {
    "core": {
        "description": "核心工具集 - agent 默认装备",
        "tools": _CORE_TOOLS,
        "includes": [],
    },
    "terminal": {
        "description": "终端和进程管理",
        "tools": ["terminal"],
        "includes": [],
    },
    "file": {
        "description": "文件操作",
        "tools": ["read_file", "write_file", "search_files"],
        "includes": [],
    },
    "minimal": {
        "description": "最小工具集（适合子代理）",
        "tools": ["terminal", "read_file"],
        "includes": [],
    },
    "mcp": {
        "description": "MCP 外部服务器工具（动态发现，通过 check_fn 门控）",
        "tools": [],  # 动态注册，工具名以 mcp__ 前缀
        "includes": [],
    },
    "bg": {
        "description": "后台任务管理（Phase 2b）",
        "tools": ["bg_start", "bg_status", "bg_result", "bg_list", "bg_stop"],
        "includes": [],
    },
    "team": {
        "description": "Team 多 agent 协作（Phase 4a）",
        "tools": ["team_send", "team_inbox", "team_members",
                  "team_spawn", "team_shutdown", "idle"],
        "includes": [],
    },
    "plan": {
        "description": "计划模式工具集（只读 + 计划工具，不能修改任何东西）",
        "tools": [
            "read_file",
            "search_files",
            "skills_list",
            "skill_view",
            "load_skill",
            "session_search",
            "exit_plan_mode",
            "plan_mode_v2_dispatch",  # P6: 多 Agent 并行（flag plan_mode_v2_parallel 门控）
        ],
        "includes": [],
    },
    "explore": {
        "description": "只读探索（Explore 子代理用）：读/搜/抓网页/查会话，无修改类",
        "tools": [
            "read_file",
            "search_files",
            "web_fetch",
            "session_search",
            "skills_list",
            "skill_view",
            "load_skill",
        ],
        "includes": [],
    },
}


# ---------------------------------------------------------------------------
# Task F: async 子代理工具白名单（ASYNC_AGENT_ALLOWED_TOOLS）
# ---------------------------------------------------------------------------
# 借鉴 claude-code-main src/constants/tools.ts:ASYNC_AGENT_ALLOWED_TOOLS
# 后台子代理在 daemon 线程里跑，用户感知不到 destructive 操作，
# 因此对 enabled_toolsets 做白名单过滤 + 对具体工具名做黑名单兜底。

ASYNC_AGENT_ALLOWED_TOOLSETS = frozenset({
    "core",      # 基础工具（read/write/search/terminal/memory/skill/...）
    "minimal",   # 子代理默认最小集
    "explore",   # 只读探索
    # 不含：mcp / team / bg / plan（这些有外部副作用或需交互）
})

ASYNC_AGENT_DISALLOWED_TOOLS = frozenset({
    # 后台任务管理（嵌套后台 → 不可控孙子进程）
    "bg_start", "bg_stop",
    # Team 多 agent 协作（影响其他 agent 进程）
    "team_spawn", "team_shutdown", "team_send",
    # 任务状态机推进（async 子代理不应修改全局任务图）
    "task_complete",
    # 子代理嵌套（防止递归派生）
    "subagent",
    # cron 调度（后台子代理不应注册定时任务）
    # idle 挂起（后台子代理不应进 IDLE 状态影响 team 协调）
    "idle",
})


def resolve_toolset(toolset_name: str) -> List[str]:
    """解析工具集，返回完整的工具名列表（递归展开 includes）。"""
    if toolset_name not in TOOLSETS:
        return []

    entry = TOOLSETS[toolset_name]
    tools = list(entry["tools"])

    # 递归展开 includes
    for included in entry.get("includes", []):
        tools.extend(resolve_toolset(included))

    # 去重（保序）
    return list(dict.fromkeys(tools))


def list_toolsets() -> List[str]:
    """返回所有工具集名。"""
    return list(TOOLSETS.keys())
