"""工具集分组。

每个场景选一组工具暴露给 LLM。
为什么不全自动暴露？每个工具 schema 都消耗 token，需要人为控制可见性。
"""

from typing import Dict, List


# 核心工具：几乎所有场景都需要
_CORE_TOOLS = [
    "terminal",        # 执行 shell 命令
    "read_file",       # 读文件
    "write_file",      # 写文件
    "search_files",    # 搜索文件内容（grep）
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
    "delegate_task",
    # 任务清单（P1 借鉴 业界 TodoWrite）
    "todo_write",
    # 持久化任务系统（P3 Task System，带 DAG 依赖）
    "task_create", "task_update", "task_complete", "task_list",
    # Python 代码沙箱（让 LLM 直接写代码执行）
    "execute_code",
    # Vision/Image 分析（B1：本地图片分析 + OCR）
    "image_analyze", "image_ocr",
    # 主动上下文压缩(借鉴 learn-claude-code s08,LLM 自己管理 context)
    "compact",
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
    "browser": {
        "description": "浏览器自动化（13 个工具，基于 Playwright）",
        "tools": [
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_press_key",
            "browser_back", "browser_forward", "browser_close",
            "browser_get_images", "browser_vision", "browser_console",
            "browser_cdp",
        ],
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
            "todo_write",
            "exit_plan_mode",
        ],
        "includes": [],
    },
}


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
