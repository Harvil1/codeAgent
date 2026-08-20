"""工具集分组——控制"哪些工具对 LLM 可见"。

像餐厅菜单：不把全部菜一次端上桌，而是按场景配成几套"套餐"（工具集），
每次只上一套。
为什么不全部端上来？因为每份工具说明书（schema）都要随每次请求发给
LLM、都花 token，可见性必须有人为控制。位于工具体系的可见性层，
被 model_tools.get_tool_definitions 调用。
"""

from typing import Dict, List


# 核心工具集：agent 默认装备，全部对 LLM 可见（浏览器的教训：不塞进 core，
# 想用浏览器就走 MCP 接外部服务——"能力放边缘"的设计原则）
_CORE_TOOLS = [
    # —— 文件与命令 ——
    "terminal",        # 跑 shell 命令
    "read_file",       # 读文件
    "write_file",      # 写文件
    "notebook_edit",   # 编辑 Jupyter 笔记本单元格（R20 #35，对齐 Claude Code）
    "search_files",    # 按内容搜文件（类似 grep）
    "glob",            # 按文件名模式找文件（CCAR11；和按内容搜的 search_files 互补）
    "lsp",             # 跳转定义/查引用（R26 #17；依赖外部 pylsp，装不上会自动隐身）
    "str_replace",     # 对文件做定点替换编辑
    # —— 技能（05 实现）——
    "skills_list",
    "skill_view",
    "skill_manage",
    "load_skill",  # P1：LLM 主动把技能正文读进上下文
    # —— 记忆（04 实现）——
    "memory",
    # LLM 主动深挖历史记忆（CCAR8；和会话开场自动注入的那份互补，这是按需查）
    "memory_recall",
    # —— 会话搜索（07 实现）——
    "session_search",
    # —— 委托（09 实现）——
    "subagent",  # 子代理（主对话派出去帮忙干活的分身；老名字 delegate_task 也能用）
    # —— 确定性工作流编排（R28：批量派分身 + 执行日志断点续跑 + 花费封顶）——
    "workflow",
    "subagent_kill",  # 中断正在后台跑的子代理（Task K）
    "subagent_resume",  # 让中断的子代理从存档接着跑（CCAR10 Task 4）
    # —— 向用户提问（复刻 AskUserQuestion）——
    "ask_user",
    # —— 结构化简报（CCAR8：重要操作前先给用户过目一遍）——
    "brief",
    # —— 持久化任务清单（P3 Task System，任务之间能有先后依赖）——
    "task_create", "task_update", "task_complete", "task_list",
    # —— 任务系统的看板扩展 ——
    "task_block", "task_unblock", "task_link", "task_comment",
    "task_heartbeat", "task_artifacts",
    # —— 图片分析（B1：本地图片理解 + 文字识别 OCR）——
    "image_analyze", "image_ocr",
    # —— 主动压缩上下文（借鉴 learn-claude-code s08：让 LLM 自己管窗口大小）——
    "compact",
    # —— LLM 主动剪掉早期历史（snip）+ 查看上下文现状（ctx_inspect）（Task L）——
    "snip",
    "ctx_inspect",
    # —— 抓网页（对齐 Claude Code WebFetch）——
    "web_fetch",
    # —— 搜网络（对齐 Claude Code WebSearch；Tavily 后端，没配 key 自动隐身）——
    "web_search",
    # —— MCP 工具说明书按需加载（对齐 Claude Code ToolSearch）——
    "tool_search",
    # —— 后台任务（Phase 2b；后台组件没起时自动隐身）——
    "bg_start", "bg_status", "bg_result", "bg_list", "bg_stop",
    # —— 多 agent 团队协作（Phase 4a）——
    "team_send", "team_inbox", "team_members",
    "team_spawn", "team_shutdown", "idle",
    # —— 队友间的异步邮箱（CCAR8：寄完就走的通信；和 team 总线的"一问一答"分工）——
    "mailbox_send", "mailbox_check", "mailbox_clear",
    # —— 定时任务（CCAR12 Task 3：LLM 自己创建/查看/删除 cron）——
    "cron_create", "cron_list", "cron_delete",
    # —— 目标驱动模式（CCAR12 Task 4：LLM 自己启动/管理 goal，和 CLI 的 /goal 同一套）——
    "goal_start", "goal_status", "goal_pause", "goal_resume", "goal_clear",
    # —— 进出独立工作副本（CCAR12 Task 6，对齐 CCB 的 EnterWorktree/ExitWorktree）——
    "worktree_enter", "worktree_exit",
    # —— 配置读写（CCAR12 Task 7：只允许白名单里的 7 个键；set 既落盘又当场生效）——
    "config_get", "config_set",
]

# 所有"套餐"的登记表：名字 → {描述, 包含哪些工具, 还捎带哪些别的套餐}
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
        # MCP = 接外部工具服务器的协议。这个套餐是"动态菜单"：
        # 固定清单为空，实际工具在连接服务器后现场登记（名字带 mcp__ 前缀，
        # 服务器断线时对应工具自动隐身）
        "description": "MCP 外部服务器工具（动态发现，通过 check_fn 门控）",
        "tools": [],
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
        # 计划模式：先只许看不许动，产出计划等用户批准后才放开手脚
        "description": "计划模式工具集（只读 + 计划工具，不能修改任何东西）",
        "tools": [
            "read_file",
            "search_files",
            "skills_list",
            "skill_view",
            "load_skill",
            "session_search",
            "exit_plan_mode",
            "plan_mode_v2_dispatch",  # P6：多 Agent 并行做计划（开关 plan_mode_v2_parallel 控制）
        ],
        "includes": [],
    },
    "explore": {
        # 给 Explore 子代理（专职"只看不改"的侦察兵分身）用的套餐
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
# Task F：后台（async）子代理的工具门禁
# ---------------------------------------------------------------------------
# 借鉴 claude-code-main 的 ASYNC_AGENT_ALLOWED_TOOLS。
# 为什么要拦：后台子代理跑在无人盯守的线程里，闯了祸用户根本看不见，
# 所以要"套餐级白名单放行 + 单个工具黑名单兜底"两道闸。

# 第一道闸：后台子代理只许用这几个套餐
ASYNC_AGENT_ALLOWED_TOOLSETS = frozenset({
    "core",      # 基础工具（读写/搜索/终端/记忆/技能……）
    "minimal",   # 子代理默认最小集
    "explore",   # 只读探索
    # 故意不含：mcp / team / bg / plan（这些有外部副作用或需要人机交互）
})

# 第二道闸：即使在放行的套餐里，这些具体工具也不许后台子代理碰
ASYNC_AGENT_DISALLOWED_TOOLS = frozenset({
    # 后台任务管理（后台里再开后台 → 孙子进程没人管得住）
    "bg_start", "bg_stop",
    # 团队协作（会牵动其他 agent 进程）
    "team_spawn", "team_shutdown", "team_send",
    # 异步邮箱投递（和 team_send 同理：后台线程打扰别的 agent；
    # 收信/清信箱只动自己的邮箱，留给子代理自救）
    "mailbox_send",
    # 把任务标记为完成（后台子代理不该擅自推进全局任务图）
    "task_complete",
    # 再派子代理（防套娃式无限派生）
    "subagent",
    # 定时任务（后台子代理不该注册/删除 cron）
    "cron_create", "cron_delete",
    # 挂起等待（后台子代理不该进入 IDLE 状态干扰团队协调）
    "idle",
    # 启动/恢复 goal 循环（CCAR12 Task 4 复审定案：goal 循环没有
    # "套娃层数"守卫，后台线程里激活会不可中断地烧 token；
    # 暂停/清除是刹车工具，保留给子代理自救）
    "goal_start", "goal_resume",
    # 进入会话级工作副本（CCAR12 Task 6 修复：后台子代理一进入就会
    # 占住模块级的 _session_worktree 标记——主对话随后再进会被
    # "已在副本中"卡死；退出工具留给子代理自救，处置思路同上）
    "worktree_enter",
    # workflow 编排（防 workflow 里再嵌 workflow；subagent 已在列）
    "workflow",
})


def resolve_toolset(toolset_name: str) -> List[str]:
    """把一个工具集名展开成具体的工具名清单。

    背景：套餐可以用 includes 说"我还捎带另一个套餐"，所以要一层层
    递归展开直到拿到全部工具名。

    参数：
        toolset_name: 套餐名（如 "core"、"minimal"）。

    返回：工具名列表（按首次出现去重、保序）；套餐名不存在返回空列表。
    """
    if toolset_name not in TOOLSETS:
        return []

    entry = TOOLSETS[toolset_name]
    tools = list(entry["tools"])

    # 把捎带的套餐也递归展开进来
    for included in entry.get("includes", []):
        tools.extend(resolve_toolset(included))

    # 去重（保持原有顺序；dict.fromkeys 是"按首次出现去重"的惯用写法）
    return list(dict.fromkeys(tools))


def list_toolsets() -> List[str]:
    """列出所有可用的工具集名。

    返回：套餐名列表（供 UI 或配置校验用）。
    """
    return list(TOOLSETS.keys())
