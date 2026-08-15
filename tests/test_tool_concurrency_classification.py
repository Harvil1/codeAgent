"""验证每个工具的 isConcurrencySafe 分类合理（Task F1）。

分类规则（spec §6.3 + 决策 5）：
- Safe=True：只读 / 无副作用工具（read_file/list_dir/grep 等读类），可 asyncio.gather 并发
- Safe=False：有副作用工具（write_file/terminal/memory_save 等写/执行/外部调用），必须串行
- 默认 False（安全默认 > 事后补救）—— 未显式标注的工具一律串行

补齐原则：新增工具必须在此处两个集合之一登记，否则 test_all_tools_classified 会 fail。
"""
from tools.registry import registry, discover_builtin_tools

# 触发所有自注册工具模块的 import（register 调用）
discover_builtin_tools()


# ---------------------------------------------------------------------------
# 分类清单（与 tools/*.py 的 isConcurrencySafe 参数一一对应）
# ---------------------------------------------------------------------------

# Safe=True：只读 / 无副作用工具。
# 12 个，全是查询类（读文件、查状态、列目录、搜历史、查上下文）。
SAFE_TOOLS = {
    # 文件类（只读）
    "read_file",        # 读文件内容
    "search_files",     # grep 文件内容
    "glob",             # CCAR11 Task 1: 文件名模式匹配（只读，不读内容）
    # 技能类（只读）
    "skills_list",      # 列技能目录
    "skill_view",       # 看技能正文（bump view 是小副作用，对并发不致命）
    # 会话搜索（只读）
    "session_search",   # FTS 查历史对话
    # 任务类（只读）
    "task_list",        # 列任务
    # 工具目录（只读）
    "tool_search",      # 查 MCP 工具 schema
    # 后台任务（只读）
    "bg_status",        # 查任务状态
    "bg_result",        # 取任务输出
    "bg_list",          # 列后台任务
    # 团队（只读）
    "team_members",     # 列成员状态
    # 上下文自查（只读）
    "ctx_inspect",      # Task L: 读 messages_count / cache_stats / token 估算
    "brief",            # CCAR8 Task 2: 纯 echo 输出格式约定，无副作用
    # mailbox（只读）
    "mailbox_check",    # CCAR8 Task 8: 读自己 mailbox 的邮件
}

# Safe=False：有副作用 / 外部调用 / 交互式 / 状态变更工具。
# 34 个，覆盖写入、执行、外部 API、子进程、消息总线等。
UNSAFE_TOOLS = {
    # 文件类（写入）
    "write_file",       # 写文件
    "str_replace",      # 改文件内容
    # 执行类
    "terminal",         # shell 命令（最强副作用）
    # 交互式
    "ask_user",         # 阻塞等用户输入（并发会导致提示交错）
    # 上下文管理（副作用）
    "compact",          # 触发 LLM 压缩（改消息历史）
    "snip",             # Task L: 剪早期历史（改 conversation_history）
    # 子代理（副作用）
    "subagent",         # spawn 子 agent（重资源 + 改子任务状态）
    "delegate_task",    # subagent 的 _compat alias
    "subagent_kill",    # Task K: set cancel_event + 改注册表（副作用）
    "subagent_resume",  # CCAR10 Task 4: 重启 AIAgent 子代理（重资源 + 写 transcript）
    # 图像（外部 API）
    "image_analyze",    # vision API 调用
    "image_ocr",        # OCR API 调用
    # 记忆（混合 action，save/update/delete 有写副作用，不能拆）
    "memory",
    # 技能加载（副作用：可能改 agent._skill_tool_scope）
    "load_skill",
    # 技能管理（写入）
    "skill_manage",     # 创建/更新/归档/删除技能文件
    # Plan Mode（状态变更）
    "exit_plan_mode",   # 触发审批流程
    "plan_mode_v2_dispatch",  # P6: 起 N 个子代理 + 合并，串行更稳
    # 任务类（写入）
    "task_create",
    "task_update",
    "task_complete",
    "task_heartbeat",
    "task_comment",
    "task_artifacts",
    "task_block",
    "task_unblock",
    "task_link",
    # 后台任务（副作用）
    "bg_start",         # 启动子进程
    "bg_stop",          # 杀子进程
    # 团队（副作用）
    "team_send",        # 发消息到 bus
    "team_inbox",       # 读后清空（消费式）
    "team_spawn",       # 启动子 agent 进程
    "team_shutdown",    # 关子 agent 进程
    "idle",             # 改 worker 状态机
    # mailbox（副作用）
    "mailbox_send",     # CCAR8 Task 8: 写投递邮件
    "mailbox_clear",    # CCAR8 Task 8: 清空 mailbox
    # 网络（外部调用）
    "web_fetch",        # 抓 URL
    "web_search",       # Tavily 搜索
    # 记忆召回（外部 LLM）
    "memory_recall",    # CCAR8 Task 3: 调 aux_llm（retrieve_relevant），消耗配额
    # Cron 定时任务（CCAR12 Task 3：写 jobs.json + 影响调度行为）
    "cron_create",
    "cron_list",        # 按 brief 归 UNSAFE（与 create/delete 同组管理）
    "cron_delete",
}


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_safe_tools_marked_correctly():
    """所有 SAFE_TOOLS 必须已注册且 isConcurrencySafe=True。"""
    for name in SAFE_TOOLS:
        entry = registry.get(name)
        assert entry is not None, f"工具 {name} 未注册"
        assert entry.isConcurrencySafe is True, (
            f"{name} 应标 isConcurrencySafe=True（只读工具）"
        )


def test_unsafe_tools_marked_correctly():
    """所有 UNSAFE_TOOLS 必须已注册且 isConcurrencySafe=False。"""
    for name in UNSAFE_TOOLS:
        entry = registry.get(name)
        assert entry is not None, f"工具 {name} 未注册"
        assert entry.isConcurrencySafe is False, (
            f"{name} 应标 isConcurrencySafe=False（有副作用）"
        )


def test_all_tools_classified():
    """所有注册的 built-in 工具必须在 SAFE 或 UNSAFE 集合里（防止漏标）。

    注意：MCP 工具（mcp__ 前缀）通过 register_mcp_tools 动态注册，不在本测试范围。
    本测试只覆盖 discover_builtin_tools() 发现的静态注册工具。
    """
    all_names = set(registry.list_all())
    # 排除 MCP 工具（动态注册，不在本测试范围）
    builtin_names = {n for n in all_names if not n.startswith("mcp__")}
    classified = SAFE_TOOLS | UNSAFE_TOOLS
    unclassified = builtin_names - classified
    assert not unclassified, (
        f"以下工具未分类（请加入 SAFE_TOOLS 或 UNSAFE_TOOLS）: {sorted(unclassified)}"
    )


def test_no_overlap_between_safe_and_unsafe():
    """SAFE_TOOLS 和 UNSAFE_TOOLS 不能有交集（防笔误）。"""
    overlap = SAFE_TOOLS & UNSAFE_TOOLS
    assert not overlap, f"以下工具同时出现在两个集合: {overlap}"


def test_safe_subset_consistent_with_plan():
    """Sanity check：SAFE 工具数量符合预期（12 个，纯读类）。

    如果新增了只读工具，记得更新本 expected 值 + SAFE_TOOLS 集合。
    """
    expected_safe_count = 15  # CCAR11 Task 1: glob 加入（只读文件名匹配）
    assert len(SAFE_TOOLS) == expected_safe_count, (
        f"SAFE_TOOLS 数量变了（{len(SAFE_TOOLS)} != {expected_safe_count}），"
        "如果新增了只读工具，更新 expected_safe_count；如果是误删，请补回。"
    )


def test_all_builtin_schemas_use_openai_parameters_key():
    """【CCAR11 防回归】所有内置工具 schema 必须用 "parameters" 键（OpenAI 格式）。

    背景：registry.get_definitions 直接 {"type":"function","function":schema}
    塞给 LLM——参数键必须是 "parameters"（OpenAI 标准）。CCAR8-10 曾有 5 个
    工具误用 Anthropic 风格 "inputSchema"，导致参数定义对 LLM 不可见
    （handler 靠 args.get 能跑，测试全过——silent-dead-code 第 5 例）。
    """
    builtin_names = {
        n for n in registry.list_all() if not n.startswith("mcp__")
    }
    bad = []
    for name in builtin_names:
        entry = registry.get(name)
        if entry is None or not entry.schema:
            continue
        if "parameters" not in entry.schema and "inputSchema" in entry.schema:
            bad.append(name)
    assert not bad, (
        f"以下工具 schema 用了 inputSchema（LLM 看不到参数定义），"
        f"改成 parameters: {sorted(bad)}"
    )
