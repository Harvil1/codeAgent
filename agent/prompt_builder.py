"""System prompt 组装器。

关键原则：
1. 会话开始时构建一次，后续缓存（_cached_system_prompt）
2. 必须 byte-stable（同一会话内字节级不变）
3. 记忆是 frozen 快照（本次会话不更新）
4. 技能只列名字和描述，不包含正文

05 升级：拆成 stable/context/volatile 三层，让 prompt cache 命中率最大化。
- stable：跨会话不变（身份、指导）
- context：单会话内不变（记忆、技能、OMNIMATE.md）
- volatile：每轮可变（todo、reminder）
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from agent.skill_commands import parse_frontmatter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 指导文本（注入 system prompt，告诉 LLM 怎么用记忆/技能/搜索）
# ---------------------------------------------------------------------------

MEMORY_GUIDANCE = (
    "## 记忆系统使用指南\n"
    "用 memory 工具保存稳定事实:用户偏好、环境细节、工具怪癖、约定。\n"
    "记忆每轮注入,保持精简——只存一周后仍重要的事实。\n\n"
    "❌ 不存任务进度/会话结果/日志/临时状态、PR/issue 号、commit SHA、"
    "'修了 bug X'、'阶段 N 完成'、文件计数等易过时内容"
    "(这些用 session_search 从历史查)。\n"
    "✅ 写成声明式事实:'项目使用 pytest' 而非 '必须用 pytest 跑'。\n"
    "新方法、解过的问题 → 用 skill_manage 存成技能。\n\n"
    "### 主动学习\n"
    "反复观察到某模式(如连续 3 次用 uv 而非 pip)但不确定是否长期偏好时,"
    "先问用户确认,再调 memory 保存(type=feedback, confidence=1.0)。"
)

SESSION_SEARCH_GUIDANCE = (
    "## 会话搜索指南\n"
    "当用户提到过去的对话内容，或你怀疑有相关的跨会话上下文时，"
    "先用 session_search 查找，再问用户。"
)

SKILLS_GUIDANCE = (
    "## 技能系统指南\n"
    "复杂任务(5+ 工具调用)、棘手错误、非平凡工作流完成后,"
    "用 skill_manage 存成技能复用;发现技能过时/不完整/错误,"
    "立即 skill_manage(action='patch') 修复。\n\n"
    "### 调用技能的 3 条铁律\n"
    "1. **load_skill 后照搬 Quick Start**:按模板改参数执行,不要自己发明命令格式。\n"
    "2. **不确定就 help**:外部 CLI 格式不确定时,先调 `<cli> --help` 一次,别瞎猜。\n"
    "3. **查完就动手**:只读命令(get/query/describe)用 1-2 次后即用写命令"
    "(add/set/update)动手;连续 3+ 次只读还在查 = 分析瘫痪,停下规划或问用户。"
)

TOOL_USAGE_GUIDANCE = (
    "## 工具使用规范\n"
    "- 工具结果都是 JSON 字符串,解析后使用;失败返回 {\"error\":...},据此自我修正\n"
    "- 文件操作必须指定 encoding='utf-8';不要假设工具可用(check_fn 可能隐藏)\n"
    "- **临时文件清理**:write_file 建的临时脚本/中间文件,执行完立即用 terminal 删除\n"
    "- **上下文占位消息识别**:看到 [snip_compact] / micro_compacted / "
    "[紧急上下文压缩] 占位消息且需更早上下文时,按占位消息里的路径"
    "(通常是 .transcripts/latest.jsonl 或 .task_outputs/tool-results/ 下的文件)"
    "用 read_file 读回(在 agent_home 下,默认安全)\n"
    "- **写入白名单**:write_file 默认只允许 cwd 和 ~/.OmniMate;写其他位置会触发审批,"
    "同意后进持久化白名单。不要绕过——用户拒绝就换个白名单内位置写\n"
    "- **自身源码保护**:不要修改 OmniMate 自身源码"
    "(开发=项目根,打包=site-packages 安装目录)。你是工具,用户用你改**他们的项目**,不是改你自己\n"
    "- **依赖安装(跟随 cwd)**:在用户当前目录用其项目环境装包:"
    "`uv pip install <pkg>`(推荐,尊重用户项目约定)或 `pip install <pkg>`;"
    "优先复用已有 .venv/pyproject.toml,不污染系统 Python;"
    "连续 2 次装失败 → 停下问用户\n"
    "- **代码输出(跟随 cwd)**:脚本写到当前目录,输出(PPT/Excel/Word 等)写到 "
    "`<cwd>/outputs/`;用户指定别的位置就照做(白名单自动处理审批)\n"
    "- **先查技能**:任何任务前先扫技能索引,判断有无适用流程技能"
    "(设计先行/调试/写计划等),有就先 load_skill('using-omnimate') 看总纲"
)


TASK_GUIDANCE = (
    "## 任务追踪（Task System）\n"
    "3 步以上的任务必须先调 task_create 创建任务列表，每步完成调 task_complete；"
    "任务之间的依赖用 blocked_by 字段声明；多步并行用 task_list 查看 ready 任务。"
    "Task System 跨会话持久化（~/.OmniMate/.tasks/）。"
)


DELEGATE_GUIDANCE = (
    "## 子代理委托（subagent）\n"
    "大项目探索、多个独立子任务、或单子任务预计 10+ 次工具调用,必须用 subagent "
    "省父 context(可 tasks=[...] 批量并行)。\n"
    "不要委托:强依赖父上下文(如基于已读内容做判断)、短任务(<5 次调用)、顺序型任务。\n"
    "用法:subagent(prompt=..., role='leaf', summary_only=True);"
    "summary_only 自动把结果压缩成 300 字摘要回填。\n"
    "**并行铁律**:需要同时派多个子代理(如并行探索各模块)时,必须用 tasks=[...] "
    "一次调用;禁止发多个独立 subagent——独立调用是串行执行的,逐个等待,不会并行。\n"
    "判断标准:同一领域打算连续 read_file 5+ 次,立刻停,委托子代理——"
    "父 context 留给综合判断。"
)


# ---------------------------------------------------------------------------
# 三层结构（05）
# ---------------------------------------------------------------------------

@dataclass
class SystemPromptLayers:
    """三层系统提示（05）。

    - stable: 跨会话不变（身份、指导、工具文档）
    - context: 单会话内不变（记忆索引、技能索引、OMNIMATE.md）
    - volatile: 每轮可变（todo、reminder、extra_instructions）

    API 厂商 prompt cache 按"前缀哈希"匹配，分层能让 stable 跨会话命中、
    context 单会话命中，整体命中率上去。
    """
    stable: str
    context: str
    volatile: str

    def render_flat(self) -> str:
        """合并成单个字符串（向后兼容老接口）。"""
        parts = [self.stable, self.context, self.volatile]
        return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 主构建函数（05 三层版）
# ---------------------------------------------------------------------------

def build_system_prompt_layers(
    *,
    memory_store=None,
    memory_manager=None,
    enabled_toolsets: Optional[List[str]] = None,
    skills_dir: Optional[Path] = None,
    context_files: Optional[List[Path]] = None,
    extra_instructions: str = "",
    include_guidance: bool = True,
    # volatile 来源（运行时传入）
    task_state: Optional[str] = None,
    reminder: Optional[str] = None,
    # === Task N NEW: 自定义子代理可跳过项目 OMNIMATE.md（省 token）===
    omit_project_memory: bool = False,
) -> SystemPromptLayers:
    """构建三层 system prompt（05）。

    分层动机：
      - stable：跨会话不变（同版本同一台机器，几乎 100% 命中 cache）
      - context：单会话内不变（记忆/技能/OMNIMATE.md，会话内 80%+ 命中）
      - volatile：每轮可变（todo / reminder），不期望 cache 命中

    Task N: omit_project_memory=True 时跳过项目 OMNIMATE.md 注入，
    用于 read-only / 轻量子代理（对齐 Claude Code omitClaudeMd 字段）。
    """
    # ---- stable 层 ----
    stable_parts = []
    if include_guidance:
        stable_parts.extend([
            MEMORY_GUIDANCE, SKILLS_GUIDANCE,
            SESSION_SEARCH_GUIDANCE, TOOL_USAGE_GUIDANCE, TASK_GUIDANCE,
            DELEGATE_GUIDANCE,
        ])
    stable = "\n\n".join(stable_parts)

    # ---- context 层 ----
    context_parts = []
    # 当前工作目录（log.log 案例：恢复历史会话后 LLM 顺着旧项目的路径
    # 模仿填 cwd，跑去探索别的项目。明确注入当前目录 + "以当前为准"）
    try:
        from agent.workspace_context import get_workspace_cwd
        context_parts.append(
            "## 当前工作目录\n"
            f"{get_workspace_cwd()}\n\n"
            "用户在此目录启动了会话。用户说\"这个项目\"时指当前工作目录；"
            "恢复的历史会话如果提到其他项目路径，以当前目录为准"
            "（除非用户明确要求切到别的目录）。"
        )
    except Exception as e:
        logger.debug("当前工作目录注入失败(可忽略): %s", e)
    if skills_dir is None:
        try:
            from constants import all_skills_dirs as _asd
            skills_dir = _asd()  # 内置 + 用户两个目录
        except Exception:
            skills_dir = None
    if skills_dir:
        skill_index = _build_skill_index(skills_dir)
        if skill_index:
            context_parts.append(f"## 可用技能\n{skill_index}")
    if memory_store:
        try:
            index_block = memory_store.snapshot_for_prompt()
            if index_block:
                context_parts.append(f"## 记忆索引\n{index_block}")
        except Exception as e:
            logger.warning("读取记忆索引失败: %s", e)
    if memory_manager:
        try:
            ext_block = memory_manager.build_system_prompt()
            if ext_block:
                context_parts.append(ext_block)
        except Exception:
            pass

    # 用户画像(自动归纳,每 5 次反思后更新)
    try:
        from constants import get_omnimate_home
        profile_path = get_omnimate_home() / "USER_PROFILE.md"
        if profile_path.exists():
            profile_text = profile_path.read_text(encoding="utf-8").strip()
            if profile_text:
                context_parts.append(profile_text)
    except Exception:
        pass

    # MCP routing hints(借鉴 DeerFlow):用户配 .mcp.json 时可加 keywords 字段,
    # 帮助 LLM 看到关键词就知道用哪个 MCP server。如:
    #   "postgres": {"keywords": ["订单", "数据库", "SQL"]}
    try:
        from agent.mcp_client import collect_routing_hints
        hints_block = collect_routing_hints()
        if hints_block:
            context_parts.append(hints_block)
    except Exception as e:
        logger.debug("MCP routing hints 收集失败(可忽略): %s", e)

    # 项目记忆：递归扫 cwd → root 收集 OMNIMATE.md
    # 对齐 Claude Code 的 "recursive CLAUDE.md lookup" 语义
    # Round 1 fix: Path.cwd() 也是进程级（等价 os.getcwd），并发子代理会踩。
    # 改走 get_workspace_cwd()（线程局部 ContextVar）。
    # Task N: omit_project_memory=True 时跳过（自定义子代理 omitClaudeMd=true）
    if not omit_project_memory:
        try:
            from agent.workspace_context import get_workspace_cwd
            scan_root = Path(get_workspace_cwd())
            project_mds = _scan_project_memory_files(scan_root)
            for pmd in project_mds:
                try:
                    content = pmd.read_text(encoding="utf-8")
                    content = _expand_imports(content, pmd.parent)
                    if content.strip():
                        # 显示相对路径，便于调试（绝对路径太长）
                        try:
                            rel = pmd.relative_to(scan_root)
                        except ValueError:
                            rel = pmd
                        context_parts.append(f"## 项目记忆: {rel}\n{content}")
                except Exception as e:
                    logger.warning("读取项目记忆失败 %s: %s", pmd, e)
        except Exception as e:
            logger.debug("项目记忆扫描失败(可忽略): %s", e)

    if context_files:
        for cf in context_files:
            cf = Path(cf)
            if cf.exists():
                try:
                    content = cf.read_text(encoding="utf-8")
                    context_parts.append(f"## 上下文文件: {cf.name}\n{content}")
                except Exception as e:
                    logger.warning("读取上下文文件失败 %s: %s", cf, e)
    context = "\n\n".join(context_parts)

    # ---- volatile 层 ----
    volatile_parts = []
    if task_state:
        volatile_parts.append(f"<current_tasks>{task_state}</current_tasks>")
    if reminder:
        volatile_parts.append(reminder)
    if extra_instructions:
        volatile_parts.append(extra_instructions)
    volatile = "\n\n".join(volatile_parts)

    return SystemPromptLayers(stable=stable, context=context, volatile=volatile)


# ---------------------------------------------------------------------------
# 主构建函数（向后兼容旧接口）
# ---------------------------------------------------------------------------

def build_system_prompt(
    *,
    memory_store=None,
    memory_manager=None,
    enabled_toolsets: Optional[List[str]] = None,
    skills_dir: Optional[Path] = None,
    context_files: Optional[List[Path]] = None,
    extra_instructions: str = "",
    include_guidance: bool = True,
) -> str:
    """组装 system prompt（向后兼容旧接口）。

    内部走 build_system_prompt_layers 然后合并成单个字符串。
    新代码应直接调 build_system_prompt_layers 拿三层。
    """
    layers = build_system_prompt_layers(
        memory_store=memory_store,
        memory_manager=memory_manager,
        enabled_toolsets=enabled_toolsets,
        skills_dir=skills_dir,
        context_files=context_files,
        extra_instructions=extra_instructions,
        include_guidance=include_guidance,
    )
    return layers.render_flat()


def _current_cwd() -> str:
    from agent.workspace_context import get_workspace_cwd
    return get_workspace_cwd()


def _paths_match(paths: list, cwd: str) -> bool:
    """简化版 path glob 匹配：支持前缀目录 + 后缀扩展名 + * 通配。

    基于 brief 的简化版（startswith 语义），并扩展支持绝对路径 cwd：
    当 cwd 是绝对路径（os.getcwd() 返回值）时，检查 base 是否作为路径段出现。
    """
    import fnmatch
    cwd_norm = cwd.replace("\\", "/")
    for pat in paths or []:
        pat = pat.replace("\\", "/")
        # src/** → cwd 在 src/ 下即匹配
        if pat.endswith("/**"):
            base = pat[:-3]
            # brief 原始语义：相对 cwd 前缀匹配（接受 srcfoo 边界，风险 2）
            if cwd_norm.startswith(base):
                return True
            # 绝对 cwd：检查 base 作为路径段出现（/src/ 或末尾 /src）
            if f"/{base}/" in cwd_norm or cwd_norm.endswith(f"/{base}"):
                return True
        # *.py → cwd 下有 .py 文件？简化：cwd 路径段不匹配，但保留技能（保守显示）
        # 用 fnmatch 兜底
        if fnmatch.fnmatch(cwd_norm, f"*/{pat}") or fnmatch.fnmatch(cwd_norm, pat):
            return True
    return False


def _expand_imports(
    content: str,
    base_dir: Path,
    depth: int = 0,
    _visited: Optional[set] = None,
) -> str:
    """展开 OMNIMATE.md 里的 `@path/to/file` 引用（对齐 Claude Code `@import` 语义）。

    规则：
    - `@path/to/file` 相对 base_dir 解析，递归展开（max_depth=5）
    - `@~/foo/bar` 展开 home 目录
    - **跳过 code span 和 code block**（避免 `@anthropic-ai/sdk` 被误判）
    - 文件不存在 / 不是文件 → 原样保留（不抛错）
    - 同一文件多次引用只展开一次（防环）
    """
    import re
    if _visited is None:
        _visited = set()
    if depth > 5:
        logger.warning("@import 递归深度超过 5，跳过剩余展开")
        return content

    pattern = re.compile(r"@(~?[\w./-]+)")

    def _resolve_and_read(path_str: str):
        # `@foo` 单 token 没 / 也没 ~  → 当作 @username，不当 import
        if not path_str.startswith("~") and "/" not in path_str \
                and not path_str.endswith((".md", ".txt", ".rst")):
            return None
        target = Path(path_str).expanduser() if path_str.startswith("~") \
            else (base_dir / path_str).resolve()
        if not target.is_file():
            return None
        key = str(target)
        if key in _visited:
            return f"<!-- @import 已展开过: {path_str} -->"
        _visited.add(key)
        try:
            return target.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("@import 读取失败 %s: %s", target, e)
            return None

    def _expand_text_segment(text: str) -> str:
        """扫描文本段里的 @path（已经被切除了 code span/block）。"""
        def replace(m):
            path_str = m.group(1)
            inner = _resolve_and_read(path_str)
            if inner is None:
                return m.group(0)  # 原样
            # 递归展开引用文件里的 @path
            return _expand_imports(inner, (base_dir / path_str).resolve().parent
                                   if not path_str.startswith("~")
                                   else Path(path_str).expanduser().parent,
                                   depth + 1, _visited)
        return pattern.sub(replace, text)

    # 按行扫描，跳过 fenced code block；inline code span 用正则切分保护
    out_lines = []
    in_fence = False
    fence_marker = None
    inline_code_re = re.compile(r"(`[^`]*`)")

    for line in content.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            out_lines.append(line)
            continue
        if in_fence:
            out_lines.append(line)
            continue
        # 不在 code block：拆出 inline code span 保护，剩余段做 @path 展开
        segments = inline_code_re.split(line)
        processed = [
            seg if (seg.startswith("`") and seg.endswith("`") and len(seg) >= 2)
            else _expand_text_segment(seg)
            for seg in segments
        ]
        out_lines.append("".join(processed))
    return "\n".join(out_lines)


def _scan_project_memory_files(cwd: Path) -> List[Path]:
    """从 cwd 向上扫到磁盘根或 .git 目录，收集所有 OMNIMATE.md。

    返回顺序：**从根到 cwd**（外层先注入，内层覆盖语义）。
    停止规则：遇到含 .git 的目录就停（含该层），不再向上。
    这是 monorepo 友好的设计：在子项目里跑 omnimate 时，
    扫到 monorepo 根（含 .git）就停，不会越过到无关目录。

    对齐 Claude Code "从 cwd 向上递归读 CLAUDE.md" 的语义，
    但品牌用 OMNIMATE.md。
    """
    found: List[Path] = []
    try:
        current = Path(cwd).resolve()
    except Exception:
        return found
    while True:
        omninate_md = current / "OMNIMATE.md"
        if omninate_md.exists():
            found.append(omninate_md)
        # .git 所在层 = 仓库根，含这一层即可，不再向上
        if (current / ".git").exists():
            break
        parent = current.parent
        if parent == current:
            break  # 磁盘根
        current = parent
    found.reverse()  # 从根到 cwd
    return found


def _build_skill_index(skills_dirs) -> str:
    """构建技能索引(名字 + 描述),支持多目录(内置 + 用户)。

    只列出 active 状态的技能,跳过归档的。
    多目录场景:按列表顺序扫描,后者覆盖前者(用户目录优先)。
    """
    import json

    # 兼容单目录输入
    if isinstance(skills_dirs, (str, Path)):
        skills_dirs = [skills_dirs]

    lines = ["使用 /技能名 触发对应技能。"]

    # 收集所有目录的 .usage.json(用户目录的覆盖内置的)
    usage = {}
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        usage_path = skills_dir / ".usage.json"
        if usage_path.exists():
            try:
                usage.update(json.loads(usage_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    # 扫描所有目录的技能,后者覆盖前者
    seen = {}  # name → skill_md 路径
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        if not skills_dir.exists():
            continue
        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            name = skill_md.parent.name
            seen[name] = skill_md  # 后扫的覆盖先扫的

    for name in sorted(seen.keys()):
        skill_md = seen[name]
        rec = usage.get(name, {})

        # 跳过归档的
        if rec.get("state") == "archived":
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, _ = parse_frontmatter(content)
            # disable-model-invocation: true → 不注入索引（模型不能自动触发）
            if frontmatter.get("disable-model-invocation", False) is True:
                continue
            # paths frontmatter：cwd 不匹配则不注入索引
            paths = frontmatter.get("paths")
            if paths:
                if not _paths_match(paths, _current_cwd()):
                    continue
            # 优先用 frontmatter 的 description，否则回退到 _extract_description
            description = frontmatter.get("description", "") or _extract_description(content)
            if description:
                lines.append(f"- /{name}: {description}")
            else:
                lines.append(f"- /{name}")
        except Exception:
            lines.append(f"- /{name}")

    if len(lines) == 1:  # 只有标题行
        return ""

    return "\n".join(lines)


def _extract_description(content: str) -> str:
    """从 SKILL.md 提取 description 字段。"""
    if not content.startswith("---"):
        return ""
    parts = content.split("---", 2)
    if len(parts) < 3:
        return ""
    for line in parts[1].splitlines():
        if line.strip().startswith("description:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return ""
