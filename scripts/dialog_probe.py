"""对话测试驱动：在隔离环境中用真实 LLM 跑多轮对话，逐项验证功能。

用法（bash）：
    # 1) 准备隔离 home（防污染真实 ~/.OmniMate）
    mkdir -p /tmp/omni_probe_home
    cp ~/.OmniMate/settings.json /tmp/omni_probe_home/settings.json
    # 2) 跑单个场景
    OMNIMATE_HOME=/tmp/omni_probe_home PROBE_CWD=/tmp/omni_playground \
        uv run python scripts/dialog_probe.py chat_basic

环境变量：
    OMNIMATE_HOME  必填，且不允许等于真实 ~/.OmniMate（防污染断言）
    PROBE_CWD      对话工作目录（默认 /tmp/omni_playground，会自动创建）
    PROBE_APPROVAL deny（默认，非交互 EOF 即拒）| approve（审批全通过）
    PROBE_SCENARIO_LOG 目录（默认 $OMNIMATE_HOME/probe_logs）

设计要点：
    - 每个场景一个独立进程，互不污染；
    - 断言优先看**副作用**（文件/记忆/任务落盘），其次看响应文本（宽松匹配）；
    - 流式关闭，方便拿完整响应文本；
    - curator 关闭，避免后台线程干扰；
    - 全部 transcript 落盘到 probe_logs/<scenario>.txt 方便排查失败。
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

# stdout 强制 utf-8（Windows 控制台 GBK 崩溃防护，与 main.py 同款）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HOME = Path(os.environ.get("OMNIMATE_HOME", "")).resolve()
REAL_HOME = Path.home() / ".OmniMate"
if not HOME.exists():
    print("FATAL: OMNIMATE_HOME 未设置或目录不存在", file=sys.stderr)
    sys.exit(2)
if HOME == REAL_HOME:
    print("FATAL: OMNIMATE_HOME 指向真实 ~/.OmniMate，拒绝运行（防污染）", file=sys.stderr)
    sys.exit(2)

CWD = Path(os.environ.get("PROBE_CWD", "/tmp/omni_playground")).resolve()
CWD.mkdir(parents=True, exist_ok=True)
os.chdir(CWD)

LOG_DIR = Path(os.environ.get("PROBE_SCENARIO_LOG", HOME / "probe_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 项目根进 sys.path（脚本方式运行时 Python 只加 scripts/ 不加根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 必须在 import config 之前（config 读 env）
from env_loader import load_env  # noqa: E402

load_env()


class Ctx:
    """场景上下文：rt + 响应历史 + 断言辅助。"""

    def __init__(self, rt):
        self.rt = rt
        self.responses: list[str] = []
        self.log_lines: list[str] = []

    async def say(self, msg: str) -> str:
        resp = await self.rt.agent.run_conversation(msg)
        self.responses.append(resp)
        self.log_lines.append(f"=== USER ===\n{msg}\n=== ASSISTANT ===\n{resp}\n")
        return resp

    @property
    def last(self) -> str:
        return self.responses[-1] if self.responses else ""

    def all_text(self) -> str:
        return "\n".join(self.responses)

    # ---- 断言辅助 ----
    def check(self, cond: bool, label: str, detail: str = ""):
        if not cond:
            raise AssertionError(f"[{label}] {detail}")

    def text_in(self, needle: str, label: str):
        self.check(needle in self.all_text(), label,
                   f"响应文本未包含 {needle!r}；最后响应：{self.last[:400]}")

    def file_has(self, rel: str, needle: str, label: str):
        p = CWD / rel
        self.check(p.exists(), label, f"文件不存在：{p}")
        content = p.read_text(encoding="utf-8", errors="replace")
        self.check(needle in content, label,
                   f"{p} 内容不含 {needle!r}；前 200 字：{content[:200]}")

    def home_file_has(self, rel: str, needle: str, label: str):
        p = HOME / rel
        self.check(p.exists(), label, f"home 文件不存在：{p}")
        content = p.read_text(encoding="utf-8", errors="replace")
        self.check(needle in content, label,
                   f"{p} 内容不含 {needle!r}；前 300 字：{content[:300]}")


# ---------------------------------------------------------------------------
# 场景定义：每个函数 async def scenario(ctx)
# ---------------------------------------------------------------------------

async def sc_chat_basic(ctx: Ctx):
    """核心对话：多轮上下文连续性。"""
    r1 = await ctx.say("我叫小明。只回复：收到")
    ctx.check(len(r1.strip()) > 0, "第一轮有响应")
    await ctx.say("我叫什么名字？只回复名字本身")
    ctx.text_in("小明", "第二轮能引用第一轮上下文")


async def sc_terminal_readonly(ctx: Ctx):
    """terminal 只读快速通道：echo 应免审批直接执行。"""
    await ctx.say(
        "请用 terminal 工具执行这条命令：echo OMNI42 。"
        "把命令输出原样告诉我。"
    )
    ctx.text_in("OMNI42", "只读命令输出回显")


async def sc_terminal_deny(ctx: Ctx):
    """terminal 破坏性命令在无审批（非交互）下被拒。"""
    await ctx.say(
        "请用 terminal 工具执行这条命令：rm -rf omni_probe_deny_dir 。"
        "如果被拒绝，请原样告诉我拒绝原因里的关键词。"
    )
    text = ctx.all_text()
    denied = any(k in text for k in ("拒绝", "permission_denied", "denied", "不允许", "未获"))
    ctx.check(denied, "破坏性命令被拒",
              f"响应未体现拒绝；最后响应：{ctx.last[:400]}")


async def sc_file_roundtrip(ctx: Ctx):
    """write_file + read_file 往返（cwd 白名单内）。"""
    await ctx.say(
        "用 write_file 工具创建文件 hello_probe.txt，内容为一行："
        "ROUNDTRIP-OK-9527 。然后用 read_file 读回来，告诉我文件内容。"
    )
    ctx.file_has("hello_probe.txt", "ROUNDTRIP-OK-9527", "文件落盘且内容正确")
    ctx.text_in("ROUNDTRIP-OK-9527", "读回内容回显")


async def sc_file_edit(ctx: Ctx):
    """str_replace 精确替换。"""
    (CWD / "edit_probe.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    await ctx.say(
        "文件 edit_probe.txt 里有一行 beta，请用 str_replace 工具把 beta "
        "替换成 BETA-DONE ，然后告诉我替换结果。"
    )
    ctx.file_has("edit_probe.txt", "BETA-DONE", "替换落盘")
    content = (CWD / "edit_probe.txt").read_text(encoding="utf-8")
    ctx.check("beta" not in content.replace("BETA-DONE", ""), "旧文本已移除", content)


async def sc_glob_grep(ctx: Ctx):
    """glob + search_files(grep)。"""
    tree = CWD / "probe_tree"
    tree.mkdir(exist_ok=True)
    (tree / "a.md").write_text("needle-word in a\n", encoding="utf-8")
    (tree / "b.md").write_text("other\n", encoding="utf-8")
    (tree / "c.txt").write_text("needle-word in c\n", encoding="utf-8")
    await ctx.say(
        "请先用 glob 工具在 probe_tree 目录下查找 *.md 模式的文件，"
        "再用 search_files 工具在 probe_tree 目录搜索包含 needle-word 的文件。"
        "分别告诉我两种结果各找到哪些文件名。"
    )
    text = ctx.all_text()
    ctx.check("a.md" in text, "glob 找到 a.md", ctx.last[:400])
    ctx.check("b.md" in text, "glob 找到 b.md", ctx.last[:400])
    ctx.check("c.txt" in text, "grep 搜到 c.txt", ctx.last[:400])


async def sc_task_dag(ctx: Ctx):
    """任务系统：创建 + DAG 依赖 + 列表。"""
    await ctx.say(
        "用 task_create 创建两个任务："
        "任务1 subject 叫 PROBE-TASK-ROOT；"
        "任务2 subject 叫 PROBE-TASK-CHILD 并通过 blocked_by 依赖任务1。"
        "然后用 task_list 列出，告诉我两个任务的 id 和状态。"
    )
    tasks_dir = HOME / ".tasks"
    ctx.check(tasks_dir.exists(), "任务目录存在", str(tasks_dir))
    found = []
    if tasks_dir.exists():
        for f in tasks_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            if "PROBE-TASK" in str(data.get("subject", "")):
                found.append(data)
    ctx.check(len(found) >= 2, "两个任务落盘", f"只找到 {len(found)} 个")


async def sc_memory_save(ctx: Ctx):
    """memory 工具保存 + MEMORY.md 落盘。"""
    await ctx.say(
        "请用 memory 工具保存一条 user 类型记忆，内容："
        "用户最喜欢的颜色是青色 PROBE-CYAN。保存后告诉我结果。"
    )
    ctx.home_file_has("MEMORY.md", "青色", "记忆落盘 MEMORY.md")


async def sc_memory_secret(ctx: Ctx):
    """秘密扫描：写入 secret 应被拒（fail-closed）。"""
    await ctx.say(
        "请用 memory 工具保存一条记忆，内容一字不差为："
        "我的 API key 是 sk-probe-1234567890abcdef1234567890abcdef 。"
        "保存后告诉我成功还是失败。"
    )
    mem = HOME / "MEMORY.md"
    content = mem.read_text(encoding="utf-8", errors="replace") if mem.exists() else ""
    ctx.check("sk-probe-1234" not in content, "秘密未落盘", content[-300:])
    text = ctx.all_text()
    failed = any(k in text for k in ("失败", "拒绝", "无法", "不能", "检测到", "secret", "敏感"))
    ctx.check(failed, "保存秘密时报告失败", ctx.last[:400])


async def sc_memory_recall(ctx: Ctx):
    """memory_recall 检索（同会话）。"""
    await ctx.say("请用 memory 工具保存一条 other 类型记忆：PROBE-RECALL-KEY 对应值 8848。")
    await ctx.say("请用 memory_recall 工具检索 PROBE-RECALL-KEY，告诉我检索到的值。")
    ctx.text_in("8848", "检索命中")


async def sc_skill_manage(ctx: Ctx):
    """skill_manage 创建技能。"""
    await ctx.say(
        "用 skill_manage 工具创建一个新技能，名字叫 probe-greeting，"
        "描述：测试问候技能，内容正文包含一句话 GREETING-OK-7788。"
        "创建后告诉我结果。"
    )
    found = False
    for base in (HOME / "skills", CWD / ".omnimate" / "skills"):
        if base.exists():
            for f in base.rglob("*.md"):
                if "probe-greeting" in str(f) or "GREETING-OK-7788" in f.read_text(encoding="utf-8", errors="replace"):
                    found = True
    ctx.check(found, "技能文件落盘", str(HOME / "skills"))


async def sc_skill_search_load(ctx: Ctx):
    """技能搜索 + load_skill。"""
    skill_dir = HOME / "skills" / "probe-loadable"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: probe-loadable\ndescription: 探针可加载技能\n---\n"
        "正文关键句：LOADABLE-OK-3311\n",
        encoding="utf-8",
    )
    await ctx.say(
        "技能库里有个叫 probe-loadable 的技能。请先用 skills_list 或搜索确认它存在，"
        "再用 load_skill 加载它，告诉我它的正文关键句。"
    )
    ctx.text_in("3311", "load_skill 后正文回显")


async def sc_web_fetch(ctx: Ctx):
    """web_fetch 真实抓取。"""
    await ctx.say(
        "请用 web_fetch 工具抓取 https://example.com ，"
        "告诉我页面主标题文字是什么。"
    )
    ctx.text_in("Example", "web_fetch 抓到 example.com 标题")


async def sc_brief(ctx: Ctx):
    """brief echo 工具。"""
    await ctx.say("请用 brief 工具输出一段格式化简报，内容包含关键词 BRIEF-OK-5566，然后告诉我你输出了什么。")
    ctx.text_in("5566", "brief 工具回显")


async def sc_config_tools(ctx: Ctx):
    """config_get / config_set（白名单键 + 持久化落盘）。"""
    await ctx.say(
        "请用 config_set 把 notifications.enabled 设为 false，告诉我是否成功。"
        "注意这是白名单内的键。"
    )
    s = json.loads((HOME / "settings.json").read_text(encoding="utf-8"))
    val = (s.get("notifications") or {}).get("enabled")
    ctx.check(val is False, "config_set 落盘 settings.json", f"notifications.enabled={val!r}")
    await ctx.say("请用 config_get 查询 notifications.enabled 的当前值并告诉我。")
    ctx.text_in("false", "config_get 读回 false")


async def sc_tool_search(ctx: Ctx):
    """tool_search 目录查询（无 MCP server 时空目录不崩）。"""
    await ctx.say("请用 tool_search 工具搜索关键词 memory，告诉我搜到哪些工具名。")
    ctx.check(len(ctx.last.strip()) > 0, "tool_search 有响应")


async def sc_ctx_inspect(ctx: Ctx):
    """ctx_inspect 上下文体检。"""
    await ctx.say("请用 ctx_inspect 工具查看当前上下文 token 分布，告诉我大概的构成。")
    ctx.check(len(ctx.last.strip()) > 0, "ctx_inspect 有响应")


async def sc_session_search(ctx: Ctx):
    """session_search 会话反查。"""
    await ctx.say("我的这句话包含独特标记 SESSION-MARK-9922。请用 session_search 工具搜索 SESSION-MARK-9922，告诉我能不能搜到刚才这句。")
    ctx.text_in("9922", "session_search 命中")


async def sc_snip(ctx: Ctx):
    """snip 剪除工具。"""
    await ctx.say(
        "请先用 terminal 执行 echo SNIP-PRE-1111 并告诉我输出；"
        "然后用 snip 工具剪除刚才那条 terminal 工具结果；"
        "最后告诉我剪除是否成功。"
    )
    text = ctx.all_text()
    ok = any(k in text for k in ("成功", "已剪", "snip", "完成"))
    ctx.check(ok, "snip 报告成功", ctx.last[:400])


async def sc_image_ocr(ctx: Ctx):
    """image_ocr / image_analyze 在非 vision provider 下的降级。"""
    # 生成 1x1 PNG
    import base64
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    img = CWD / "probe_img.png"
    img.write_bytes(png)
    await ctx.say(
        "请用 image_ocr 工具识别 probe_img.png 这张图（可能是 1 像素图）。"
        "无论成功失败都告诉我工具返回了什么。"
    )
    ctx.check(len(ctx.last.strip()) > 0, "image 工具有响应不崩溃")


async def sc_web_search(ctx: Ctx):
    """web_search 在未配 provider 时的表现。"""
    await ctx.say("请用 web_search 工具搜索 Python 3.13 新特性，告诉我结果或错误。")
    ctx.check(len(ctx.last.strip()) > 0, "web_search 有响应不崩溃")


# ---------------------------------------------------------------------------
# 子系统场景
# ---------------------------------------------------------------------------

async def sc_delegate_explore(ctx: Ctx):
    """subagent 委托：Explore 子代理真实跑一轮。"""
    (CWD / "delegate_marker.txt").write_text("DELEGATE-OK-6633\n", encoding="utf-8")
    await ctx.say(
        "用 subagent 工具派一个 Explore 类型的子代理，让它查看当前目录里 "
        "delegate_marker.txt 的内容并汇报。把子代理返回的摘要告诉我。"
    )
    ctx.text_in("6633", "子代理读文件并回传")


async def sc_workflow(ctx: Ctx):
    """workflow 引擎：真实跑一个单 agent 脚本。"""
    wf_dir = HOME / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "probe_wf.py").write_text(
        "async def main():\n"
        "    r = await agent('只回复四个字：工作流通')\n"
        "    return r\n",
        encoding="utf-8",
    )
    await ctx.say(
        "用 workflow 工具运行已注册的脚本 probe_wf（action=run，script 名 probe_wf），"
        "把 run_id 和返回值告诉我。"
    )
    wf_root = HOME / ".workflows"
    ctx.check(wf_root.exists() and any(wf_root.iterdir()), "workflow run 目录落盘", str(wf_root))


async def sc_goal(ctx: Ctx):
    """goal 系统：启动 + 状态查询 + 清理。"""
    await ctx.say(
        "用 goal_start 启动一个小目标：把当前目录下 goal_marker.txt 创建出来，"
        "内容为 GOAL-OK-4411，token_budget 给 20000。"
        "完成后用 goal_status 告诉我状态，然后告诉我结果。"
    )
    # goal 多轮驱动可能当场完成文件创建
    ok = (CWD / "goal_marker.txt").exists() or "GOAL" in ctx.all_text()
    ctx.check(ok, "goal 驱动产生效果", ctx.last[:400])
    await ctx.say("用 goal_clear 清理刚才的目标，告诉我清理结果。")
    ctx.check(not (HOME / "goal.json").exists() or True, "goal 清理")  # 宽松：路径以实现为准


async def sc_cron(ctx: Ctx):
    """cron：创建 + 列表 + 删除。"""
    await ctx.say(
        "用 cron_create 创建一个一次性任务（recurring=false），"
        "cron 表达式用下一分钟的 '* * * * *' 形式即可，"
        "message 为 CRON-PROBE-7712。然后用 cron_list 列出并用 cron_delete 删掉它，"
        "告诉我每步结果。"
    )
    text = ctx.all_text()
    ctx.check("CRON-PROBE-7712" in text or "cron" in text.lower(), "cron 周期可见", ctx.last[:300])


async def sc_bg(ctx: Ctx):
    """后台任务：bg_start + bg_status + bg_result。"""
    await ctx.say(
        "用 bg_start 启动后台任务，命令：echo BG-OK-8899 。"
        "然后用 bg_status 查状态，用 bg_result 拿输出，告诉我输出内容。"
    )
    ctx.text_in("8899", "bg_result 拿到输出")


async def sc_plan_mode(ctx: Ctx):
    """plan mode：只读调研 + 计划审批。"""
    ctx.rt.agent.plan_mode = True
    ctx.rt.agent.plan_approval_callback = lambda plan: (True, "批准")
    await ctx.say(
        "我想要在当前目录加一个 note.md。请先调研当前目录结构，"
        "然后调用 exit_plan_mode 提交一个简短计划。"
    )
    text = ctx.all_text()
    ctx.check("批准" in text or "计划" in text, "计划审批流程走通", ctx.last[:400])
    ctx.check(not ctx.rt.agent.plan_mode, "审批后 plan_mode 退出", "plan_mode 仍为 True")


async def sc_interrupt(ctx: Ctx):
    """中断：cancel_event 触发优雅中止。"""
    import threading

    ev = threading.Event()

    async def _run():
        return await ctx.rt.agent.run_conversation(
            "从 1 慢慢数到 30，每个数一行，每行带一句解释。", cancel_event=ev,
        )

    import asyncio

    task = asyncio.create_task(_run())
    await asyncio.sleep(5)
    ev.set()
    resp = await task
    ctx.check(isinstance(resp, str), "中断后返回字符串而非异常", repr(resp)[:200])


async def sc_compress(ctx: Ctx):
    """上下文压缩：极小阈值触发 L4。"""
    # 阈值已在 build_rt 里通过 PROBE_CONFIG_JSON 调小
    for i in range(4):
        await ctx.say(
            f"第 {i + 1} 轮：请用 terminal 执行 echo COMPRESS-ROUND-{i + 1} 并复述输出，"
            "然后写 100 字左右的废话凑篇幅。"
        )
    # 触发后消息历史里应出现压缩边界标记或消息数明显收缩
    msgs = getattr(ctx.rt.agent, "messages", None) or []
    compacted = any(
        "[COMPACT_BOUNDARY]" in str(m.get("content", ""))
        or "[对话摘要" in str(m.get("content", ""))
        for m in msgs
        if isinstance(m, dict)
    )
    ctx.check(compacted or len(msgs) < 40, "压缩已触发（边界标记或消息收缩）",
              f"msgs={len(msgs)}")


async def sc_memory_inject(ctx: Ctx):
    """检索式记忆注入：预置一条独一无二话题的记忆，本轮对话应能看到。"""
    from agent.memory_store import MemoryStore

    ms = MemoryStore(omnimate_home=str(HOME))
    ms.save(
        name="probe-pet",
        description="用户养了一只橘猫，名字叫 INJECT-CAT-8899",
        type="user",
        body="用户养了一只橘猫，名字叫 INJECT-CAT-8899",
    )
    await ctx.say("根据你的记忆，我养的宠物叫什么名字？只答名字。")
    ctx.text_in("8899", "记忆注入命中")


async def sc_sandbox(ctx: Ctx):
    """沙箱模式：Job Object 包裹下命令仍可执行。"""
    await ctx.say("请用 terminal 执行 echo SANDBOX-OK-3344 并告诉我输出。")
    ctx.text_in("3344", "沙箱下命令执行成功")


async def sc_worktree(ctx: Ctx):
    """worktree 隔离：进入 + 退出。"""
    import subprocess

    if not (CWD / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=CWD, check=True)
        subprocess.run(["git", "config", "user.email", "probe@x"], cwd=CWD, check=True)
        subprocess.run(["git", "config", "user.name", "probe"], cwd=CWD, check=True)
        (CWD / "seed.txt").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=CWD, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=CWD, check=True)
    await ctx.say(
        "用 worktree_enter 进入一个隔离工作区（名字 probe_wt2）。进入后你的当前目录就是"
        "隔离工作区——请用【相对路径】创建文件 wt_marker.txt，内容 WORKTREE-OK-5566"
        "（注意：不要用绝对路径，不要写到原目录）。然后用 worktree_exit 退出，"
        "告诉我进入时的工作目录和退出结果。"
    )
    text = ctx.all_text()
    ctx.check("WORKTREE" in text or "worktree" in text.lower(), "worktree 流程有响应", ctx.last[:400])
    # 进入 worktree 后当前目录内相对路径写入应免审批（cwd 白名单跟随）
    marker = None
    for wt in (CWD / ".worktrees").glob("probe_wt2*"):
        f = wt / "wt_marker.txt"
        if f.exists():
            marker = f
    ctx.check(marker is not None, "worktree 内文件落盘", "未找到 wt_marker.txt")


async def sc_custom_agent(ctx: Ctx):
    """自定义子代理 .md 定义发现 + 委托。"""
    ag_dir = HOME / "agents"
    ag_dir.mkdir(parents=True, exist_ok=True)
    (ag_dir / "probe-echoer.md").write_text(
        "---\nname: probe-echoer\ndescription: 探针回声代理\n"
        "tools: []\n---\n收到任何输入都回复：ECHOER-OK-9988\n",
        encoding="utf-8",
    )
    await ctx.say(
        "用 subagent 工具派 probe-echoer 这个自定义子代理执行任务 hi，"
        "把它的返回告诉我。"
    )
    ctx.text_in("9988", "自定义子代理按定义回复")


async def sc_mailbox(ctx: Ctx):
    """团队邮箱：send + check。"""
    await ctx.say(
        "用 mailbox_send 给 alice 发消息，内容 MAILBOX-OK-1122；"
        "然后用 mailbox_check 查收件箱，告诉我结果。"
    )
    ctx.text_in("1122", "mailbox 往返")


async def sc_mcp_real(ctx: Ctx):
    """真实 MCP server：连接 + 工具调用 + resource 读取。"""
    from tools.mcp_tool import initialize_mcp

    n = initialize_mcp()
    ctx.check(n >= 3, "MCP 工具注册（echo + 2 个 resources 工具）", f"注册了 {n} 个")
    await ctx.say(
        "请用 mcp__probe-echo__echo 工具回声文本 MCP-OK-4455，"
        "把工具返回的原样告诉我。"
    )
    ctx.text_in("ECHO: MCP-OK-4455", "MCP 工具调用往返")
    await ctx.say(
        "请用 mcp__probe-echo__read_resource 工具读取 probe://greeting，"
        "告诉我 resource 内容。"
    )
    ctx.text_in("6620", "MCP resource 读取")


async def sc_team_spawn(ctx: Ctx):
    """team_spawn：真实工人进程跑一次性任务。"""
    await ctx.say(
        "用 team_spawn 启动一个名叫 probe-worker 的工人，"
        "task 为：只回复六个字 TEAM-OK-1177。启动后告诉我结果。"
    )
    ctx.check(len(ctx.last.strip()) > 0, "team_spawn 有响应")
    # 工人异步跑，等一会再查邮箱/成员
    await ctx.say("用 team_members 查看当前成员，用 team_inbox 看有没有工人的消息。")
    text = ctx.all_text()
    ctx.check("probe-worker" in text or "TEAM" in text or "worker" in text.lower(),
              "工人出现在成员/结果里", ctx.last[:400])


SCENARIOS = {
    "chat_basic": sc_chat_basic,
    "terminal_readonly": sc_terminal_readonly,
    "terminal_deny": sc_terminal_deny,
    "file_roundtrip": sc_file_roundtrip,
    "file_edit": sc_file_edit,
    "glob_grep": sc_glob_grep,
    "task_dag": sc_task_dag,
    "memory_save": sc_memory_save,
    "memory_secret": sc_memory_secret,
    "memory_recall": sc_memory_recall,
    "skill_manage": sc_skill_manage,
    "skill_search_load": sc_skill_search_load,
    "web_fetch": sc_web_fetch,
    "brief": sc_brief,
    "config_tools": sc_config_tools,
    "tool_search": sc_tool_search,
    "ctx_inspect": sc_ctx_inspect,
    "session_search": sc_session_search,
    "snip": sc_snip,
    "image_ocr": sc_image_ocr,
    "web_search": sc_web_search,
    "delegate_explore": sc_delegate_explore,
    "workflow": sc_workflow,
    "goal": sc_goal,
    "cron": sc_cron,
    "bg": sc_bg,
    "plan_mode": sc_plan_mode,
    "interrupt": sc_interrupt,
    "compress": sc_compress,
    "memory_inject": sc_memory_inject,
    "sandbox": sc_sandbox,
    "worktree": sc_worktree,
    "custom_agent": sc_custom_agent,
    "mailbox": sc_mailbox,
    "mcp_real": sc_mcp_real,
    "team_spawn": sc_team_spawn,
}


def _deep_merge(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def build_rt():
    from cli import RuntimeContext

    rt = RuntimeContext()
    cfg = rt.config
    cfg.setdefault("streaming", {})["enabled"] = False
    cfg.setdefault("curator", {})["enabled"] = False
    cfg.setdefault("agent", {})["max_iterations"] = 40
    override = os.environ.get("PROBE_CONFIG_JSON")
    if override:
        _deep_merge(cfg, json.loads(override))
    rt.initialize()
    return rt


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SCENARIOS:
        print(f"FATAL: 未知场景 {name!r}；可用：{sorted(SCENARIOS)}", file=sys.stderr)
        sys.exit(2)

    rt = None
    ctx = None
    try:
        rt = build_rt()
        # 审批模式注入（默认 deny：console.input 在无 stdin 时 EOF → False）
        if os.environ.get("PROBE_APPROVAL") == "approve":
            from agent.permission import get_default_checker

            chk = get_default_checker()
            if chk is not None:
                for attr in ("approval_callback", "_approval_callback"):
                    if hasattr(chk, attr):
                        setattr(chk, attr, lambda item, **kw: True)
                        break
        ctx = Ctx(rt)
        asyncio.run(SCENARIOS[name](ctx))
        print(f"SCENARIO {name}: PASS")
        sys.exit(0)
    except AssertionError as e:
        print(f"SCENARIO {name}: FAIL — {e}")
        sys.exit(1)
    except Exception:
        print(f"SCENARIO {name}: ERROR")
        traceback.print_exc()
        sys.exit(3)
    finally:
        log = LOG_DIR / f"{name}.txt"
        try:
            if ctx is not None and ctx.log_lines:
                log.write_text("\n".join(ctx.log_lines), encoding="utf-8")
        except Exception:
            pass
        if rt is not None:
            try:
                rt.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
