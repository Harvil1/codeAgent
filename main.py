"""程序总入口：整个 OmniMate 命令行工具从 `python main.py` 这里启动。

这个文件是"点火器"，只做四件事：
1. 把 Windows 控制台输出切成 UTF-8（防止中文/emoji 乱码崩溃）；
2. 提前建好 agent home 目录（默认 ~/.OmniMate）、加载 .env、初始化配置；
3. 尝试连接 MCP 外部工具（失败只警告不阻断启动）；
4. 把剩下的活全交给 cli.main（参数解析和分发都在那边）。

用法：
    python main.py                 # 交互模式（启动时提示恢复历史）
    python main.py -c              # 自动恢复最近会话（continue）
    python main.py --continue      # 同上
    python main.py chat <msg>      # 非交互模式（一次性问答）
"""

import sys

# Windows 控制台默认用 GBK 编码，遇到 emoji 或特殊字符（如 \u26a0 警告符）
# 会直接 UnicodeEncodeError 崩溃。所以启动第一件事就是把输出流强制切成 UTF-8，
# 并且遇到编不出的字符用 ? 顶替而不是抛异常。
# 输入流同理：用管道/重定向喂数时编码跟系统走（GBK），UTF-8 中文会被解码成乱码
# （例如 /handoff save "中文标题" 存盘就是乱码）。
# 注意只重配"非终端"的 stdin——真终端（PEP 528 的 WindowsConsoleIO）本来就是
# unicode 的，动了反而可能出问题。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # 某些环境（输出被重定向/捕获）不支持 reconfigure，切不了就算了

from constants import get_omnimate_home, skills_dir, logs_dir

# 启动前先把 agent home（~/.OmniMate）的目录骨架建好，后面代码好往里写东西
get_omnimate_home().mkdir(parents=True, exist_ok=True)
skills_dir().mkdir(parents=True, exist_ok=True)
logs_dir().mkdir(parents=True, exist_ok=True)

# 初始化配置：确保 settings.json 存在（第一次运行会自动生成/迁移旧配置）
from agent.settings import ensure_default_settings
ensure_default_settings()

# 下面是 MCP 外部工具的初始化部分（只在配置了 .mcp.json 时才有动作）
def _mcp_server_approval(name: str, desc: str) -> bool:
    """项目级 .mcp.json 的首次连接审批。

    背景：项目目录里的 .mcp.json 可能指定任意外部 server，直接连有安全风险，
    所以第一次连接前要问用户一句"允许吗"。拿不到用户输入（非交互环境，
    比如管道喂数）就一律拒绝——宁可不用，不可乱连（fail-closed）。

    参数：
        name：MCP server 的名字（配置里写的键名）
        desc：这个 server 的描述信息（给用户看的）

    返回：
        bool —— True 表示用户允许连接，False 表示拒绝（含非交互环境）。
    """
    import sys
    if not sys.stdin.isatty():
        return False
    try:
        from rich.prompt import Confirm
        print(f"[MCP 审批] 当前项目的 .mcp.json 请求连接 server {name!r}: {desc}")
        return Confirm.ask("允许连接该 MCP server？", default=False)
    except Exception:
        return False


def _init_mcp_safely():
    """带兜底的 MCP 初始化包装函数：失败了也要让用户在终端看得见。

    背景：MCP 连接失败如果只写一条 debug 日志，用户
    根本不知道为什么 mcp__ 开头的工具全消失了，所以改成终端上直接警告。
    注意失败不阻断启动——MCP 只是可选扩展，主功能不依赖它。
    """
    try:
        from tools.mcp_tool import initialize_mcp
        initialize_mcp(approval_callback=_mcp_server_approval)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("MCP 初始化失败（可忽略）: %s", e)
        # 用 print 而不是 logger，保证用户终端一定能看到这行警告
        # （main.py 顶部已把 stdout 强制成 utf-8，不怕乱码）
        print(f"⚠️  MCP 初始化失败（可忽略）: {e}", file=sys.stderr)


_init_mcp_safely()


def main():
    """主入口函数：把启动流程收尾，实际分发交给 cli 模块。

    背景与分工：参数解析和"该进交互模式还是一次性问答"
    的判断在 cli.main，这里只剩"转手调用"。main.py 本身只负责
    三件事：输出编码设置、MCP 初始化、调用 cli.main。

    支持的调用形式（向后兼容）：
        python main.py                         # 交互模式
        python main.py -c / --continue         # 自动恢复最近会话
        python main.py chat <msg>              # 非交互一次性问答
        python main.py --agents '{json}'       # 从命令行注入子代理定义
        python main.py --agents '{json}' chat <msg>

    一个设计细节：asyncio.run（启动异步事件循环的开关）不在本层调用，而是
    放在 cli 里的 run_interactive / run_one_shot 内部、紧贴真正要用异步的
    run_conversation 调用点。这样 cli.main 保持纯同步，避免出现"事件循环
    里再套事件循环"的嵌套问题。
    """
    # 参数解析 + 分发逻辑在 cli.main，这里只转手
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
