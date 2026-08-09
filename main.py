"""启动入口。

用法：
    python main.py                 # 交互模式（启动时提示恢复历史）
    python main.py -c              # 自动恢复最近会话（continue）
    python main.py --continue      # 同上
    python main.py chat <msg>      # 非交互模式（一次性问答）
"""

import sys

# Windows 控制台默认 GBK，遇 emoji/特殊字符（\u26a0 等）会 UnicodeEncodeError 崩。
# 启动时强制 stdout/stderr 为 utf-8 + errors='replace'，保证任意 unicode 都能输出（不可编码字符替换为 ?）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # 某些环境（重定向/捕获）不支持 reconfigure，忽略

from constants import get_omnimate_home, skills_dir, logs_dir

# 启动前确保 agent home 目录结构存在
get_omnimate_home().mkdir(parents=True, exist_ok=True)
skills_dir().mkdir(parents=True, exist_ok=True)
logs_dir().mkdir(parents=True, exist_ok=True)

# 加载 .env（必须在 import config 之前，因为 config 可能读 env）
from env_loader import load_env
load_env()

# 初始化配置（settings.json，含自动迁移）
from agent.settings import ensure_default_settings
ensure_default_settings()

# 初始化 MCP（如果有 .mcp.json 配置）
def _init_mcp_safely():
    """Bug #1 fix: MCP 初始化包裹函数，失败时用户终端可见。

    之前只 debug log，用户看不到 mcp__ 工具为何消失。
    现在 console 警告（不阻断启动，MCP 是可选扩展）。
    """
    try:
        from tools.mcp_tool import initialize_mcp
        initialize_mcp()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("MCP 初始化失败（可忽略）: %s", e)
        # 用户终端可见（用 print，main.py 顶部 stdout 已强制 utf-8）
        print(f"⚠️  MCP 初始化失败（可忽略）: {e}", file=sys.stderr)


_init_mcp_safely()


def main():
    """主入口。

    支持的调用形式（向后兼容）：
        python main.py                         # 交互模式
        python main.py -c / --continue         # 自动恢复最近会话
        python main.py chat <msg>              # 非交互一次性问答
        python main.py --agents '{json}'       # CLI 注入子代理（阶段 6 NEW）
        python main.py --agents '{json}' chat <msg>

    Task E1: 参数解析 + 分发逻辑已抽到 cli.main。
    main.py 只负责 stdout 编码 + MCP 初始化 + 调 cli.main。
    asyncio.run 包装发生在 run_interactive / run_one_shot 内部（紧贴 async
    run_conversation 调用点），cli.main 本身保持同步（避免嵌套 asyncio.run）。
    """
    # Task E1: 把参数解析 + 分发改到 cli.main
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
