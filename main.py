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
try:
    from tools.mcp_tool import initialize_mcp
    initialize_mcp()
except Exception as e:
    import logging
    logging.getLogger(__name__).debug("MCP 初始化失败（可忽略）: %s", e)


def main():
    """主入口。"""
    args = sys.argv[1:]

    # 非交互模式：python main.py chat "你好"
    if args and args[0] == "chat":
        from cli import run_one_shot
        run_one_shot(" ".join(args[1:]))
        return

    # 交互模式：检查 -c / --continue 标志
    resume_last = "-c" in args or "--continue" in args
    from cli import run_interactive
    run_interactive(resume_last=resume_last)


if __name__ == "__main__":
    main()
