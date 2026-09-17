"""常量和路径函数——整个项目最底层的模块，谁都不依赖、谁都用到它。

集中提供各种数据文件的路径（agent home、技能目录、记忆文件、会话
数据库等），让全项目"问一个地方"就能拿到统一路径。

所有数据（API key、记忆、工具、技能、会话、任务）统一放在 ~/.codeAgent/ 下：
  - Linux/macOS: ~/.codeAgent/
  - Windows:     C:\\Users\\<user>\\.codeAgent\\

支持 CODEAGENT_HOME 环境变量覆盖默认位置（测试和多配置隔离用）。
"""

import os
from pathlib import Path

# 应用版本号（启动横幅用；跟 pyproject.toml 的 [project].version 保持同步）
APP_VERSION = "0.1.0"


def _default_codeagent_home() -> Path:
    """算出 agent home（数据总目录）的位置。

    优先级：
      1. CODEAGENT_HOME 环境变量（测试/开发时用来切一个临时目录）
      2. 都没设就用 ~/.codeAgent/

    返回：
        Path 对象，指向 agent home 目录。
    """
    env_override = os.environ.get("CODEAGENT_HOME")
    if env_override:
        return Path(env_override).expanduser()
    return Path.home() / ".codeAgent"


def _default_logs_dir() -> Path:
    """日志目录（统一放 agent home 下，出问题好找）。

    返回：
        Path 对象，指向 <agent home>/logs。
    """
    return _default_codeagent_home() / "logs"


def get_codeagent_home() -> Path:
    """拿到 agent home 目录（全项目数据的根目录）。

    每次调用都重新读 CODEAGENT_HOME 环境变量（测试可随时切换目录）。

    返回：
        Path 对象，指向 agent home 目录。
    """
    return _default_codeagent_home()


def display_codeagent_home() -> str:
    """拿到适合展示给用户看的 home 路径字符串。

    返回：
        路径的字符串形式（用于界面显示/日志）。
    """
    return str(get_codeagent_home())


def skills_dir() -> Path:
    """用户技能库根目录（每个子目录是一个技能）。

    用户自己创建、agent 自动沉淀的技能都放这里。
    换机器时需要跟着用户数据一起 rsync 搬走。

    返回：
        Path 对象，指向 <agent home>/skills。
    """
    return get_codeagent_home() / "skills"


def builtin_skills_dir() -> Path:
    """内置技能目录（项目代码自带，跟着 git 走）。

    与用户技能分开存；两边有同名技能时用户目录优先（可覆盖内置的）。

    返回：
        Path 对象，指向项目根下的 skills/。
    """
    # 项目根 = constants.py 的父目录（本文件就放在项目根）
    return Path(__file__).resolve().parent / "skills"


def project_root() -> Path:
    """项目根目录（constants.py 所在目录），用于 agent 自身代码的写保护判定。

    返回：
        Path 对象，指向项目根。
    """
    return Path(__file__).resolve().parent


def all_skills_dirs() -> list:
    """列出所有要扫描的技能目录（内置 + 用户 + 已启用插件）。

    返回 [内置, 用户, 插件1/skills, ...]，排在后面的同名技能覆盖前面的；
    插件通过各自 plugin.json 的 enabled 字段控制是否参与。

    返回：
        Path 列表，顺序即优先级（低→高）。
    """
    dirs = [builtin_skills_dir(), skills_dir()]
    # 再把已启用插件的 skills 目录追加进来
    plugins_root = plugins_dir()
    if plugins_root.exists():
        import json as _json
        for plugin_dir in sorted(plugins_root.iterdir()):
            if not plugin_dir.is_dir():
                continue
            # 清单认两个位置：根下 plugin.json（本项目约定）或
            # .claude-plugin/plugin.json（官方 claude code 插件布局，
            # 从官方市场装的插件长这样）
            manifest = plugin_dir / "plugin.json"
            if not manifest.exists():
                manifest = plugin_dir / ".claude-plugin" / "plugin.json"
            if not manifest.exists():
                continue
            try:
                data = _json.loads(manifest.read_text(encoding="utf-8"))
                if data.get("enabled", True):  # manifest 没写 enabled 就默认启用
                    skills = plugin_dir / "skills"
                    if skills.is_dir():
                        dirs.append(skills)
            except Exception:
                continue
    return dirs


def plugins_dir() -> Path:
    """插件目录（每个子目录是一个插件：plugin.json + skills/）。

    插件的结构长这样：
        ~/.codeAgent/plugins/<name>/plugin.json   （清单：名称/版本/描述/是否启用）
        ~/.codeAgent/plugins/<name>/skills/<skill>/SKILL.md
    程序启动时由 all_skills_dirs() 扫描已启用插件的 skills/ 子目录。

    返回：
        Path 对象，指向 <agent home>/plugins。
    """
    return get_codeagent_home() / "plugins"


def logs_dir() -> Path:
    """日志目录（统一在 ~/.codeAgent/logs 下）。

    返回：
        Path 对象，指向日志目录。
    """
    return _default_logs_dir()


def archive_dir() -> Path:
    """归档目录——curator 把不用的技能挪到这里，只搬家不删除（完全可逆）。

    返回：
        Path 对象，指向 <agent home>/skills/.archive。
    """
    return get_codeagent_home() / "skills" / ".archive"


def memory_file() -> Path:
    """MEMORY.md 的路径（agent 的笔记本：环境事实、项目约定）。

    返回：
        Path 对象，指向 <agent home>/MEMORY.md。
    """
    return get_codeagent_home() / "MEMORY.md"


def config_path() -> Path:
    """config.yaml 的路径（主要作为迁移源使用）。

    返回：
        Path 对象，指向 <agent home>/config.yaml。
    """
    return get_codeagent_home() / "config.yaml"


def sessions_db_path() -> Path:
    """会话数据库文件的路径。

    返回：
        Path 对象，指向 <agent home>/sessions.db。
    """
    return get_codeagent_home() / "sessions.db"


def session_dir() -> Path:
    """会话级临时数据目录（放各会话自己的 env 文件等）。

    每个会话一个专属 env 文件（路径经 CODEAGENT_ENV_FILE 传给 hook）：
    SessionStart hook 追加 `export K=V` 行，terminal 执行命令时把内容
    合并进子进程的环境变量。

    返回：
        Path 对象，指向 <agent home>/.session。
    """
    return get_codeagent_home() / ".session"


def session_env_file(session_id: str) -> Path:
    """某个会话专属的 env 文件路径（即 CODEAGENT_ENV_FILE 指向的文件）。

    参数：
        session_id：会话 ID。为空用 "default"；里面的非法字符
            （字母数字和 -_ 之外的）会被清掉——防止有人拿
            "../../" 之类的会话 ID 把文件写到目录外面去。

    返回：
        Path 对象，指向 <agent home>/.session/<安全化的会话ID>.env。
    """
    sid = session_id or "default"
    # 清理非法字符（防路径穿越攻击）
    safe_sid = "".join(c for c in sid if c.isalnum() or c in "-_")
    if not safe_sid:
        safe_sid = "default"
    return session_dir() / f"{safe_sid}.env"
