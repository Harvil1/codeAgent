"""沙箱环境变量策略:执行外部命令前洗掉密钥类环境变量。

防止 agent 调 subprocess 时把 API key / 数据库密码等泄漏给子进程。
子进程可能是用户脚本、第三方包、或 LLM 写的代码——任何 `print(os.environ)`
都会把所有环境变量打到 stdout,然后进 messages 历史,可能被保存/分享。

策略(借鉴 DeerFlow env_policy):
1. 检查变量名是否含密钥关键词(KEY/SECRET/TOKEN/PASS/CREDENTIAL/DSN 等)
2. 命中则从环境变量里删除
3. 保留良性变量(PATH/HOME/LANG/VIRTUAL_ENV/AGENT_HOME 等)
4. 支持显式 allow_keys 白名单(技能声明的 required-secrets 用)
"""
import os
import re
from typing import Dict, Optional, Set

# 密钥关键词正则(大小写不敏感)
_SECRET_KEYWORDS = re.compile(
    r"(?i)(KEY|SECRET|TOKEN|PASS|PASSWORD|CREDENTIAL|DSN|AUTH|PRIVATE)"
)

# 连接串黑名单(变量名不在密钥关键词里但含敏感信息的)
_CONNECTION_STRING_VARS = {
    "DATABASE_URL", "REDIS_URL", "MONGODB_URL",
    "GH_PAT", "MYSQL_PWD", "REDISCLI_AUTH", "PGPASSFILE",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
}

# 永远保留的良性变量(命令执行需要的)
_ALWAYS_KEEP = {
    "PATH", "HOME", "USER", "USERNAME", "LANG", "LC_ALL", "LC_CTYPE",
    "VIRTUAL_ENV", "PYTHONPATH", "PYTHONIOENCODING", "PYTHONHOME",
    "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "APPDATA", "LOCALAPPDATA",
    "COMSPEC", "PATHEXT", "PROCESSOR_ARCHITECTURE", "OS",
    "AGENT_HOME",  # 让子进程能找到 agent home
}


def _looks_secret(name: str) -> bool:
    """判断环境变量名是否是密钥类。"""
    upper = name.upper()
    # 在连接串黑名单里
    if upper in _CONNECTION_STRING_VARS:
        return True
    # 命中密钥关键词
    if _SECRET_KEYWORDS.search(name):
        return True
    return False


def build_safe_env(
    inherited: Optional[Dict[str, str]] = None,
    *,
    allow_keys: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """构造安全的环境变量 dict(洗掉密钥)。

    参数:
        inherited: 继承的环境变量(默认 os.environ)
        allow_keys: 显式允许的密钥变量名白名单(技能 required-secrets)
                    这些变量即使是密钥也保留

    返回:
        新的 dict,不含密钥(除非在 allow_keys 里)

    用法:
        # 完全清理
        safe = build_safe_env()

        # 技能要求 OPENAI_API_KEY 可见
        safe = build_safe_env(allow_keys={"OPENAI_API_KEY"})

        # 然后传给 subprocess
        subprocess.run(cmd, env=safe)
    """
    if inherited is None:
        inherited = dict(os.environ)
    else:
        inherited = dict(inherited)
    allow_keys = {k.upper() for k in (allow_keys or set())}

    safe: Dict[str, str] = {}
    for name, value in inherited.items():
        # 永远保留的良性变量
        if name.upper() in {k.upper() for k in _ALWAYS_KEEP}:
            safe[name] = value
            continue
        # 显式 allow 的(技能声明 required-secrets)
        if name.upper() in allow_keys:
            safe[name] = value
            continue
        # 密钥类删除
        if _looks_secret(name):
            continue
        # 其他普通变量保留(可能命令需要)
        safe[name] = value

    return safe
