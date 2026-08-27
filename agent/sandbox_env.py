"""外部命令的"环境变量安检"：跑命令前把密钥类变量从环境里洗掉。

给谁用：terminal 等会启动子进程（subprocess）的工具，在启动前调这里的
build_safe_env 拿一份"干净"的环境变量再传进去。

为什么需要：主进程的环境里有 API key、数据库密码这类敏感信息。而子进程
可能是用户脚本、第三方包、甚至 LLM 自己写的代码——只要它来一句
`print(os.environ)`，所有环境变量就全打到了输出里，接着进入对话历史，
可能被保存或分享出去，密钥就漏了。

做法（借鉴 DeerFlow 项目的 env_policy 思路）——像机场安检逐个检查：
1. 看变量名里是否带密钥关键词（KEY/SECRET/TOKEN/PASS/CREDENTIAL/DSN 等）
2. 命中的直接从环境里删掉，子进程看不见
3. 命令运行必需的良性变量（PATH/HOME/LANG/VIRTUAL_ENV/OMNIMATE_HOME 等）保留
4. 另有显式白名单 allow_keys——技能声明了 required-secrets 时，点名要的
   密钥可以放行
"""
import os
import re
from typing import Dict, Optional, Set

# 密钥关键词正则（不区分大小写，变量名里出现这些词就按密钥嫌疑人处理）
_SECRET_KEYWORDS = re.compile(
    r"(?i)(KEY|SECRET|TOKEN|PASS|PASSWORD|CREDENTIAL|DSN|AUTH|PRIVATE)"
)

# 连接串黑名单：名字里不含关键词、但值明显是敏感信息的变量
# （比如 DATABASE_URL 里装着带密码的数据库地址）
_CONNECTION_STRING_VARS = {
    "DATABASE_URL", "REDIS_URL", "MONGODB_URL",
    "GH_PAT", "MYSQL_PWD", "REDISCLI_AUTH", "PGPASSFILE",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
}

# 永远放行的良性变量（跑命令离不开它们：找可执行文件的 PATH、家目录、
# 编码设置、Python 环境等）
_ALWAYS_KEEP = {
    "PATH", "HOME", "USER", "USERNAME", "LANG", "LC_ALL", "LC_CTYPE",
    "VIRTUAL_ENV", "PYTHONPATH", "PYTHONIOENCODING", "PYTHONHOME",
    "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "APPDATA", "LOCALAPPDATA",
    "COMSPEC", "PATHEXT", "PROCESSOR_ARCHITECTURE", "OS",
    "OMNIMATE_HOME",  # 让子进程能找到 agent home
}
# 提前算好全大写版本的白名单集合——省得 build_safe_env 每检查一个变量
# 都现场重新建一遍集合
_ALWAYS_KEEP_UPPER = {k.upper() for k in _ALWAYS_KEEP}


def _looks_secret(name: str) -> bool:
    """判断一个环境变量的名字像不像密钥。像就返回 True。

    两步判断：名字在连接串黑名单里 → 是；名字里带密钥关键词 → 是；
    都不沾边 → 不是。

    参数：
        name: 环境变量名（如 "OPENAI_API_KEY"）
    """
    upper = name.upper()
    # 第一步：查连接串黑名单
    if upper in _CONNECTION_STRING_VARS:
        return True
    # 第二步：查密钥关键词
    if _SECRET_KEYWORDS.search(name):
        return True
    return False


def build_safe_env(
    inherited: Optional[Dict[str, str]] = None,
    *,
    allow_keys: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """生成一份"洗过"的环境变量字典：密钥被剔除，其余保留。

    子进程不能信任，不能把主进程的密钥原样递过去；但也不能全删——
    把 PATH 之类删了命令根本跑不起来。所以按"白名单放行 + 黑名单剔除 +
    普通变量默认保留"的安检规则筛一遍。

    参数：
        inherited: 出发点环境变量（默认取当前进程的 os.environ）；
            传自定义字典方便测试
        allow_keys: 点名放行的密钥变量名集合——技能声明了 required-secrets
            时用；名单里的变量即使是密钥也原样保留

    返回：
        新字典，不含任何密钥（除非它在 allow_keys 里）。原字典不被修改。

    用法示例：
        # 全量清洗
        safe = build_safe_env()

        # 某技能明确需要读到 OPENAI_API_KEY
        safe = build_safe_env(allow_keys={"OPENAI_API_KEY"})

        # 然后传给子进程
        subprocess.run(cmd, env=safe)
    """
    if inherited is None:
        inherited = dict(os.environ)
    else:
        inherited = dict(inherited)
    allow_keys = {k.upper() for k in (allow_keys or set())}

    safe: Dict[str, str] = {}
    for name, value in inherited.items():
        # 安检第 1 关：白名单良性变量，直接放行
        if name.upper() in _ALWAYS_KEEP_UPPER:
            safe[name] = value
            continue
        # 安检第 2 关：点名放行的（技能声明了 required-secrets），放行
        if name.upper() in allow_keys:
            safe[name] = value
            continue
        # 安检第 3 关：看着像密钥的，没收（不进 safe 就是子进程看不见）
        if _looks_secret(name):
            continue
        # 其余普通变量放行——误删了可能命令就跑不动
        safe[name] = value

    return safe
