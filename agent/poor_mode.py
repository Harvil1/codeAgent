"""Poor Mode（省钱模式）：一键关掉所有"额外花 LLM token"的功能。

平时跑的很多功能（反思、记忆提取……）都要额外调 LLM，每次调用都花钱。
用户想省 token 时打开这个模式，烧钱开关一次性全关。

关闭清单（都是"锦上添花"型功能，关了不影响核心对话）：
- reflection 引擎（事后复盘总结经验的辅助 LLM 调用）
- auto memory 提取（自动从对话里挖记忆）

注意：这里只登记真正有运行时读取点的开关。曾经的 9 段摘要 / cache
监控 / verification / curator / reactive_compact 五个键在对应子系统里
已经没有读取点（写了等于没关），等哪天真接上读取点再登记回来。

绝不关的安全底线：fatal 拦截（rm -rf / 任何模式下都拒绝）、路径白名单
（safe_path）、异步子代理自动拒绝审批（autoDeny）、基础 LLM 重试。

实现方式：只改内存里这份配置（runtime 翻转 flag），不写回配置文件——
重启会话就自动恢复默认，不留永久痕迹。

在项目里的位置：给 cli.py 的 /poor 命令用，改完的 config 传给主循环。
"""
import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


# 开关清单：点号分隔的配置键 → 要设成的值（全是 False = 全关）。
# 两条坑记一下：
# - memory.auto_extract 必须写嵌套的 enabled：读取方期望该节点是 dict
#   （.get("enabled")），整节点设成 False 会让它每轮 AttributeError
# - reflection.enabled 的读取点是 AIAgent 构造时的快照（_reflection_enabled），
#   光改内存配置关不住已经活着的 agent，cli 的 /poor 会顺手翻快照属性
POOR_PRESET: dict[str, Any] = {
    "reflection.enabled": False,
    "memory.auto_extract.enabled": False,
}


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """把 "a.b.c" 这种点号键写进嵌套字典，相当于 d[a][b][c] = value。

    点号键"展开"成一层层的字典赋值；中间某一层不存在时先建一个
    空字典再往下走。

    参数：
        d: 要写入的目标字典（直接在它上面改）
        dotted_key: 点号分隔的键，如 "reflection.enabled"
        value: 要设置的值
    """
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_poor_preset(config: dict, on: bool = True) -> dict:
    """生成一份"省钱模式"的新配置并返回。

    深拷贝一份再改（不直接动调用方手里的配置字典，别处可能还在用），
    原配置原封不动。

    参数：
        config: 当前完整配置字典
        on: True = 开启省钱模式（把 POOR_PRESET 里的开关全设 False）；
            False = 关闭（直接原样返回，等用户重启会话自然恢复默认）

    返回：应用了预设的新配置字典（on=False 时就是原 config 本身）。
    """
    if not on:
        return config
    new_config = copy.deepcopy(config)
    for dotted_key, value in POOR_PRESET.items():
        _set_dotted(new_config, dotted_key, value)
    logger.info("Poor Mode 已启用，关闭 %d 项功能", len(POOR_PRESET))
    return new_config
