"""Poor Mode（穷鬼模式）：一键关闭所有烧钱功能。

关闭清单：
- reflection 引擎（aux_llm 反思）
- 9 段式摘要（降级为简单拼接）
- auto memory 提取
- cache 监控
- verification 子代理
- Curator 维护
- reactive_compact

不关：fatal 底线（rm -rf / 任何模式都拒）、safe_path、autoDeny、基础 LLM 重试。

设计：runtime 翻转 config flag，不写盘，重启恢复默认。
"""
import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


# 7 个开关：dotted key → 目标值（全 False = 全关）
# review 修正：reactive_compact 的真实开关是 features.reactive_compact.enabled
# （agent/__init__.py 用 is_feature_enabled 读）——旧键 context.reactive_compact_enabled
# 全仓无读取点，是 dead write。写 features 键保持 False 语义一致（reactive 默认
# OFF，poor 是"全关"，即使 features 里被用户开过也强制压回 False）。
POOR_PRESET: dict[str, Any] = {
    "reflection.enabled": False,
    "context.summarize_9section": False,
    "memory.auto_extract": False,
    "cache_monitor.enabled": False,
    "verification_agent.enabled": False,
    "curator.enabled": False,
    "features.reactive_compact.enabled": False,
}


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """a.b.c = value → d[a][b][c] = value（中间节点不存在则建空 dict）。"""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_poor_preset(config: dict, on: bool = True) -> dict:
    """返回应用了 poor preset 的新 config（不修改入参）。

    on=True: 把 POOR_PRESET 里的 flag 全设 False
    on=False: 直接返回原 config（用户重启会话恢复默认）
    """
    if not on:
        return config
    new_config = copy.deepcopy(config)
    for dotted_key, value in POOR_PRESET.items():
        _set_dotted(new_config, dotted_key, value)
    logger.info("Poor Mode 已启用，关闭 %d 项功能", len(POOR_PRESET))
    return new_config
