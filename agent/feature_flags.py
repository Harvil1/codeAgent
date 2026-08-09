"""Feature flags 读取 API。

设计原则（spec §5.4）：
- 不破坏现有 config.py 只放数据、不放逻辑的分工
- 启动时加载一次（保护 prompt cache），运行时不热加载
- fail-safe：未知 flag 返回 False + log warning，不崩

使用方式：
    from agent.feature_flags import is_feature_enabled, get_feature_config

    if is_feature_enabled(config, "bash_llm_classifier"):
        # 调 aux_llm 做兜底分类
        ...

    cfg = get_feature_config(config, "bash_llm_classifier")
    whitelist = cfg.get("whitelist", [])
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def is_feature_enabled(config: Dict[str, Any], name: str) -> bool:
    """快速判断 feature flag 是否开启。

    Args:
        config: 配置字典（通常是 RuntimeContext.config 或 agent.config）
        name: flag 名（如 "bash_llm_classifier"）

    Returns:
        True 如果 flag 开启；False 如果关闭/未知/类型错。

    Fail-safe 行为：
        - config 不是 dict → False
        - config 没有 features 节 → False
        - flag 名不存在 → False + log warning（防 typo）
        - flag 值类型错（如字符串 "true"）→ False
    """
    if not isinstance(config, dict):
        return False

    features = config.get("features")
    if not isinstance(features, dict):
        return False

    flag = features.get(name)
    if flag is None:
        logger.warning("未知 feature flag: %s（已忽略，返回 False）", name)
        return False

    if isinstance(flag, dict):
        return bool(flag.get("enabled", False))
    if isinstance(flag, bool):
        return flag
    # 类型错（字符串、数字等）→ fail-safe 返回 False
    logger.warning(
        "feature flag %s 类型异常（%s），应为 dict 或 bool，返回 False",
        name, type(flag).__name__,
    )
    return False


def get_feature_config(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """拿到该 flag 的完整配置（含 enabled 之外的参数）。

    Args:
        config: 配置字典
        name: flag 名

    Returns:
        flag 的完整配置字典；未知 flag / 非 dict 值 / 缺 features 节 → 返回空字典。
    """
    if not isinstance(config, dict):
        return {}

    features = config.get("features")
    if not isinstance(features, dict):
        return {}

    flag = features.get(name)
    if isinstance(flag, dict):
        return flag
    return {}
