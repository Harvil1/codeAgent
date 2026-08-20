"""功能开关（feature flag）的读取接口。

是什么：功能开关就是配置文件里的「这个新功能开不开」小开关，方便
灰度试跑——出了问题关掉就行，不用改代码。

三条设计原则（出自 spec §5.4）：
- 不破坏分工：config.py 只放数据不放逻辑，判断逻辑集中在本文件
- 只在启动时读一次，运行时不热加载（改动配置会破坏 prompt 缓存，成本翻倍）
- fail-safe：查到不存在的开关名不报错不崩溃，返回 False 并打条 warning 日志

在项目里的位置：被各功能模块调用（如 bash 命令的 LLM 分类器），
判断某个灰度功能当前该不该启用。

用法示例：
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
    """快速判断某个功能开关是否打开。

    背景：新功能先挂开关再上线，运行时到处要问「这个开关开了吗」，
    这个函数就是那个统一的问法。

    参数：
        config —— 配置字典（通常是 RuntimeContext.config 或 agent.config）
        name   —— 开关名（如 "bash_llm_classifier"）

    返回：True 表示开着；False 表示关着/名字不认识/值类型不对。

    Fail-safe 行为（任何异常形态都不崩，一律当「关」处理）：
        - config 不是 dict → False
        - config 里没有 features 这一节 → False
        - 开关名不存在 → False + 打 warning（帮发现 typo 拼错名）
        - 开关值类型错（比如写成字符串 "true"）→ False
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
    # 值类型不对（字符串、数字等）→ 保险起见返回 False
    logger.warning(
        "feature flag %s 类型异常（%s），应为 dict 或 bool，返回 False",
        name, type(flag).__name__,
    )
    return False


def get_feature_config(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """拿到某个开关的完整配置（enabled 之外还带参数的情况用这个）。

    背景：有些开关不只是开/关，还带自己的配置项（如白名单、阈值），
    开关值是一个 dict，这里把整个 dict 取出来。

    参数：
        config —— 配置字典
        name   —— 开关名

    返回：该开关的完整配置字典；开关不存在 / 值不是 dict / 没有
    features 节时，返回空字典（调用方用 .get 带默认值即可安全消费）。
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
