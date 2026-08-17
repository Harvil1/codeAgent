"""bashlex AST wrapper（R27 #21）——bash 命令的结构化解析，权限系统专用接触点。

为什么需要：正则切分不认引号——``echo "a && rm -rf /"`` 会被 && 误切成两段；
``echo hi && rm -rf build`` 的 deny 规则整串匹配不上后半段。AST 能分清
"命令分隔符"和"参数里的文本"。

渐进路线（fail-open 铁律）：bashlex 缺失 / 解析失败 → parse_info 返回
None → 调用方（permission / tool_permissions）回落现状正则行为，
不引入新拒绝面。

方向铁律：AST 结果只用于「收紧」（deny/ask 逐段命中）与「只读正判」
（动词仍须在既有白名单表内），不用于放宽（allow 保持整串匹配）。
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def parse_info(command: str) -> Optional[dict]:
    """解析命令 → {"segments": [[token...]], "has_redirect": bool, "has_substitution": bool}。

    segments：所有简单命令段（kind == 'command' 节点）的 token 列表，
    按出现序。token 来自节点 parts 的 word/assignment 文本。
    解析失败 / 缺依赖 / 空命令 → None。
    """
    cmd = (command or "").strip()
    if not cmd:
        return None
    try:
        import bashlex
    except ImportError:
        logger.debug("bashlex 未安装，AST 解析跳过（fail-open）")
        return None
    try:
        nodes = bashlex.parse(cmd)
    except Exception as e:
        logger.debug("bashlex 解析失败（fail-open 回落正则）: %r → %s", cmd[:80], e)
        return None

    segments: List[List[str]] = []
    has_redirect = False
    has_substitution = False

    def _walk(node):
        nonlocal has_redirect, has_substitution
        kind = getattr(node, "kind", "")
        if kind == "redirect":
            has_redirect = True
        # bashlex 实测 kind 是 "processsubstitution"；"procsubstitute" 兜底别名保留
        if kind in ("commandsubstitution", "procsubstitute", "processsubstitution"):
            has_substitution = True
        if kind == "command":
            toks = []
            # spike 实测：word 节点文本在 .word 属性（.text 不存在），assignment 在 .text
            for part in getattr(node, "parts", []):
                if getattr(part, "kind", "") == "redirect":
                    has_redirect = True  # redirect 是 command 的直接 part，叶子短路前先记
                text = getattr(part, "word", None) or getattr(part, "text", "")
                if text:
                    toks.append(text)
            if toks:
                segments.append(toks)
        # R27 复审 Critical 修复：command 不再提前 return——word 子节点的 parts
        # 里嵌着 commandsubstitution/procsubstitute/redirect，必须全树递归扫到，
        # 否则 `echo $(rm -rf /)` 的替换体漏检 → has_substitution 恒 False。
        # 语义变化（收紧，方向正确）：替换体内的嵌套 command 节点也进 segments，
        # 逐段 deny/只读判定能命中替换体本身。
        for child in getattr(node, "parts", []):
            _walk(child)
        # bashlex 实测：commandsubstitution/processsubstitution 的子命令挂在
        # .command 属性（不在 .parts），必须单独递归才能扫到替换体内部
        inner = getattr(node, "command", None)
        if inner is not None:
            _walk(inner)

    try:
        for n in nodes:
            _walk(n)
    except Exception as e:
        logger.debug("bashlex AST 遍历异常（fail-open）: %s", e)
        return None

    if not segments:
        return None  # 解析成功但没有命令段（纯注释等）→ 交回正则路径
    return {
        "segments": segments,
        "has_redirect": has_redirect,
        "has_substitution": has_substitution,
    }
