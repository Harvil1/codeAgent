"""bash 命令的结构化解析器——把命令拆成语法树，供权限系统用。

大白话：判断一条命令危不危险，最土的办法是用正则去切字符串。但正则有个致命
盲区——它不认引号。比如 ``echo "a && rm -rf /"`` 里的 && 只是引号里的一段
文字，正却会当成"两条命令的分界"误切两段；反过来 ``echo hi && rm -rf build``
的后半段危险命令，整串匹配的黑名单规则又逮不到。语法树（AST，把命令按
bash 的语法规则拆成一棵树）能分清"真正的命令分隔符"和"参数里的文本"，
谁在引号里、谁是要执行的命令，一目了然。

本项目只在权限检查（permission.py / tool_permissions.py）里用它，是唯一的
解析入口。

容错铁律（fail-open，出错就当没看见）：bashlex 这个库没装、
或者某条命令解析失败 → parse_info 返回 None → 调用方退回正则判断，
绝不会因为引入它而多拒绝命令。

方向铁律（只收紧不放宽）：AST 的结果只用来"抓坏"——deny/ask 规则逐段
命中即命中、只读判定仍要求动词在既有白名单表里；从不用来"放行"
（allow 规则维持整串匹配语义）。
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def parse_info(command: str) -> Optional[dict]:
    """把一条 bash 命令解析成结构化信息。

    干什么：调用 bashlex 把命令拆成语法树，再遍历树提取权限检查要用的三样东西。

    为什么需要：正则分不清引号里的文字和真正的命令分隔符（见模块头说明），
    只有语法树能可靠地"逐段"检查命令。

    参数：
        command: 要解析的 bash 命令字符串。

    返回：一个字典，包含三个键——
        - "segments"：所有"简单命令段"（即每个真正要执行的命令，如
          `echo hi && rm x` 里的 `echo hi` 和 `rm x`）各自拆成的词列表，按出现顺序排；
        - "has_redirect"：命令里有没有重定向（> >>，会写文件）；
        - "has_substitution"：有没有命令替换/进程替换（$() 或 <()，实际执行
          结果会被拼进命令，属于注入面）。
    解析失败、bashlex 没装、命令为空时返回 None（调用方退回正则判断）。
    """
    cmd = (command or "").strip()
    if not cmd:
        return None
    # bash 把 \r 也当命令分隔符，但 bashlex 不认——
    # 必须先归一成 \n，否则 "ls \r rm xxx" 会被当成一条命令，
    # 危险的后半段就溜进只读通道了
    cmd = cmd.replace("\r\n", "\n").replace("\r", "\n")
    try:
        import bashlex
    except ImportError:
        logger.warning("bashlex 未安装，AST 解析跳过（fail-open）")
        return None
    try:
        nodes = bashlex.parse(cmd)
    except Exception as e:
        logger.warning("bashlex 解析失败（fail-open 回落正则）: %r → %s", cmd[:80], e)
        return None

    segments: List[List[str]] = []
    has_redirect = False
    has_substitution = False

    def _walk(node):
        nonlocal has_redirect, has_substitution
        kind = getattr(node, "kind", "")
        if kind == "redirect":
            has_redirect = True
        # 实测 bashlex 的进程替换节点 kind 叫 "processsubstitution"；"procsubstitute" 是保险起见留的别名
        if kind in ("commandsubstitution", "procsubstitute", "processsubstitution"):
            has_substitution = True
        if kind == "command":
            toks = []
            # 实测注意：普通词节点的文本在 .word 属性（没有 .text），
            # 变量赋值节点的文本才在 .text——所以两个属性都试
            for part in getattr(node, "parts", []):
                if getattr(part, "kind", "") == "redirect":
                    has_redirect = True  # 重定向节点是 command 的直接子节点，先记下再继续
                text = getattr(part, "word", None) or getattr(part, "text", "")
                if text:
                    toks.append(text)
            if toks:
                segments.append(toks)
        # 这里不能扫到 command 节点就提前 return——
        # word 子节点的更深层还嵌着命令替换/进程替换/重定向节点，必须
        # 把整棵树都走完，否则 `echo $(rm -rf /)` 里的替换体漏检。
        # 替换体里的嵌套 command 节点也进 segments，逐段 deny/只读判定能逮到它。
        for child in getattr(node, "parts", []):
            _walk(child)
        # 实测注意：命令替换/进程替换节点的子命令挂在 .command 属性（不在
        # .parts 里），得单独递归一次才能钻进替换体内部
        inner = getattr(node, "command", None)
        if inner is not None:
            _walk(inner)

    try:
        for n in nodes:
            _walk(n)
    except Exception as e:
        logger.warning("bashlex AST 遍历异常（fail-open）: %s", e)
        return None

    if not segments:
        return None  # 解析成功但一条命令段都没有（比如纯注释）→ 退回正则判断
    return {
        "segments": segments,
        "has_redirect": has_redirect,
        "has_substitution": has_substitution,
    }
