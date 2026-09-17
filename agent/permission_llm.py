"""LLM 分类辅助（从 permission.py 簇 E 纯搬迁而来）。

大白话：权限闸门 4 要请"副 LLM"（一个帮忙干杂活的便宜小模型）当裁判，
看一条 shell 命令安不安全。这个文件装的就是裁判的全部家当：提问的
台词模板（提示词）、什么时候不再信裁判（拒绝回落阈值：连续误拒 3 次
或累计 20 次就停用闸门 4）、哪些命令长得危险不能走快速通道（危险
前缀表——python/eval/curl 这类能执行任意代码的入口）、白名单怎么
比对，以及真正打电话问裁判的函数（_classify_bash_command）。

为什么单独一个文件：permission.py 是全项目安全地基，太重了；这一簇
只依赖 json/logging/typing，不依赖 permission 任何东西——本模块不
反向 import permission（叶子模块，谁都可靠它，它不靠别人）。
副 LLM 实例不在这边拿——_classify_bash_command 的 aux_llm_router 是
参数不是全局，由 PermissionChecker._check_llm_classifier（R4-T5 热区，
留在类内一字未动）读配置后注入；自然语言规则（nl_rules）也是类内
读 settings 后借 router 上的临时属性 nl_rules_cache 传进来的。

外部契约：permission 主文件 re-export 类内消费的五个符号
（_classify_bash_command、LLM_DENIAL_MAX_CONSECUTIVE/LLM_DENIAL_MAX_TOTAL、
_is_dangerous_whitelist_entry、_matches_whitelist），闸门 4 的调用点零感知。
"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 闸门 4:aux_llm 分类器(让副 LLM 判断命令安全性;feature flag 控制开关,
# 默认关)
# ---------------------------------------------------------------------------
#
# 什么时候触发:前三道闸门都没拒绝、也没要求审批(也就是命令快落到
# check() 末尾"默认通过"了),且 feature flag
# bash_llm_classifier.enabled=True(配置里开了)。
# 白名单快速通道(spec 第 4 节决策 4):命令匹配白名单前缀(如 "ls"、
# "git status")就直接放行,0 次 LLM 调用。白名单从
# config.features.bash_llm_classifier.whitelist 读。
# AI 判断:safe → 放行;unsafe → 拒绝并给原因。容错 fail-open:
# LLM 调用失败就放行(记 log warning,别把用户卡死)。
#
# 注:aux_llm_router.chat_completions 是 async 的;
# 而 PermissionChecker.check 是同步函数 → 交给进程级常驻循环宿主
# loop_host.run_async 桥接(协程跑在常驻循环上,aux client
# 绑定稳定不漂移,不再每条命令现搭一个临时循环)。

# 分类用的提示词模板(三向):要求 LLM 输出严格的 JSON,方便解析。
# {nl_rules} 由 _check_llm_classifier 读 settings.json 的
# permissions.nl_rules 后注入(通过 aux_llm_router.nl_rules_cache 这个属性
# 传递——因为分类函数这边拿不到 config)。
_CLASSIFY_PROMPT_TEMPLATE = """你是命令安全分类器。判断下面这条 shell 命令该放行、拒绝还是交人工审批。

用户自然语言权限规则（优先级最高，逐条对照）：
{nl_rules}

分类标准：
- allow：只读或明显无破坏性（查状态/跑测试/编译）
- deny：明确破坏性（删数据/覆盖系统文件/提权）
- ask：拿不准、有副作用但意图不明、规则未覆盖

只输出 JSON：{{"verdict": "allow"|"deny"|"ask", "reason": "一句话理由", "confidence": 0.0-1.0}}
confidence < 0.7 时请直接给 ask。

命令：{cmd}
"""

# ---------------------------------------------------------------------------
# LLM 分类器增强(拒绝回落 + 危险前缀剥离)
# ---------------------------------------------------------------------------

# 拒绝回落阈值:连续 3 次或累计 20 次判 unsafe
# → 本会话停用闸门 4。为什么:分类器明显和用户意图不合拍了,与其反复
# 误拒,不如回落人工审批。
LLM_DENIAL_MAX_CONSECUTIVE = 3
LLM_DENIAL_MAX_TOTAL = 20

# 危险前缀表(跨平台代码执行入口 + Bash 扩展):白名单
# 不许给"任意代码执行入口"开快速通道——`Bash(python:*)` 这类 allow/白名单
# 条目会让分类器形同虚设(解释器、包 runner、eval、env、xargs 都能执行
# 任意代码)。白名单里的这类条目只在闸门 4 内部被忽略(不改用户的 config,
# 只是不让它们享受免 LLM 的快速通道)。
_DANGEROUS_CLASSIFIER_PREFIXES = frozenset({
    # 解释器
    "python", "python3", "python2", "node", "deno", "tsx", "ruby", "perl",
    "php", "lua",
    # 包 runner
    "npx", "bunx", "npm run", "yarn run", "pnpm run", "bun run",
    # shell（Git Bash/WSL 跨平台可达）
    "bash", "sh", "zsh", "fish",
    # 远程任意命令包装
    "ssh",
    # 任意代码执行/提权原语
    "eval", "exec", "env", "xargs", "sudo",
    # 网络外泄面
    "curl", "wget",
})


def _is_dangerous_whitelist_entry(entry: str) -> bool:
    """判断一个白名单条目是否命中危险前缀(即"能执行任意代码"的入口)。

    参数:
        entry: 白名单条目字符串(如 "python" / "python -c" / "npm run")。

    返回:True 表示危险(不给开快速通道)。按四种形态比对:和表里精确
    相等;x:* 旧前缀形态(取前段比对);首词命中(python -c、npx pkg);
    前两个词的组合命中(npm run build——"npm run" 本身是表里的独立条目)。
    空条目/纯空白也算危险(这种本来就不该进白名单)。
    """
    if not entry or not entry.strip():
        return True  # 空条目/纯空白条目本身就不该进白名单
    e = entry.strip().lower()
    if e in _DANGEROUS_CLASSIFIER_PREFIXES:
        return True
    # x:* 旧前缀形态 → 去掉尾部的 :* 取前段再比对
    if e.endswith(":*"):
        e = e[:-2].strip()
        if e in _DANGEROUS_CLASSIFIER_PREFIXES:
            return True
    words = e.split()
    if not words:
        return True
    # 首词比对(python -c / npx pkg 这类)
    if words[0] in _DANGEROUS_CLASSIFIER_PREFIXES:
        return True
    # 前两词组合比对(npm run build / npm run:*——"npm run" 在表里是独立条目)
    if len(words) >= 2 and f"{words[0]} {words[1]}" in _DANGEROUS_CLASSIFIER_PREFIXES:
        return True
    return False


def _matches_whitelist(command: str, whitelist: List[str]) -> bool:
    """检查命令是否匹配白名单前缀(命中就走快速通道,0 次 LLM 调用)。

    参数:
        command: 待检查的命令。
        whitelist: 白名单条目列表(已经剥掉危险前缀后的)。

    返回:True 表示命中。规则(spec 第 4 节决策 4):
    - 命令去掉前导空白后,要么和白名单条目一模一样,要么以"条目 + 空格"开头;
    - 例:白名单 "ls" 匹配 "ls -la"、"ls /tmp";不匹配 "ls; rm -rf /"
      (含复合操作符的不算);
    - 含 shell 复合操作符(&&/||/;/|/反引号/$())的命令一律不匹配
      (保守,交给 LLM 判)。
    """
    if not command or not whitelist:
        return False
    cmd = command.strip()
    # 含复合操作符的命令不走快速通道(后半段可能藏着危险子命令)
    if any(op in cmd for op in ("&&", "||", ";", "|", "`", "$(")):
        return False
    for entry in whitelist:
        if not entry:
            continue
        # 精确匹配(命令和白名单条目一模一样)
        if cmd == entry:
            return True
        # 前缀 + 空格("ls -la" 命中白名单 "ls" 加一个空格)
        if cmd.startswith(entry) and len(cmd) > len(entry) and cmd[len(entry)].isspace():
            return True
    return False


def _parse_classify_response(text: str) -> Dict[str, Any]:
    """解析 LLM 分类器的回答(三向 + 置信度门控)。

    参数:
        text: LLM 返回的原始文本。

    返回:{"verdict": "allow"/"deny"/"ask"}(可能带 "reason"),解析不了
    返回 {"error": ...}。规则:
    - 兼容旧的 safe 字段(true→allow / false→deny);
    - confidence < 0.7 一律改成 ask(LLM 自己都没把握就回落问人);
    - 不是合法 JSON / 缺 verdict / verdict 值不认识 → ask 或 error(保守)。
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        parsed = json.loads(t)
    except json.JSONDecodeError:
        return {"error": f"输出非 JSON: {t[:80]}"}
    if not isinstance(parsed, dict):
        return {"error": "输出非对象"}
    # 兼容两态格式(safe: true/false)
    if "verdict" not in parsed and "safe" in parsed:
        parsed["verdict"] = "allow" if parsed["safe"] else "deny"
    verdict = str(parsed.get("verdict", "")).lower()
    try:
        conf = float(parsed.get("confidence", 1.0))
    except (TypeError, ValueError):
        conf = 1.0
    if verdict not in ("allow", "deny", "ask"):
        verdict = "ask"
    if verdict in ("allow", "deny") and conf < 0.7:
        verdict = "ask"
    out = {"verdict": verdict}
    if parsed.get("reason"):
        out["reason"] = str(parsed["reason"])
    return out


async def _classify_bash_command(command: str, aux_llm_router: Any) -> Dict[str, Any]:
    """调副 LLM(aux_llm)判断一条命令安不安全。

    为什么需要:前三道闸门只看"长相"(正则/模式),有些命令长相普通但
    语义危险(或反之),让 LLM 从意图层面再判一次。

    参数:
        command: 要判断的 shell 命令。
        aux_llm_router: AuxLLMRouter 实例(必须带 async 的 chat_completions 方法)。

    返回:字典,四种可能——
    - {"verdict": "allow"} 判放行;
    - {"verdict": "deny", "reason": "..."} 判拒绝;
    - {"verdict": "ask", "reason": "..."} 拿不准(升人工审批);
    - {"error": "..."} 调用失败(fail-open,调用方放行)。

    LLM 没输出合法 JSON 也按 {"error": ...} 返回,调用方 fail-open 放行。
    """
    if aux_llm_router is None:
        # 没有副 LLM → 不分类(check() 那边会 fail-open 放行)
        return {"error": "aux_llm_router 未注入"}

    rules: list = []
    # 自然语言规则由调用方(_check_llm_classifier)读 config 后注入——这里从
    # router 侧拿不到 config,所以借 aux_llm_router 上的一个临时属性传递
    rules = list(getattr(aux_llm_router, "nl_rules_cache", []) or [])
    prompt = _CLASSIFY_PROMPT_TEMPLATE.format(
        cmd=command,
        nl_rules="\n".join(f"- {r}" for r in rules) or "（无）",
    )
    try:
        resp = await aux_llm_router.chat_completions(
            [{"role": "user", "content": prompt}],
        )
    except Exception as e:
        # 调用本身就抛异常 → fail-open 放行
        logger.warning("bash_llm_classifier: aux_llm 调用异常（fail-open 放行）: %s", e)
        return {"error": f"aux_llm 调用异常: {e}"}

    # 从响应里取正文:resp.choices[0].message.content
    try:
        text = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as e:
        logger.warning("bash_llm_classifier: aux_llm 响应格式异常（fail-open）: %s", e)
        return {"error": f"响应格式异常: {e}"}

    return _parse_classify_response(text)
