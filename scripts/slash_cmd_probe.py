# -*- coding: utf-8 -*-
"""slash 命令全量探针：隔离环境里逐个跑注册表里的命令，抓崩溃/挂起。

大白话：像质检员给 42 个命令挨个通电——不通电的真 bug 当场抓现行，
要人陪的（交互式选择器）标记 INTERACTIVE 不算错。

用法（自动建临时 agent home + 临时工作区，跑完即删，不碰真实数据）：
    uv run python scripts/slash_cmd_probe.py            # 全量
    uv run python scripts/slash_cmd_probe.py /stats /usage  # 只跑指定的

隔离手段：
    CODEAGENT_HOME → 临时目录（settings.json 从真 home 拷一份，凭证能用、
    落盘全进沙盒）；工作目录切到临时 workspace（/init /diff 这类写 cwd 的
    命令不污染项目）。
交互替身：cli_ui.input / console.input 一律回空串，编辑器打开改成 no-op，
question 选择器返回首项——命令的"无人值守路径"能不能跑通也是质量。
"""
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path

# 项目根钉进 sys.path（后面要 chdir 去沙盒，cwd 查找会失效）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 先搭隔离环境（必须在 import cli 系列之前设好 CODEAGENT_HOME）──
_REAL_HOME = Path(os.environ.get("CODEAGENT_HOME", Path.home() / ".codeAgent"))
_TMP = Path(tempfile.mkdtemp(prefix="slash_probe_"))
_TMP_HOME = _TMP / "home"
_TMP_WS = _TMP / "ws"
_TMP_HOME.mkdir(parents=True, exist_ok=True)
_TMP_WS.mkdir(parents=True, exist_ok=True)
os.environ["CODEAGENT_HOME"] = str(_TMP_HOME)
os.chdir(_TMP_WS)

# 凭证沿用真 home 的 settings.json（没 key 的话 initialize 会 SystemExit，
# 这里提前复制一份；拷不上就裸跑，让报告里能看到真实失败原因）
try:
    _real_settings = _REAL_HOME / "settings.json"
    if _real_settings.exists():
        (_TMP_HOME / "settings.json").write_text(
            _real_settings.read_text(encoding="utf-8"), encoding="utf-8",
        )
except Exception as e:
    print(f"[probe] 拷贝 settings.json 失败（继续裸跑）: {e}")

# 无头：布局构建相关命令别真起全屏界面
os.environ["CODEAGENT_LAYOUT_HEADLESS"] = "1"

results = []  # (name, status, detail)


def run_one(entry, rt):
    """单命令跑在带看门狗的线程里：OK / EXCEPTION / TIMEOUT(交互挂起)。"""
    box = {}

    def _work():
        try:
            entry.handler("", rt)
            box["status"] = "OK"
        except SystemExit:
            box["status"] = "OK"          # /quit 这类用标志位/退出请求，正常
        except Exception:
            box["status"] = "EXCEPTION"
            box["detail"] = traceback.format_exc(limit=8)

    t = threading.Thread(target=_work, daemon=True, name=f"probe-{entry.name}")
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        return "TIMEOUT", "交互式挂起（等待用户输入/全屏界面）"
    return box.get("status", "?"), box.get("detail", "")


def main():
    only = [a for a in sys.argv[1:] if a.startswith("/")]
    args_mode = "--args" in sys.argv

    # 交互替身：没人按键时当"用户直接回车"
    import cli_ui
    cli_ui.input = lambda *a, **k: ""
    cli_ui.console.input = lambda *a, **k: ""
    try:
        import cli_skill_memory_cmds
        cli_skill_memory_cmds._open_in_editor = lambda path: None
    except Exception:
        pass
    # question 选择器（/sessions /resume 这类）：给个"取消"应答
    try:
        import cli_live
        if hasattr(cli_live, "InteractiveQuestion"):
            cli_live.InteractiveQuestion.ask = classmethod(
                lambda cls, *a, **k: None)
    except Exception:
        pass

    # 触发命令模块注册（import 即登记），再建装配完整的 rt
    import cli  # noqa: F401（cli 顶部会连带 import 各 *_cmds）
    from cli import RuntimeContext
    import cli_commands

    try:
        rt = RuntimeContext()
        rt.initialize()
    except SystemExit as e:
        print(f"[probe] RuntimeContext 初始化被拒（多半是没配 API key）: {e}")
        return 2
    except Exception:
        print("[probe] RuntimeContext 初始化失败：")
        traceback.print_exc()
        return 2

    entries = [c for c in cli_commands.all_commands() if not only or c.name in only]
    print(f"[probe] 共 {len(entries)} 个命令，逐个通电（10s 看门狗）...\n")

    for entry in entries:
        status, detail = run_one(entry, rt)
        results.append((entry.name, status, detail))
        mark = {"OK": "  OK ", "EXCEPTION": " FAIL", "TIMEOUT": " WAIT"}[status] \
            if status in ("OK", "EXCEPTION", "TIMEOUT") else " ????"
        print(f"[{mark}] {entry.name}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")

    # ── 带参数变体轮：覆盖子命令/未知参数/开关类分支（仍在沙盒）──
    if args_mode:
        import cli_commands as _cc

        class _Entry:
            pass

        variants = [
            "/model 不存在的模型", "/model opus",
            "/search 关键词", "/history 1", "/history 99",
            "/goal status", "/goal clear",
            "/compact --yes",
            "/plugin list", "/plugins",
            "/approved remove 1", "/approved remove-root zz",
            "/add-dir D:/probews_x", "/sandbox on", "/sandbox off",
            "/permission default",
            "/rewind 1", "/trace today", "/trace 999",
            "/output-style 不存在",
            "/handoff save", "/handoff list",
            "/mailbox list", "/mailbox clear",
            "/skill-learning status",
            "/skin list", "/stats", "/usage detail",
            "/resume_bundle list",
            "/resumable list",
            "/hooks diff",
        ]
        print(f"\n[probe] 带参变体 {len(variants)} 条 ...\n")
        for line in variants:
            token = line.split()[0]
            entry = _cc.lookup(token)
            if entry is None:
                results.append((line, "EXCEPTION", "注册表中没有该命令"))
                print(f"[ FAIL] {line}  —— 未注册")
                continue
            rest = line[len(token):].strip()
            shadow = _Entry()
            shadow.name = line
            shadow.handler = (lambda r: (lambda _a, _rt: entry.handler(r, _rt)))(rest)
            status, detail = run_one(shadow, rt)
            results.append((line, status, detail))
            mark = {"OK": "  OK ", "EXCEPTION": " FAIL", "TIMEOUT": " WAIT"}.get(status, " ????")
            print(f"[{mark}] {line}")
            if detail:
                for ln in detail.splitlines():
                    print(f"         {ln}")

    n_fail = sum(1 for _, s, _ in results if s == "EXCEPTION")
    n_wait = sum(1 for _, s, _ in results if s == "TIMEOUT")
    print(f"\n[probe] 汇总：{len(results)} 跑完，OK={len(results)-n_fail-n_wait} "
          f"FAIL={n_fail} TIMEOUT(交互)={n_wait}")
    print(f"[probe] 沙盒目录（自动保留供排查，可手删）：{_TMP}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
