# agent/team/worker.py
"""工人进程的启动入口：被当命令行程序拉起来干活的子 agent。

谁拉它：coordinator.py 的 spawn 用 subprocess 启动本模块。
它干什么：组装一个独立的 AIAgent 实例（带团队总线和协调员），
把命令行传来的任务跑完，把结果发回给主 agent，然后退出。

用法：
    python -m agent.team.worker --name X --task "..." \
        --team-dir ~/.OmniMate/.team --agent-home ~/.OmniMate \
        [--autonomous] [--depth N]
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OmniMate team worker")
    parser.add_argument("--name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--team-dir", required=True)
    parser.add_argument("--agent-home", required=True)
    parser.add_argument("--config", default=None,
                        help="可选 config.yaml 路径")
    parser.add_argument("--autonomous", action="store_true",
                        help="启用 IDLE 轮询模式")
    parser.add_argument("--depth", type=int, default=1,
                        help="递归 spawn 深度（主 agent=0，子 agent=1+）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [worker:%(process)d] %(message)s")
    logger.info("worker %s 启动 (depth=%d, autonomous=%s)",
                args.name, args.depth, args.autonomous)

    team_dir = Path(args.team_dir)
    agent_home = Path(args.agent_home)

    # 导入放函数内（延迟导入）：这些模块反过来会引用 worker 链上的东西，顶部导入会循环
    from config import load_config
    from agent.memory_store import MemoryStore
    from agent.team.bus import MessageBus
    from agent.team.coordinator import TeamCoordinator
    from agent import AIAgent

    config = load_config(args.config) if args.config else load_config()
    memory_store = MemoryStore(omnimate_home=agent_home)
    bus = MessageBus(team_dir=team_dir)
    coordinator = TeamCoordinator(
        team_dir=team_dir, omnimate_home=agent_home, config=config,
    )

    # 从配置抠出模型三件套，构造 AIAgent
    api_base = config.get("model", {}).get("base_url") or ""
    api_key = config.get("model", {}).get("api_key") or ""
    model_name = config.get("model", {}).get("name", "deepseek-chat")
    agent = AIAgent(
        base_url=api_base, api_key=api_key, model=model_name,
        enabled_toolsets=config.get("agent", {}).get("enabled_toolsets", ["core"]),
        omnimate_home=str(agent_home),
        memory_store=memory_store,
        team_bus=bus, team_coordinator=coordinator, team_name=args.name,
        spawn_depth=args.depth,
        config=config,
    )

    if args.autonomous:
        # === autonomous 模式（P4b-T3 引入） ===
        # 历史踩坑（P4b final-fix I2）：lifecycle.run 必须 try/except 包住，
        # 否则异常会让进程静默崩掉，主 agent 完全不知情
        from agent.team.lifecycle import AutonomousLifecycle
        team_cfg = config.get("team", {})

        def _run_work(task):
            """干活回调：每轮 WORK 前清空对话历史，再跑一遍任务。

            参数：task：本轮任务文本。
            返回：run_conversation 的最终回复文本。

            为什么清空历史：规范 §9.3 要求每个 WORK 周期独立——
            上一轮的对话不该渗进这一轮。

            历史踩坑（Task E2）：run_conversation 已经改成 async 了，但
            lifecycle 的 work_fn 签名要求同步函数（lifecycle.run 是同步
            状态机），所以这里用 asyncio.run 桥接。每个周期各起一个
            独立 event loop（跑完即弃），不存在嵌套 loop 的风险。
            """
            agent.conversation_history = []  # 周期独立：跨周期累积会污染上下文
            agent._idle_requested = False    # run_conversation 开头本会重置，这里是双保险
            return asyncio.run(agent.run_conversation(task))

        lifecycle = AutonomousLifecycle(
            work_fn=_run_work,
            poll_inbox_fn=lambda: bus.read_inbox(args.name),
            poll_tasks_fn=lambda: [],  # 暂不接入 task_store
            claim_task_fn=lambda tid: False,
            on_shutdown_fn=lambda: coordinator.update_status(args.name, "completed"),
            idle_timeout=team_cfg.get("autonomous_idle_timeout", 60.0),
            poll_interval=team_cfg.get("autonomous_poll_interval", 5.0),
        )
        try:
            lifecycle.run(initial_task=args.task)
            coordinator.update_status(args.name, "completed")
            logger.info("worker %s autonomous 生命周期结束", args.name)
        except Exception as e:
            logger.exception("worker %s autonomous 异常", args.name)
            try:
                bus.send(
                    from_=args.name, to="main",
                    type_="message",
                    content=f"[worker autonomous crashed: {e}]",
                )
            except Exception:
                pass
            coordinator.update_status(args.name, "failed")
            sys.exit(1)
    else:
        # 一次性模式（Phase 4a 最早形态）：跑完一个任务就退，不进入轮询等活
        try:
            # 历史踩坑（Task E2）：run_conversation 已改 async。worker 是 CLI
            # 子进程入口（由 coordinator.spawn 启动），main 必须保持同步签名，
            # 所以在调用点用 asyncio.run 驱动（对齐 E1 模式）。
            response = asyncio.run(agent.run_conversation(args.task))
            bus.send(
                from_=args.name, to="main",
                type_="response", content=response or "",
            )
            coordinator.update_status(args.name, "completed")
            logger.info("worker %s 任务完成", args.name)
        except Exception as e:
            logger.exception("worker %s 异常", args.name)
            bus.send(
                from_=args.name, to="main",
                type_="message",
                content=f"[worker crashed: {e}]",
            )
            coordinator.update_status(args.name, "failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
