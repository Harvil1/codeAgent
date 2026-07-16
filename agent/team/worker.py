# agent/team/worker.py
"""子 agent CLI 入口。

用法：
    python -m agent.team.worker --name X --task "..." \
        --team-dir ~/.agent/.team --agent-home ~/.agent \
        [--autonomous] [--depth N]
"""
import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HarvilAgent team worker")
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

    # 延迟导入避免循环
    from config import load_config
    from agent.memory_store import MemoryStore
    from agent.team.bus import MessageBus
    from agent.team.coordinator import TeamCoordinator
    from agent import AIAgent

    config = load_config(args.config) if args.config else load_config()
    memory_store = MemoryStore(harvil_home=agent_home)
    bus = MessageBus(team_dir=team_dir)
    coordinator = TeamCoordinator(
        team_dir=team_dir, harvil_home=agent_home, config=config,
    )

    # 构造 AIAgent
    api_base = config.get("model", {}).get("base_url") or ""
    api_key = config.get("model", {}).get("api_key") or ""
    model_name = config.get("model", {}).get("name", "deepseek-chat")
    agent = AIAgent(
        base_url=api_base, api_key=api_key, model=model_name,
        enabled_toolsets=config.get("agent", {}).get("enabled_toolsets", ["core"]),
        harvil_home=str(agent_home),
        memory_store=memory_store,
        team_bus=bus, team_coordinator=coordinator, team_name=args.name,
        spawn_depth=args.depth,
        config=config,
    )

    if args.autonomous:
        # === P4b-T3: autonomous 模式 ===
        from agent.team.lifecycle import AutonomousLifecycle
        team_cfg = config.get("team", {})
        lifecycle = AutonomousLifecycle(
            work_fn=lambda task: agent.run_conversation(task),
            poll_inbox_fn=lambda: bus.read_inbox(args.name),
            poll_tasks_fn=lambda: [],  # 暂不接入 task_store
            claim_task_fn=lambda tid: False,
            on_shutdown_fn=lambda: coordinator.update_status(args.name, "completed"),
            idle_timeout=team_cfg.get("autonomous_idle_timeout", 60.0),
            poll_interval=team_cfg.get("autonomous_poll_interval", 5.0),
        )
        lifecycle.run(initial_task=args.task)
        logger.info("worker %s autonomous 生命周期结束", args.name)
    else:
        # 一次性模式（Phase 4a）
        try:
            response = agent.run_conversation(args.task)
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
