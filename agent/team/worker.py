# agent/team/worker.py
"""子 agent CLI 入口。

用法：
    python -m agent.team.worker --name X --task "..." \
        --team-dir ~/.agent/.team --agent-home ~/.agent

流程：
1. 加载 config + MemoryStore + MessageBus + TeamCoordinator
2. 构造 AIAgent（team_name=name）
3. run_conversation(task)
4. 把响应 send 给 "main"
5. update_status(name, "completed")
6. 退出
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [worker:%(process)d] %(message)s")
    logger.info("worker %s 启动", args.name)

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
        config=config,
    )

    # 跑任务
    try:
        response = agent.run_conversation(args.task)
        # 把响应发给 main
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
