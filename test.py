from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from agents.dqn_agent import DQNAgent, load_checkpoint_file
from config import Config
from envs.gridworld import GridWorld
from train import make_agent, make_env, resolve_output_dir
from utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent


def normalize_config_values(config: Config) -> Config:
    config.start = tuple(config.start)
    config.goal = tuple(config.goal)
    config.obstacles = tuple(tuple(item) for item in config.obstacles)
    return config


def load_config_from_checkpoint(checkpoint_path: Path) -> Config:
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("config", {})
    config = Config()
    editable_fields = {field.name for field in fields(Config)}
    for key, value in saved_config.items():
        if key in editable_fields:
            setattr(config, key, value)
    return normalize_config_values(config)


def run_test_episode(env: GridWorld, agent: DQNAgent) -> dict:
    state = env.reset()
    total_reward = 0.0
    actions = []
    done = False
    info = {}

    while not done:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        actions.append(env.ACTION_NAMES[action])
        total_reward += reward
        state = next_state

    return {
        "reached_goal": info.get("reached_goal", False),
        "steps": info.get("steps", 0),
        "total_reward": total_reward,
        "actions": actions,
        "path": env.path,
        "map": env.render(),
    }


def test_agent(checkpoint_path: Path, episodes: int) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Please run `python train.py` first."
        )

    config = load_config_from_checkpoint(checkpoint_path)
    set_seed(config.seed)
    env = make_env(config)
    agent = make_agent(config)
    agent.load(checkpoint_path)

    output_dir = resolve_output_dir(config.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Testing episodes: {episodes}\n")

    for episode in range(1, episodes + 1):
        result = run_test_episode(env, agent)
        header = (
            f"Episode {episode}: reached_goal={result['reached_goal']}, "
            f"steps={result['steps']}, reward={result['total_reward']:.3f}"
        )
        print(header)
        print(result["map"])
        print()

        report_lines.append(header)
        report_lines.append("Actions: " + " -> ".join(result["actions"]))
        report_lines.append(result["map"])
        report_lines.append("")

    report_path = logs_dir / "test_trajectory.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved test trajectory: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained DQN agent in GridWorld.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "dqn_gridworld.pt"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_agent(Path(args.checkpoint), args.episodes)
