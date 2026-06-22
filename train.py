from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from agents.dqn_agent import DQNAgent
from config import Config
from envs.gridworld import GridWorld
from utils.plot import plot_training_curves
from utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_config(args: argparse.Namespace) -> Config:
    config = Config()
    config.episodes = args.episodes
    config.output_dir = args.output_dir
    config.seed = args.seed
    return config


def make_env(config: Config) -> GridWorld:
    return GridWorld(
        size=config.grid_size,
        start=config.start,
        goal=config.goal,
        obstacles=config.obstacles,
        max_steps=config.max_steps_per_episode,
    )


def make_agent(config: Config) -> DQNAgent:
    return DQNAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        device=config.device,
        seed=config.seed,
    )


def run_greedy_episode(env: GridWorld, agent: DQNAgent) -> dict:
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
        "total_reward": total_reward,
        "steps": info.get("steps", 0),
        "reached_goal": info.get("reached_goal", False),
        "actions": actions,
        "path": env.path,
        "map": env.render(),
    }


def save_training_log(rows: list[dict], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["episode", "reward", "success", "steps", "epsilon", "loss"],
        )
        writer.writeheader()
        writer.writerows(rows)


def train(config: Config) -> None:
    set_seed(config.seed)
    output_dir = resolve_output_dir(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    figures_dir = output_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config)
    agent = make_agent(config)

    rewards = []
    successes = []
    losses = []
    log_rows = []

    print(f"Training on device: {agent.device}")
    print(f"Grid size: {config.grid_size}x{config.grid_size}, episodes: {config.episodes}")

    for episode in range(1, config.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        episode_losses = []
        done = False
        info = {}

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            total_reward += reward
            state = next_state

        agent.decay_epsilon()
        if episode % config.target_update_interval == 0:
            agent.update_target_network()

        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        success = int(info.get("reached_goal", False))
        rewards.append(total_reward)
        successes.append(success)
        losses.append(mean_loss)
        log_rows.append(
            {
                "episode": episode,
                "reward": round(total_reward, 6),
                "success": success,
                "steps": info.get("steps", 0),
                "epsilon": round(agent.epsilon, 6),
                "loss": round(mean_loss, 8),
            }
        )

        if episode == 1 or episode % 25 == 0 or episode == config.episodes:
            recent_success = np.mean(successes[-25:])
            recent_reward = np.mean(rewards[-25:])
            print(
                f"Episode {episode:4d} | reward {total_reward:6.2f} | "
                f"recent_reward {recent_reward:6.2f} | recent_success {recent_success:4.2f} | "
                f"epsilon {agent.epsilon:5.3f}"
            )

    checkpoint_path = checkpoint_dir / "dqn_gridworld.pt"
    agent.save(checkpoint_path, config=config.as_dict())
    save_training_log(log_rows, logs_dir / "training_log.csv")

    with (logs_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(config.as_dict(), file, ensure_ascii=False, indent=2)

    plot_training_curves(rewards, successes, losses, figures_dir / "training_curves.png")

    greedy_result = run_greedy_episode(env, agent)
    with (logs_dir / "greedy_path.txt").open("w", encoding="utf-8") as file:
        file.write("GridWorld greedy policy after training\n")
        file.write(f"Reached goal: {greedy_result['reached_goal']}\n")
        file.write(f"Total reward: {greedy_result['total_reward']:.3f}\n")
        file.write(f"Steps: {greedy_result['steps']}\n")
        file.write(f"Actions: {' -> '.join(greedy_result['actions'])}\n\n")
        file.write(greedy_result["map"])
        file.write("\n")

    print("\nTraining finished.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training log: {logs_dir / 'training_log.csv'}")
    print(f"Figure: {figures_dir / 'training_curves.png'}")
    print(f"Greedy path: {logs_dir / 'greedy_path.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent in GridWorld.")
    parser.add_argument("--episodes", type=int, default=Config.episodes, help="Number of training episodes.")
    parser.add_argument("--output-dir", type=str, default=Config.output_dir, help="Directory for outputs.")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    train(build_config(parse_args()))

