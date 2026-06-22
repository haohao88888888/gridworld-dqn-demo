from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: Sequence[float], window: int = 20) -> np.ndarray:
    if not values:
        return np.array([])
    values_arr = np.asarray(values, dtype=np.float32)
    window = max(1, min(window, len(values_arr)))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values_arr, kernel, mode="valid")


def plot_training_curves(
    rewards: Sequence[float],
    successes: Sequence[int],
    losses: Sequence[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    episodes = np.arange(1, len(rewards) + 1)
    axes[0].plot(episodes, rewards, color="#2f6fed", alpha=0.35, label="Reward")
    reward_ma = moving_average(rewards)
    if reward_ma.size:
        axes[0].plot(np.arange(len(reward_ma)) + 1, reward_ma, color="#123c9c", label="Reward MA")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    success_ma = moving_average(successes)
    axes[1].plot(episodes, successes, color="#1b9e77", alpha=0.25, label="Success")
    if success_ma.size:
        axes[1].plot(np.arange(len(success_ma)) + 1, success_ma, color="#08724f", label="Success MA")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(episodes, losses, color="#d95f02", alpha=0.8, label="Loss")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle("GridWorld DQN Training Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

