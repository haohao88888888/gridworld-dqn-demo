from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Deque

import numpy as np


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, batch_size)
        states = np.stack([item.state for item in batch]).astype(np.float32)
        actions = np.array([item.action for item in batch], dtype=np.int64)
        rewards = np.array([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch]).astype(np.float32)
        dones = np.array([item.done for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

