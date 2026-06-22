from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Position = Tuple[int, int]
DEFAULT_OBSTACLES: Tuple[Position, ...] = (
    (1, 1),
    (1, 2),
    (2, 4),
    (3, 1),
    (4, 3),
)


class GridWorld:
    """A small deterministic grid environment for DQN experiments."""

    ACTIONS: Dict[int, Position] = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }
    ACTION_NAMES: Dict[int, str] = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    def __init__(
        self,
        size: int = 6,
        start: Position = (0, 0),
        goal: Position = (5, 5),
        obstacles: Iterable[Position] | None = DEFAULT_OBSTACLES,
        max_steps: int = 60,
    ) -> None:
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.max_steps = max_steps

        self._validate_layout()
        self.agent_pos: Position = self.start
        self.steps = 0
        self.path: List[Position] = []

    def reset(self) -> np.ndarray:
        self.agent_pos = self.start
        self.steps = 0
        self.path = [self.start]
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action {action}. Valid actions: {sorted(self.ACTIONS)}")

        row_delta, col_delta = self.ACTIONS[action]
        next_pos = (self.agent_pos[0] + row_delta, self.agent_pos[1] + col_delta)
        reward = -0.02
        hit_wall = False
        hit_obstacle = False

        if not self._inside(next_pos):
            next_pos = self.agent_pos
            reward = -0.20
            hit_wall = True
        elif next_pos in self.obstacles:
            next_pos = self.agent_pos
            reward = -0.20
            hit_obstacle = True

        self.agent_pos = next_pos
        self.steps += 1
        self.path.append(self.agent_pos)

        reached_goal = self.agent_pos == self.goal
        if reached_goal:
            reward = 1.0

        truncated = self.steps >= self.max_steps and not reached_goal
        done = reached_goal or truncated

        info = {
            "position": self.agent_pos,
            "action_name": self.ACTION_NAMES[action],
            "hit_wall": hit_wall,
            "hit_obstacle": hit_obstacle,
            "reached_goal": reached_goal,
            "truncated": truncated,
            "steps": self.steps,
        }
        return self._get_state(), reward, done, info

    def render(self) -> str:
        return self.render_path(self.path)

    def render_path(self, path: Sequence[Position] | None = None) -> str:
        path_set = set(path or [])
        lines = []
        for row in range(self.size):
            cells = []
            for col in range(self.size):
                pos = (row, col)
                if pos == self.agent_pos:
                    cells.append("A")
                elif pos == self.start:
                    cells.append("S")
                elif pos == self.goal:
                    cells.append("G")
                elif pos in self.obstacles:
                    cells.append("#")
                elif pos in path_set:
                    cells.append("*")
                else:
                    cells.append(".")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def _get_state(self) -> np.ndarray:
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        for row, col in self.obstacles:
            grid[row, col] = -1.0
        grid[self.goal] = 0.5
        grid[self.agent_pos] = 1.0
        return grid.flatten()

    def _inside(self, pos: Position) -> bool:
        row, col = pos
        return 0 <= row < self.size and 0 <= col < self.size

    def _validate_layout(self) -> None:
        important_positions = [self.start, self.goal, *self.obstacles]
        for pos in important_positions:
            if not self._inside(pos):
                raise ValueError(f"Position {pos} is outside a {self.size}x{self.size} grid.")
        if self.start == self.goal:
            raise ValueError("Start and goal must be different positions.")
        if self.start in self.obstacles:
            raise ValueError("Start position cannot be an obstacle.")
        if self.goal in self.obstacles:
            raise ValueError("Goal position cannot be an obstacle.")
