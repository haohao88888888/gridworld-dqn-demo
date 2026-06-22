from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass
class Config:
    grid_size: int = 6
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (5, 5)
    obstacles: Tuple[Tuple[int, int], ...] = (
        (1, 1),
        (1, 2),
        (2, 4),
        (3, 1),
        (4, 3),
    )
    max_steps_per_episode: int = 60

    episodes: int = 400
    batch_size: int = 64
    buffer_size: int = 5000
    learning_rate: float = 1e-3
    gamma: float = 0.95

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_interval: int = 20

    hidden_dim: int = 128
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"

    @property
    def state_dim(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def action_dim(self) -> int:
        return 4

    def as_dict(self) -> dict:
        data = asdict(self)
        data["state_dim"] = self.state_dim
        data["action_dim"] = self.action_dim
        return data

