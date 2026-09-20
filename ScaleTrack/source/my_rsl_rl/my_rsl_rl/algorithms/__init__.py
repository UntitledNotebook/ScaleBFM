"""Implementation of different learning algorithms."""

from .ppo import PPO
from .distillation import Distillation

__all__ = ["PPO", "Distillation"]
