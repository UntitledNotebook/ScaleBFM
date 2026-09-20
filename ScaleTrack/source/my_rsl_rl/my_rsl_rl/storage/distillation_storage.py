from __future__ import annotations

import torch
from tensordict import TensorDict


class DistillationStorage:
    """One horizon of student observations, teacher labels, and post-step dones."""

    def __init__(
        self,
        num_envs: int,
        num_steps_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        obs_keys: list[str],
        device: str = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.device = device
        # These tensors must remain usable by autograd after inference-mode collection.
        with torch.inference_mode(False):
            self.observations = TensorDict(
                {
                    key: torch.empty(num_steps_per_env, *obs[key].shape, dtype=obs[key].dtype, device=device)
                    for key in obs_keys
                },
                batch_size=[num_steps_per_env, num_envs],
                device=device,
            )
            self.teacher_actions = torch.empty(num_steps_per_env, num_envs, *actions_shape, device=device)
            self.dones = torch.empty(num_steps_per_env, num_envs, dtype=torch.bool, device=device)
        self.step = 0

    def save_observations(self, obs: TensorDict, teacher_actions: torch.Tensor) -> None:
        """Copy before env.step(), which may reuse or modify observation tensors."""
        for key in self.observations.keys():
            self.observations[key][self.step].copy_(obs[key])
        self.teacher_actions[self.step].copy_(teacher_actions)

    def add_dones(self, dones: torch.Tensor) -> None:
        self.dones[self.step].copy_(dones.reshape(self.num_envs))
        self.step += 1

    def get_batch(self) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten in environment-major order, retaining within-episode adjacent pairs."""
        horizon = self.step
        obs = TensorDict(
            {
                key: value[:horizon].transpose(0, 1).reshape(self.num_envs * horizon, *value.shape[2:])
                for key, value in self.observations.items()
            },
            batch_size=[self.num_envs * horizon],
            device=self.device,
        )
        actions = self.teacher_actions[:horizon].transpose(0, 1).reshape(
            self.num_envs * horizon, *self.teacher_actions.shape[2:]
        )
        indices = torch.arange(self.num_envs * horizon, device=self.device).reshape(self.num_envs, horizon)
        valid = ~self.dones[:horizon - 1].transpose(0, 1)
        previous = indices[:, :-1][valid]
        current = indices[:, 1:][valid]
        return obs, actions, previous, current

    def clear(self) -> None:
        self.step = 0
