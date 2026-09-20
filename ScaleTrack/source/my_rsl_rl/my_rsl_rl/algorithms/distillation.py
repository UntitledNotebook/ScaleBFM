from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn, optim
from torch.nn import functional as F

from my_rsl_rl.storage import DistillationStorage


class Distillation:
    """Full-horizon VAE behavior cloning on student-visited states."""

    def __init__(
        self,
        policy: nn.Module,
        device: str = "cpu",
        num_learning_epochs: int = 5,
        learning_rate: float = 8.0e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        kl_weight: float = 0.01,
        temporal_weight: float = 0.0,
        schedule: str = "cosine_annealing",
        schedule_iterations: int = 400000,
        min_learning_rate: float = 1.0e-6,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.policy = policy.to(device)
        self.num_learning_epochs = num_learning_epochs
        self.max_grad_norm = max_grad_norm
        self.kl_weight = kl_weight
        self.temporal_weight = temporal_weight
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_world_size = multi_gpu_cfg["world_size"] if self.is_multi_gpu else 1
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if self.is_multi_gpu else 0
        self.parameters = [param for param in self.policy.parameters() if param.requires_grad]
        self.optimizer = optim.AdamW(self.parameters, lr=learning_rate, weight_decay=weight_decay)
        if schedule == "cosine_annealing":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=schedule_iterations, eta_min=min_learning_rate
            )
        elif schedule == "constant":
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        else:
            raise ValueError(f"Unknown distillation schedule: {schedule}")
        self.storage: DistillationStorage | None = None

    def init_storage(
        self,
        num_envs: int,
        num_steps_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        obs_keys = list(dict.fromkeys(key for group in self.policy.obs_groups.values() for key in group))
        self.storage = DistillationStorage(
            num_envs, num_steps_per_env, obs, actions_shape, obs_keys, self.device
        )

    @torch.no_grad()
    def act(self, obs: TensorDict, teacher_actions: torch.Tensor) -> torch.Tensor:
        self.storage.save_observations(obs, teacher_actions)
        return self.policy.act(obs).detach()

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        self.storage.add_dones(dones)
        self.policy.reset(dones)

    def update(self) -> dict[str, float]:
        obs, labels, previous, current = self.storage.get_batch()
        # Gradients are averaged across ranks; compensate for different local pair counts.
        counts = torch.tensor([labels.shape[0], current.numel()], dtype=torch.float32, device=self.device)
        if self.is_multi_gpu:
            torch.distributed.all_reduce(counts)
        sample_scale = labels.shape[0] * self.gpu_world_size / counts[0]
        pair_scale = self.gpu_world_size / counts[1].clamp_min(1)
        totals = torch.zeros(5, device=self.device)
        for _ in range(self.num_learning_epochs):
            actions, (mu_q, logvar_q), (mu_p, logvar_p) = self.policy(obs)
            behavior_loss = F.mse_loss(actions, labels) * sample_scale
            # Avoid forming individual variances, which may overflow even when their ratio is finite.
            with torch.autocast(device_type=mu_q.device.type, enabled=False):
                mq, mp = mu_q.float(), mu_p.float()
                lq, lp = logvar_q.float(), logvar_p.float()
                log_ratio = lq - lp
                normalized_delta = (mq - mp) * torch.exp(-0.5 * lp)
                kl_loss = 0.5 * (
                    torch.expm1(log_ratio) - log_ratio + normalized_delta.square()
                ).mean() * sample_scale
            temporal_loss = (mu_q[current] - mu_q[previous].detach()).norm(dim=-1).sum() * pair_scale
            loss = behavior_loss + self.kl_weight * kl_loss + self.temporal_weight * temporal_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            grad_norm = nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
            self.optimizer.step()
            totals += torch.stack((behavior_loss, kl_loss, temporal_loss, loss, grad_norm)).detach()

        self.scheduler.step()
        self.storage.clear()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(totals)
            totals /= self.gpu_world_size
        totals /= self.num_learning_epochs
        metrics = dict(zip(("behavior_loss", "kl_loss", "temporal_loss", "total_loss", "grad_norm"), totals.tolist()))
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        for tensor in self.policy.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Average a fixed parameter layout, including gradients absent on a rank."""
        gradients = torch.cat(
            [param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1)
             for param in self.parameters]
        )
        torch.distributed.all_reduce(gradients)
        gradients /= self.gpu_world_size
        offset = 0
        for param in self.parameters:
            param.grad = gradients[offset:offset + param.numel()].view_as(param)
            offset += param.numel()
