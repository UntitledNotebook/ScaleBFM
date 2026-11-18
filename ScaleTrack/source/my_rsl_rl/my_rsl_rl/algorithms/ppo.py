from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from tensordict import TensorDict

from my_rsl_rl.modules import ActorCritic, ActorCriticHumanoidTransformer
from my_rsl_rl.storage import RolloutStorage
from my_rsl_rl.utils import string_to_callable


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticHumanoidTransformer
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticHumanoidTransformer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        actor_learning_rate: float = 0.001,
        critic_learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Create optimizer
        self.actor_optimizer = optim.Adam(self.policy.actor_parameters, lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.policy.critic_parameters, lr=critic_learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.clean_teacher = None

    def freeze_clean_teacher(self, coefficient: float) -> None:
        """Snapshot the imported XL actor. Supervise only pre-action clean observations."""
        if not isinstance(self.policy, ActorCriticHumanoidTransformer) or self.policy.state_dependent_std:
            raise ValueError("The clean anchor requires the XL transformer with state-independent action noise.")
        if self.policy.noise_std_type != "scalar" or not (self.policy.std > 0).all():
            raise ValueError("The XL checkpoint must have positive, scalar-parameterized action standard deviations.")
        self.clean_kl_coef = coefficient
        self.clean_teacher = deepcopy(self.policy).requires_grad_(False).eval()
        # The critic is never evaluated by the teacher.
        del self.clean_teacher.critic, self.clean_teacher.critic_task_embedder
        shape = self.storage.observations.batch_size
        self.storage.observations["xl_clean"] = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.storage.observations["xl_mean"] = torch.zeros(*shape, self.policy.std.numel(), device=self.device)

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict, clean_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        if self.clean_teacher is not None:
            if clean_mask is None or clean_mask.dtype != torch.bool or clean_mask.shape != obs.batch_size:
                raise ValueError("Expected a per-environment, pre-action boolean clean mask.")
            clean_mask = clean_mask.to(self.device).clone()
            target = torch.zeros_like(self.transition.action_mean)
            if clean_mask.any():
                teacher = self.clean_teacher
                prop, task, action = teacher.get_actor_obs(obs[clean_mask])
                with torch.no_grad():
                    target[clean_mask] = teacher.actor(prop, action, teacher.actor_task_embedder(task))
            # These two fields belong to storage only, never to the environment or the XL input.
            self.transition.observations = obs.copy()
            self.transition.observations["xl_clean"] = clean_mask
            self.transition.observations["xl_mean"] = target
        return self.transition.actions

    def _clean_kl(self, obs: TensorDict, mean: torch.Tensor, sigma: torch.Tensor):
        mask = obs["xl_clean"]
        student_std = sigma[mask]
        teacher_std = self.clean_teacher.std
        # KL(teacher || student), averaged over the 29 joints.
        kl_sum = (
            torch.log(student_std / teacher_std)
            + (teacher_std.square() + (obs["xl_mean"][mask] - mean[mask]).square()) / (2 * student_std.square())
            - 0.5
        ).mean(dim=-1).sum()
        local_count = mask.sum()
        global_count = local_count.clone()
        if self.is_multi_gpu:
            torch.distributed.all_reduce(global_count)
        # reduce_parameters averages rank gradients, including ranks with zero clean samples.
        loss = kl_sum * (self.gpu_world_size / global_count.clamp_min(1))
        return loss, kl_sum.detach(), local_count

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, gpu_global_rank=self.gpu_global_rank, gpu_world_size=self.gpu_world_size,
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        teacher_stats = torch.zeros(3, device=self.device)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.actor_learning_rate = max(1e-5, self.actor_learning_rate/1.5)
                            self.critic_learning_rate = max(1e-5, self.critic_learning_rate/1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.actor_learning_rate = min(1e-2, self.actor_learning_rate*1.5)
                            self.critic_learning_rate = min(1e-2, self.critic_learning_rate*1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        actor_lr_tensor = torch.tensor(self.actor_learning_rate, device=self.device)
                        critic_lr_tensor = torch.tensor(self.critic_learning_rate, device=self.device)
                        torch.distributed.broadcast(actor_lr_tensor, src=0)
                        torch.distributed.broadcast(critic_lr_tensor, src=0)
                        self.actor_learning_rate = actor_lr_tensor.item()
                        self.critic_learning_rate = critic_lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = self.actor_learning_rate
                    for param_group in self.critic_optimizer.param_groups:
                        param_group["lr"] = self.critic_learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            if self.clean_teacher is not None:
                clean_loss, kl_sum, count = self._clean_kl(obs_batch, mu_batch, sigma_batch)
                loss = loss + self.clean_kl_coef * clean_loss
                teacher_stats[0] += kl_sum
                teacher_stats[1] += count
                teacher_stats[2] += original_batch_size

            # Compute the gradients for PPO
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            
        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.clean_teacher is not None:
            if self.is_multi_gpu:
                torch.distributed.all_reduce(teacher_stats)
            loss_dict["clean_teacher_kl"] = (teacher_stats[0] / teacher_stats[1].clamp_min(1)).item()
            loss_dict["clean_fraction"] = (teacher_stats[1] / teacher_stats[2]).item()

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Keep each receiving tensor on its own device. Pickling a CUDA state_dict
        # can otherwise materialize rank 0 device tensors in every process.
        for tensor in self.policy.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
