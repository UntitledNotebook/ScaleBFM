# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import math
import hashlib
import statistics
import tempfile
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict
from rich.progress import track
from copy import deepcopy

import my_rsl_rl
from my_rsl_rl.algorithms import PPO
from my_rsl_rl.env import VecEnv
from my_rsl_rl.modules import (
    ActorCritic, 
    ActorCriticHumanoidTransformer,
)
from my_rsl_rl.utils import resolve_obs_groups, store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Check if multi-GPU is enabled
        self._configure_multi_gpu()

        # Store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # Create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # Note: We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.switch_training = None
        self.switch_command = None
        self.git_status_repos = [my_rsl_rl.__file__]

        self.eval_during_training = self.cfg["eval_during_training"]
        self.eval_interval = self.cfg["eval_interval"]
        self.eval_metric_keys = self.cfg.get('eval_metric_keys', [])
        self.eval_max_steps = self.cfg.get('eval_max_steps', None)
        self.success_metric_dict = self.cfg.get('success_metric_dict', {})
        self.command_name = self.cfg.get('command_name', 'motion')
        self.success_discount_coef = self.cfg.get("success_discount_coef", 0.999)

    def initialize_switch_finetuning(self, xl_checkpoint: str, clean_kl_coef: float = 1.0) -> None:
        """Start at update zero with strict XL weights, fresh Adam and a frozen clean teacher."""
        if not math.isfinite(clean_kl_coef) or clean_kl_coef < 0:
            raise ValueError("clean_kl_coef must be finite and nonnegative.")
        policy = self.alg.policy
        if not isinstance(policy, ActorCriticHumanoidTransformer) or policy.actor.transformer_blocks[0].self_attention.num_heads != 6:
            raise ValueError("Expected the six-head ScaleTrack XL transformer.")
        command = self.env.unwrapped.command_manager.get_term(self.command_name)
        if not hasattr(command, "clean_teacher_mask") or not hasattr(command, "set_training_iteration"):
            raise ValueError("XL fine-tuning requires the switch-aware motion command.")
        if self.current_learning_iteration or self.alg.actor_optimizer.state or self.alg.critic_optimizer.state:
            raise RuntimeError("Initialize XL fine-tuning before learning or restoring an experiment.")
        with open(xl_checkpoint, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        xl = torch.load(xl_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.alg.policy.load_state_dict(xl["model_state_dict"], strict=True)
        self.alg.freeze_clean_teacher(clean_kl_coef)
        self.switch_command = command
        self.switch_training = {"xl_sha256": digest, "clean_kl_coef": clean_kl_coef, "xl_iteration": xl["iter"]}
        self.cfg["scalar_metrics_only"] = True
        if not self.disable_logs:
            print(f"[XL] Imported update {xl['iter']}, SHA256 {digest}, fresh PPO update 0", flush=True)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):

            start = time.time()
            if self.switch_command is not None:
                self.switch_command.set_training_iteration(it)
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    clean_mask = self.switch_command.clean_teacher_mask if self.switch_command is not None else None
                    actions = self.alg.act(obs, clean_mask=clean_mask)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        
                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.switch_training is not None:
                self.current_learning_iteration = it + 1
                if it == start_iter or (it + 1) % self.save_interval == 0:
                    print(f"[PPO] rank={self.gpu_global_rank}/{self.gpu_world_size} device={self.device} "
                          f"completed={it + 1} local_steps={self.num_steps_per_env * self.env.num_envs} "
                          f"value_loss={loss_dict['value_function']:.6g}", flush=True)

            if self.eval_during_training and (it + 1) % self.eval_interval == 0:
                if self.log_dir is not None and not self.disable_logs:
                    self.save(os.path.join(self.log_dir, f"model_{it+1}.pt"))

                with torch.inference_mode():
                    eval_dict = self.evaluate_policy()
                    eval_test_dict = self.evaluate_policy(test_set=True)
                    self.env.unwrapped.command_manager.get_term(self.command_name).resample_motions()
                    self.env.unwrapped.command_manager.get_term(self.command_name).randomize_next_resampling = True
                    obs, _ = self.env.reset()

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if self.current_learning_iteration % self.save_interval == 0 or (self.switch_training and it == start_iter):
                    self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and self.log_dir is not None and not self.disable_logs and self.switch_training is None:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:

        if 'eval_dict' in locs:
            for key, value in locs['eval_dict'].items():
                self.writer.add_scalar(f"Eval/{key}", value, locs['it'])
        
        if 'eval_test_dict' in locs:
            for key, value in locs['eval_test_dict'].items():
                self.writer.add_scalar(f"Eval_Test/{key}", value, locs['it'])

        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # Handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])

        self.writer.add_scalar("Loss/actor_learning_rate", self.alg.actor_learning_rate, locs["it"])
        self.writer.add_scalar("Loss/critic_learning_rate", self.alg.critic_learning_rate, locs["it"])

        # Log noise std
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if self.switch_training is not None:
            self.writer.add_scalar("Train/global_env_steps", self.tot_timesteps, locs["it"])
            self.writer.add_scalar("Train/world_size", self.gpu_world_size, locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            # Everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            # Print losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            # Print rewards
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # Print episode information
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        if self.switch_training is not None:
            saved_dict["switch_training"] = self.switch_training
            saved_dict["global_steps"] = self.tot_timesteps
            saved_dict["training_time"] = self.tot_time
        # A preemption must leave the last complete checkpoint loadable.
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(path)), delete=False) as stream:
            tmp_path = stream.name
        try:
            torch.save(saved_dict, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Upload model to external logging service
        # if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
        #     self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location or self.device)
        if self.switch_training is not None:
            if loaded_dict.get("switch_training") != self.switch_training:
                raise ValueError("Resume checkpoint has a different or missing frozen XL anchor or KL coefficient.")
            self.tot_timesteps = loaded_dict["global_steps"]
            self.tot_time = loaded_dict["training_time"]
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.alg.actor_learning_rate = self.alg.actor_optimizer.param_groups[0]["lr"]
            self.alg.critic_learning_rate = self.alg.critic_optimizer.param_groups[0]["lr"]
        # Load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        cpu_gloo = self.device == "cpu" and torch.distributed.is_initialized() and torch.distributed.get_backend() == "gloo"
        if self.device != f"cuda:{self.gpu_local_rank}" and not cpu_gloo:
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-critic algorithm."""

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticHumanoidTransformer = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self) -> None:
        """Prepare the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune or Tensorboard summary writer, default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from my_rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from my_rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def evaluate_policy(self, test_set=False):
        if test_set and not self.env.unwrapped.command_manager.get_term(self.command_name).has_test_set:
            return {}

        self.eval_mode()

        self._set_env_is_evaluating(test_set)

        num_motions = self.env.unwrapped.command_manager.get_term(self.command_name).num_motion
        motion_lengths = self.env.unwrapped.command_manager.get_term(self.command_name).time_totals
        motion_range = torch.argsort(motion_lengths)
        
        if self.is_distributed and not test_set: # weishuai: we still evaluate the whole test set for each rank
            world_size = self.gpu_world_size
            rank = self.gpu_global_rank
            motion_range = motion_range[rank::world_size] # weishuai: This is to ensure balanced spread of motions of all lengths; Otherwise, some motions may be too long and torch.distributed.barriar may get timed out.
            num_motions_to_eval = len(motion_range)
        else:
            num_motions_to_eval = num_motions

        assert num_motions_to_eval > 0, f"There should be at least one motion to evaluate!"

        metrics = {}
        metrics["motion_ids"] = motion_range
        for k in self.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions_to_eval)
            metrics[f"{k}_max"] = torch.zeros(num_motions_to_eval)
            # metrics[f"{k}_min"] = torch.zeros(num_motions)

        motion_map = []
        for i in range(0, len(motion_range), self.env.num_envs):
            batch_idx = torch.arange(i, min(i+self.env.num_envs, len(motion_range)))
            batch_motion_ids = motion_range[i: i+self.env.num_envs]
            motion_map.append((batch_idx, batch_motion_ids))
        num_iterations = len(motion_map)

        policy = self.get_inference_policy(device=self.device)

        for iter in track(
            range(num_iterations), total=num_iterations, description="Evaluting..."
        ):
            local_idx, motion_ids = motion_map[iter]
            num_motions_this_iter = len(motion_ids)
            
            self.env.unwrapped.command_manager.get_term(self.command_name).motion_ids[:num_motions_this_iter] = motion_ids
            self.env.unwrapped.command_manager.get_term(self.command_name).motion_ids[num_motions_this_iter:] = motion_ids[0]
            elapsed_time = torch.zeros_like(motion_ids)

            motion_lengths = self.env.unwrapped.command_manager.get_term(self.command_name).time_totals[motion_ids]
            if self.eval_max_steps:
                motion_lengths = torch.clamp(motion_lengths, max=self.eval_max_steps*torch.ones_like(motion_lengths))
            max_length = motion_lengths.max().item()
            
            obs, extras = self.env.reset()

            for l in track(range(max_length), total=max_length, transient=True):
                actions = policy(obs.to(self.device))
                obs, _, _, _ = self.env.step(actions.to(self.env.device))

                elapsed_time += 1

                clip_done = (elapsed_time >= motion_lengths).cpu()
                clip_not_done = torch.logical_not(clip_done)

                env_metric_dict = self.env.unwrapped.command_manager.get_term(self.command_name).metrics
                for k in self.eval_metric_keys:
                    if k not in env_metric_dict:
                        raise ValueError(f"key {k} not found in command manager!")
                    value = env_metric_dict[k].cpu()

                    metric = value[:num_motions_this_iter]
                    metrics[k][local_idx[clip_not_done]] += metric[clip_not_done]
                    metrics[f"{k}_max"][local_idx[clip_not_done]] = torch.maximum(
                        metrics[f"{k}_max"][local_idx[clip_not_done]],
                        metric[clip_not_done]
                    )
                    # metrics[f"{k}_min"][motion_ids[clip_not_done]] = torch.minimum(
                    #     metrics[f"{k}_min"][motion_ids[clip_not_done]],
                    #     metric[clip_not_done]
                    # )
                    
        motion_lengths = self.env.unwrapped.command_manager.get_term(self.command_name).time_totals[motion_range]
        if self.eval_max_steps:
            motion_lengths = torch.clamp(motion_lengths, max=self.eval_max_steps*torch.ones_like(motion_lengths))
        for k in self.eval_metric_keys:
            metrics[k] =  metrics[k] / motion_lengths

        if self.success_metric_dict:
            tracking_failures = torch.zeros(num_motions_to_eval, dtype=torch.bool)
            for k in self.success_metric_dict:
                if k in metrics:
                    tracking_failures = torch.logical_or(tracking_failures, metrics[f"{k}_max"] > self.success_metric_dict[k])
            tracking_failures = tracking_failures.float()
            failed_motions_index = torch.nonzero(tracking_failures).flatten().tolist()
            failed_motions_id = motion_range[failed_motions_index]
            motion_names = self.env.unwrapped.command_manager.get_term(self.command_name).motion_names
            
            set_name = "test" if test_set else 'train'
            if self.is_distributed:
                fail_save_path = os.path.join(self.log_dir, f"failed_{set_name}_motions_rank_{self.gpu_global_rank}.txt")
            else:
                fail_save_path = os.path.join(self.log_dir, f"failed_{set_name}_motions.txt")
            
            with open(fail_save_path, 'w') as f:
                for index in failed_motions_id:
                    f.write(f"{motion_names[index]}\n")

        if self.is_distributed:
            with open(os.path.join(self.log_dir, f"{self.gpu_global_rank}_metrics.pt"), "wb") as f:
                torch.save(metrics, f)
            
            torch.distributed.barrier()

            metric_dict = {}

            if self.gpu_global_rank == 0:
                
                if not test_set:
                    gathered_metrics = {k: torch.zeros(self.env.unwrapped.command_manager.get_term(self.command_name).num_motion_train) for k in metrics}

                    for rank in range(torch.distributed.get_world_size()):
                        with open(os.path.join(self.log_dir, f"{rank}_metrics.pt"), 'rb') as f:
                            other_metrics = torch.load(f, map_location="cpu")
                        
                        for k in gathered_metrics:
                            if k == "motion_ids":
                                gathered_metrics[k][other_metrics["motion_ids"]] += 1
                            else:
                                gathered_metrics[k][other_metrics["motion_ids"]] = other_metrics[k]
                        
                        os.unlink(os.path.join(self.log_dir, f"{rank}_metrics.pt"))
                    
                    metrics = gathered_metrics
                    assert torch.all(metrics["motion_ids"] == 1.0).item(), f"Some motions in the training set are not evaluated in the training set or evaluated several times!"
                
                else:
                    gathered_metrics = {k: [] for k in metrics}

                    for rank in range(torch.distributed.get_world_size()):
                        with open(os.path.join(self.log_dir, f"{rank}_metrics.pt"), 'rb') as f:
                            other_metrics = torch.load(f, map_location="cpu")
                        
                        for k in gathered_metrics:
                            gathered_metrics[k].append(other_metrics[k])
                        
                        os.unlink(os.path.join(self.log_dir, f"{rank}_metrics.pt"))
                    
                    metrics = {k: torch.cat(gathered_metrics[k], dim=0) for k in gathered_metrics}

                if self.success_metric_dict:
                    example_key = list(self.success_metric_dict.keys())[0]
                    tracking_failures = torch.zeros_like(metrics[example_key], dtype=torch.bool)
                    for k in self.success_metric_dict:
                        if k in metrics:
                            tracking_failures = torch.logical_or(tracking_failures, metrics[f"{k}_max"] > self.success_metric_dict[k])
                    tracking_failures = tracking_failures.float()
                    metric_dict["success_rate"] = 1.0 - tracking_failures.detach().mean().item()

                    for k in self.eval_metric_keys:
                        mask = (tracking_failures == 0)
                        if mask.any():
                            result = metrics[k][mask].detach().mean().item()
                        else:
                            result = 0.0
                        metric_dict[f"{k}_success"] = result

                for k in self.eval_metric_keys:
                    metric_dict[k] = metrics[k].detach().mean().item()
            
            if not test_set and self.success_metric_dict:
                if self.gpu_global_rank == 0:
                    failed_idx = (tracking_failures == 1)
                    success_discount = math.pow(self.success_discount_coef, self.eval_interval)
                    new_sampling_prob = self.env.unwrapped.command_manager.get_term(self.command_name).motion_sampling_prob.clone()
                    new_sampling_prob[failed_idx] /= success_discount
                    new_sampling_prob[~failed_idx] *= success_discount
                    new_sampling_prob.clamp_(min=0.03, max=1.0)
                    new_sampling_prob_cuda = new_sampling_prob.to(self.device)
                else:
                    new_sampling_prob_cuda = torch.zeros(self.env.unwrapped.command_manager.get_term(self.command_name).num_motion_train, dtype=torch.float, device=self.device)
                
                torch.distributed.broadcast(new_sampling_prob_cuda, src=0)

                new_sampling_prob = new_sampling_prob_cuda.detach().cpu()
                self.env.unwrapped.command_manager.get_term(self.command_name).motion_sampling_prob[:] = new_sampling_prob

            torch.distributed.barrier()

        else:
            metric_dict = {}
            if self.success_metric_dict:
                tracking_failures = torch.zeros(num_motions, dtype=torch.bool)
                for k in self.success_metric_dict:
                    if k in metrics:
                        tracking_failures = torch.logical_or(tracking_failures, metrics[f"{k}_max"] > self.success_metric_dict[k])
                tracking_failures = tracking_failures.float()
                metric_dict["success_rate"] = 1.0 - tracking_failures.detach().mean().item()
                for k in self.eval_metric_keys:
                    mask = (tracking_failures == 0)
                    if mask.any():
                        result = metrics[k][mask].detach().mean().item()
                    else:
                        result = 0.0
                    metric_dict[f"{k}_success"] = result
                if not test_set:
                    failed_idx = (tracking_failures == 1)
                    success_discount = math.pow(self.success_discount_coef, self.eval_interval)
                    new_sampling_prob = self.env.unwrapped.command_manager.get_term(self.command_name).motion_sampling_prob.clone()
                    new_sampling_prob[failed_idx] /= success_discount
                    new_sampling_prob[~failed_idx] *= success_discount
                    new_sampling_prob.clamp_(min=0.03, max=1.0)
                    self.env.unwrapped.command_manager.get_term(self.command_name).motion_sampling_prob[:] = new_sampling_prob
            for k in self.eval_metric_keys:
                metric_dict[k] = metrics[k].detach().mean().item()
    
        self._set_env_no_evaluating(test_set)

        self.train_mode()

        return metric_dict
        
    def _set_env_is_evaluating(self, test_set = False):
        self.env.unwrapped.command_manager.get_term(self.command_name).is_evaluating = True
        if test_set:
            self.env.unwrapped.command_manager.get_term(self.command_name).switch_motion_set(True)

        # Disable reset
        for key in self.env.unwrapped.termination_manager.active_terms:
            term_cfg = self.env.unwrapped.termination_manager.get_term_cfg(key)
            if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                term_cfg.params['disable_flag'] = True
            self.env.unwrapped.termination_manager.set_term_cfg(key, term_cfg)

        for key in self.env.unwrapped.event_manager.active_terms:
            if key in ['interval', 'reset']:
                for term_name in self.env.unwrapped.event_manager.active_terms[key]:
                    term_cfg = self.env.unwrapped.event_manager.get_term_cfg(term_name)
                    if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                        term_cfg.params['disable_flag'] = True
                    self.env.unwrapped.event_manager.set_term_cfg(term_name, term_cfg)
        
        self.obs_noise_cfg = {}

        for group_name in self.env.unwrapped.observation_manager.active_terms:
            obs_group_names = self.env.unwrapped.observation_manager._group_obs_term_names[group_name]
            obs_group_terms = self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name]
            self.obs_noise_cfg[group_name] = {}

            for i in range(len(obs_group_names)):
                name, term = obs_group_names[i], obs_group_terms[i]
                if hasattr(term, 'noise') and term.noise:
                    self.obs_noise_cfg[group_name][name] = deepcopy(term.noise)
                    self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name][i].noise = None
        
    def _set_env_no_evaluating(self, test_set=False):
        self.env.unwrapped.command_manager.get_term(self.command_name).is_evaluating = False
        if test_set:
            self.env.unwrapped.command_manager.get_term(self.command_name).switch_motion_set(False)

        for key in self.env.unwrapped.termination_manager.active_terms:
            term_cfg = self.env.unwrapped.termination_manager.get_term_cfg(key)
            if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                term_cfg.params['disable_flag'] = False
            self.env.unwrapped.termination_manager.set_term_cfg(key, term_cfg)
        
        for key in self.env.unwrapped.event_manager.active_terms:
            if key in ['interval', 'reset']:
                for term_name in self.env.unwrapped.event_manager.active_terms[key]:
                    term_cfg = self.env.unwrapped.event_manager.get_term_cfg(term_name)
                    if hasattr(term_cfg, "params") and 'disable_flag' in term_cfg.params:
                        term_cfg.params['disable_flag'] = False
                    self.env.unwrapped.event_manager.set_term_cfg(term_name, term_cfg)

        for group_name in self.env.unwrapped.observation_manager.active_terms:
            obs_group_names = self.env.unwrapped.observation_manager._group_obs_term_names[group_name]
            # obs_group_terms = self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name]

            for i in range(len(obs_group_names)):
                name = obs_group_names[i]
                if name in self.obs_noise_cfg[group_name]:
                    self.env.unwrapped.observation_manager._group_obs_term_cfgs[group_name][i].noise = self.obs_noise_cfg[group_name][name]
        self.obs_noise_cfg = {}
