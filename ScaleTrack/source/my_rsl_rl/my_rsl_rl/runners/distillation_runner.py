from __future__ import annotations

import os
import time
from collections import deque
from copy import deepcopy

import torch
from tensordict import TensorDict

from my_rsl_rl.algorithms import Distillation
from my_rsl_rl.env import VecEnv
from my_rsl_rl.modules import ActorCriticHumanoidTransformer, VAEPolicy
from my_rsl_rl.utils import store_code_state


class DistillationRunner:
    """Collect student rollouts and distill a frozen tracking teacher."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = deepcopy(train_cfg)
        self.alg_cfg = self.cfg["algorithm"]
        self.policy_cfg = self.cfg["policy"]
        self.device = device
        self.env = env

        self._configure_multi_gpu()
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        obs = self.env.get_observations().to(self.device)
        self.teacher = self._construct_teacher(obs)
        self.alg = self._construct_algorithm(obs)

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)

        if self.is_distributed:
            self.alg.broadcast_parameters()
        if self.log_dir is not None and not self.disable_logs:
            for path in store_code_state(self.log_dir, self.git_status_repos):
                if self.logger_type in ("wandb", "neptune"):
                    self.writer.save_file(path)

        for it in range(num_learning_iterations):
            start = time.time()
            # Rollout: labels and student actions use the same pre-step state.
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    teacher_actions = self.teacher.act_inference(obs)
                    actions = self.alg.act(obs, teacher_actions)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    if self.log_dir is not None:
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        finished = dones.bool()
                        num_finished = int(finished.sum())
                        if num_finished:
                            ep_infos.append((extras.get("episode", extras.get("log", {})), num_finished))
                        rewbuffer.extend(cur_reward_sum[finished].tolist())
                        lenbuffer.extend(cur_episode_length[finished].tolist())
                        cur_reward_sum[finished] = 0
                        cur_episode_length[finished] = 0

            collection_time = time.time() - start
            start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it + 1

            if self.log_dir is not None:
                self.log(locals())  # All ranks contribute episode metrics.
                if not self.disable_logs and (it + 1) % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
            ep_infos.clear()

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.writer.flush()

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_time += iteration_time

        # Gather sums and counts so different episode counts do not bias rank averages.
        episode_metrics = {}
        for info, num_finished in locs["ep_infos"]:
            for key, value in info.items():
                tag = key if "/" in key else f"Episode/{key}"
                values = torch.as_tensor(value).float()
                total, count = episode_metrics.get(tag, (0.0, 0))
                episode_metrics[tag] = (total + values.mean().item() * num_finished, count + num_finished)
        for tag, buffer in (("Train/mean_reward", locs["rewbuffer"]), ("Train/mean_episode_length", locs["lenbuffer"])):
            episode_metrics[tag] = (sum(buffer), len(buffer))
        gathered_metrics = [episode_metrics]
        if self.is_distributed:
            gathered_metrics = [None] * self.gpu_world_size
            torch.distributed.all_gather_object(gathered_metrics, episode_metrics)
        if self.disable_logs:
            return

        metrics = {f"Loss/{key}": value for key, value in locs["loss_dict"].items()}
        for key in {key for rank_metrics in gathered_metrics for key in rank_metrics}:
            total = sum(rank_metrics.get(key, (0.0, 0))[0] for rank_metrics in gathered_metrics)
            count = sum(rank_metrics.get(key, (0.0, 0))[1] for rank_metrics in gathered_metrics)
            if count:
                metrics[key] = total / count
        fps = int(collection_size / iteration_time)
        metrics.update({
            "Perf/total_fps": fps,
            "Perf/collection_time": locs["collection_time"],
            "Perf/learning_time": locs["learn_time"],
        })
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.current_learning_iteration)

        title = f" Learning iteration {self.current_learning_iteration}/{locs['num_learning_iterations']} "
        log_string = f"{'#' * width}\n{title.center(width)}\n\n"
        log_string += f"{'Computation:':>{pad}} {fps} steps/s\n"
        for key, value in locs["loss_dict"].items():
            log_string += f"{key + ':':>{pad}} {value:.6f}\n"
        for key in ("Train/mean_reward", "Train/mean_episode_length"):
            if key in metrics:
                log_string += f"{key + ':':>{pad}} {metrics[key]:.2f}\n"
        log_string += f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
        print(log_string)

    def save(self, path: str) -> None:
        """Save only the policy and environment contract needed for playback."""
        torch.save({
            "format_version": 1,
            "model_state_dict": self.alg.policy.inference_state_dict(),
            "policy_config": self.alg.policy.policy_config,
            "observation_config": self.env.observation_config,
            "action_config": self.env.action_config,
            "task_name": self.cfg["task_name"],
        }, path)

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()
        self.teacher.eval()

    def eval_mode(self) -> None:
        self.alg.policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.is_distributed = self.gpu_world_size > 1
        self.multi_gpu_cfg = None
        if self.is_distributed:
            self.multi_gpu_cfg = {
                "global_rank": self.gpu_global_rank,
                "local_rank": self.gpu_local_rank,
                "world_size": self.gpu_world_size,
            }

    def _construct_teacher(self, obs: TensorDict) -> ActorCriticHumanoidTransformer:
        teacher_cfg = self.cfg["teacher"]
        policy_cfg = dict(teacher_cfg["policy"])
        teacher_class = {"ActorCriticHumanoidTransformer": ActorCriticHumanoidTransformer}[policy_cfg.pop("class_name")]
        teacher = teacher_class(obs, {"policy": ["policy"], "critic": ["critic"]}, self.env.num_actions, **policy_cfg)
        checkpoint = torch.load(teacher_cfg["checkpoint"], map_location="cpu", weights_only=True)
        teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
        teacher.requires_grad_(False)
        return teacher.to(self.device).eval()

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        policy_cfg = dict(self.policy_cfg)
        policy_class = {"VAEPolicy": VAEPolicy}[policy_cfg.pop("class_name")]
        policy = policy_class(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        alg_cfg = dict(self.alg_cfg)
        alg_class = {"Distillation": Distillation}[alg_cfg.pop("class_name")]
        alg = alg_class(policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **alg_cfg)
        alg.init_storage(self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def _prepare_logging_writer(self) -> None:
        if self.log_dir is None or self.disable_logs or self.writer is not None:
            return
        if self.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        elif self.logger_type == "wandb":
            from my_rsl_rl.utils.wandb_utils import WandbSummaryWriter

            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
        elif self.logger_type == "neptune":
            from my_rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

            self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
        else:
            raise ValueError(f"Unknown logger: {self.logger_type}")
        if self.logger_type in ("wandb", "neptune"):
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
