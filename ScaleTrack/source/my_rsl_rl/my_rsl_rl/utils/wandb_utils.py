# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs=flush_secs)
        self.scalar_metrics_only = cfg.get("scalar_metrics_only", False)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        if self.scalar_metrics_only:
            # Console output and SDK metadata can contain asset paths and launch arguments.
            settings = wandb.Settings(
                console="off", disable_git=True, disable_code=True, save_code=False,
                disable_job_creation=True, x_disable_meta=True, x_disable_machine_info=True,
                x_disable_stats=True, x_save_requirements=False,
            )
            run = wandb.init(project=project, entity=entity, name=run_name, settings=settings,
                             save_code=False, sync_tensorboard=False, monitor_gym=False)
            if run.url:
                print(f"[W&B] {run.url}", flush=True)
        else:
            wandb.init(project=project, entity=entity, name=run_name)
            wandb.config.update({"log_dir": log_dir})

    def store_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        if self.scalar_metrics_only:
            return
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        wandb.finish()

    def log_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path: str, iter: int) -> None:
        if not self.scalar_metrics_only:
            wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        if not self.scalar_metrics_only:
            wandb.save(path, base_path=os.path.dirname(path))
