# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_file", type=str, required=True, help="The name of the motion file.")
parser.add_argument("--test_motion_file", type=str, default="", help="The name of the motion file for test set.")
parser.add_argument("--distributed", action="store_true", default=False, help="Distributed training.")
parser.add_argument("--xl_checkpoint", help="Original XL weights used to initialize switch-aware PPO and its clean teacher.")
parser.add_argument("--clean_kl_coef", type=float, default=1.0, help="Clean teacher KL weight per action dimension.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
switch_training = args_cli.task == "G1-BFM-Transformer-Switch-Tracking"
if switch_training != bool(args_cli.xl_checkpoint):
    parser.error("The switch-aware task requires --xl_checkpoint; use that option only with this task.")
if args_cli.xl_checkpoint:
    args_cli.xl_checkpoint = str(Path(args_cli.xl_checkpoint).resolve(strict=True))

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import scaletrack.tasks  # noqa: F401

from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.test_motion_file = args_cli.test_motion_file

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + int(os.environ["RANK"])
        env_cfg.seed = seed
        agent_cfg.seed = seed

        gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        gpu_global_rank = int(os.getenv("RANK", "0"))
        gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        assert gpu_world_size > 1, f"You have enabled distributed training but there is only {gpu_world_size} GPUs detected!"
        
        # weishuai: We init communication at the very beginning instead of the runner as the initialization of some environment components may need sync.
        torch.cuda.set_device(gpu_local_rank)
        torch.distributed.init_process_group(backend="nccl", rank=gpu_global_rank, world_size=gpu_world_size)
        gpu_info = torch.cuda.get_device_properties(gpu_local_rank)
        print(f"[NCCL] rank={gpu_global_rank}/{gpu_world_size} local_rank={gpu_local_rank} "
              f"device=cuda:{torch.cuda.current_device()} gpu={gpu_info.name} uuid={gpu_info.uuid} "
              f"nccl={torch.cuda.nccl.version()}", flush=True)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    # log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # if agent_cfg.run_name:
    #     log_dir += f"_{agent_cfg.run_name}"
    log_dir = agent_cfg.run_name
    log_dir = os.path.join(log_root_path, log_dir)
    if switch_training and not agent_cfg.resume and any(Path(log_dir).glob("model_*.pt")):
        raise ValueError("Run already contains checkpoints. Choose a new --run_name or use --resume.")
    if switch_training and agent_cfg.logger not in {"tensorboard", "wandb"}:
        raise ValueError("The switch task supports tensorboard and scalar-only wandb logging.")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    if switch_training:
        runner.initialize_switch_finetuning(args_cli.xl_checkpoint, args_cli.clean_kl_coef)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()
    if runner.writer is not None:
        if runner.logger_type == "wandb":
            runner.writer.stop()
        runner.writer.close()
    if args_cli.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
