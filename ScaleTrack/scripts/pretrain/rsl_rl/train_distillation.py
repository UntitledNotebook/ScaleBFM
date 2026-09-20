"""Train a PULSE student with a frozen ScaleTrack tracking teacher."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-BFM-Distillation-Tracking")
parser.add_argument("--teacher_checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments per process.")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run_name", default=None)
parser.add_argument("--experiment_name", default=None)
parser.add_argument("--logger", choices=("tensorboard", "wandb", "neptune"), default=None)
parser.add_argument("--log_project_name", default=None)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
# AppLauncher pre-parses arguments, so register required inputs afterward.
parser.add_argument("--motion_file", required=True)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import scaletrack.tasks  # noqa: F401
from scaletrack.tasks.tracking.config.g1_29dof.agents.rsl_rl_distillation_cfg import G1DistillationRunnerCfg
from scaletrack.utils.distillation_wrapper import DistillationVecEnvWrapper
from my_rsl_rl.runners import DistillationRunner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: G1DistillationRunnerCfg):
    for name in ("max_iterations", "seed", "run_name", "experiment_name", "logger"):
        value = getattr(args_cli, name)
        if value is not None:
            setattr(agent_cfg, name, value)
    if args_cli.teacher_checkpoint is not None:
        agent_cfg.teacher.checkpoint = args_cli.teacher_checkpoint
    if args_cli.log_project_name is not None:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    agent_cfg.task_name = args_cli.task

    rank = int(os.getenv("RANK", "0"))
    if args_cli.distributed:
        device = f"cuda:{app_launcher.local_rank}"
        env_cfg.sim.device = agent_cfg.device = device
        torch.cuda.set_device(app_launcher.local_rank)
        # Motion loading uses collectives during environment construction.
        torch.distributed.init_process_group(backend="nccl")
    elif int(os.getenv("WORLD_SIZE", "1")) > 1:
        raise ValueError("Use --distributed when launching with torchrun.")
    agent_cfg.seed += rank
    env_cfg.seed = agent_cfg.seed
    torch.manual_seed(agent_cfg.seed)

    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, agent_cfg.run_name))
    if rank == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video and rank == 0:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = DistillationVecEnvWrapper(RslRlVecEnvWrapper(env), agent_cfg.observation.to_dict())
    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    if args_cli.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
    simulation_app.close()
