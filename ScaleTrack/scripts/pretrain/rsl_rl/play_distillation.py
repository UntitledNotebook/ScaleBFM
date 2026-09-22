"""Play a distilled PULSE checkpoint without loading its teacher."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", required=True, help="Path to a student checkpoint.")
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import scaletrack.tasks  # noqa: F401
from scaletrack.utils.distillation_wrapper import DistillationVecEnvWrapper
from my_rsl_rl.modules import VAEPolicy


checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=True)
if checkpoint["format_version"] != 1:
    raise ValueError(f"Unsupported distillation checkpoint version: {checkpoint['format_version']}")


@hydra_task_config(checkpoint["task_name"], "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.enable_reset_disturbance = False
    for group in vars(env_cfg.observations).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    for term in vars(env_cfg.terminations).values():
        if hasattr(term, "params") and "disable_flag" in term.params:
            term.params["disable_flag"] = True
    for term in vars(env_cfg.events).values():
        if getattr(term, "mode", None) in ("reset", "interval") and "disable_flag" in term.params:
            term.params["disable_flag"] = True

    env = gym.make(checkpoint["task_name"], cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.unwrapped.command_manager.get_term("motion").is_evaluating = True
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = DistillationVecEnvWrapper(RslRlVecEnvWrapper(env), checkpoint["observation_config"])
    env.validate_action_config(checkpoint["action_config"])
    policy = VAEPolicy.from_checkpoint(checkpoint, device=env.device)

    obs, _ = env.reset()
    timestep = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy.act_inference(obs)
            obs, _, _, _ = env.step(actions)
            timestep += 1
            if args_cli.video and timestep >= args_cli.video_length:
                break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
