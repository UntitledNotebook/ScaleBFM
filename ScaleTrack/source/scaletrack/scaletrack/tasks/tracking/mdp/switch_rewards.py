"""Switch-only objectives. Rewards are evaluated after physics, before command advancement."""

import torch
from isaaclab.utils.math import quat_apply

from . import rewards
from .switch_math import excess_impact


def switch_tracking(env, command_name: str, std: float, target: str):
    fn = {
        "height": rewards.global_body_height, "pos": rewards.global_body_pos,
        "rot": rewards.global_body_rot, "lin_vel": rewards.global_body_lin_vel,
        "ang_vel": rewards.global_body_ang_vel,
    }[target]
    fraction = env.command_manager.get_term(command_name).recovery_fraction
    return fn(env, command_name, std) * (0.25 + 0.75 * fraction)


def switch_body_error(env, command_name: str, threshold: float, disable_flag: bool = False):
    cmd = env.command_manager.get_term(command_name)
    relaxed = threshold + 0.7 * (1 - cmd.recovery_fraction)
    error = (cmd.robot_body_pos_w - cmd.body_pos_w).norm(dim=-1).amax(dim=-1)
    return (error > relaxed) & (not disable_flag)


def switch_fall(env, command_name: str, disable_flag: bool = False):
    cmd = env.command_manager.get_term(command_name)
    pelvis_height = cmd.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    fallen = (pelvis_height < 0.35) | (cmd.robot.data.projected_gravity_b[:, 2] > -0.20)
    return (cmd.switch_age >= 0) & fallen & (not disable_flag)


def switch_slip(env, command_name: str, asset_cfg, sensor_cfg):
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name].data
    # Rigid point velocity at the lowest of four sole landmarks (heel/toe edges).
    # Gating uses simulated ankle normal force. This follows the physical stance
    # foot even when a B reference calls for the opposite support.
    ids = asset_cfg.body_ids
    q = robot.body_link_quat_w[:, ids]
    local = q.new_tensor([[-.054, -.018, -.035], [-.054, .018, -.035],
                          [.132, -.018, -.035], [.132, .018, -.035]])
    offsets = quat_apply(q.unsqueeze(2).expand(-1, -1, 4, -1), local.expand(*q.shape[:2], -1, -1))
    lowest = offsets[..., 2].argmin(-1)
    point = offsets.gather(2, lowest[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
    velocity = robot.body_link_lin_vel_w[:, ids] + torch.linalg.cross(robot.body_link_ang_vel_w[:, ids], point)
    force = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    slip = ((force > 15) * velocity[..., :2].norm(dim=-1)).sum(dim=-1)
    cmd.metrics["stance_sole_speed_sum_mps"] = slip
    return slip * (1 - cmd.recovery_fraction)


def switch_impact(env, command_name: str, sensor_cfg):
    cmd = env.command_manager.get_term(command_name)
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    excess = excess_impact(forces, cmd.body_weight)
    cmd.metrics["ankle_peak_excess_bw"] = excess
    return excess * (1 - cmd.recovery_fraction)


def switch_torque_change(env, command_name: str):
    cmd = env.command_manager.get_term(command_name)
    # Isaac's implicit PD actuator exposes its clipped torque estimate.
    delta = cmd.robot.data.applied_torque - cmd.previous_torque
    rms = delta.square().mean(dim=-1).sqrt() * cmd._previous_torque_valid
    cmd.metrics["pd_torque_change_rms_nm"] = rms
    return (rms / 20) * (1 - cmd.recovery_fraction)
