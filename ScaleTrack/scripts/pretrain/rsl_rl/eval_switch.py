"""Evaluate two XL policies on identical, causally scheduled, held-out raw reference switches.

Run from ScaleTrack with --motion_file, --baseline, --candidate and --output. Use
--plan-only to inspect case coverage without launching Isaac Sim or loading policies.
The report contains motion keys and individual case definitions; keep it with private run outputs.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from switch_eval_protocol import Protocol, build_cases, entry_table, paired_summary, summarize_trial


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as out:
        json.dump(report, out, allow_nan=False, indent=2)


def contact_metrics(view, position, linear, angular, dt):
    """Force-weighted speed of material points in contact with the static plane, per foot."""
    import torch

    normal_force, points, normals, _, counts, starts = view.get_contact_data(dt=dt)
    counts, starts = counts.flatten().long(), starts.flatten().long()
    rows = torch.arange(len(counts), device=counts.device).repeat_interleave(counts)
    load = linear.new_zeros(len(counts))
    weighted_speed = torch.zeros_like(load)
    if len(rows):
        local = torch.arange(len(rows), device=rows.device) - (counts.cumsum(0) - counts)[rows]
        indices = starts[rows] + local
        fz = (normal_force.reshape(-1)[indices] * normals[indices, 2]).abs()
        contact_v = linear[rows] + torch.linalg.cross(angular[rows], points[indices] - position[rows])
        weighted_speed.index_add_(0, rows, fz * torch.linalg.vector_norm(contact_v[:, :2], dim=-1))
        load.index_add_(0, rows, fz)
    in_stance = load > 5.0
    speed = torch.where(in_stance, weighted_speed / load.clamp_min(1e-6), 0.0)
    return speed, in_stance.float(), load


class PhysicsSamples:
    def __init__(self, env, cmd):
        import torch
        from isaaclab.sim.utils import get_first_matching_child_prim
        from isaacsim.core.simulation_manager import SimulationManager

        self.env, self.cmd = env, cmd
        ground = get_first_matching_child_prim(env.cfg.scene.terrain.prim_path, lambda prim: prim.GetTypeName() == "Plane")
        if not ground:
            raise RuntimeError("Ground collider for foot contact measurement was not found")
        sim = SimulationManager.get_physics_sim_view()
        self.feet = []
        prefix = env.cfg.scene.contact_forces.prim_path.rsplit("/", 1)[0].replace("{ENV_REGEX_NS}", env.scene.env_regex_ns)
        for link in ("left_ankle_roll_link", "right_ankle_roll_link"):
            view = sim.create_rigid_contact_view(f"{prefix}/{link}".replace(".*", "*"),
                                                 filter_patterns=[str(ground.GetPath())],
                                                 max_contact_data_count=64 * env.num_envs)
            if view.sensor_count != env.num_envs or view.filter_count != 1:
                raise RuntimeError(f"Foot-ground contact view has unexpected shape for {link}")
            self.feet.append((cmd.robot.body_names.index(link), view))
        self.values = torch.zeros((env.num_envs, 5), device=env.device)
        self.fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_acc = None
        self.previous_torque = None
        self.update = env.scene.update

    def sample_after_physics(self, dt):
        import torch

        self.update(dt)
        data = self.cmd.robot.data
        force = torch.zeros_like(self.values[:, 2])
        for foot, view in self.feet:
            speed, stance, fz = contact_metrics(view, data.body_com_pos_w[:, foot], data.body_com_lin_vel_w[:, foot],
                                                data.body_com_ang_vel_w[:, foot], dt)
            self.values[:, 0] += speed * dt
            self.values[:, 1] += stance * dt
            force += fz
        self.values[:, 2] = torch.maximum(self.values[:, 2], force)
        acc = data.body_com_lin_acc_w[:, self.cmd.robot_anchor_body_index]
        torque = data.applied_torque
        if self.previous_acc is not None:
            jerk = torch.linalg.vector_norm(acc - self.previous_acc, dim=-1) / dt
            rate = ((torque - self.previous_torque).square().mean(-1)).sqrt() / dt
            self.values[:, 3] = torch.maximum(self.values[:, 3], jerk)
            self.values[:, 4] = torch.maximum(self.values[:, 4], rate)
        self.previous_acc, self.previous_torque = acc.clone(), torque.clone()
        height = self.cmd.robot_anchor_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
        self.fell |= (height < 0.35) | (data.projected_gravity_b[:, 2] > -0.20)


def tracking_error(cmd):
    import torch

    def yaw(q):
        w, x, y, z = q.unbind(dim=-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    angle = yaw(cmd.robot_anchor_quat_w) - yaw(cmd.anchor_quat_w)
    return torch.stack((
        torch.linalg.vector_norm(cmd.robot_body_pos_w - cmd.body_pos_w, dim=-1).mean(-1),
        torch.atan2(torch.sin(angle), torch.cos(angle)).abs(),
        torch.linalg.vector_norm(cmd.robot_anchor_lin_vel_w[:, :2] - cmd.anchor_lin_vel_w[:, :2], dim=-1),
    ), dim=1)


def braking_state(cmd):
    import torch

    return torch.cat((cmd.anchor_pos_w[:, :2], cmd.robot_anchor_pos_w[:, :2],
                      cmd.robot_anchor_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)), dim=1)


def rollout(env, cmd, sensors, policy, batch, seed, p, paired_start=None):
    import torch

    real_count = len(batch)
    cases = batch + [batch[-1]] * (env.num_envs - real_count)
    ids = torch.arange(env.num_envs)
    env.unwrapped.seed(seed)
    cmd.set_reset_reference(ids, [c["a"] for c in cases], [c["a_frame"] for c in cases])
    obs, _ = env.reset()
    critic_delta = {}
    if paired_start is not None:
        start_state = dict(obs.items(), reset_root=cmd.robot.data.root_state_w,
                           reset_joint_pos=cmd.robot_joint_pos, reset_joint_vel=cmd.robot_joint_vel)
        if not paired_start:
            paired_start.update({key: value.detach().clone() for key, value in start_state.items()})
        else:
            for key, value in start_state.items():
                if key in ("critic", "critic_task"):
                    # PhysX updates derived non-root link velocities on its first physics step.
                    # These privileged history values are never consumed by the inference actor.
                    critic_delta[key] = (value - paired_start[key]).abs().flatten(1).amax(1).cpu().tolist()
                else:
                    torch.testing.assert_close(value, paired_start[key], rtol=1e-4, atol=1e-4,
                                               msg=lambda msg: f"Paired evaluation reset differs in {key}. {msg}")
    expected_ids = torch.tensor([c["a"] for c in cases])
    expected_frame = torch.tensor([c["a_frame"] for c in cases])
    if not (torch.equal(cmd.motion_ids.cpu(), expected_ids) and torch.equal(cmd.time_steps.cpu(), expected_frame)):
        raise RuntimeError("The requested initial references were not installed by the environment reset")
    scheduled = {}
    for slot, case in enumerate(cases):
        for step, motion, frame, yaw in case["switches"]:
            scheduled.setdefault(step, []).append((slot, motion, frame, yaw))
    last_cut = max(scheduled, default=p.warmup)
    length = last_cut + p.horizon + 1
    trace = torch.zeros(length, env.num_envs, 13, device=env.device)
    trace[0, :, 8:] = braking_state(cmd)
    trace[0, :, :3] = tracking_error(cmd)
    ended = torch.zeros(env.num_envs, dtype=torch.bool)
    last = torch.full((env.num_envs,), length - 1)
    fallen = torch.zeros_like(ended)
    sensors.previous_acc = sensors.previous_torque = None
    with patch.object(env.unwrapped.scene, "update", sensors.sample_after_physics):
        for t in range(1, length):
            expected_frame += 1
            targets = [entry for entry in scheduled.get(t, []) if not ended[entry[0]]]
            if targets:
                slots, motions, frames, yaws = zip(*targets)
                cmd.queue_switch(torch.tensor(slots), torch.tensor(motions), torch.tensor(frames), torch.tensor(yaws))
                expected_ids[list(slots)] = torch.tensor(motions)
                expected_frame[list(slots)] = torch.tensor(frames)
            sensors.values.zero_()
            sensors.fell.zero_()
            obs, _, done, _ = env.step(policy(obs.to(env.device)))
            trace[t, :, 3:8] = sensors.values
            done = done.cpu().bool()
            fallen |= ~ended & sensors.fell.cpu()
            last[~ended & done] = t
            ended |= done
            active = ~ended
            if (not torch.equal(cmd.motion_ids.cpu()[active], expected_ids[active])
                    or not torch.equal(cmd.time_steps.cpu()[active], expected_frame[active])):
                raise RuntimeError("Observed command clock or clip differs from the paired switch schedule")
            trace[t, :, :3] = tracking_error(cmd)
            trace[t, :, 8:] = braking_state(cmd)
            trace[t, ended.to(env.device), 8:] = float("nan")
            # Auto-reset observations belong to the next episode. Terminal substep physics above is still valid.
            trace[t, ended.to(env.device), :3] = float("nan")
            if ended[:real_count].all():
                break
    values = trace.cpu().numpy()
    rows = [summarize_trial(values[:int(last[i]) + 1, i], p, last_cut, ended[i], fallen[i], cases[i]["stratum"])
            for i in range(real_count)]
    for key, deltas in critic_delta.items():
        for row, delta in zip(rows, deltas):
            row[f"paired_reset_{key}_max_abs_delta"] = delta
    return rows


def run(args, motions, plans, report, p):
    import gymnasium as gym
    import torch

    import scaletrack.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

    task = "G1-BFM-Transformer-Switch-Tracking"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.seed, cfg.scene.num_envs, cfg.sim.device = args.seed, args.num_envs, args.device
    cfg.commands.motion.motion_file = str(args.motion_file)
    cfg.commands.motion.test_motion_file = ""
    cfg.commands.motion.automatic_switches = cfg.commands.motion.enable_reset_disturbance = False
    cfg.commands.motion.debug_vis = False
    cfg.commands.motion.rand_timestep_range = (16, 17)
    cfg.observations.policy.enable_corruption = cfg.observations.policy_task.enable_corruption = False
    cfg.events.push_robot = None
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
    try:
        cmd = env.unwrapped.command_manager.get_term("motion")
        if abs(env.unwrapped.step_dt - 0.02) > 1e-7 or list(cmd.motion_names) != list(motions):
            raise RuntimeError("Holdout ID order or control frequency differs from the case plan")
        agents = {}
        for name, checkpoint in (("raw", args.baseline), ("candidate", args.candidate)):
            cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
            cfg.num_steps_per_env = 1
            cfg.policy.embedding_dim, cfg.policy.num_heads, cfg.policy.ff_dim, cfg.policy.num_layers = 384, 6, 384, 6
            runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device=env.device)
            runner.load(str(checkpoint), load_optimizer=False, map_location="cpu")
            agents[name] = runner.get_inference_policy(device=env.device)
        sensors = PhysicsSamples(env.unwrapped, cmd)
        rows = []
        with torch.inference_mode():
            for stratum, cases in plans.items():
                for start in range(0, len(cases), args.num_envs):
                    batch = cases[start:start + args.num_envs]
                    paired_start = {}
                    results = {name: rollout(env, cmd, sensors, actor, batch, args.seed + start, p, paired_start)
                               for name, actor in agents.items()}
                    rows.extend({"case": case, "raw": base, "candidate": trained}
                                for case, base, trained in zip(batch, results["raw"], results["candidate"]))
                print(f"Completed {stratum}: {len(cases)} paired cases", flush=True)
        report.update(status="evaluated", rows=rows, summary=paired_summary(rows, p),
                      baseline_sha256=digest(args.baseline), candidate_sha256=digest(args.candidate))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_file", type=Path, required=True, help="Held-out split YAML, paths relative to ScaleTrack")
    parser.add_argument("--baseline", type=Path, help="Frozen released XL checkpoint")
    parser.add_argument("--candidate", type=Path, help="Fine-tuned XL checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New local JSON path")
    parser.add_argument("--seed", type=int, default=81017)
    parser.add_argument("--per_stratum", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--plan-only", action="store_true")
    initial, _ = parser.parse_known_args()
    if not initial.plan_only:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.per_stratum <= 0 or args.num_envs <= 0 or args.output.exists():
        parser.error("Use positive sample counts and a new output path")
    if not args.plan_only and not all(path and path.is_file() for path in (args.baseline, args.candidate)):
        parser.error("Evaluation requires existing --baseline and --candidate checkpoints")
    p = Protocol()
    motions = yaml.safe_load(args.motion_file.read_text())
    plans = build_cases(entry_table(motions, p), args.seed, args.per_stratum, p, list(motions))
    report = dict(status="plan_only", seed=args.seed, protocol=asdict(p), split_sha256=digest(args.motion_file),
                  ordered_keys=list(motions), plans=plans, requested_per_stratum=args.per_stratum,
                  coverage={name: len(cases) for name, cases in plans.items()})
    if not args.plan_only:
        app = AppLauncher(args).app
        run(args, motions, plans, report, p)
        write_report(args.output, report)
        app.close()
    else:
        write_report(args.output, report)
    print(json.dumps(report["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
