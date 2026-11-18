"""Paired 50 Hz switch evaluation protocol. All distances and times are SI."""

from dataclasses import dataclass
from math import pi

import numpy as np


@dataclass(frozen=True)
class Protocol:
    warmup: int = 25
    deadline: int = 50
    hold: int = 10
    horizon: int = 65
    rapid_gap: int = 12
    position_m: float = 0.15
    heading_rad: float = 0.35
    velocity_m_s: float = 0.35
    clean_margin_m: float = 0.015
    clean_failure_margin: float = 0.02


STRATA = ("clean", "sham", "matched", "opposite_support", "random_phase", "sharp_turn", "running_brake", "rapid")


def entry_table(motions, p):
    """Index actual clip phases. Support labels are kinematic proxies used only to choose cases."""
    table = []
    for motion_id, path in enumerate(motions.values()):
        with np.load(path, allow_pickle=False) as clip:
            if np.asarray(clip["fps"]).item() != 50:
                raise ValueError("Switch evaluation requires native 50 Hz inputs")
            size = len(clip["joint_pos"])
            # Every initial clip, B and C has room for warm-up, both windows and final XL lookahead.
            frames = np.arange(p.warmup, size - p.horizon - p.rapid_gap - 33, 5)
            if not len(frames):
                continue
            if len(frames) > 512:
                frames = frames[np.linspace(0, len(frames) - 1, 512, dtype=int)]
            joint = clip["joint_pos"][frames]
            poses = clip["body_pos_w"]
            velocities = clip["body_lin_vel_w"]
            z = poses[frames][:, [18, 19], 2]
            foot_speed = np.linalg.norm(velocities[frames][:, [18, 19], :], axis=-1)
            # Unambiguous, slow, low stance ankle. Exclude near-level double support.
            left = (z[:, 0] + 0.035 < z[:, 1]) & (foot_speed[:, 0] < 0.45)
            right = (z[:, 1] + 0.035 < z[:, 0]) & (foot_speed[:, 1] < 0.45)
            support = left.astype(int) - right.astype(int)
            root = velocities[:, 0, :2]
            speed = np.linalg.norm(root, axis=1)
            quat = clip["body_quat_w"][frames, 0]
            w, x, y, zq = quat.T
            yaw = np.arctan2(2 * (w * zq + x * y), 1 - 2 * (y**2 + zq**2))
            local_angle = np.arctan2(root[frames, 1], root[frames, 0]) - yaw
            # Require a measured braking distance, a moving entry and a sustained stopped window.
            stopped = speed[frames[:, None] + np.arange(32, 42)].max(axis=1) < 0.30
            distance = np.linalg.norm(poses[frames + 36, 0, :2] - poses[frames, 0, :2], axis=1)
            brake = (speed[frames] >= 0.65) & stopped & (distance >= 0.12)
            table.append((np.full(len(frames), motion_id), frames, joint, support, speed[frames], local_angle, brake))
    if not table:
        raise ValueError("No held-out clip has enough native frames for this protocol")
    return tuple(np.concatenate(fields, axis=0) for fields in zip(*table))


def build_cases(table, seed, count, p, motion_names):
    mids, frames, joints, support, speed, angle, brake = table
    sources = np.array([key.split("__", 1)[0] for key in motion_names])[mids]
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(sources == source)).tolist() for source in np.unique(sources)]
    # Interleave sources so rare recordings can be tested; short and ambiguous clips stay excluded.
    anchors = []
    while groups and len(anchors) < max(1024, 16 * count):
        for group in groups:
            anchors.append(group.pop())
        groups = [group for group in groups if group]
    plans = {name: [] for name in STRATA}
    for name in STRATA:
        for a in anchors:
            case = {"stratum": name, "a": int(mids[a]), "a_frame": int(frames[a]) - p.warmup, "switches": []}
            if name not in ("clean", "sham"):
                eligible = mids != mids[a]
                yaw = 0.0
                if name == "matched":
                    eligible &= (support == support[a]) & (np.abs(speed - speed[a]) < 0.40)
                    if speed[a] >= 0.4:
                        eligible &= np.cos(angle - angle[a]) > np.cos(pi / 5)
                    idx = np.flatnonzero(eligible)
                    eligible[idx] = np.mean((joints[idx] - joints[a]) ** 2, axis=1) < 0.30**2
                elif name == "opposite_support":
                    eligible &= (support[a] != 0) & (support == -support[a])
                elif name == "sharp_turn":
                    yaw = float(rng.choice([-pi / 2, pi / 2]))
                    eligible &= (speed[a] > 0.65) & (speed > 0.65) & (np.cos(angle - angle[a] + yaw) < np.cos(pi / 3))
                elif name == "running_brake":
                    eligible &= (speed[a] >= 1.0) & brake & (np.abs(speed - speed[a]) < 0.6)
                    eligible &= np.cos(angle - angle[a]) > np.cos(pi / 4)
                available = np.flatnonzero(eligible)
                if not len(available):
                    continue
                target_source = rng.choice(np.unique(sources[available]))
                b = rng.choice(available[sources[available] == target_source])
                case["switches"].append([p.warmup, int(mids[b]), int(frames[b]), yaw])
                if name == "rapid":
                    c = rng.choice(np.flatnonzero(mids != mids[b]))
                    case["switches"].append([p.warmup + p.rapid_gap, int(mids[c]), int(frames[c]), 0.0])
            plans[name].append(case)
            if len(plans[name]) == count:
                break
    return plans


def summarize_trial(trace, p, last_cut, terminated, fall, stratum="clean"):
    """Track the unsmoothed active target. First confirmed hold must finish by the deadline."""
    pos, heading, velocity, slip, contact, force, jerk, torque = trace[:, :8].T
    post = slice(last_cut, last_cut + p.horizon + 1)
    pos, heading, velocity = pos[post], heading[post], velocity[post]
    inside = (pos <= p.position_m) & (heading <= p.heading_rad) & (velocity <= p.velocity_m_s)
    confirm = None
    streak = 0
    for t, good in enumerate(inside):
        streak = streak + 1 if good else 0
        if streak >= p.hold:
            confirm = t / 50
            break
    # Clipping and imputing missing frames prevents early terminations looking accurate.
    slots = p.horizon + 1
    observed = pos[np.isfinite(pos)]
    penalized = (np.minimum(observed, 0.5).sum() + 0.5 * (slots - len(observed))) / slots
    # Burst metrics start at the first action under B, include a second switch and terminal physics.
    burst = slice(p.warmup + 1, last_cut + p.horizon + 1)
    valid_force, valid_jerk, valid_torque = force[burst], jerk[burst], torque[burst]

    def peak(values):
        return float(np.max(values)) if len(values) else None

    # A stationary body can be within a loose MPJPE threshold on short braking segments.
    # Require initial movement, controlled travel along the original B and an actual stop.
    brake_ref_m = brake_actual_m = None
    brake_ok = True
    if stratum == "running_brake":
        window = trace[last_cut:last_cut + 42, 8:13]
        brake_ok = False
        if len(window) == 42 and np.isfinite(window).all():
            ref = window[36, :2] - window[0, :2]
            brake_ref_m = float(np.linalg.norm(ref))
            actual = window[36, 2:4] - window[0, 2:4]
            brake_actual_m = float(actual @ ref / max(brake_ref_m, 1e-6))
            brake_ok = (brake_ref_m >= 0.119 and window[0, 4] >= 0.65 and window[32:42, 4].max() <= 0.45
                        and 0.5 * brake_ref_m <= brake_actual_m <= 1.5 * brake_ref_m)

    first_window = trace[p.warmup:p.warmup + p.rapid_gap, :3]
    observed_first = first_window[np.isfinite(first_window).all(axis=1)]
    return {
        "terminated": bool(terminated), "fell": bool(fall), "post_samples": len(observed),
        "convergence_confirmed_s": confirm,
        "deadline_success": bool(confirm is not None and confirm <= p.deadline / 50 and len(inside) == slots
                                 and inside[-p.hold:].all() and brake_ok and not terminated and not fall),
        "mpjpe_m": float(observed.mean()) if len(observed) else None, "penalized_mpjpe_m": float(penalized),
        "early_velocity_error_m_s": float(observed_first[:, 2].mean()) if len(observed_first) else None,
        "early_mpjpe_m": float(observed_first[:, 0].mean()) if len(observed_first) else None,
        "braking_reference_travel_m": brake_ref_m, "braking_actual_along_reference_m": brake_actual_m,
        "peak_vertical_grf_N": peak(valid_force), "peak_pelvis_jerk_m_s3": peak(valid_jerk),
        "peak_pd_torque_rate_rms_Nm_s": peak(valid_torque),
        "slip_distance_m": float(slip[burst].sum()) if len(valid_force) else None,
        "foot_contact_s": float(contact[burst].sum()),
        "stance_slip_m_s": float(slip[burst].sum() / contact[burst].sum()) if contact[burst].sum() else None,
    }


def paired_summary(rows, p):
    summary = {}
    measures = ("terminated", "fell", "deadline_success", "convergence_confirmed_s", "mpjpe_m",
                "penalized_mpjpe_m", "early_velocity_error_m_s", "early_mpjpe_m", "braking_reference_travel_m",
                "braking_actual_along_reference_m", "peak_vertical_grf_N", "peak_pelvis_jerk_m_s3",
                "peak_pd_torque_rate_rms_Nm_s", "stance_slip_m_s", "slip_distance_m", "foot_contact_s")
    for name in STRATA:
        cases = [row for row in rows if row["case"]["stratum"] == name]
        stats = {"paired_cases": len(cases)}
        for metric in measures:
            values = {}
            for policy in ("raw", "candidate"):
                observed = [float(c[policy][metric]) for c in cases if c[policy][metric] is not None]
                values[policy] = {
                    "observed": len(observed), "mean": float(np.mean(observed)) if observed else None,
                    "p95": float(np.percentile(observed, 95)) if observed else None,
                }
            paired = [float(c["candidate"][metric]) - float(c["raw"][metric]) for c in cases
                      if c["candidate"][metric] is not None and c["raw"][metric] is not None]
            values["paired_count"] = len(paired)
            values["paired_delta_mean"] = float(np.mean(paired)) if paired else None
            stats[metric] = values
        if name == "clean" and cases:
            stats["within_observed_clean_margins"] = (
                stats["penalized_mpjpe_m"]["paired_delta_mean"] <= p.clean_margin_m
                and stats["terminated"]["paired_delta_mean"] <= p.clean_failure_margin
                and stats["fell"]["paired_delta_mean"] <= p.clean_failure_margin
            )
        summary[name] = stats
    return summary
