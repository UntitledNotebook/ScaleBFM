"""Torch-only geometry and entry selection for the 50 Hz motion command."""

import torch


def heading(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def yaw_coefficients(angle):
    return torch.stack((angle.cos(), angle.sin(), (angle / 2).cos(), (angle / 2).sin()), dim=-1)


def transform(value, field, yaw, translation):
    # Broadcast one fixed rigid transform across every lookahead and every tracked link.
    shape = (len(value),) + (1,) * (value.ndim - 2)
    c, s, cw, sz = (v.reshape(shape) for v in yaw.unbind(-1))
    if field == "body_quat_w":
        w, x, y, z = value.unbind(-1)
        return torch.stack((cw*w - sz*z, cw*x - sz*y, cw*y + sz*x, cw*z + sz*w), dim=-1)
    x, y, z = value.unbind(-1)
    mapped = torch.stack((c*x - s*y, s*x + c*y, z), dim=-1)
    if field == "body_pos_w":
        mapped += translation.reshape(*shape, 3)
    return mapped


def registration(a_position, a_quat, b_position, b_quat, turn):
    yaw = yaw_coefficients(heading(a_quat) + turn - heading(b_quat))
    translation = a_position - transform(b_position, "vector", yaw, None)
    # Both clips were retargeted on the same z=0 floor. Matching pelvis height
    # here would raise/lower the entire new floor during jumps or deep crouches.
    translation[:, 2] = 0
    return yaw, translation


def source_weights(names, masses):
    prefixes = [name.partition("__")[0] for name in names]
    # Original ScaleTrack identifiers can themselves contain "__" (e.g. take suffixes).
    groups = [prefix if prefix in masses else "original" for prefix in prefixes]
    counts = {group: groups.count(group) for group in set(groups)}
    if any(group not in masses for group in counts):
        raise ValueError("Unrecognized motion source prefix in the split manifest")
    weights = torch.tensor([masses[group] / counts[group] for group in groups])
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Source sampling masses must be finite, nonnegative and cover this split")
    return weights / weights.sum()


def support_side(foot_pos, foot_vel):
    # Kinematic proxy for retargeted reference clips, which have no force sensor.
    # Exclude flight/double support from the opposite-support curriculum.
    stance = (foot_pos[..., 2] < 0.11) & (foot_vel[..., :2].norm(dim=-1) < 0.45)
    return stance[..., 0].int() - stance[..., 1].int()


def recovery_fraction(age, duration):
    return torch.where(age < 0, 1.0, (age / duration).clamp(0.0, 1.0))


def excess_impact(force_history, body_weight):
    """Peak vertical force on either ankle over one policy step, in body weights."""
    peak = force_history[..., 2].clamp_min(0).amax(dim=(1, 2)) / body_weight
    return (peak - 1.8).clamp_min(0)
