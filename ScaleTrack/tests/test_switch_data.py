from collections import defaultdict
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/data_process"))
import build_switch_dataset as data


def test_actor_and_synchronous_take_are_isolated_after_bridge_tracks_removed():
    takes = [data.Take("source", Path(f"{k}_{a}.npz"), a, k, "")
             for k, cast in {"a": "xy", "b": "xz", "c": "yz", "d": "x", "e": "y"}.items()
             for a in cast]
    selected = data.group_small(takes, 42)
    assert selected == data.group_small(list(reversed(takes)), 42)
    assert 3 <= len(selected) < len(takes)
    assert set(selected.values()) == set(data.SPLITS)
    for attr in ("actor", "capture"):
        by_id = defaultdict(set)
        for t in takes:
            if t.key in selected:
                by_id[getattr(t, attr)].add(selected[t.key])
        assert all(len(folds) == 1 for folds in by_id.values())


def test_named_mujoco_conversion_uses_world_com_and_wxyz():
    n = 14
    joints = np.tile(np.arange(29), (n, 1))
    angle = np.arange(n) / data.FPS
    root = np.column_stack((angle, np.zeros(n), np.full(n, .8)))
    quat = Rotation.from_euler("z", angle).as_quat()[:, [3, 0, 1, 2]]
    names = ["world", *reversed(data.BODIES)]
    xpos = np.zeros((n, 31, 3))
    pelvis = names.index("pelvis")
    xpos[:, pelvis] = root
    quat_body = np.repeat(quat[:, None], 31, axis=1)
    # Quaternion sign changes represent the same pose and angular speed.
    quat_body[1::2] *= -1
    raw = dict(frequency=50, split_points=np.array([0, n]), joint_names=np.array(["root", *data.MJ_JOINTS]),
               body_names=np.array(names), qpos=np.column_stack((root, quat, joints)),
               xpos=xpos, xquat=quat_body, body_ipos=np.repeat([[0.1, 0, 0]], 31, axis=0))
    arr = data.lafan_arrays(raw)
    np.testing.assert_array_equal(arr["joint_pos"], joints[:, data.BFS])
    np.testing.assert_allclose(arr["body_ang_vel_w"][:, 0, 2], 1)
    np.testing.assert_allclose(arr["body_lin_vel_w"][1:-1, 0, 1], .1 * np.cos(angle[1:-1]), atol=1e-4)
    assert data.quality(arr)["speed_p95_mps"] == pytest.approx(1.)
    raw["split_points"] = [0, 7, n]
    with pytest.raises(ValueError, match="internal splits"):
        data.lafan_arrays(raw)


def test_brake_requires_motion_settling_and_nonzero_travel():
    n = 170
    arr = {k: np.zeros((n, *shape)) for k, shape in data.FIELDS.items()}
    arr["body_quat_w"][..., 0] = 1
    arr["body_pos_w"][:, :, 0] = (np.minimum(np.arange(n), 120) / data.FPS)[:, None]
    assert data.quality(arr)["braking_path_m"] > .01
    arr["body_pos_w"][:] = 0
    assert data.quality(arr)["braking_path_m"] is None
    arr["joint_vel"][3, 0] = np.nan
    with pytest.raises(ValueError, match="joint_vel"):
        data.quality(arr)
