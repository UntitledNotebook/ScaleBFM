import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from eval_switch import contact_metrics, rollout
from switch_eval_protocol import Protocol, build_cases, paired_summary, summarize_trial


def test_phase_plan_is_paired_and_conditioned():
    p = Protocol()
    mids = np.repeat([0, 1, 2], 4)
    frames = np.tile([40, 60, 80, 100], 3)
    stance = np.tile([-1, 1, 0, 1], 3)
    speed = np.tile([1.1, 0.8, 0.0, 1.2], 3)
    brake = np.tile([False, True, False, False], 3)
    table = mids, frames, np.zeros((12, 29)), stance, speed, np.zeros(12), brake
    names = ["original__a", "lch__b", "lafan__c"]
    plan = build_cases(table, 31, 4, p, names)
    assert plan == build_cases(table, 31, 4, p, names)
    assert all(len(cases) == 4 for cases in plan.values())
    for clean, sham in zip(plan["clean"], plan["sham"]):
        assert (clean["a"], clean["a_frame"]) == (sham["a"], sham["a_frame"])
        assert clean["switches"] == sham["switches"] == []
    for name, cases in plan.items():
        for case in cases:
            if not case["switches"]:
                continue
            _, b, bframe, yaw = case["switches"][0]
            arow = np.flatnonzero((mids == case["a"]) & (frames == case["a_frame"] + p.warmup))[0]
            brow = np.flatnonzero((mids == b) & (frames == bframe))[0]
            assert case["a"] != b
            if name == "opposite_support":
                assert stance[arow] * stance[brow] == -1
            if name == "running_brake":
                assert brake[brow] and speed[arow] >= 1.0
            if name == "sharp_turn":
                assert abs(yaw) == pytest.approx(np.pi / 2)
            if name == "rapid":
                assert case["switches"][1][0] - p.warmup == p.rapid_gap
    no_brakes = (*table[:-1], np.zeros_like(brake))
    assert build_cases(no_brakes, 31, 4, p, names)["running_brake"] == []


def test_delay_retention_terminal_and_no_contact():
    p = Protocol()
    t = np.zeros((p.warmup + p.horizon + 1, 8))
    assert summarize_trial(t, p, p.warmup, False, False)["deadline_success"]
    # A robot can look smooth while missing B for too long.
    t[p.warmup:p.warmup + p.deadline, 2] = 1
    slow = summarize_trial(t, p, p.warmup, False, False)
    assert not slow["deadline_success"] and slow["convergence_confirmed_s"] > 1
    assert slow["stance_slip_m_s"] is None
    t[:] = 0
    t[-1, 1] = 1
    assert not summarize_trial(t, p, p.warmup, False, False)["deadline_success"]
    terminal = t[:p.warmup + 5].copy()
    terminal[-1, :3] = np.nan  # returned auto-reset observation
    terminal[-1, 3:] = [0.02, 0.02, 1400.0, 300, 50]
    failed = summarize_trial(terminal, p, p.warmup, True, True)
    assert failed["mpjpe_m"] == 0 and failed["penalized_mpjpe_m"] > 0.4
    assert failed["peak_vertical_grf_N"] == 1400 and failed["stance_slip_m_s"] == 1
    stats = paired_summary([{"case": {"stratum": "clean"}, "raw": slow, "candidate": failed}], p)
    assert not stats["clean"]["within_observed_clean_margins"]
    assert stats["clean"]["stance_slip_m_s"]["paired_count"] == 0


def test_contact_point_velocity_and_buffer_offsets():
    class ContactView:
        def get_contact_data(self, dt):
            forces = torch.tensor([0, 40, 80, 1], dtype=torch.float)
            points = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 0, -1], [0, 0, -1]], dtype=torch.float)
            normals = torch.tensor([[0, 0, 1]], dtype=torch.float).expand(4, -1)
            return forces, points, normals, None, torch.tensor([[1], [1], [1]]), torch.tensor([[2], [1], [3]])

    linear = torch.tensor([[1, 0, 0], [0.5, 0, 0], [2, 0, 0]], dtype=torch.float)
    angular = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float)
    slip, stance, force = contact_metrics(ContactView(), torch.zeros_like(linear), linear, angular, 0.005)
    # A foot rolling around a stationary contact point has a moving ankle and zero slip.
    torch.testing.assert_close(slip, torch.tensor([0, 0.5, 0]))
    torch.testing.assert_close(stance, torch.tensor([1, 1, 0.]))
    torch.testing.assert_close(force, torch.tensor([80, 40, 1.]))


def test_rollout_ignores_reset_state_and_executes_second_switch(monkeypatch):
    import eval_switch
    from types import SimpleNamespace

    p = Protocol(warmup=2, rapid_gap=2, horizon=6, hold=2, deadline=4)
    plan = [dict(stratum="rapid", a=0, a_frame=6, switches=[[2, 1, 20, 0], [4, 2, 30, 0]])] * 2

    class Command:
        def set_reset_reference(self, ids, mids, frames):
            self.motion_ids, self.time_steps = torch.tensor(mids), torch.tensor(frames)
            self.pending = None

        def queue_switch(self, ids, mids, frames, yaws):
            self.pending = ids, mids, frames

    cmd = Command()
    sensors = SimpleNamespace(values=torch.ones(2, 5), fell=torch.zeros(2, dtype=torch.bool),
                              sample_after_physics=lambda dt: None)

    class Env:
        num_envs = 2
        device = "cpu"

        def __init__(self):
            self.unwrapped = SimpleNamespace(seed=lambda x: None, scene=SimpleNamespace(update=lambda dt: None))

        def reset(self):
            self.step_no = 0
            return torch.zeros(2), {}

        def step(self, action):
            self.step_no += 1
            cmd.time_steps += 1
            if cmd.pending is not None:
                ids, mids, frames = cmd.pending
                cmd.motion_ids[ids], cmd.time_steps[ids] = mids, frames
                cmd.pending = None
            done = torch.tensor([self.step_no == 3, False])
            cmd.time_steps[done] = 0
            sensors.fell.copy_(done)
            sensors.values.fill_(1)
            return torch.zeros(2), None, done, {}

    monkeypatch.setattr(eval_switch, "tracking_error", lambda c: torch.zeros(2, 3))
    monkeypatch.setattr(eval_switch, "braking_state", lambda c: torch.zeros(2, 5))
    failed, stable = rollout(Env(), cmd, sensors, lambda x: x, plan, 1, p)
    assert failed["fell"] and failed["post_samples"] == 0
    assert stable["deadline_success"] and stable["post_samples"] == p.horizon + 1


def test_braking_requires_motion_and_measured_braking_distance():
    p = Protocol()
    t = np.zeros((p.warmup + p.horizon + 1, 13))
    t[p.warmup:p.warmup + 37, 8] = np.linspace(0, 0.2, 37)
    t[p.warmup + 37:, 8] = 0.2
    t[p.warmup, 12] = 1
    # Tracking thresholds alone pass on this constructed trace. A frozen plant cannot pass braking.
    assert not summarize_trial(t, p, p.warmup, False, False, "running_brake")["deadline_success"]
    t[:, 10] = t[:, 8]
    valid = summarize_trial(t, p, p.warmup, False, False, "running_brake")
    assert valid["deadline_success"] and valid["braking_actual_along_reference_m"] == pytest.approx(0.2)
