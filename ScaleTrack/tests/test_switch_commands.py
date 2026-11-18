"""CPU command regressions. Only Kit's scene/manager interfaces are replaced."""
import importlib.util
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

TRACK = Path(__file__).resolve().parents[1]
MDP = TRACK / "source/scaletrack/scaletrack/tasks/tracking/mdp"
A_LOOK = [0, 1, 2, 3, 4, -1]
C_LOOK = [0, 1, 2, 4, 8, 16, 32]


@pytest.fixture
def code(monkeypatch):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    def module(name, **members):
        mod = ModuleType(name)
        mod.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("isolated_tracking", __path__=[])
    module("isaaclab.assets", Articulation=object)
    module("isaaclab.managers", CommandTerm=object, CommandTermCfg=object, SceneEntityCfg=object)
    module("isaaclab.sensors", ContactSensor=object)
    module("isaaclab.markers", VisualizationMarkers=object, VisualizationMarkersCfg=object)
    module("isaaclab.markers.config", FRAME_MARKER_CFG=MagicMock())
    module("isaaclab.utils", configclass=lambda cls: cls)
    # Isaac's actual quaternion math has no simulator dependency.
    path = Path(importlib.util.find_spec("isaaclab").submodule_search_locations[0])
    load("isaaclab.utils.math", path / "utils/math.py")
    core = load("isolated_tracking.switch_math", MDP / "switch_math.py")
    base = load("isolated_tracking.commands", MDP / "commands.py")
    switch = load("isolated_tracking.switch_commands", MDP / "switch_commands.py")
    monkeypatch.setitem(sys.modules, "scaletrack.tasks.tracking.mdp.commands", base)
    load("scaletrack.utils.torch_utils", MDP.parents[2] / "utils/torch_utils.py")
    obs = load("isolated_tracking.observations", MDP / "observations.py")
    load("isolated_tracking.rewards", MDP / "rewards.py")
    rewards = load("isolated_tracking.switch_rewards", MDP / "switch_rewards.py")
    return SimpleNamespace(core=core, base=base, switch=switch, obs=obs, rewards=rewards)


def command(code):
    cmd = object.__new__(code.switch.SwitchMotionCommand)
    cmd.num_envs, cmd.num_motion, cmd.device = 2, 3, "cpu"
    cmd.motion_anchor_body_index = cmd.robot_anchor_body_index = 0
    paths = sorted((TRACK / "source/scaletrack/data/example").glob("*.npz"))
    assert len(paths) == 3
    fields = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    ids = [0, 4, 10, 18, 5, 11, 19, 9, 16, 20, 28, 17, 21, 29]
    cmd._foot_indices = [3, 6]
    arrays = cmd._load_motion_arrays({p.stem: p for p in paths}, ids)
    for name, data in zip(fields, arrays[1:-1]):
        setattr(cmd, f"cat_{name}", torch.from_numpy(data))
    cmd.time_totals = torch.from_numpy(arrays[-1])
    cmd.time_offsets = torch.tensor([0, *cmd.time_totals.cumsum(0)[:-1]])
    cmd.motion_ids = torch.zeros(2, dtype=torch.long)
    cmd.time_steps = torch.tensor([20, 20])
    cmd._yaw = code.core.yaw_coefficients(torch.zeros(2))
    cmd._translation = torch.zeros(2, 3)
    cmd._rand_timestep = torch.full((2, 1), 12)
    cmd._future_manual_cache = {}
    cmd.cfg = SimpleNamespace(automatic_switches=False, body_names=[f"body_{i}" for i in ids],
                              ref_frame_buffer_size=33, recovery_steps=35, max_burst_grace_steps=50, entry_candidates=16, matched_q_rms=.45, matched_speed_error=.4,
                              first_switch_range=(5, 5), clean_probability=0, curriculum_iterations=(2, 3),
                              rapid_probability=0.2, sham_probability=0.0)
    cmd._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    cmd.body_indexes = torch.arange(len(ids))
    cmd.robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=cmd.body_pos_w.clone(),
                                body_quat_w=cmd.body_quat_w.clone(), body_lin_vel_w=cmd.body_lin_vel_w.clone(),
                                body_ang_vel_w=cmd.body_ang_vel_w.clone(), applied_torque=torch.zeros(2, 29)))
    cmd.previous_torque = torch.zeros(2, 29)
    cmd.switch_age = torch.full((2,), -1)
    cmd._burst_age = cmd.switch_age.clone()
    cmd._countdown = cmd.switch_age.clone()
    cmd._fixed_motion = cmd.switch_age.clone()
    cmd._fixed_frame = cmd.switch_age.clone()
    cmd._pending_motion = cmd.switch_age.clone()
    cmd._pending_frame = cmd.switch_age.clone()
    cmd._pending_yaw = torch.zeros(2)
    cmd._previous_torque_valid = torch.ones(2, dtype=torch.bool)
    cmd.metrics = {key: torch.zeros(2) for key in (
        "hard_switches", "switch_rapid", "switch_matched", "switch_random_phase",
        "switch_opposite_support", "switch_turn", "switch_braking", "switch_match_unavailable", "sham_switches")}
    cmd.is_evaluating = False
    cmd._rng = torch.Generator().manual_seed(9)
    cmd._weights = cmd._training_weight = torch.ones(3)
    cmd._braking_frames = None
    return cmd


def observations(cmd, code, offsets):
    env = SimpleNamespace(num_envs=2, command_manager=SimpleNamespace(get_term=lambda name: cmd))
    names = ["target_body_pos_future_to_robot_base_manual", "target_body_pos_future_rel_to_robot_base_manual",
             "target_body_rot_future_to_robot_base_manual", "target_body_rot_future_rel_to_robot_base_manual"]
    if offsets == C_LOOK:
        names += ["target_body_vel_future_rel_to_robot_base_manual", "target_body_ang_vel_future_rel_to_robot_base_manual"]
    names += ["ref_motion_time_offsets_future_manual"]
    return torch.cat([getattr(code.obs, fn)(env, "motion", offsets) for fn in names], dim=-1)


def test_atomic_causal_switch_and_native_clock(code):
    cmd = command(code)
    # The same A will receive distinct Bs. Filling both caches before commit tests stale lookahead leakage.
    cmd.queue_switch([0, 1], [1, 2], [31, 74], [math.pi / 2, 0])
    raw_cmd = deepcopy(cmd)
    raw_cmd.__class__ = code.base.MotionCommand
    robot_before = deepcopy(cmd.robot.data)
    for look in (A_LOOK, C_LOOK):
        obs = observations(cmd, code, look)
        assert torch.equal(obs[0], obs[1])
        assert torch.equal(obs, observations(raw_cmd, code, look))
    # No reset/write methods are available on this robot.
    cmd._update_command()
    assert cmd.motion_ids.tolist() == [1, 2] and cmd.time_steps.tolist() == [31, 74]
    assert cmd.switch_age.tolist() == [0, 0] and not cmd.clean_teacher_mask.any()
    for field in vars(robot_before):
        assert torch.equal(getattr(cmd.robot.data, field), getattr(robot_before, field))
    for offsets in (A_LOOK, C_LOOK):
        assert not torch.equal(observations(cmd, code, offsets)[0], observations(cmd, code, offsets)[1])
        idx = cmd._global_time_indices_manual(offsets)
        for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            actual = getattr(cmd, f"{field}_future_manual")(offsets)
            expected = code.core.transform(getattr(cmd, f"cat_{field}")[idx], field, cmd._yaw, cmd._translation)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual[:, 0], getattr(cmd, field))
            torch.testing.assert_close(actual[:, :2], getattr(cmd, f"{field}_future")[:, :2])
    # A's native heading on the would-have-been A frame is the registration anchor.
    torch.testing.assert_close(cmd.anchor_pos_w[:, :2], cmd.cat_body_pos_w[21, 0, :2].expand(2, -1))
    a_heading = code.core.heading(cmd.cat_body_quat_w[21, 0])
    difference = code.core.heading(cmd.anchor_quat_w) - a_heading
    torch.testing.assert_close(torch.atan2(difference.sin(), difference.cos()), torch.tensor([math.pi / 2, 0.]))
    yaw, translation = cmd._yaw.clone(), cmd._translation.clone()
    cmd._update_command()
    assert cmd.time_steps.tolist() == [32, 75] and cmd.switch_age.tolist() == [1, 1]
    torch.testing.assert_close(cmd._yaw, yaw)
    torch.testing.assert_close(cmd._translation, translation)
    torch.testing.assert_close(cmd.joint_vel, cmd.cat_joint_vel[cmd.time_offsets[cmd.motion_ids] + cmd.time_steps])


def test_released_xl_same_a_different_b(code):
    from my_rsl_rl.modules.actor_critic_humanoid_transformer import ActorCriticHumanoidTransformer

    default = TRACK / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_xl/model_22200.pt"
    checkpoint = Path(os.environ.get("SCALETRACK_XL_CHECKPOINT", default))
    if not checkpoint.exists():
        pytest.skip("Set SCALETRACK_XL_CHECKPOINT for the released policy causality test")
    cmd = command(code)
    # Identical robot and three-sample histories for two different prospective Bs.
    obs = {"policy_task": observations(cmd, code, A_LOOK), "critic_task": observations(cmd, code, C_LOOK),
           "policy": torch.zeros(2, 3, 64), "critic": torch.zeros(2, 3, 266), "action": torch.zeros(2, 3, 29),
           "mode": torch.ones(2, 14), "mode_mapping": torch.ones(2, 253)}
    policy = ActorCriticHumanoidTransformer(obs, {}, 29, embedding_dim=384, num_heads=6, ff_dim=384, num_layers=6)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    policy.load_state_dict(state["model_state_dict"], strict=True)
    policy.eval()
    with torch.inference_mode():
        before = policy.act_inference(obs)
        cmd.queue_switch([0, 1], [1, 2], [31, 74], [math.pi / 2, 0])
        obs["policy_task"], obs["critic_task"] = observations(cmd, code, A_LOOK), observations(cmd, code, C_LOOK)
        queued = policy.act_inference(obs)
        assert torch.equal(queued, before) and torch.equal(queued[0], queued[1])
        cmd._update_command()
        obs["policy_task"], obs["critic_task"] = observations(cmd, code, A_LOOK), observations(cmd, code, C_LOOK)
        after = policy.act_inference(obs)
        assert torch.isfinite(after).all() and not torch.allclose(after[0], after[1])


def test_reset_cancels_queued_event_and_curriculum(code):
    cmd = command(code)
    cmd.cfg.automatic_switches = True
    cmd.queue_switch([0], [2], [53])
    cmd.set_reset_reference([0], [1], [73])
    cmd._reset_reference_phase(torch.tensor([0]))
    assert cmd.motion_ids[0] == 1 and cmd.time_steps[0] == 73
    assert cmd._pending_motion[0] == -1 and cmd.clean_teacher_mask.all()
    cmd.set_training_iteration(0)
    cmd._countdown = torch.ones(2, dtype=torch.long)
    cmd._update_command()
    assert (cmd.metrics["switch_matched"] + cmd.metrics["switch_match_unavailable"]).tolist() == [1, 1]
    cmd.cfg.sham_probability = 1
    before = (cmd.motion_ids.clone(), cmd.time_steps.clone())
    cmd._countdown.fill_(1)
    cmd._update_command()
    assert torch.equal(before[0], cmd.motion_ids)
    assert torch.equal(before[1] + 1, cmd.time_steps)
    assert cmd.metrics["sham_switches"].tolist() == [1, 1]
    cmd.set_training_iteration(1000)
    for k in range(5):
        _, entries, realized = cmd._select_entries(torch.tensor([0, 1]), torch.full((2,), k))
        assert (entries >= 0).all() and ((realized == k) | (realized == 2) | (realized == -1)).all()


def test_rapid_switches_cannot_extend_total_grace(code):
    cmd = command(code)
    cmd.switch_to([0], [1], [20])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: cmd), scene=cmd._env.scene)
    cmd.robot.data.projected_gravity_b = torch.tensor([[0., 0., 0.], [0., 0., 0.]])
    # Identical physical tilt: immediately terminal during the grace period for
    # a hard switch, while the clean env retains the native task's termination.
    assert code.rewards.switch_fall(env, "motion").tolist() == [True, False]
    cmd.robot.data.projected_gravity_b[:, 2] = -1
    cmd.robot.data.body_pos_w[:, 0, 2] = 0.30
    assert code.rewards.switch_fall(env, "motion").tolist() == [True, False]
    for step in range(1, 56):
        cmd._update_command()
        if step % 5 == 0:
            cmd.switch_to([0], [1], [20 + step])
            assert cmd.time_steps[0] == 20 + step
            assert cmd.switch_age.tolist() == [0, -1]
        assert cmd.recovery_fraction[0] >= min(1, step / 50)
        assert cmd.recovery_fraction[1] == 1 and cmd.clean_teacher_mask.tolist() == [False, True]
    # Full tracking and termination thresholds now apply despite a just-installed B.
    assert cmd.recovery_fraction.tolist() == [1, 1]
    for _ in range(50):
        cmd._update_command()
    cmd.switch_to([0], [2], [30])
    assert cmd.recovery_fraction.tolist() == [0, 1]


def test_stage_costs_and_source_balance(code):
    core = code.core
    p = core.source_weights(["lch__a", "lafan__b", "lafan__c", "bones__d"], {"lch": .3, "lafan": .3, "bones": .4})
    torch.testing.assert_close(p, torch.tensor([.3, .15, .15, .4]))
    torch.testing.assert_close(core.source_weights(["take__A424", "original__b"], {"original": 1.}), torch.tensor([.5, .5]))
    torch.testing.assert_close(core.recovery_fraction(torch.tensor([-1, 0, 7, 35, 55]), 35),
                               torch.tensor([1, 0, .2, 1, 1.]))
    force = torch.zeros(2, 4, 2, 3)
    force[1, 2, 0, 2] = 1000
    torch.testing.assert_close(core.excess_impact(force, torch.full((2,), 400)), torch.tensor([0, .7]))
    cmd = command(code)
    with pytest.raises(IndexError):
        cmd.switch_to([0, 1], [0, 3], [0, 0])
    assert cmd.motion_ids.tolist() == [0, 0] and cmd.time_steps.tolist() == [20, 20]


def test_shared_native_split_two_local_ranks(code, tmp_path):
    """Opt in on hosts that allow local Gloo sockets."""
    import yaml

    if os.getenv("SCALETRACK_TEST_GLOO") != "1":
        pytest.skip("Set SCALETRACK_TEST_GLOO=1 to run the two-rank local transport test")
    with open(TRACK / "source/scaletrack/data/example/example.yaml") as stream:
        motions = {name: str(TRACK / path) for name, path in yaml.safe_load(stream).items()}

    def rank(r):
        torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=r, world_size=2)
        cmd = object.__new__(code.switch.SwitchMotionCommand)
        cmd.is_distributed = True
        cmd.gpu_global_rank = cmd.gpu_local_rank = r
        arrays = cmd.load_motions(motions, [0, 18, 19], "cpu")
        assert len(arrays[0]) == 3 and arrays[1].shape[1] == 29
        payload = torch.stack([arrays[1].sum(), arrays[-1].sum()])
        checksums = [torch.empty_like(payload) for _ in range(2)]
        torch.distributed.all_gather(checksums, payload)
        assert torch.equal(*checksums)
        torch.distributed.destroy_process_group()

    torch.multiprocessing.start_processes(rank, nprocs=2, start_method="fork", join=True)
