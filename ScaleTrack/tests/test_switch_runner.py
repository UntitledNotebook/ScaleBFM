"""CPU regressions for the original XL import, clean-only anchor, resume and rank synchronization."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from my_rsl_rl.runners import OnPolicyRunner


class Motion:
    def __init__(self, mask):
        self.clean_teacher_mask = torch.tensor(mask)
        self.iterations = []

    def set_training_iteration(self, iteration):
        self.iterations.append(iteration)


class TensorEnv:
    """Known XL input shapes; this fixture exercises PPO, not the robot simulator."""

    num_envs = 2
    num_actions = 29
    max_episode_length = 5
    device = "cpu"
    cfg = {}

    def __init__(self, mask):
        self.motion = Motion(mask)
        self.unwrapped = self
        self.command_manager = SimpleNamespace(get_term=lambda name: self.motion)
        self.obs = TensorDict({
            "policy": torch.randn(2, 3, 64),
            "action": torch.randn(2, 3, 29),
            "policy_task": torch.randn(2, 6, 253),
            "mode_mapping": torch.randint(2, (2, 253)).float(),
            "mode": torch.randint(2, (2, 14)).float(),
            "critic": torch.randn(2, 3, 266),
            "critic_task": torch.randn(2, 7, 337),
        }, batch_size=[2])
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)

    def get_observations(self):
        return self.obs

    def step(self, actions):
        return self.obs, -actions.square().mean(dim=-1), torch.zeros(2), {}


def runner_cfg(width=384, layers=6):
    return {
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1,
                      "actor_learning_rate": 1e-5, "critic_learning_rate": 1e-4, "schedule": "fixed"},
        "policy": {"class_name": "ActorCriticHumanoidTransformer", "embedding_dim": width, "num_heads": 6,
                   "ff_dim": width, "num_layers": layers},
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]}, "num_steps_per_env": 2,
        "save_interval": 100, "eval_during_training": False, "eval_interval": 200, "logger": "tensorboard",
    }


def released_checkpoint():
    default = Path(__file__).resolve().parents[1] / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_xl/model_22200.pt"
    path = Path(os.environ.get("SCALETRACK_XL_CHECKPOINT", default))
    if not path.exists():
        pytest.skip("Set SCALETRACK_XL_CHECKPOINT to run the real released XL regression.")
    return path


def test_xl_exact_import_update_and_resume(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(1806)
    env = TensorEnv([True, False])
    path = released_checkpoint()
    runner = OnPolicyRunner(env, runner_cfg(), log_dir=str(tmp_path))
    original_obs = env.obs.clone()
    ref = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model_state_dict"]
    runner.alg.policy.load_state_dict(ref, strict=True)
    with torch.inference_mode():
        expected_action = runner.alg.policy.act_inference(env.obs).clone()
        expected_value = runner.alg.policy.evaluate(env.obs).clone()
    runner.initialize_switch_finetuning(path)
    assert runner.current_learning_iteration == 0 and not runner.alg.actor_optimizer.state
    with torch.inference_mode():
        torch.testing.assert_close(runner.alg.policy.act_inference(env.obs), expected_action, rtol=0, atol=0)
        torch.testing.assert_close(runner.alg.policy.evaluate(env.obs), expected_value, rtol=0, atol=0)
        runner.alg.act(env.obs, env.motion.clean_teacher_mask)
        transition = runner.alg.transition.observations
        student_mean = runner.alg.transition.action_mean
        torch.testing.assert_close(transition["xl_mean"][0], student_mean[0], atol=2e-5, rtol=1e-5)
        assert not transition["xl_mean"][1].any()
        env.motion.clean_teacher_mask.logical_not_()
        assert transition["xl_clean"].tolist() == [True, False]  # pre-action mask survives command update
        for key in env.obs.keys():
            torch.testing.assert_close(env.obs[key], original_obs[key], atol=0, rtol=0)

    runner.learn(1)
    assert env.motion.iterations == [0]
    student = runner.alg.policy.actor.projection_head.weight
    assert not torch.equal(student, ref["actor.projection_head.weight"])
    torch.testing.assert_close(runner.alg.clean_teacher.actor.projection_head.weight,
                               ref["actor.projection_head.weight"], rtol=0, atol=0)
    assert all(param.grad is None for param in runner.alg.clean_teacher.parameters())
    saved_action = runner.alg.policy.act_inference(env.obs).detach().clone()
    runner.writer.close()
    del runner
    restored = OnPolicyRunner(env, runner_cfg(), log_dir=str(tmp_path))
    restored.initialize_switch_finetuning(path)
    restored.load(tmp_path / "model_1.pt")
    assert restored.current_learning_iteration == 1 and restored.tot_timesteps == 4
    assert restored.alg.actor_optimizer.state and restored.alg.critic_optimizer.state
    torch.testing.assert_close(restored.alg.policy.act_inference(env.obs), saved_action, rtol=0, atol=0)
    restored.learn(1)
    restored.writer.close()
    assert env.motion.iterations == [0, 1] and (tmp_path / "model_2.pt").exists()


def _distributed_step(rank, rendezvous, directory):
    # The hardware launcher always uses NCCL. Exercise the identical collective sequence on CPU/Gloo.
    from torch import distributed as dist

    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(rank + 100)
        env = TensorEnv([bool(rank)] * 2)
        runner = OnPolicyRunner(env, runner_cfg(12, 1), log_dir=str(directory))
        runner.alg.broadcast_parameters()
        runner.alg.freeze_clean_teacher(1.0)
        runner.switch_training = {"fixture": True}
        runner.switch_command = env.motion
        # Closed-form global KL gradient for one clean rank and one rank with zero clean samples.
        mask = env.motion.clean_teacher_mask
        mean = torch.ones(2, 29, requires_grad=True)
        std = torch.ones_like(mean)
        obs = TensorDict({"xl_clean": mask, "xl_mean": torch.zeros_like(mean)}, batch_size=[2])
        loss, _, count = runner.alg._clean_kl(obs, mean, std)
        loss.backward()
        assert count == (2 if rank else 0)
        grad = mean.grad.sum()
        dist.all_reduce(grad)
        torch.testing.assert_close(grad / 2, torch.tensor(1.0))
        runner.learn(1)
        assert (runner.writer is None) == bool(rank)
        if runner.writer:
            runner.writer.close()
        for parameter in runner.alg.policy.parameters():
            ref = parameter.detach().clone()
            dist.broadcast(ref, src=0)
            torch.testing.assert_close(parameter, ref, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_empty_clean_slice(tmp_path):
    torch.multiprocessing.spawn(_distributed_step, args=(tmp_path / "rdv", tmp_path), nprocs=2)
    checkpoint = torch.load(tmp_path / "model_1.pt", weights_only=True, map_location="cpu")
    assert checkpoint["global_steps"] == 8
    assert len(list(tmp_path.glob("events.out.tfevents.*"))) == 1


def test_scalar_only_wandb(monkeypatch, tmp_path):
    from my_rsl_rl.utils import wandb_utils

    calls = []
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **opts: calls.append(opts) or SimpleNamespace(url=None))
    monkeypatch.setattr(wandb_utils.wandb, "save", lambda *args, **kwargs: pytest.fail("unexpected file upload"))
    writer = wandb_utils.WandbSummaryWriter(str(tmp_path / "smoke"), 10,
                                            {"scalar_metrics_only": True, "wandb_project": "test"})
    writer.log_config({"asset": "/private"}, {}, {}, {})
    writer.save_file("private")
    writer.save_model("private", 0)
    writer.close()
    options, = calls
    assert "config" not in options and options["save_code"] is False and options["sync_tensorboard"] is False
    settings = options["settings"]
    assert settings.console == "off" and settings.disable_git and settings.disable_code
    assert settings.x_disable_machine_info and settings.x_disable_meta and settings.x_disable_stats
