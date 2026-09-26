r"""Evaluate PULSE latent continuity and perturbation sensitivity in ScaleTrack.

Run from ScaleTrack using the Isaac Lab Python environment::

    python scripts/eval/eval_latent_continuity.py \
        --checkpoint logs/rsl_rl/g1_bfm_distillation/<run> \
        --motion_file source/scaletrack/data/example/example.yaml --headless

B1 records prior means, log variances, sampled latents and decoded actions.
B2 measures single-step action sensitivity and H-step COM divergence after a
one-step latent perturbation, scaled by the conditional prior standard deviation.
Paired rollouts use the same motion start and sampling noise. Fall/invalid-state
termination is distinguished from motion completion, and COM is captured before
Isaac Lab's automatic reset. COM error uses the last common finite step.

A run directory selects its highest-numbered model_<iteration>.pt. Results and
plots go directly to <checkpoint_dir>/eval/latent_continuity/. Every fresh rollout
requires a complete distillation checkpoint (posterior, prior and decoder).
Use --temperature 0 for deterministic prior actions. --from_debug_logs analyzes
B1 only, without starting Isaac Lab; --debug_logs_dir defaults to the output
folder and accepts both legacy 'latent' and this script's 'latent_z' NPZ keys.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__:
    from .eval_posterior_prior_match import latent_eval_failure, prepare_motion_file, resolve_checkpoint
else:
    from eval_posterior_prior_match import latent_eval_failure, prepare_motion_file, resolve_checkpoint


@dataclass
class EvalArgs:
    checkpoint: str
    motion_file: str | None = None
    num_envs: int = 1
    first_n_traj: int = 3
    max_steps: int = 0  # B1 per-motion cap; 0 evaluates complete trajectories.
    temperature: float = 1.0
    seed: int = 42
    perturbation_scales: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    n_perturbation_samples_B2a: int = 10
    n_perturbation_samples_B2b: int = 10
    rollout_horizon: int = 50
    root_height_threshold: float = 0.3
    B1_from_debug_logs: bool = False
    debug_logs_dir: str | None = None
    save_plots: bool = True

    @property
    def exp_name(self) -> str:
        return Path(self.checkpoint).parent.name

    @property
    def save_dir(self) -> Path:
        return Path(self.checkpoint).parent / "eval" / "latent_continuity"


@dataclass
class TrajectoryLatentData:
    """Latent time-series data for a single trajectory."""
    traj_name: str

    # Time series (T, latent_dim)
    prior_mu: np.ndarray = field(default_factory=lambda: np.array([]))
    prior_logvar: np.ndarray = field(default_factory=lambda: np.array([]))
    latent_z: np.ndarray = field(default_factory=lambda: np.array([]))
    action: np.ndarray = field(default_factory=lambda: np.array([]))

    # Computed deltas (T-1,)
    delta_z: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_mu: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_a: np.ndarray = field(default_factory=lambda: np.array([]))

    def compute_deltas(self):
        """Compute L2 differences between consecutive timesteps."""
        if len(self.latent_z) > 1:
            self.delta_z = np.linalg.norm(np.diff(self.latent_z, axis=0), axis=1)
        if len(self.prior_mu) > 1:
            self.delta_mu = np.linalg.norm(np.diff(self.prior_mu, axis=0), axis=1)
        if len(self.action) > 1:
            self.delta_a = np.linalg.norm(np.diff(self.action, axis=0), axis=1)

    # Current policy inputs, retained for the fixed-observation B2a test.
    proprio: np.ndarray = field(default_factory=lambda: np.array([]))

    @staticmethod
    def from_npz(npz_path: str) -> "TrajectoryLatentData":
        """Read legacy debug files or this evaluator's saved trajectories."""
        with np.load(npz_path, allow_pickle=False) as data:
            latent_key = "latent_z" if "latent_z" in data else "latent"
            traj = TrajectoryLatentData(
                traj_name=str(data["traj_name"].item()) if "traj_name" in data else Path(npz_path).stem,
                prior_mu=data["prior_mu"], prior_logvar=data["prior_logvar"],
                latent_z=data[latent_key], action=data["action"],
            )
        arrays = (traj.prior_mu, traj.prior_logvar, traj.latent_z, traj.action)
        if any(a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"Expected finite, nonempty (time, features) arrays in {npz_path}")
        if any(a.shape != traj.prior_mu.shape for a in arrays[1:3]) or len(traj.action) != len(traj.prior_mu):
            raise ValueError(f"Unaligned latent/action time series in {npz_path}")
        traj.compute_deltas()
        return traj


@dataclass
class ContinuityMetrics:
    """Aggregated continuity metrics across trajectories."""
    n_trajectories: int = 0
    n_steps: int = 0

    # Δz statistics
    mean_delta_z: float = 0.0
    std_delta_z: float = 0.0
    p50_delta_z: float = 0.0
    p95_delta_z: float = 0.0
    max_delta_z: float = 0.0

    # Δμ statistics
    mean_delta_mu: float = 0.0
    std_delta_mu: float = 0.0
    p50_delta_mu: float = 0.0
    p95_delta_mu: float = 0.0
    max_delta_mu: float = 0.0

    # Δa statistics
    mean_delta_a: float = 0.0
    std_delta_a: float = 0.0
    p50_delta_a: float = 0.0
    p95_delta_a: float = 0.0
    max_delta_a: float = 0.0

    # Correlations
    corr_delta_z_delta_a: float = 0.0
    corr_delta_mu_delta_a: float = 0.0

    # Interpretation
    latent_is_smooth: bool = False  # Δz is small relative to latent std
    noise_dominates_jitter: bool = False  # corr(Δz,Δa) >> corr(Δμ,Δa)

    @staticmethod
    def from_trajectories(trajectories: List[TrajectoryLatentData]) -> "ContinuityMetrics":
        """Aggregate metrics from multiple trajectories."""
        metrics = ContinuityMetrics(n_trajectories=len(trajectories))

        if not trajectories:
            return metrics

        # Collect all deltas
        all_delta_z = []
        all_delta_mu = []
        all_delta_a = []

        for traj in trajectories:
            if len(traj.delta_z) > 0:
                all_delta_z.extend(traj.delta_z.tolist())
            if len(traj.delta_mu) > 0:
                all_delta_mu.extend(traj.delta_mu.tolist())
            if len(traj.delta_a) > 0:
                all_delta_a.extend(traj.delta_a.tolist())

        metrics.n_steps = len(all_delta_z)

        # Δz statistics
        if all_delta_z:
            arr = np.array(all_delta_z)
            metrics.mean_delta_z = float(np.mean(arr))
            metrics.std_delta_z = float(np.std(arr))
            metrics.p50_delta_z = float(np.percentile(arr, 50))
            metrics.p95_delta_z = float(np.percentile(arr, 95))
            metrics.max_delta_z = float(np.max(arr))

        # Δμ statistics
        if all_delta_mu:
            arr = np.array(all_delta_mu)
            metrics.mean_delta_mu = float(np.mean(arr))
            metrics.std_delta_mu = float(np.std(arr))
            metrics.p50_delta_mu = float(np.percentile(arr, 50))
            metrics.p95_delta_mu = float(np.percentile(arr, 95))
            metrics.max_delta_mu = float(np.max(arr))

        # Δa statistics
        if all_delta_a:
            arr = np.array(all_delta_a)
            metrics.mean_delta_a = float(np.mean(arr))
            metrics.std_delta_a = float(np.std(arr))
            metrics.p50_delta_a = float(np.percentile(arr, 50))
            metrics.p95_delta_a = float(np.percentile(arr, 95))
            metrics.max_delta_a = float(np.max(arr))

        # Correlations
        if len(all_delta_z) > 2 and len(all_delta_a) > 2:
            min_len = min(len(all_delta_z), len(all_delta_a))
            z_arr = np.array(all_delta_z[:min_len])
            a_arr = np.array(all_delta_a[:min_len])
            if np.std(z_arr) > 1e-8 and np.std(a_arr) > 1e-8:
                metrics.corr_delta_z_delta_a = float(np.corrcoef(z_arr, a_arr)[0, 1])

        if len(all_delta_mu) > 2 and len(all_delta_a) > 2:
            min_len = min(len(all_delta_mu), len(all_delta_a))
            mu_arr = np.array(all_delta_mu[:min_len])
            a_arr = np.array(all_delta_a[:min_len])
            if np.std(mu_arr) > 1e-8 and np.std(a_arr) > 1e-8:
                metrics.corr_delta_mu_delta_a = float(np.corrcoef(mu_arr, a_arr)[0, 1])

        # Interpretations
        # Latent is smooth if mean Δz < 1.0 (relative to unit normal prior)
        metrics.latent_is_smooth = metrics.n_steps > 0 and metrics.mean_delta_z < 1.0

        # Noise dominates if corr(Δz,Δa) is significantly higher than corr(Δμ,Δa)
        # This means the sampling noise component contributes more to action jitter
        # which indicates decoder is sensitive to noise (not ideal for RL)
        metrics.noise_dominates_jitter = (
            metrics.corr_delta_z_delta_a > metrics.corr_delta_mu_delta_a + 0.1
        )

        return metrics

    def to_dict(self) -> dict:
        return {
            "n_trajectories": int(self.n_trajectories),
            "n_steps": int(self.n_steps),
            "mean_delta_z": float(self.mean_delta_z),
            "std_delta_z": float(self.std_delta_z),
            "p50_delta_z": float(self.p50_delta_z),
            "p95_delta_z": float(self.p95_delta_z),
            "max_delta_z": float(self.max_delta_z),
            "mean_delta_mu": float(self.mean_delta_mu),
            "std_delta_mu": float(self.std_delta_mu),
            "p50_delta_mu": float(self.p50_delta_mu),
            "p95_delta_mu": float(self.p95_delta_mu),
            "max_delta_mu": float(self.max_delta_mu),
            "mean_delta_a": float(self.mean_delta_a),
            "std_delta_a": float(self.std_delta_a),
            "p50_delta_a": float(self.p50_delta_a),
            "p95_delta_a": float(self.p95_delta_a),
            "max_delta_a": float(self.max_delta_a),
            "corr_delta_z_delta_a": float(self.corr_delta_z_delta_a),
            "corr_delta_mu_delta_a": float(self.corr_delta_mu_delta_a),
            "latent_is_smooth": bool(self.latent_is_smooth),
            "noise_dominates_jitter": bool(self.noise_dominates_jitter),
        }


@dataclass
class PerturbationResult:
    """Results from a single perturbation test."""
    scale: float
    n_samples: int

    # Single-step action difference: ||a(z') - a(z)||
    mean_action_diff: float = 0.0
    std_action_diff: float = 0.0
    max_action_diff: float = 0.0

    # H-step rollout divergence
    mean_terminal_com_error: float = 0.0  # COM position error after H steps
    mean_survival_steps: float = 0.0      # Average steps before termination
    termination_rate: float = 0.0         # Fraction of rollouts that terminated early

    # B2a and B2b have independent sample counts.
    n_rollout_samples: int = 0
    n_com_samples: int = 0
    mean_comparison_steps: float = 0.0
    baseline_termination_rate: float = 0.0
    motion_completion_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_rollout_samples": int(self.n_rollout_samples),
            "n_com_samples": int(self.n_com_samples),
            "mean_comparison_steps": float(self.mean_comparison_steps),
            "baseline_termination_rate": float(self.baseline_termination_rate),
            "motion_completion_rate": float(self.motion_completion_rate),
            "scale": float(self.scale),
            "n_samples": int(self.n_samples),
            "mean_action_diff": float(self.mean_action_diff),
            "std_action_diff": float(self.std_action_diff),
            "max_action_diff": float(self.max_action_diff),
            "mean_terminal_com_error": float(self.mean_terminal_com_error),
            "mean_survival_steps": float(self.mean_survival_steps),
            "termination_rate": float(self.termination_rate),
        }



def load_continuity_from_debug_logs(debug_dir: str, first_n_traj: int = 0) -> List[TrajectoryLatentData]:
    """Use the report manifest when present, excluding stale files from older runs."""
    directory = Path(debug_dir).expanduser().resolve()
    manifest = directory / "latent_continuity_results.json"
    if manifest.is_file():
        entries = json.loads(manifest.read_text()).get("trajectories")
    else:
        entries = None
    files = [directory / entry["file"] for entry in entries] if entries is not None else sorted(directory.glob("*.npz"))
    if first_n_traj > 0:
        files = files[:first_n_traj]
    if not files:
        raise FileNotFoundError(f"No trajectory NPZ files found in {directory}")
    trajectories = [TrajectoryLatentData.from_npz(str(path)) for path in files]
    print(f"Loaded {len(trajectories)} trajectories from {directory}")
    return trajectories


def checked_prior(policy, proprio):
    import torch

    if not torch.isfinite(proprio).all():
        raise RuntimeError("Non-finite policy observations; evaluation stopped.")
    mu, logvar = policy.prior.encode(proprio)
    std = (0.5 * logvar).exp()
    if not all(torch.isfinite(value).all() for value in (mu, logvar, std)):
        raise RuntimeError("Non-finite prior outputs; evaluation stopped.")
    return mu, logvar, std


def checked_decode(policy, latent, proprio):
    import torch

    actions = policy.decoder(latent, proprio)
    if not torch.isfinite(latent).all() or not torch.isfinite(actions).all():
        raise RuntimeError("Non-finite latent/actions; stopped before stepping the simulator.")
    return actions


def run_continuity_rollout(env, policy, args: EvalArgs, simulation_app) -> List[TrajectoryLatentData]:
    """Evaluate each selected motion once, excluding padding and auto-reset rows."""
    import torch

    motion = env.unwrapped.command_manager.get_term("motion")
    motion.is_evaluating = True
    n_motions = min(motion.num_motion, args.first_n_traj) if args.first_n_traj > 0 else motion.num_motion
    generator = torch.Generator(device=env.device).manual_seed(args.seed)
    trajectories = []
    with torch.inference_mode():
        for start in range(0, n_motions, env.num_envs):
            ids = torch.arange(start, min(start + env.num_envs, n_motions))
            batch_size = len(ids)
            motion.motion_ids[:batch_size] = ids
            motion.motion_ids[batch_size:] = ids[0]
            lengths = motion.time_totals[ids].clone()
            if args.max_steps > 0:
                lengths.clamp_(max=args.max_steps)
            active = torch.arange(env.num_envs, device=env.device) < batch_size
            buffers = [[] for _ in ids]
            obs, _ = env.reset()
            for step in range(int(lengths.max())):
                if not simulation_app.is_running():
                    raise RuntimeError("Simulation closed before evaluation finished.")
                rows = active.nonzero(as_tuple=False).squeeze(-1)
                if not rows.numel():
                    break
                proprio = torch.cat([obs[key][rows] for key in policy.obs_groups["policy"]], dim=-1)
                mu, logvar, std = checked_prior(policy, proprio)
                noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
                latent = mu + args.temperature * std * noise
                decoded = checked_decode(policy, latent, proprio)
                values = [x.cpu().numpy() for x in (mu, logvar, latent, decoded, proprio)]
                for index, row in enumerate(rows.cpu().tolist()):
                    buffers[row].append(tuple(value[index].copy() for value in values))
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                actions[rows] = decoded
                obs, _, dones, _ = env.step(actions)
                active &= ~dones.bool()
                active[:batch_size] &= (step + 1 < lengths).to(env.device)
            for row, motion_id in enumerate(ids.tolist()):
                if not buffers[row]:
                    raise ValueError(f"No samples for motion {motion.motion_names[motion_id]}")
                mu, logvar, latent, actions, proprio = [np.stack(x) for x in zip(*buffers[row])]
                traj = TrajectoryLatentData(
                    str(motion.motion_names[motion_id]), mu, logvar, latent, actions, proprio=proprio
                )
                traj.compute_deltas()
                trajectories.append(traj)
                print(f"  {traj.traj_name}: {len(mu)}/{int(lengths[row])} steps")
    if not trajectories:
        raise ValueError("No trajectories were collected.")
    return trajectories


def run_perturbation_test(policy, trajectories, args: EvalArgs) -> Dict[float, PerturbationResult]:
    """B2a: perturb fixed observations sampled uniformly from the B1 frames."""
    import torch

    rng = np.random.default_rng(args.seed + 1)
    observations = np.concatenate([t.proprio for t in trajectories])
    indices = rng.integers(len(observations), size=args.n_perturbation_samples_B2a)
    device = next(policy.parameters()).device
    proprio = torch.as_tensor(observations[indices], device=device)
    results = {}
    with torch.inference_mode():
        mu, _, std = checked_prior(policy, proprio)
        noise = torch.as_tensor(rng.standard_normal(mu.shape), dtype=mu.dtype, device=device)
        delta = torch.as_tensor(rng.standard_normal(mu.shape), dtype=mu.dtype, device=device)
        latent = mu + args.temperature * std * noise
        baseline = checked_decode(policy, latent, proprio)
        for scale in args.perturbation_scales:
            perturbed = checked_decode(policy, latent + scale * std * delta, proprio)
            diffs = torch.linalg.vector_norm(perturbed - baseline, dim=-1).cpu().numpy()
            results[scale] = PerturbationResult(
                scale, len(diffs), float(diffs.mean()), float(diffs.std()), float(diffs.max())
            )
    return results


def capture_com_and_failure(env, root_height_threshold: float):
    """Termination hook: preserve terminal physics state before Isaac Lab resets."""
    robot = env.command_manager.get_term("motion").robot.data
    # Startup mass randomization is disabled in this evaluator.
    masses = robot.default_mass.to(robot.body_com_pos_w.device).unsqueeze(-1)
    env._latent_continuity_com = ((robot.body_com_pos_w * masses).sum(1) / masses.sum(1)).clone()
    env._latent_continuity_failed = latent_eval_failure(env, root_height_threshold).clone()
    return env._latent_continuity_failed


def rollout_from_motion_starts(env, policy, args, simulation_app, ids, noise, delta, scale, reset_seed):
    """One B2b branch; return pre-reset COM paths and true failure flags."""
    import torch

    motion = env.unwrapped.command_manager.get_term("motion")
    motion.is_evaluating = True
    batch_size = len(ids)
    motion.motion_ids[:batch_size] = ids
    motion.motion_ids[batch_size:] = ids[0]
    lengths = motion.time_totals[ids].clamp(max=args.rollout_horizon)
    torch.manual_seed(reset_seed)
    obs, _ = env.reset()
    initial_proprio = torch.cat([obs[key][:batch_size] for key in policy.obs_groups["policy"]], dim=-1).clone()
    active = torch.arange(env.num_envs, device=env.device) < batch_size
    com = np.full((args.rollout_horizon, batch_size, 3), np.nan)
    steps = np.zeros(batch_size, dtype=int)
    failed = np.zeros(batch_size, dtype=bool)
    completed = np.zeros(batch_size, dtype=bool)
    for step in range(int(lengths.max())):
        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed before perturbation evaluation finished.")
        rows = active.nonzero(as_tuple=False).squeeze(-1)
        if not rows.numel():
            break
        proprio = torch.cat([obs[key][rows] for key in policy.obs_groups["policy"]], dim=-1)
        mu, _, std = checked_prior(policy, proprio)
        latent = mu + args.temperature * std * noise[step, rows]
        if step == 0:
            latent = latent + scale * std * delta[rows]
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        actions[rows] = checked_decode(policy, latent, proprio)
        obs, _, dones, _ = env.step(actions)
        row_ids = rows.cpu().numpy()
        com[step, row_ids] = env.unwrapped._latent_continuity_com[rows].cpu().numpy()
        steps[row_ids] += 1
        failures = env.unwrapped._latent_continuity_failed[rows].cpu().numpy()
        failed[row_ids] |= failures
        completed[row_ids] |= dones[rows].bool().cpu().numpy() & ~failures
        active &= ~dones.bool()
        active[:batch_size] &= (step + 1 < lengths).to(env.device)
    return com, steps, failed, completed, initial_proprio


def run_rollout_divergence_test(env, policy, args: EvalArgs, simulation_app) -> Dict[float, PerturbationResult]:
    """B2b: matching resets/noise isolate the effect of a t=0 perturbation."""
    import torch

    motion = env.unwrapped.command_manager.get_term("motion")
    n_motions = min(motion.num_motion, args.first_n_traj) if args.first_n_traj > 0 else motion.num_motion
    rng = np.random.default_rng(args.seed + 1000)
    # Cycle through selected motions so short runs cover distinct initial states.
    motion_ids = torch.arange(args.n_perturbation_samples_B2b) % n_motions
    records = {scale: {key: [] for key in ("errors", "comparison_steps", "steps", "failed", "baseline_failed", "completed")}
               for scale in args.perturbation_scales}
    latent_dim = policy.policy_config["latent_dim"]
    with torch.inference_mode():
        for start in range(0, len(motion_ids), env.num_envs):
            ids = motion_ids[start:start + env.num_envs]
            noise = torch.as_tensor(
                rng.standard_normal((args.rollout_horizon, len(ids), latent_dim)),
                dtype=torch.float32, device=env.device,
            )
            delta = torch.as_tensor(rng.standard_normal((len(ids), latent_dim)), dtype=torch.float32, device=env.device)
            reset_seed = args.seed + 1000 + start
            baseline = rollout_from_motion_starts(
                env, policy, args, simulation_app, ids, noise, delta, 0.0, reset_seed
            )
            for scale, record in records.items():
                branch = rollout_from_motion_starts(
                    env, policy, args, simulation_app, ids, noise, delta, scale, reset_seed
                )
                if not torch.allclose(baseline[4], branch[4], atol=1e-5, rtol=1e-5):
                    raise RuntimeError("Paired resets produced different initial policy observations.")
                for row in range(len(ids)):
                    common = np.isfinite(baseline[0][:, row]).all(-1) & np.isfinite(branch[0][:, row]).all(-1)
                    valid_steps = np.flatnonzero(common)
                    if len(valid_steps):
                        last = valid_steps[-1]
                        record["errors"].append(float(np.linalg.norm(baseline[0][last, row] - branch[0][last, row])))
                        record["comparison_steps"].append(int(last) + 1)
                for key, values in zip(("steps", "failed", "baseline_failed", "completed"),
                                       (branch[1], branch[2], baseline[2], branch[3])):
                    record[key].extend(values.tolist())
    results = {}
    for scale, record in records.items():
        if not record["errors"]:
            raise RuntimeError(f"No finite paired COM samples at perturbation scale {scale}.")
        result = PerturbationResult(scale, 0)
        result.n_rollout_samples = len(record["steps"])
        result.n_com_samples = len(record["errors"])
        result.mean_terminal_com_error = float(np.mean(record["errors"]))
        result.mean_comparison_steps = float(np.mean(record["comparison_steps"]))
        result.mean_survival_steps = float(np.mean(record["steps"]))
        result.termination_rate = float(np.mean(record["failed"]))
        result.baseline_termination_rate = float(np.mean(record["baseline_failed"]))
        result.motion_completion_rate = float(np.mean(record["completed"]))
        results[scale] = result
        print(f"  η={scale:g}: COM error={result.mean_terminal_com_error:.5f}, "
              f"survival={result.mean_survival_steps:.1f}, failures={result.termination_rate:.1%}")
    return results

def plot_continuity_analysis(
    trajectories: List[TrajectoryLatentData],
    metrics: ContinuityMetrics,
    save_dir: Path,
    exp_name: str,
):
    """Generate B1 visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== Plot 1: Delta time series for each trajectory =====
    n_trajs = len(trajectories)
    if n_trajs > 0:
        fig, axes = plt.subplots(n_trajs, 3, figsize=(15, 4 * n_trajs))
        if n_trajs == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f"Latent Continuity Time Series: {exp_name}", fontsize=14)

        for i, traj in enumerate(trajectories):
            # Δz time series
            ax = axes[i, 0]
            if len(traj.delta_z) > 0:
                ax.plot(traj.delta_z, alpha=0.7)
                ax.axhline(y=np.mean(traj.delta_z), color='r', linestyle='--',
                          label=f'mean={np.mean(traj.delta_z):.3f}')
            ax.set_ylabel('||Δz||')
            ax.set_title(f'{traj.traj_name} - Latent Jump')
            if ax.lines:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Δμ time series
            ax = axes[i, 1]
            if len(traj.delta_mu) > 0:
                ax.plot(traj.delta_mu, alpha=0.7, color='orange')
                ax.axhline(y=np.mean(traj.delta_mu), color='r', linestyle='--',
                          label=f'mean={np.mean(traj.delta_mu):.3f}')
            ax.set_ylabel('||Δμ||')
            ax.set_title(f'{traj.traj_name} - Prior Mean Jump')
            if ax.lines:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Δa time series
            ax = axes[i, 2]
            if len(traj.delta_a) > 0:
                ax.plot(traj.delta_a, alpha=0.7, color='green')
                ax.axhline(y=np.mean(traj.delta_a), color='r', linestyle='--',
                          label=f'mean={np.mean(traj.delta_a):.3f}')
            ax.set_ylabel('||Δa||')
            ax.set_title(f'{traj.traj_name} - Action Jump')
            if ax.lines:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            if i == n_trajs - 1:
                axes[i, 0].set_xlabel('Time step')
                axes[i, 1].set_xlabel('Time step')
                axes[i, 2].set_xlabel('Time step')

        plt.tight_layout()
        plt.savefig(save_dir / "continuity_time_series.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ===== Plot 2: Delta histograms =====
    all_delta_z = np.concatenate([t.delta_z for t in trajectories]) if trajectories else np.array([])
    all_delta_mu = np.concatenate([t.delta_mu for t in trajectories]) if trajectories else np.array([])
    all_delta_a = np.concatenate([t.delta_a for t in trajectories]) if trajectories else np.array([])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Delta Distributions: {exp_name}", fontsize=14)

    # Δz histogram
    ax = axes[0]
    if len(all_delta_z) > 0:
        ax.hist(all_delta_z, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=metrics.mean_delta_z, color='r', linestyle='--',
                  label=f'mean={metrics.mean_delta_z:.3f}')
        ax.axvline(x=metrics.p95_delta_z, color='orange', linestyle='--',
                  label=f'p95={metrics.p95_delta_z:.3f}')
    ax.set_xlabel('||Δz||')
    ax.set_ylabel('Count')
    ax.set_title('Latent Jump Distribution')
    if ax.lines:
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Δμ histogram
    ax = axes[1]
    if len(all_delta_mu) > 0:
        ax.hist(all_delta_mu, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(x=metrics.mean_delta_mu, color='r', linestyle='--',
                  label=f'mean={metrics.mean_delta_mu:.3f}')
        ax.axvline(x=metrics.p95_delta_mu, color='darkred', linestyle='--',
                  label=f'p95={metrics.p95_delta_mu:.3f}')
    ax.set_xlabel('||Δμ||')
    ax.set_ylabel('Count')
    ax.set_title('Prior Mean Jump Distribution')
    if ax.lines:
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Δa histogram
    ax = axes[2]
    if len(all_delta_a) > 0:
        ax.hist(all_delta_a, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=metrics.mean_delta_a, color='r', linestyle='--',
                  label=f'mean={metrics.mean_delta_a:.3f}')
        ax.axvline(x=metrics.p95_delta_a, color='darkgreen', linestyle='--',
                  label=f'p95={metrics.p95_delta_a:.3f}')
    ax.set_xlabel('||Δa||')
    ax.set_ylabel('Count')
    ax.set_title('Action Jump Distribution')
    if ax.lines:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "continuity_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== Plot 3: Correlation scatter plots =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Latent-Action Correlation: {exp_name}", fontsize=14)

    # Δz vs Δa
    ax = axes[0]
    if len(all_delta_z) > 0 and len(all_delta_a) > 0:
        min_len = min(len(all_delta_z), len(all_delta_a))
        ax.scatter(all_delta_z[:min_len], all_delta_a[:min_len], alpha=0.3, s=5)
        ax.set_xlabel('||Δz|| (Latent Jump)')
        ax.set_ylabel('||Δa|| (Action Jump)')
        ax.set_title(f'corr(Δz, Δa) = {metrics.corr_delta_z_delta_a:.3f}')
    ax.grid(True, alpha=0.3)

    # Δμ vs Δa
    ax = axes[1]
    if len(all_delta_mu) > 0 and len(all_delta_a) > 0:
        min_len = min(len(all_delta_mu), len(all_delta_a))
        ax.scatter(all_delta_mu[:min_len], all_delta_a[:min_len], alpha=0.3, s=5, color='orange')
        ax.set_xlabel('||Δμ|| (Prior Mean Jump)')
        ax.set_ylabel('||Δa|| (Action Jump)')
        ax.set_title(f'corr(Δμ, Δa) = {metrics.corr_delta_mu_delta_a:.3f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "continuity_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Continuity plots saved to {save_dir}")


def plot_perturbation_results(
    perturbation_results: Dict[float, PerturbationResult],
    save_dir: Path,
    exp_name: str,
):
    """Generate B2 visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    scales = sorted(perturbation_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Local Perturbation Test (B2): {exp_name}", fontsize=14)

    # 1. Action difference vs perturbation scale
    ax = axes[0]
    mean_diffs = [perturbation_results[s].mean_action_diff for s in scales]
    std_diffs = [perturbation_results[s].std_action_diff for s in scales]
    ax.errorbar(scales, mean_diffs, yerr=std_diffs, marker='o', capsize=5)
    ax.set_xlabel('Perturbation Scale η')
    ax.set_ylabel('||a(z\') - a(z)||')
    ax.set_title('Single-Step Action Sensitivity')
    ax.grid(True, alpha=0.3)

    # 2. COM error vs perturbation scale
    ax = axes[1]
    com_errors = [perturbation_results[s].mean_terminal_com_error for s in scales]
    ax.plot(scales, com_errors, marker='s', color='orange')
    ax.set_xlabel('Perturbation Scale η')
    ax.set_ylabel('Terminal COM Error')
    ax.set_title(f'H-Step Rollout Divergence')
    ax.grid(True, alpha=0.3)

    # 3. Termination rate vs perturbation scale
    ax = axes[2]
    term_rates = [perturbation_results[s].termination_rate for s in scales]
    ax.bar(scales, term_rates, width=0.03, color='red', alpha=0.7)
    ax.set_xlabel('Perturbation Scale η')
    ax.set_ylabel('Termination Rate')
    ax.set_title('Early Termination Under Perturbation')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "perturbation_test.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Perturbation plots saved to {save_dir}")


def save_results(args, trajectories, metrics, perturbation_results):
    """Keep the source metric schema and list the current run's trajectory files."""
    args.save_dir.mkdir(parents=True, exist_ok=True)
    config = asdict(args)
    config.update(exp_name=args.exp_name, from_debug_logs=args.B1_from_debug_logs)
    report = {
        "config": config,
        "continuity_metrics": metrics.to_dict(),
        "perturbation_results": {str(float(k)): v.to_dict() for k, v in perturbation_results.items()},
        "definitions": {
            "n_steps": "Number of within-trajectory consecutive-frame differences (B1).",
            "com": "Mass-weighted world-space center of mass of all robot bodies, in meters.",
            "mean_terminal_com_error": "COM distance at the last common finite post-step state, before auto-reset.",
            "termination_rate": "Fraction of perturbed rollouts with fall/invalid state; excludes motion completion.",
            "mean_survival_steps": "Executed steps through failure, motion completion, or rollout_horizon.",
            "n_samples": "Number of fixed-observation B2a tests; n_rollout_samples counts B2b pairs.",
            "perturbation": "At t=0 only: z += scale * prior_std * delta; paired branches share sampling noise.",
        },
        "trajectories": [],
    }
    json.dumps(report, allow_nan=False)
    for index, traj in enumerate(trajectories):
        safe_name = re.sub(r"[^\w.-]+", "_", traj.traj_name)[:160]
        filename = f"traj_{index:05d}_{safe_name}_latent_data.npz"
        report["trajectories"].append({"name": traj.traj_name, "n_steps": len(traj.latent_z), "file": filename})
        np.savez(
            args.save_dir / filename, traj_name=traj.traj_name,
            prior_mu=traj.prior_mu, prior_logvar=traj.prior_logvar,
            latent_z=traj.latent_z, action=traj.action,
            delta_z=traj.delta_z, delta_mu=traj.delta_mu, delta_a=traj.delta_a,
        )
    with (args.save_dir / "latent_continuity_results.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Results saved to: {args.save_dir}")


def finish_evaluation(args, trajectories, perturbation_results):
    metrics = ContinuityMetrics.from_trajectories(trajectories)
    print(f"B1: {metrics.n_trajectories} trajectories, {metrics.n_steps} frame differences")
    print(f"  Mean Δz={metrics.mean_delta_z:.4f}, Δμ={metrics.mean_delta_mu:.4f}, Δa={metrics.mean_delta_a:.4f}")
    print(f"  corr(Δz, Δa)={metrics.corr_delta_z_delta_a:.4f}, corr(Δμ, Δa)={metrics.corr_delta_mu_delta_a:.4f}")
    if metrics.n_steps == 0:
        print("  No consecutive frames; continuity statistics are unavailable (stored as zeros).")
    save_results(args, trajectories, metrics, perturbation_results)
    if args.save_plots:
        plot_continuity_analysis(trajectories, metrics, args.save_dir, args.exp_name)
        if perturbation_results:
            plot_perturbation_results(perturbation_results, args.save_dir, args.exp_name)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion_file", help="Motion YAML; defaults to params/env.yaml's training motion file.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--first_n_traj", type=int, default=3, help="First N motions in YAML order; <= 0 selects all.")
    parser.add_argument("--max_steps", type=int, default=0, help="B1 per-motion cap; 0 plays the full motion.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Prior sampling temperature; 0 uses the mean.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturbation_scales", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--n_perturbation_samples_B2a", type=int, default=10)
    parser.add_argument("--n_perturbation_samples_B2b", type=int, default=10)
    parser.add_argument("--rollout_horizon", type=int, default=50)
    parser.add_argument("--root_height_threshold", type=float, default=0.3, help="Fall cutoff in meters.")
    parser.add_argument("--from_debug_logs", "--B1_from_debug_logs", dest="B1_from_debug_logs", action="store_true",
                        help="Analyze saved B1 NPZs without loading the policy or simulator; skip B2.")
    parser.add_argument("--debug_logs_dir", help="NPZ directory; defaults to <checkpoint_dir>/eval/latent_continuity.")
    parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(parser, args):
    if args.num_envs < 1 or args.max_steps < 0:
        parser.error("--num_envs must be positive and --max_steps must be nonnegative")
    if not np.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    if not np.isfinite(args.root_height_threshold) or args.root_height_threshold <= 0:
        parser.error("--root_height_threshold must be finite and positive")
    if not args.perturbation_scales or any(not np.isfinite(s) or s < 0 for s in args.perturbation_scales):
        parser.error("--perturbation_scales must be finite and nonnegative")
    if min(args.n_perturbation_samples_B2a, args.n_perturbation_samples_B2b, args.rollout_horizon) < 1:
        parser.error("Perturbation sample counts and --rollout_horizon must be positive")
    if args.debug_logs_dir and not args.B1_from_debug_logs:
        parser.error("--debug_logs_dir requires --from_debug_logs")


def main():
    # Discover offline mode without importing Isaac Lab, including for --help.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--from_debug_logs", "--B1_from_debug_logs", dest="offline", action="store_true")
    offline = probe.parse_known_args()[0].offline
    parser = create_parser()
    if offline:
        parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    # AppLauncher pre-parses arguments: register required inputs afterward.
    parser.add_argument("--checkpoint", required=True, help="Student .pt checkpoint or run directory (latest iteration).")
    args_cli, hydra_args = parser.parse_known_args()
    args = EvalArgs(**{name: getattr(args_cli, name) for name in EvalArgs.__dataclass_fields__})
    validate_args(parser, args)
    try:
        args.checkpoint = str(resolve_checkpoint(args.checkpoint))
        if offline:
            if hydra_args:
                parser.error(f"Unrecognized offline arguments: {' '.join(hydra_args)}")
            args.debug_logs_dir = str(Path(args.debug_logs_dir or args.save_dir).expanduser().resolve())
            trajectories = load_continuity_from_debug_logs(args.debug_logs_dir, args.first_n_traj)
            finish_evaluation(args, trajectories, {})
            return

        import torch
        from my_rsl_rl.modules import VAEPolicy

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError(f"Unsupported distillation checkpoint version: {checkpoint.get('format_version')}")
        policy = VAEPolicy.from_checkpoint(checkpoint)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Checkpoint: {args.checkpoint}\nOutput: {args.save_dir}\nTemperature: {args.temperature}")
    sys.argv = [sys.argv[0]] + hydra_args
    with tempfile.TemporaryDirectory(prefix="scaletrack_latent_continuity_") as temp_dir:
        motion_file = prepare_motion_file(args, Path(temp_dir) / "motions.yaml")
        simulation_app = AppLauncher(args_cli).app
        try:
            import gymnasium as gym
            from isaaclab.envs import ManagerBasedRLEnvCfg
            from isaaclab.managers import TerminationTermCfg
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from isaaclab_tasks.utils.hydra import hydra_task_config

            import scaletrack.tasks  # noqa: F401
            from scaletrack.utils.distillation_wrapper import DistillationVecEnvWrapper

            @hydra_task_config(checkpoint["task_name"], "rsl_rl_cfg_entry_point")
            def evaluate(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
                env_cfg.scene.num_envs = args.num_envs
                env_cfg.seed = args.seed
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                if args_cli.device is not None:
                    env_cfg.sim.device = args_cli.device
                env_cfg.commands.motion.motion_file = str(motion_file)
                env_cfg.commands.motion.test_motion_file = ""
                env_cfg.commands.motion.enable_reset_disturbance = False
                env_cfg.commands.motion.debug_vis = False
                for group in vars(env_cfg.observations).values():
                    if hasattr(group, "enable_corruption"):
                        group.enable_corruption = False
                for name, term in list(vars(env_cfg.events).items()):
                    if hasattr(term, "mode"):
                        setattr(env_cfg.events, name, None)
                for name, term in list(vars(env_cfg.terminations).items()):
                    if name != "motion_time_out" and hasattr(term, "func"):
                        setattr(env_cfg.terminations, name, None)
                env_cfg.terminations.latent_eval_failure = TerminationTermCfg(
                    func=capture_com_and_failure, params={"root_height_threshold": args.root_height_threshold}
                )
                env = gym.make(checkpoint["task_name"], cfg=env_cfg)
                try:
                    env = DistillationVecEnvWrapper(RslRlVecEnvWrapper(env), checkpoint["observation_config"])
                    env.validate_action_config(checkpoint["action_config"])
                    policy.to(env.device)
                    print("Running B1: prior continuity rollouts...")
                    trajectories = run_continuity_rollout(env, policy, args, simulation_app)
                    print("Running B2a: fixed-observation action sensitivity...")
                    results = run_perturbation_test(policy, trajectories, args)
                    print("Running B2b: paired rollout divergence...")
                    divergence = run_rollout_divergence_test(env, policy, args, simulation_app)
                    for scale, result in results.items():
                        combined = divergence[scale]
                        combined.n_samples = result.n_samples
                        combined.mean_action_diff = result.mean_action_diff
                        combined.std_action_diff = result.std_action_diff
                        combined.max_action_diff = result.max_action_diff
                    finish_evaluation(args, trajectories, divergence)
                finally:
                    env.close()

            evaluate()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
