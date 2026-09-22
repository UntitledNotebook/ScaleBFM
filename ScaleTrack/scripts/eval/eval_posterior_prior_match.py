r"""Evaluate PULSE posterior/prior alignment with ScaleTrack Isaac Lab rollouts.

Run from ScaleTrack (using the Isaac Lab Python environment)::

    python scripts/eval/eval_posterior_prior_match.py \
        --checkpoint logs/rsl_rl/g1_bfm_distillation/<run> \
        --motion_file source/scaletrack/data/example/example.yaml --headless

A checkpoint directory selects its highest-numbered model_<iteration>.pt. Outputs
are written directly to <checkpoint_dir>/eval/posterior_prior_match/. Re-running
replaces the summary and plots. Every load requires the complete posterior
encoder, decoder, and prior; older checkpoints missing weights are rejected.

Metrics and plots are ported from Humanoid_Pipeline's evaluation script C. OOD
scores always evaluate posterior means under the learned conditional prior,
regardless of the distribution used for rollout actions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EvalArgs:
    checkpoint: str
    motion_file: str | None = None
    num_envs: int = 1
    first_n_traj: int = 5  # <= 0 evaluates all motions, in YAML order.
    max_steps: int = 0  # 0 evaluates complete trajectories.
    decode_source: str = "posterior"
    use_stochastic: bool = False
    temperature: float = 1.0
    seed: int = 42
    active_unit_thresholds: List[float] = field(default_factory=lambda: [1e-3, 1e-2])
    ood_percentile: float = 95.0
    root_height_threshold: float = 0.3
    save_plots: bool = True

    @property
    def exp_name(self) -> str:
        return Path(self.checkpoint).parent.name

    @property
    def save_dir(self) -> Path:
        return Path(self.checkpoint).parent / "eval" / "posterior_prior_match"


def resolve_checkpoint(path: str) -> Path:
    """Resolve an explicit checkpoint or the latest numeric checkpoint in a run."""
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_dir():
        candidates = [(int(match.group(1)), file) for file in checkpoint.glob("model_*.pt")
                      if file.is_file() and (match := re.fullmatch(r"model_(\d+)\.pt", file.name))]
        if not candidates:
            raise FileNotFoundError(f"No model_<iteration>.pt checkpoints in {checkpoint}")
        checkpoint = max(candidates)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def prepare_motion_file(args: EvalArgs, destination: Path) -> Path:
    """Select motions before loading the simulator; preserve YAML order and names."""
    import yaml

    scaletrack_root = Path(__file__).resolve().parents[2]

    def existing_path(value: str, relative_to: Path) -> Path:
        path = Path(value).expanduser()
        for candidate in (path, scaletrack_root / path, relative_to / path):
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Motion file not found: {value}. Supply a local --motion_file YAML.")

    run_dir = Path(args.checkpoint).parent
    if args.motion_file is None:
        config_path = run_dir / "params" / "env.yaml"
        if not config_path.is_file():
            raise ValueError("Supply --motion_file; the checkpoint has no params/env.yaml.")
        # BaseLoader reads only strings/containers, without instantiating Isaac Lab's Python YAML tags.
        with config_path.open() as stream:
            config = yaml.load(stream, Loader=yaml.BaseLoader)
        args.motion_file = config["commands"]["motion"]["motion_file"]
    source = existing_path(args.motion_file, run_dir)
    with source.open() as stream:
        motions = yaml.safe_load(stream)
    if not isinstance(motions, dict) or not motions:
        raise ValueError(f"Expected a nonempty motion-name to NPZ-path mapping in {source}")
    selected = list(motions.items())
    if args.first_n_traj > 0:
        selected = selected[:args.first_n_traj]
    selected = {str(name): str(existing_path(path, source.parent)) for name, path in selected}
    with destination.open("w") as stream:
        yaml.safe_dump(selected, stream, sort_keys=False)
    args.motion_file = str(source)
    return destination


# ============== Data Structures ==============

@dataclass
class TrajectoryLatentData:
    """Aggregated latent data for a trajectory."""
    traj_name: str
    n_steps: int = 0

    # Time series (T, latent_dim)
    posterior_mu: np.ndarray = field(default_factory=lambda: np.array([]))
    posterior_logvar: np.ndarray = field(default_factory=lambda: np.array([]))
    prior_mu: np.ndarray = field(default_factory=lambda: np.array([]))
    prior_logvar: np.ndarray = field(default_factory=lambda: np.array([]))

    # KL per step (T, latent_dim)
    kl_per_dim: np.ndarray = field(default_factory=lambda: np.array([]))
    # KL total per step (T,)
    kl_total: np.ndarray = field(default_factory=lambda: np.array([]))

    # ||μ_q - μ_p|| per step (T,)
    mu_diff_norm: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class DistributionMetrics:
    """Aggregated metrics for posterior-prior matching."""
    n_trajectories: int = 0
    n_samples: int = 0
    latent_dim: int = 0

    # ===== C1: Basic Diagnostics =====

    # KL statistics
    kl_per_dim_mean: np.ndarray = field(default_factory=lambda: np.array([]))  # (latent_dim,)
    kl_per_dim_std: np.ndarray = field(default_factory=lambda: np.array([]))
    kl_total_mean: float = 0.0
    kl_total_std: float = 0.0

    # Active units (for each threshold)
    active_units_count: Dict[float, int] = field(default_factory=dict)  # threshold -> count
    active_units_ratio: Dict[float, float] = field(default_factory=dict)  # threshold -> ratio
    active_units_mask: Dict[float, np.ndarray] = field(default_factory=dict)  # threshold -> (latent_dim,) bool

    # Posterior variance per dimension: Var_x[μ_q_i(x)]
    posterior_mu_variance: np.ndarray = field(default_factory=lambda: np.array([]))  # (latent_dim,)

    # ||μ_q - μ_p|| statistics
    mu_diff_norm_mean: float = 0.0
    mu_diff_norm_std: float = 0.0
    mu_diff_norm_p50: float = 0.0
    mu_diff_norm_p95: float = 0.0
    mu_diff_norm_max: float = 0.0

    # ===== C2: OOD Scoring =====

    # Prior NLL statistics (computed on the selected evaluation motions)
    prior_nll_mean: float = 0.0
    prior_nll_std: float = 0.0
    prior_nll_p95: float = 0.0  # "Normal" threshold

    # Mahalanobis distance statistics
    mahal_dist_mean: float = 0.0
    mahal_dist_std: float = 0.0
    mahal_dist_p95: float = 0.0  # "Normal" threshold

    # Baseline scores (z = μ_p, should be near 0)
    prior_nll_baseline_mean: float = 0.0
    mahal_dist_baseline_mean: float = 0.0

    # ===== Interpretation flags =====

    # Prior covers posterior well
    prior_covers_posterior: bool = False
    # Has collapsed dimensions
    has_collapsed_dims: bool = False
    # Collapse severity
    n_collapsed_dims: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "n_trajectories": int(self.n_trajectories),
            "n_samples": int(self.n_samples),
            "latent_dim": int(self.latent_dim),
            # KL
            "kl_per_dim_mean": self.kl_per_dim_mean.tolist() if len(self.kl_per_dim_mean) > 0 else [],
            "kl_per_dim_std": self.kl_per_dim_std.tolist() if len(self.kl_per_dim_std) > 0 else [],
            "kl_total_mean": float(self.kl_total_mean),
            "kl_total_std": float(self.kl_total_std),
            # Active units
            "active_units_count": {str(k): int(v) for k, v in self.active_units_count.items()},
            "active_units_ratio": {str(k): float(v) for k, v in self.active_units_ratio.items()},
            # Posterior variance
            "posterior_mu_variance": self.posterior_mu_variance.tolist() if len(self.posterior_mu_variance) > 0 else [],
            # Mu diff
            "mu_diff_norm_mean": float(self.mu_diff_norm_mean),
            "mu_diff_norm_std": float(self.mu_diff_norm_std),
            "mu_diff_norm_p50": float(self.mu_diff_norm_p50),
            "mu_diff_norm_p95": float(self.mu_diff_norm_p95),
            "mu_diff_norm_max": float(self.mu_diff_norm_max),
            # OOD thresholds
            "prior_nll_mean": float(self.prior_nll_mean),
            "prior_nll_std": float(self.prior_nll_std),
            "prior_nll_p95": float(self.prior_nll_p95),
            "mahal_dist_mean": float(self.mahal_dist_mean),
            "mahal_dist_std": float(self.mahal_dist_std),
            "mahal_dist_p95": float(self.mahal_dist_p95),
            # Baseline (z = μ_p)
            "prior_nll_baseline_mean": float(self.prior_nll_baseline_mean),
            "mahal_dist_baseline_mean": float(self.mahal_dist_baseline_mean),
            # Interpretation
            "prior_covers_posterior": bool(self.prior_covers_posterior),
            "has_collapsed_dims": bool(self.has_collapsed_dims),
            "n_collapsed_dims": int(self.n_collapsed_dims),
        }


# ============== C1: KL and Active Units Computation ==============

def compute_kl_divergence(
    mu_q: np.ndarray, logvar_q: np.ndarray,
    mu_p: np.ndarray, logvar_p: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KL divergence KL(q||p) for diagonal Gaussians.

    KL(q||p) = -0.5 * Σ_i (1 + logvar_q_i - logvar_p_i
                          - (mu_q_i - mu_p_i)² / exp(logvar_p_i)
                          - exp(logvar_q_i) / exp(logvar_p_i))

    Args:
        mu_q: (*, latent_dim) posterior mean
        logvar_q: (*, latent_dim) posterior log variance
        mu_p: (*, latent_dim) prior mean
        logvar_p: (*, latent_dim) prior log variance
        eps: numerical stability

    Returns:
        kl_per_dim: (*, latent_dim) KL per dimension
        kl_total: (*,) total KL
    """
    var_p = np.exp(logvar_p) + eps
    var_q = np.exp(logvar_q) + eps

    kl_per_dim = 0.5 * (
        logvar_p - logvar_q  # log(var_p / var_q)
        + var_q / var_p  # var_q / var_p
        + (mu_q - mu_p) ** 2 / var_p  # (mu_q - mu_p)^2 / var_p
        - 1  # -1
    )

    kl_total = kl_per_dim.sum(axis=-1)
    return kl_per_dim, kl_total


def compute_prior_nll(
    z: np.ndarray,
    mu_p: np.ndarray,
    logvar_p: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute negative log likelihood under prior: -log p(z|c).

    For diagonal Gaussian:
    NLL = 0.5 * Σ_i (logvar_p_i + (z_i - mu_p_i)² / exp(logvar_p_i) + log(2π))

    Args:
        z: (*, latent_dim) latent samples
        mu_p: (*, latent_dim) prior mean
        logvar_p: (*, latent_dim) prior log variance

    Returns:
        nll: (*,) negative log likelihood
    """
    var_p = np.exp(logvar_p) + eps
    log_2pi = np.log(2 * np.pi)

    nll_per_dim = 0.5 * (logvar_p + (z - mu_p) ** 2 / var_p + log_2pi)
    nll = nll_per_dim.sum(axis=-1)
    return nll


def compute_mahalanobis_distance(
    z: np.ndarray,
    mu_p: np.ndarray,
    logvar_p: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute Mahalanobis distance (diagonal simplified version).

    D²(z; c) = Σ_i ((z_i - μ_p_i) / σ_p_i)²

    Args:
        z: (*, latent_dim) latent samples
        mu_p: (*, latent_dim) prior mean
        logvar_p: (*, latent_dim) prior log variance

    Returns:
        dist: (*,) squared Mahalanobis distance
    """
    std_p = np.exp(0.5 * logvar_p) + eps
    normalized_diff = (z - mu_p) / std_p
    dist = (normalized_diff ** 2).sum(axis=-1)
    return dist



def latent_eval_failure(env, root_height_threshold: float):
    """Match the source evaluator's fall/invalid-state guard, before auto-reset."""
    import torch

    motion = env.command_manager.get_term("motion")
    robot = motion.robot.data
    finite = torch.isfinite(robot.root_state_w).all(dim=-1)
    finite &= torch.isfinite(robot.joint_pos).all(dim=-1) & torch.isfinite(robot.joint_vel).all(dim=-1)
    height_error = (robot.root_pos_w[:, 2] - motion.anchor_pos_w[:, 2]).abs()
    return ~finite | (height_error > root_height_threshold)


def collect_latent_data(env, policy, args: EvalArgs, simulation_app) -> List[TrajectoryLatentData]:
    """Evaluate each selected motion once, excluding auto-reset and padded rows."""
    import torch

    motion = env.unwrapped.command_manager.get_term("motion")
    motion.is_evaluating = True  # Reset each assigned motion to frame zero.
    n_motions = motion.num_motion
    if args.first_n_traj > 0:
        n_motions = min(n_motions, args.first_n_traj)
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
                    raise RuntimeError("Simulation closed before evaluation finished; no complete report was written.")
                rows = active.nonzero(as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    break
                proprio, target = policy._get_obs(obs[rows])
                mu_q, logvar_q = policy.model.encoder.encode(target, proprio)
                mu_p, logvar_p = policy.model.prior_encoder.encode(proprio)
                stats = torch.stack((mu_q, logvar_q, mu_p, logvar_p), dim=1)
                if not torch.isfinite(stats).all():
                    raise RuntimeError("Non-finite encoder outputs; refusing to report invalid distribution metrics.")

                mu, logvar = (mu_q, logvar_q) if args.decode_source == "posterior" else (mu_p, logvar_p)
                latent = mu
                if args.use_stochastic:
                    noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
                    latent = mu + args.temperature * (0.5 * logvar).exp() * noise
                decoded = policy.model.decode(latent, proprio)
                if not torch.isfinite(decoded).all():
                    raise RuntimeError("Non-finite decoded actions; evaluation stopped before stepping the simulator.")

                for row, values in zip(rows.cpu().tolist(), stats.cpu().numpy()):
                    buffers[row].append(values.copy())
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                actions[rows] = decoded
                obs, _, dones, _ = env.step(actions)
                # The wrapper returns new-episode observations on done rows. Never record them.
                active &= ~dones.bool()
                active[:batch_size] &= (step + 1 < lengths).to(env.device)

            for row, motion_id in enumerate(ids.tolist()):
                values = np.stack(buffers[row])
                traj = TrajectoryLatentData(
                    traj_name=str(motion.motion_names[motion_id]),
                    n_steps=len(values),
                    posterior_mu=values[:, 0], posterior_logvar=values[:, 1],
                    prior_mu=values[:, 2], prior_logvar=values[:, 3],
                )
                traj.kl_per_dim, traj.kl_total = compute_kl_divergence(
                    traj.posterior_mu, traj.posterior_logvar, traj.prior_mu, traj.prior_logvar
                )
                traj.mu_diff_norm = np.linalg.norm(traj.posterior_mu - traj.prior_mu, axis=-1)
                trajectories.append(traj)
                print(f"  {traj.traj_name}: {traj.n_steps}/{int(lengths[row])} steps, "
                      f"mean KL={traj.kl_total.mean():.4f}")
    if not trajectories:
        raise ValueError("No trajectories were collected.")
    return trajectories


def histogram_bins(low: float, high: float, count: int) -> np.ndarray:
    """Keep histograms valid when the posterior and prior match exactly."""
    if high <= low:
        margin = max(abs(low) * 0.01, 0.5)
        low, high = low - margin, low + margin
    return np.linspace(low, high, count)

# ============== Metrics Computation ==============

def compute_distribution_metrics(
    trajectories: List[TrajectoryLatentData],
    active_thresholds: List[float],
    ood_percentile: float = 95.0,
) -> DistributionMetrics:
    """Compute aggregated metrics from trajectory data."""
    metrics = DistributionMetrics()

    if not trajectories:
        return metrics

    # Collect all data
    all_posterior_mu = np.concatenate([t.posterior_mu for t in trajectories], axis=0)
    all_prior_mu = np.concatenate([t.prior_mu for t in trajectories], axis=0)
    all_prior_logvar = np.concatenate([t.prior_logvar for t in trajectories], axis=0)
    all_kl_per_dim = np.concatenate([t.kl_per_dim for t in trajectories], axis=0)
    all_kl_total = np.concatenate([t.kl_total for t in trajectories], axis=0)
    all_mu_diff_norm = np.concatenate([t.mu_diff_norm for t in trajectories], axis=0)

    metrics.n_trajectories = len(trajectories)
    metrics.n_samples = len(all_posterior_mu)
    metrics.latent_dim = all_posterior_mu.shape[1]

    # ===== C1: KL Statistics =====
    metrics.kl_per_dim_mean = all_kl_per_dim.mean(axis=0)
    metrics.kl_per_dim_std = all_kl_per_dim.std(axis=0)
    metrics.kl_total_mean = float(all_kl_total.mean())
    metrics.kl_total_std = float(all_kl_total.std())

    # ===== C1: Active Units =====
    # Active unit: Var_x[μ_q_i(x)] > τ
    metrics.posterior_mu_variance = all_posterior_mu.var(axis=0)

    for threshold in active_thresholds:
        active_mask = metrics.posterior_mu_variance > threshold
        metrics.active_units_mask[threshold] = active_mask
        metrics.active_units_count[threshold] = int(active_mask.sum())
        metrics.active_units_ratio[threshold] = float(active_mask.mean())

    # ===== C1: ||μ_q - μ_p|| Statistics =====
    metrics.mu_diff_norm_mean = float(all_mu_diff_norm.mean())
    metrics.mu_diff_norm_std = float(all_mu_diff_norm.std())
    metrics.mu_diff_norm_p50 = float(np.percentile(all_mu_diff_norm, 50))
    metrics.mu_diff_norm_p95 = float(np.percentile(all_mu_diff_norm, 95))
    metrics.mu_diff_norm_max = float(all_mu_diff_norm.max())

    # ===== C2: OOD Scoring =====
    # Test BOTH posterior and prior as z sources (no randomness, use mean directly)
    #
    # Posterior z = μ_q: This is what tracking uses. Measures how well prior covers posterior.
    # Prior z = μ_p: This is a baseline (should score well since z is from prior itself).

    # --- Posterior as z (z = μ_q) ---
    z_posterior = all_posterior_mu

    prior_nll_posterior = compute_prior_nll(z_posterior, all_prior_mu, all_prior_logvar)
    metrics.prior_nll_mean = float(prior_nll_posterior.mean())
    metrics.prior_nll_std = float(prior_nll_posterior.std())
    metrics.prior_nll_p95 = float(np.percentile(prior_nll_posterior, ood_percentile))

    mahal_dist_posterior = compute_mahalanobis_distance(z_posterior, all_prior_mu, all_prior_logvar)
    metrics.mahal_dist_mean = float(mahal_dist_posterior.mean())
    metrics.mahal_dist_std = float(mahal_dist_posterior.std())
    metrics.mahal_dist_p95 = float(np.percentile(mahal_dist_posterior, ood_percentile))

    # --- Prior as z (z = μ_p, baseline) ---
    z_prior = all_prior_mu

    prior_nll_prior = compute_prior_nll(z_prior, all_prior_mu, all_prior_logvar)
    mahal_dist_prior = compute_mahalanobis_distance(z_prior, all_prior_mu, all_prior_logvar)

    # Store prior baseline scores (should be near 0 for Mahalanobis, low for NLL)
    metrics.prior_nll_baseline_mean = float(prior_nll_prior.mean())
    metrics.mahal_dist_baseline_mean = float(mahal_dist_prior.mean())

    # ===== Interpretation =====
    # Prior covers posterior if mean Mahalanobis distance is small
    # (z from posterior should be near prior mean)
    metrics.prior_covers_posterior = metrics.mahal_dist_mean < metrics.latent_dim * 2

    # Check for collapsed dimensions (very low variance)
    collapse_threshold = 1e-4
    collapsed_mask = metrics.posterior_mu_variance < collapse_threshold
    metrics.has_collapsed_dims = collapsed_mask.any()
    metrics.n_collapsed_dims = int(collapsed_mask.sum())

    return metrics


# ============== Visualization ==============

def plot_distribution_analysis(
    trajectories: List[TrajectoryLatentData],
    metrics: DistributionMetrics,
    save_dir: Path,
    exp_name: str,
    ood_percentile: float = 95.0,
):
    """Generate C3 visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect all data
    all_posterior_mu = np.concatenate([t.posterior_mu for t in trajectories], axis=0)
    all_prior_mu = np.concatenate([t.prior_mu for t in trajectories], axis=0)

    # ===== Plot 1: PCA 2D Projection =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Posterior vs Prior Distribution: {exp_name}", fontsize=14)

    # Fit PCA on posterior μ
    center = all_posterior_mu.mean(axis=0)
    _, _, components = np.linalg.svd(all_posterior_mu - center, full_matrices=False)
    projection = components[:2].T
    posterior_mu_2d = (all_posterior_mu - center) @ projection
    prior_mu_2d = (all_prior_mu - center) @ projection
    # A one-sample/one-dimensional smoke run still gets a valid 2D plot.
    if posterior_mu_2d.shape[1] == 1:
        posterior_mu_2d = np.pad(posterior_mu_2d, ((0, 0), (0, 1)))
        prior_mu_2d = np.pad(prior_mu_2d, ((0, 0), (0, 1)))

    # Left: scatter plot
    ax = axes[0]
    ax.scatter(posterior_mu_2d[:, 0], posterior_mu_2d[:, 1],
               alpha=0.3, s=5, c='blue', label=r'$\mu_q$ (posterior)')
    ax.scatter(prior_mu_2d[:, 0], prior_mu_2d[:, 1],
               alpha=0.3, s=5, c='red', label=r'$\mu_p$ (prior)')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Projection (fitted on μ_q)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: density contours
    ax = axes[1]

    # Simple 2D histogram for density
    h_posterior = ax.hist2d(posterior_mu_2d[:, 0], posterior_mu_2d[:, 1],
                            bins=50, cmap='Blues', alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Posterior μ_q Density (PCA)')
    plt.colorbar(h_posterior[3], ax=ax, label='Count')

    plt.tight_layout()
    plt.savefig(save_dir / "pca_projection.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== Plot 2: KL per Dimension =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"KL Divergence Analysis: {exp_name}", fontsize=14)

    latent_dim = metrics.latent_dim
    dims = np.arange(latent_dim)

    # Left: KL per dim bar plot
    ax = axes[0]
    ax.bar(dims, metrics.kl_per_dim_mean, yerr=metrics.kl_per_dim_std,
           capsize=2, alpha=0.7, color='steelblue')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('KL(q||p)')
    ax.set_title('KL per Dimension (mean ± std)')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight collapsed dims
    if metrics.has_collapsed_dims:
        collapsed_mask = metrics.posterior_mu_variance < 1e-4
        for i, collapsed in enumerate(collapsed_mask):
            if collapsed:
                ax.axvline(i, color='red', alpha=0.3, linewidth=2)

    # Right: Posterior variance per dim
    ax = axes[1]
    ax.bar(dims, metrics.posterior_mu_variance, alpha=0.7, color='coral')
    for threshold in metrics.active_units_count:
        ax.axhline(threshold, linestyle='--', label=f'τ={threshold:g}')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel(r'$Var_x[\mu_q(x)]$')
    ax.set_title('Posterior Mean Variance (Active Units)')
    if np.any(metrics.posterior_mu_variance > 0):
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / "kl_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== Plot 3: 1D Marginal Distributions =====
    # Select top 6 most active dimensions
    top_dims = np.argsort(metrics.posterior_mu_variance)[-6:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"1D Marginal Distributions (Top 6 Active Dims): {exp_name}", fontsize=14)

    for idx, dim in enumerate(top_dims):
        ax = axes.flat[idx]

        posterior_vals = all_posterior_mu[:, dim]
        prior_vals = all_prior_mu[:, dim]

        # Histograms
        bins = histogram_bins(
            min(posterior_vals.min(), prior_vals.min()),
            max(posterior_vals.max(), prior_vals.max()),
            50
        )
        ax.hist(posterior_vals, bins=bins, alpha=0.5, density=True,
                label=r'$\mu_q$', color='blue')
        ax.hist(prior_vals, bins=bins, alpha=0.5, density=True,
                label=r'$\mu_p$', color='red')

        ax.set_xlabel(f'Dim {dim}')
        ax.set_ylabel('Density')
        ax.set_title(f'Dim {dim} (var={metrics.posterior_mu_variance[dim]:.4f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[len(top_dims):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_dir / "marginal_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== Plot 4: OOD Score Distributions (C2 Visualization) =====
    # Compare z = μ_q (posterior, what tracking uses) vs z = μ_p (prior baseline)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"C2: OOD Score Distributions - Posterior vs Prior Baseline: {exp_name}", fontsize=14)

    # Compute OOD scores for BOTH z sources
    all_prior_logvar = np.concatenate([t.prior_logvar for t in trajectories], axis=0)

    # z = μ_q (posterior mean) - what tracking uses
    z_posterior = all_posterior_mu
    prior_nll_posterior = compute_prior_nll(z_posterior, all_prior_mu, all_prior_logvar)
    mahal_dist_posterior = compute_mahalanobis_distance(z_posterior, all_prior_mu, all_prior_logvar)

    # z = μ_p (prior mean) - baseline (should score well)
    z_prior = all_prior_mu
    prior_nll_prior = compute_prior_nll(z_prior, all_prior_mu, all_prior_logvar)
    mahal_dist_prior = compute_mahalanobis_distance(z_prior, all_prior_mu, all_prior_logvar)

    # Top-Left: Prior NLL comparison
    ax = axes[0, 0]
    bins_nll = histogram_bins(
        min(prior_nll_posterior.min(), prior_nll_prior.min()),
        np.percentile(prior_nll_posterior, 99),  # Clip outliers for visibility
        50
    )
    ax.hist(prior_nll_posterior, bins=bins_nll, alpha=0.7, color='steelblue',
            density=True, label=r'$z = \mu_q$ (posterior)')
    ax.hist(prior_nll_prior, bins=bins_nll, alpha=0.7, color='orange',
            density=True, label=r'$z = \mu_p$ (baseline)')
    ax.axvline(metrics.prior_nll_p95, color='red', linestyle='--',
               label=f'{ood_percentile:g}% threshold: {metrics.prior_nll_p95:.2f}')
    ax.set_xlabel('Prior NLL: -log p(z|c)')
    ax.set_ylabel('Density')
    ax.set_title('Prior NLL: Posterior vs Prior Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-Right: Mahalanobis distance comparison
    ax = axes[0, 1]
    bins_mahal = histogram_bins(
        0,
        np.percentile(mahal_dist_posterior, 99),  # Clip outliers for visibility
        50
    )
    ax.hist(mahal_dist_posterior, bins=bins_mahal, alpha=0.7, color='coral',
            density=True, label=r'$z = \mu_q$ (posterior)')
    ax.hist(mahal_dist_prior, bins=bins_mahal, alpha=0.7, color='green',
            density=True, label=r'$z = \mu_p$ (baseline)')
    ax.axvline(metrics.mahal_dist_p95, color='red', linestyle='--',
               label=f'{ood_percentile:g}% threshold: {metrics.mahal_dist_p95:.2f}')
    ax.set_xlabel(r'Mahalanobis $D^2(z; c)$')
    ax.set_ylabel('Density')
    ax.set_title('Mahalanobis Distance: Posterior vs Prior Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-Left: Prior NLL (posterior only, detailed)
    ax = axes[1, 0]
    ax.hist(prior_nll_posterior, bins=50, alpha=0.7, color='steelblue', density=True)
    ax.axvline(metrics.prior_nll_mean, color='blue', linestyle='-',
               label=f'Mean: {metrics.prior_nll_mean:.2f}')
    ax.axvline(metrics.prior_nll_p95, color='red', linestyle='--',
               label=f'P{ood_percentile:g}: {metrics.prior_nll_p95:.2f}')
    ax.set_xlabel('Prior NLL: -log p(z|c)')
    ax.set_ylabel('Density')
    ax.set_title(r'Prior NLL Distribution ($z = \mu_q$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-Right: Mahalanobis (posterior only, detailed)
    ax = axes[1, 1]
    ax.hist(mahal_dist_posterior, bins=50, alpha=0.7, color='coral', density=True)
    ax.axvline(metrics.mahal_dist_mean, color='blue', linestyle='-',
               label=f'Mean: {metrics.mahal_dist_mean:.2f}')
    ax.axvline(metrics.mahal_dist_p95, color='red', linestyle='--',
               label=f'P{ood_percentile:g}: {metrics.mahal_dist_p95:.2f}')
    ax.set_xlabel(r'Mahalanobis $D^2(z; c)$')
    ax.set_ylabel('Density')
    ax.set_title(r'Mahalanobis Distribution ($z = \mu_q$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "ood_scores.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ===== Plot 5: ||μ_q - μ_p|| Distribution =====
    fig, ax = plt.subplots(figsize=(8, 5))

    all_mu_diff = np.concatenate([t.mu_diff_norm for t in trajectories])
    ax.hist(all_mu_diff, bins=50, alpha=0.7, color='purple', density=True)
    ax.axvline(metrics.mu_diff_norm_p50, color='blue', linestyle='-',
               label=f'P50: {metrics.mu_diff_norm_p50:.3f}')
    ax.axvline(metrics.mu_diff_norm_p95, color='red', linestyle='--',
               label=f'P95: {metrics.mu_diff_norm_p95:.3f}')
    ax.set_xlabel(r'$||\mu_q - \mu_p||$')
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior-Prior Mean Distance: {exp_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "mu_diff_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved to {save_dir}")



def save_results(args: EvalArgs, trajectories: List[TrajectoryLatentData], metrics: DistributionMetrics):
    """Keep the original metric/NPZ schema, rooted in the checkpoint directory."""
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    results_data = {
        "config": {
            "exp_name": args.exp_name,
            "checkpoint": args.checkpoint,
            "motion_file": args.motion_file,
            "num_envs": args.num_envs,
            "decode_source": args.decode_source,
            "use_stochastic": args.use_stochastic,
            "temperature": args.temperature if args.use_stochastic else None,
            "first_n_traj": args.first_n_traj,
            "max_steps": args.max_steps,
            "root_height_threshold": args.root_height_threshold,
            "active_unit_thresholds": args.active_unit_thresholds,
            "ood_percentile": args.ood_percentile,
            "seed": args.seed,
        },
        "metrics": metrics.to_dict(),
        "trajectories": [],
    }
    # Validate JSON numbers before writing any report (json otherwise accepts NaN/Infinity).
    json.dumps(results_data, allow_nan=False)
    for index, traj in enumerate(trajectories):
        safe_name = re.sub(r"[^\w.-]+", "_", traj.traj_name)[:160]
        filename = f"traj_{index:05d}_{safe_name}.npz"
        results_data["trajectories"].append({"name": traj.traj_name, "n_steps": traj.n_steps, "file": filename})
        np.savez(
            save_dir / filename,
            posterior_mu=traj.posterior_mu,
            posterior_logvar=traj.posterior_logvar,
            prior_mu=traj.prior_mu,
            prior_logvar=traj.prior_logvar,
            kl_per_dim=traj.kl_per_dim,
            kl_total=traj.kl_total,
            mu_diff_norm=traj.mu_diff_norm,
        )
    with (save_dir / "posterior_prior_metrics.json").open("w") as stream:
        json.dump(results_data, stream, indent=2, allow_nan=False)
    ood_thresholds = {
        "prior_nll_threshold": metrics.prior_nll_p95,
        "mahal_dist_threshold": metrics.mahal_dist_p95,
        "decode_source": args.decode_source,
        "use_stochastic": args.use_stochastic,
        "ood_percentile": args.ood_percentile,
        "description": (
            f"{args.ood_percentile:g}th percentile thresholds for z=posterior mean from "
            f"{args.decode_source} {'stochastic' if args.use_stochastic else 'deterministic'} rollout"
        ),
    }
    with (save_dir / "ood_thresholds.json").open("w") as stream:
        json.dump(ood_thresholds, stream, indent=2, allow_nan=False)
    print(f"\nResults saved to: {save_dir}")

def print_summary(metrics: DistributionMetrics, args: EvalArgs):
    """Print formatted summary."""
    stochastic_str = "stochastic" if args.use_stochastic else "deterministic"
    decode_str = "posterior (μ_q)" if args.decode_source == "posterior" else "prior (μ_p)"

    print("\n" + "=" * 80)
    print(f"EXPERIMENT C: POSTERIOR VS PRIOR DISTRIBUTION MATCHING")
    print(f"  Decode source: {decode_str}, Mode: {stochastic_str}")
    print("=" * 80)

    print(f"\nData Summary:")
    print(f"  Trajectories: {metrics.n_trajectories}")
    print(f"  Total samples: {metrics.n_samples}")
    print(f"  Latent dimension: {metrics.latent_dim}")

    print(f"\n{'='*40}")
    print("C1: KL Divergence Statistics")
    print(f"{'='*40}")
    print(f"  KL total: {metrics.kl_total_mean:.4f} ± {metrics.kl_total_std:.4f}")
    if len(metrics.kl_per_dim_mean) > 0:
        print(f"  KL per-dim range: [{metrics.kl_per_dim_mean.min():.4f}, {metrics.kl_per_dim_mean.max():.4f}]")

    print(f"\nActive Units (Var_x[μ_q_i] > τ):")
    for threshold, count in sorted(metrics.active_units_count.items()):
        ratio = metrics.active_units_ratio[threshold]
        print(f"  τ = {threshold:.0e}: {count}/{metrics.latent_dim} dims ({ratio:.1%})")

    print(f"\n||μ_q - μ_p|| Statistics:")
    print(f"  Mean: {metrics.mu_diff_norm_mean:.4f}")
    print(f"  Std:  {metrics.mu_diff_norm_std:.4f}")
    print(f"  P50:  {metrics.mu_diff_norm_p50:.4f}")
    print(f"  P95:  {metrics.mu_diff_norm_p95:.4f}")
    print(f"  Max:  {metrics.mu_diff_norm_max:.4f}")

    print(f"\n{'='*40}")
    print("C2: OOD Scoring (z = μ_q vs z = μ_p)")
    print(f"{'='*40}")
    print(f"\n  When z = μ_q (posterior mean, used in tracking):")
    print(f"    Prior NLL:    {metrics.prior_nll_mean:.2f} ± {metrics.prior_nll_std:.2f}  (P{args.ood_percentile:g}: {metrics.prior_nll_p95:.2f})")
    print(f"    Mahalanobis:  {metrics.mahal_dist_mean:.2f} ± {metrics.mahal_dist_std:.2f}  (P{args.ood_percentile:g}: {metrics.mahal_dist_p95:.2f})")
    print(f"\n  When z = μ_p (prior mean, baseline):")
    print(f"    Prior NLL:    {metrics.prior_nll_baseline_mean:.2f}  (should be low)")
    print(f"    Mahalanobis:  {metrics.mahal_dist_baseline_mean:.2f}  (should be ~0)")
    print(f"\n  OOD Thresholds ({args.ood_percentile:g}th percentile of z=μ_q):")
    print(f"    Prior NLL threshold:    {metrics.prior_nll_p95:.2f}")
    print(f"    Mahalanobis threshold:  {metrics.mahal_dist_p95:.2f}")

    print(f"\n{'='*40}")
    print("Interpretation")
    print(f"{'='*40}")
    print(f"  Prior covers posterior: {'✓' if metrics.prior_covers_posterior else '✗'}")
    print(f"  Has collapsed dimensions: {'✗ (bad)' if metrics.has_collapsed_dims else '✓ (good)'}")
    if metrics.has_collapsed_dims:
        print(f"    → {metrics.n_collapsed_dims} dimensions with Var < 1e-4")

    print(f"\nUsage for High-Level RL:")
    print(f"  To check if sampled z is in-distribution:")
    print(f"    1. Compute Mahalanobis: D²(z;c) = Σ_i ((z_i - μ_p_i) / σ_p_i)²")
    print(f"    2. Flag OOD if D² > {metrics.mahal_dist_p95:.2f}")
    print(f"  Or use Prior NLL:")
    print(f"    1. Compute NLL = -log p(z|c)")
    print(f"    2. Flag OOD if NLL > {metrics.prior_nll_p95:.2f}")


def print_config_summary(args: EvalArgs):
    """Print experiment configuration summary."""
    stochastic_str = "stochastic (z = μ + σ·ε)" if args.use_stochastic else "deterministic (z = μ)"
    decode_str = "posterior (μ_q)" if args.decode_source == "posterior" else "prior (μ_p)"

    print(f"\n{'='*60}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Decode source:  {decode_str}")
    print(f"  Stochasticity:  {stochastic_str}")
    if args.use_stochastic:
        print(f"  Temperature:    {args.temperature}")
    print(f"  Trajectories:   {args.first_n_traj}")
    print(f"  Seed:           {args.seed}")
    print()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion_file", help="Motion YAML; defaults to params/env.yaml's training motion file.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--first_n_traj", type=int, default=5, help="First N motions in YAML order; <= 0 selects all.")
    parser.add_argument("--max_steps", type=int, default=0, help="Per-motion step cap; 0 plays the full motion.")
    parser.add_argument("--decode_source", choices=("posterior", "prior"), default="posterior")
    parser.add_argument("--use_stochastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active_unit_thresholds", type=float, nargs="+", default=[1e-3, 1e-2])
    parser.add_argument("--ood_percentile", type=float, default=95.0)
    parser.add_argument("--root_height_threshold", type=float, default=0.3, help="Fall cutoff in meters.")
    parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    AppLauncher.add_app_launcher_args(parser)
    # AppLauncher pre-parses arguments, so required inputs must be registered afterward.
    parser.add_argument("--checkpoint", required=True, help="Student .pt checkpoint or run directory (latest iteration).")
    args_cli, hydra_args = parser.parse_known_args()
    if args_cli.num_envs < 1 or args_cli.max_steps < 0:
        parser.error("--num_envs must be positive and --max_steps must be nonnegative")
    if not np.isfinite(args_cli.temperature) or args_cli.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    if not 0 < args_cli.ood_percentile < 100:
        parser.error("--ood_percentile must lie strictly between 0 and 100")
    if not np.isfinite(args_cli.root_height_threshold) or args_cli.root_height_threshold <= 0:
        parser.error("--root_height_threshold must be finite and positive")
    if any(not np.isfinite(t) or t < 0 for t in args_cli.active_unit_thresholds):
        parser.error("--active_unit_thresholds must be finite and nonnegative")

    import torch
    from my_rsl_rl.modules import VAEPolicy

    args = EvalArgs(**{name: getattr(args_cli, name) for name in EvalArgs.__dataclass_fields__})
    # Fail on legacy/malformed checkpoints before paying the simulator startup cost.
    try:
        args.checkpoint = str(resolve_checkpoint(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError(f"Unsupported distillation checkpoint version: {checkpoint.get('format_version')}")
        policy = VAEPolicy.from_checkpoint(checkpoint)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.save_dir}")
    print_config_summary(args)
    sys.argv = [sys.argv[0]] + hydra_args
    with tempfile.TemporaryDirectory(prefix="scaletrack_posterior_prior_") as temp_dir:
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
                # Only motion-end and fall/invalid-state resets should end collection.
                for name, term in list(vars(env_cfg.terminations).items()):
                    if name != "motion_time_out" and hasattr(term, "func"):
                        setattr(env_cfg.terminations, name, None)
                env_cfg.terminations.latent_eval_failure = TerminationTermCfg(
                    func=latent_eval_failure, params={"root_height_threshold": args.root_height_threshold}
                )
                env = gym.make(checkpoint["task_name"], cfg=env_cfg)
                try:
                    env = DistillationVecEnvWrapper(RslRlVecEnvWrapper(env), checkpoint["observation_config"])
                    env.validate_action_config(checkpoint["action_config"])
                    policy.to(env.device)
                    trajectories = collect_latent_data(env, policy, args, simulation_app)
                    metrics = compute_distribution_metrics(
                        trajectories, args.active_unit_thresholds, args.ood_percentile
                    )
                    print_summary(metrics, args)
                    save_results(args, trajectories, metrics)
                    if args.save_plots:
                        plot_distribution_analysis(
                            trajectories, metrics, args.save_dir, args.exp_name, args.ood_percentile
                        )
                finally:
                    env.close()

            evaluate()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
