"""Episode-internal replacement of the upstream, native-clock tracking command."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from isaaclab.utils import configclass

from .commands import MotionCommand, MotionCommandCfg, load_motion_data_worker
from .switch_math import recovery_fraction, registration, source_weights, support_side, transform, yaw_coefficients


class SwitchMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if abs(env.step_dt - 0.02) > 1e-7:
            raise ValueError("The switch task requires one policy step per native 50 Hz frame")
        self._rng = torch.Generator(device="cpu").manual_seed(int(env.cfg.seed) + 3181)
        self._weights = source_weights(self.motion_names_train, cfg.source_masses)
        self._training_weight = self._weights * (self.time_totals_train > cfg.recovery_steps + 32)
        if cfg.automatic_switches and not self._training_weight.any():
            raise ValueError("Training split needs a clip long enough for the deadline and critic lookahead")
        self._training_iteration = 0
        self._yaw = yaw_coefficients(torch.zeros(self.num_envs, device=self.device))
        self._translation = torch.zeros(self.num_envs, 3, device=self.device)
        self.switch_age = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        if cfg.recovery_steps < 1 or cfg.max_burst_grace_steps < cfg.recovery_steps:
            raise ValueError("Burst grace must be at least one positive single-switch recovery window")
        self._burst_age = torch.full_like(self.switch_age, -1)
        self._countdown = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._fixed_motion = torch.full_like(self._countdown, -1)
        self._fixed_frame = self._fixed_motion.clone()
        self._pending_motion = self._fixed_motion.clone()
        self._pending_frame = self._fixed_motion.clone()
        self._pending_yaw = torch.zeros(self.num_envs)
        self._braking_frames = None
        self._foot_indices = [self.cfg.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        self.body_weight = (self.robot.data.default_mass.sum(-1) * 9.81).to(self.device)
        self.previous_torque = torch.zeros_like(self.robot.data.joint_pos)
        self._previous_torque_valid = torch.zeros_like(self.switch_age, dtype=torch.bool)
        self.metrics["hard_switches"] = torch.zeros_like(self.switch_age, dtype=torch.float)
        self.metrics["sham_switches"] = torch.zeros_like(self.metrics["hard_switches"])
        for key in ("matched", "opposite_support", "random_phase", "turn", "braking", "rapid", "match_unavailable"):
            self.metrics[f"switch_{key}"] = torch.zeros_like(self.metrics["hard_switches"])

    def _load_motion_arrays(self, motions, body_indexes):
        # Kit is already running. Avoid forking a CUDA/PhysX process. Allocate once
        # (the original multiprocessing loader retains copies of the entire split).
        lengths = []
        for path in motions.values():
            with np.load(path, allow_pickle=False) as clip:
                lengths.append(len(clip["joint_pos"]))
        arrays = []
        fields = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        start = 0
        for i, (name, path) in enumerate(motions.items()):
            clip = load_motion_data_worker((i, name, path, body_indexes))
            if clip is None:
                raise ValueError(f"Invalid native motion at index {i}")
            if not arrays:
                arrays = [np.empty((sum(lengths), *clip[k].shape[1:]), dtype=np.float32) for k in fields]
            for out, field in zip(arrays, fields):
                out[start:start + lengths[i]] = clip[field]
            start += lengths[i]
        return list(motions), *arrays, np.asarray(lengths, dtype=np.int64)

    def load_motions(self, motions, body_indexes, set_name):
        if not self.is_distributed:
            names, *arrays = self._load_motion_arrays(motions, body_indexes)
            return names, *(torch.from_numpy(a) for a in arrays)

        # Linux node-local RAM. Open all ranks' private mappings before unlinking
        # the job-specific files. A later job cannot attach to this split, and a
        # peer's Python resource_tracker cannot prematurely unlink the mapping.
        tag = [uuid4().hex if self.gpu_global_rank == 0 else None]
        torch.distributed.broadcast_object_list(tag, src=0)
        folder = Path("/dev/shm") / f"scaletrack_{set_name}_{tag[0]}"
        paths = [folder / f"{i}.npy" for i in range(7)]
        leader = self.gpu_local_rank == 0
        if leader:
            folder.mkdir(mode=0o700)
        try:
            if leader:
                _, *arrays = self._load_motion_arrays(motions, body_indexes)
                required = sum(a.nbytes + 1024 for a in arrays)
                if shutil.disk_usage(folder).free < required:
                    raise RuntimeError(f"Native split requires {required} free bytes of node-local shared memory")
                for path, array in zip(paths, arrays):
                    np.save(path, array, allow_pickle=False)
                del arrays, array
            torch.distributed.barrier()
            tensors = [torch.from_numpy(np.load(path, mmap_mode="c", allow_pickle=False)) for path in paths]
            torch.distributed.barrier()
        finally:
            if leader:
                for path in paths:
                    path.unlink(missing_ok=True)
                folder.rmdir()
        return list(motions), *tensors

    def _map_reference(self, value, field):
        return transform(value, field, self._yaw, self._translation)

    # Current reference is read repeatedly by rewards/metrics. Reuse the same
    # native-index and mapped values until the next commit or clock tick.
    @property
    def body_pos_w(self):
        return self.body_pos_w_future_manual((0,))[:, 0]

    @property
    def body_quat_w(self):
        return self.body_quat_w_future_manual((0,))[:, 0]

    @property
    def body_lin_vel_w(self):
        return self.body_lin_vel_w_future_manual((0,))[:, 0]

    @property
    def body_ang_vel_w(self):
        return self.body_ang_vel_w_future_manual((0,))[:, 0]

    @property
    def clean_teacher_mask(self):
        return self.switch_age < 0

    @property
    def recovery_fraction(self):
        # A new target may soften an individual transition, but cannot repeatedly
        # restart the deadline. The burst clock imposes a monotonic weight floor.
        single = recovery_fraction(self.switch_age, self.cfg.recovery_steps)
        burst = recovery_fraction(self._burst_age, self.cfg.max_burst_grace_steps)
        return torch.maximum(single, burst)

    def set_training_iteration(self, iteration):
        self._training_iteration = int(iteration)

    def _check_selection(self, env_ids, motion_ids, frames):
        env_ids, motion_ids, frames = (
            torch.as_tensor(v, dtype=torch.long, device="cpu").reshape(-1) for v in (env_ids, motion_ids, frames)
        )
        if len(env_ids) != len(motion_ids) or len(env_ids) != len(frames):
            raise ValueError("Environment, clip and frame vectors must have the same length")
        if ((env_ids < 0) | (env_ids >= self.num_envs)).any() or env_ids.unique().numel() != len(env_ids):
            raise IndexError("Invalid or duplicate switch environment")
        if ((motion_ids < 0) | (motion_ids >= self.num_motion)).any():
            raise IndexError("Invalid switch motion")
        if ((frames < 0) | (frames >= self.time_totals[motion_ids])).any():
            raise IndexError("Switch entry is outside its native clip")
        return env_ids, motion_ids, frames

    def set_reset_reference(self, env_ids, motion_ids, frames):
        ids, mids, frames = self._check_selection(env_ids, motion_ids, frames)
        self._fixed_motion[ids], self._fixed_frame[ids] = mids, frames

    def queue_switch(self, env_ids, motion_ids, frames, yaw_offset=0.0):
        """Submit after the next physics step and before its single observation/history update."""
        ids, mids, frames = self._check_selection(env_ids, motion_ids, frames)
        yaw = torch.as_tensor(yaw_offset, dtype=torch.float32, device="cpu").expand(len(ids))
        if not torch.isfinite(yaw).all():
            raise ValueError("Switch heading must be finite")
        self._pending_motion[ids], self._pending_frame[ids], self._pending_yaw[ids] = mids, frames, yaw

    def switch_to(self, env_ids, motion_ids, frames, yaw_offset=0.0):
        """Commit a reference packet. Only an actual episode reset may touch the articulation."""
        ids, mids, entry = self._check_selection(env_ids, motion_ids, frames)
        d = ids.to(self.device)
        turn = torch.as_tensor(yaw_offset, dtype=self._yaw.dtype, device=self.device).expand(len(ids))
        if not torch.isfinite(turn).all():
            raise ValueError("Switch heading must be finite")
        anchor = self.motion_anchor_body_index
        old = self._global_time_index()[ids]
        target = self.time_offsets[mids] + entry
        old_pos = transform(self.cat_body_pos_w[old, anchor].to(self.device), "body_pos_w", self._yaw[d], self._translation[d])
        old_rot = transform(self.cat_body_quat_w[old, anchor].to(self.device), "body_quat_w", self._yaw[d], None)
        yaw, offset = registration(old_pos, old_rot, self.cat_body_pos_w[target, anchor].to(self.device),
                                   self.cat_body_quat_w[target, anchor].to(self.device), turn)
        rapid = (self.switch_age[d] > 0) & (self.switch_age[d] <= 20)
        self.metrics["switch_rapid"][d] += rapid.float()
        self._yaw[d], self._translation[d] = yaw, offset
        self.motion_ids[ids], self.time_steps[ids] = mids, entry
        fresh_burst = (self.switch_age[d] < 0) | (self.switch_age[d] >= self.cfg.max_burst_grace_steps)
        self._burst_age[d] = torch.where(fresh_burst, 0, self._burst_age[d])
        self.switch_age[d] = 0
        self.metrics["hard_switches"][d] += 1
        self._future_manual_cache.clear()

    def _reset_reference_phase(self, env_ids_cpu):
        # Parent's actual reset writes the selected native state once. A cancelled
        # scheduled B can never escape into the following episode.
        d = env_ids_cpu.to(self.device)
        self._future_manual_cache.clear()
        self._yaw[d] = self._yaw.new_tensor((1, 0, 1, 0))
        self._translation[d] = 0
        self.switch_age[d] = -1
        self._burst_age[d] = -1
        self._previous_torque_valid[d] = False
        self._pending_motion[env_ids_cpu] = -1
        self._countdown[env_ids_cpu] = -1
        if self.cfg.automatic_switches and not self.is_evaluating:
            mids = torch.multinomial(self._weights, len(d), replacement=True, generator=self._rng)
            self.motion_ids[env_ids_cpu] = mids
            self.time_steps[env_ids_cpu] = (torch.rand(len(d), generator=self._rng) * self.time_totals[mids]).long()
            delay = torch.randint(self.cfg.first_switch_range[0], self.cfg.first_switch_range[1] + 1, (len(d),), generator=self._rng)
            active = torch.rand(len(d), generator=self._rng) >= self.cfg.clean_probability
            self._countdown[env_ids_cpu] = torch.where(active, delay, -1)
        else:
            super()._reset_reference_phase(env_ids_cpu)
        fixed = env_ids_cpu[self._fixed_motion[env_ids_cpu] >= 0]
        self.motion_ids[fixed], self.time_steps[fixed] = self._fixed_motion[fixed], self._fixed_frame[fixed]
        self._fixed_motion[fixed] = -1

    def _index_braking_frames(self):
        """Keep native phases with an observed, nonzero half-second braking path."""
        anchor = self.motion_anchor_body_index
        phases = []
        for start, length in zip(self.time_offsets.tolist(), self.time_totals.tolist()):
            count = max(0, length - self.cfg.recovery_steps - 32)
            speed = self.cat_body_lin_vel_w[start:start + length, anchor, :2].norm(dim=-1)
            pos = self.cat_body_pos_w[start:start + length, anchor, :2]
            horizon = min(25, self.cfg.recovery_steps)
            distance = (pos[horizon:horizon+count] - pos[:count]).norm(dim=-1)
            valid = (speed[:count] >= 0.55) & (speed[horizon:horizon+count] <= 0.30) & (distance >= 0.10)
            phases.append(valid.nonzero().flatten())
        self._braking_frames = phases
        self._braking_weights = self._weights * torch.tensor([bool(len(phase)) for phase in phases])

    def _select_entries(self, ids, kinds):
        # Apply source balancing before phase selection. Hard objectives cannot
        # silently move to an easier recording or change the target after t0.
        mids = torch.multinomial(self._training_weight, len(ids), replacement=True, generator=self._rng)
        lengths = self.time_totals[mids] - self.cfg.recovery_steps - 32
        entries = (torch.rand(len(ids), self.cfg.entry_candidates, generator=self._rng) * lengths[:, None]).long()
        indices = self.time_offsets[mids, None] + entries
        old = self._global_time_index()[ids]
        anchor = self.motion_anchor_body_index
        feet = self._foot_indices
        root_speed = self.cat_body_lin_vel_w[indices, anchor, :2].norm(dim=-1)
        old_speed = self.cat_body_lin_vel_w[old, anchor, :2].norm(dim=-1)
        q_error = (self.cat_joint_pos[indices] - self.cat_joint_pos[old, None]).square().mean(dim=-1)
        speed_error = (root_speed - old_speed[:, None]).abs()
        match_score = q_error + speed_error
        support = support_side(self.cat_body_pos_w[indices[..., None], feet], self.cat_body_lin_vel_w[indices[..., None], feet])
        old_support = support_side(self.cat_body_pos_w[old[:, None], feet], self.cat_body_lin_vel_w[old[:, None], feet])
        opposite = (old_support[:, None] * support == -1)
        similar = (support == old_support[:, None]) & (q_error <= self.cfg.matched_q_rms**2) & (
            speed_error <= self.cfg.matched_speed_error)
        matched = match_score.masked_fill(~similar, float("inf"))
        # Reject unavailable matched cases. A difficult jump mislabeled "matched"
        # would defeat the initial clean/easy regression.
        kinds = torch.where((kinds == 0) & ~similar.any(1), -1, kinds)
        kinds = torch.where((kinds == 1) & ~opposite.any(1), 2, kinds)
        choice = torch.zeros_like(kinds)
        choice = torch.where(kinds == 0, matched.argmin(1), choice)
        choice = torch.where(kinds == 1, opposite.int().argmax(1), choice)
        entry = entries[torch.arange(len(ids)), choice]

        kinds = torch.where((kinds == 4) & (old_speed < 0.65), 2, kinds)
        brake_ids = (kinds == 4).nonzero().flatten()
        if len(brake_ids):
            if self._braking_frames is None:
                self._index_braking_frames()
            if self._braking_weights.any():
                braking_clips = torch.multinomial(self._braking_weights, len(brake_ids), replacement=True, generator=self._rng)
                u = torch.rand(len(brake_ids), generator=self._rng).tolist()
                phases = [self._braking_frames[m] for m in braking_clips.tolist()]
                mids[brake_ids] = braking_clips
                entry[brake_ids] = torch.stack([p[int(v * len(p))] for v, p in zip(u, phases)])
            else:
                kinds[brake_ids] = 2
        return mids, entry, kinds

    def _automatic_switch(self, ids):
        stage = sum(self._training_iteration >= n for n in self.cfg.curriculum_iterations)
        # Early episodes mix matched and sham. Add arbitrary phase and opposite
        # single support, then turns, observed braking and short inter-switch gaps.
        if stage == 0:
            kinds = torch.zeros(len(ids), dtype=torch.long)
        else:
            kinds = torch.randint(0, 3 if stage == 1 else 5, (len(ids),), generator=self._rng)
        sham = torch.rand(len(ids), generator=self._rng) < self.cfg.sham_probability
        self.metrics["sham_switches"][ids[sham].to(self.device)] += 1
        ids, kinds = ids[~sham], kinds[~sham]
        if not len(ids):
            return
        mids, entry, kinds = self._select_entries(ids, kinds)
        accepted = kinds >= 0
        self.metrics["switch_match_unavailable"][ids[~accepted].to(self.device)] += 1
        ids, mids, entry, kinds = ids[accepted], mids[accepted], entry[accepted], kinds[accepted]
        if not len(ids):
            return
        turns = (torch.rand(len(ids), generator=self._rng) * 0.5 + 0.25) * math.pi
        sign = torch.where(torch.rand(len(ids), generator=self._rng) < 0.5, -1, 1)
        self.switch_to(ids, mids, entry, torch.where(kinds == 3, turns * sign, 0.0))
        for k, label in enumerate(("matched", "opposite_support", "random_phase", "turn", "braking")):
            self.metrics[f"switch_{label}"][ids[kinds == k].to(self.device)] += 1
        if stage >= 2:
            rapid_ids = ids[torch.rand(len(ids), generator=self._rng) < self.cfg.rapid_probability]
            self._countdown[rapid_ids] = torch.randint(8, 21, (len(rapid_ids),), generator=self._rng)

    def _update_command(self):
        super()._update_command()
        self.switch_age += (self.switch_age >= 0).long()
        self._burst_age += (self._burst_age >= 0).long()
        if self.cfg.automatic_switches and not self.is_evaluating:
            self._countdown -= (self._countdown > 0).long()
            due = (self._countdown == 0).nonzero().flatten()
            self._countdown[due] = -1
            if len(due):
                self._automatic_switch(due)
        queued = (self._pending_motion >= 0).nonzero().flatten()
        if len(queued):
            self.switch_to(queued, self._pending_motion[queued], self._pending_frame[queued], self._pending_yaw[queued])
            self._pending_motion[queued] = -1
        self.previous_torque.copy_(self.robot.data.applied_torque)
        self._previous_torque_valid[:] = True


@configclass
class SwitchMotionCommandCfg(MotionCommandCfg):
    class_type: type = SwitchMotionCommand
    automatic_switches: bool = True
    source_masses: dict = {"bones": 0.45, "lafan": 0.25, "lch": 0.25, "original": 0.05}
    clean_probability: float = 0.35
    sham_probability: float = 0.15
    first_switch_range: tuple[int, int] = (25, 100)
    recovery_steps: int = 35
    max_burst_grace_steps: int = 50
    entry_candidates: int = 24
    matched_q_rms: float = 0.45
    matched_speed_error: float = 0.4
    curriculum_iterations: tuple[int, int] = (50, 150)
    rapid_probability: float = 0.2
