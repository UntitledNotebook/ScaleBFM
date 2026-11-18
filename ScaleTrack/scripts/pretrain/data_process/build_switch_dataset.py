"""Assemble a relocatable 50 Hz G1 corpus with capture- and performer-disjoint splits.

Run from ScaleTrack. All inputs are read-only. Output (including provenance) is
private. --mount-prefix is the path seen by the original motion loader from its
training working directory; symlink the finished dataset there on each host.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml


FPS = 50
SPLITS = ("train", "validation", "holdout")
# The input retarget uses MJCF depth-first hinge order. Isaac exposes the
# articulation breadth-first, including the pelvis before the 29 child links.
MJ_JOINTS = tuple(
    [f"{side}_{part}_joint" for side in ("left", "right")
     for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{part}_joint" for side in ("left", "right")
       for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                    "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
BFS = (0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
       17, 24, 18, 25, 19, 26, 20, 27, 21, 28)
JOINTS = tuple(MJ_JOINTS[i] for i in BFS)
BODIES = ("pelvis",) + tuple(
    "torso_link" if j == "waist_pitch_joint" else j.replace("_joint", "_link") for j in JOINTS
)
FIELDS = {"joint_pos": (29,), "joint_vel": (29,), "body_pos_w": (30, 3),
          "body_quat_w": (30, 4), "body_lin_vel_w": (30, 3), "body_ang_vel_w": (30, 3)}


def ident(s):
    return hashlib.sha256(s.encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Take:
    source: str
    path: Path
    actor: str
    capture: str
    category: str
    native: Path | None = None
    entry: dict | None = None

    @property
    def key(self):
        return f"{self.source}__{ident(self.path.stem)[:20]}"


def canonical_name(name):
    return re.sub(r"(?:^|[_-])mirror(?:ed)?(?=[_-]|$)", "", name, flags=re.I).strip("_-")


def group_small(takes, seed):
    """Keep as many tracks as possible under both actor and synchronous-take isolation.

    A take can have several performers, with performers appearing in other takes.
    Splitting whole connected components would leave some sources just one fold.
    Pick actor folds and one fold per recording and exclude conflicting tracks.
    """
    actors = sorted({t.actor for t in takes})
    if not 3 <= len(actors) <= 8:
        raise ValueError("Expected 3–8 named performers for the small retarget collection")
    by_take = defaultdict(list)
    for t in takes:
        by_take[t.capture].append(t)
    ordered = sorted(by_take, key=lambda k: ident(f"{seed}:{k}"))
    n_eval = max(1, round(len(actors) * .2))
    best = None
    for val in itertools.combinations(actors, n_eval):
        for hold in itertools.combinations(sorted(set(actors) - set(val)), n_eval):
            fold = {a: 1 if a in val else 2 if a in hold else 0 for a in actors}
            # Many optimal assignments discard the same number of bridges.
            # Among them, minimize deviation from 70/15/15 clip counts.
            states = {(0, 0, 0): ()}
            for k in ordered:
                votes = Counter(fold[t.actor] for t in by_take[k])
                next_states = {}
                for counts, choices in states.items():
                    for i, v in sorted(votes.items()):
                        sizes = list(counts)
                        sizes[i] += v
                        next_states.setdefault(tuple(sizes), choices + (i,))
                states = next_states
            for counts, choices in states.items():
                if min(counts) == 0:
                    continue
                total = sum(counts)
                # Integer percentages keep tie handling exact.
                miss = sum(abs(100 * n - pct * total)
                           for n, pct in zip(counts, (70, 15, 15)))
                score = (total, -miss, min(counts), ident(f"{seed}:{val}:{hold}:{choices}"))
                if best is None or score > best[0]:
                    selected = {t.key: SPLITS[i] for k, i in zip(ordered, choices)
                                for t in by_take[k] if fold[t.actor] == i}
                    best = (score, selected)
    if best is None:
        raise ValueError("No train/validation/holdout assignment isolates both actor and take")
    return best[1]


def small_collection(source, root, packed=None):
    takes = []
    for path in sorted(root.glob("*.npz")):
        name = canonical_name(path.stem)
        capture, actor = name.rsplit("_", 1)
        # Exclude named mirror variants when an unmirrored original is present.
        if name != path.stem and (root / f"{name}.npz").exists():
            continue
        takes.append(Take(source, path, source + "/" + actor, source + "/" + capture,
                          re.sub(r"\d+$", "", capture).rstrip("_"),
                          (packed / path.name) if packed else None))
    return takes


def inventory(args):
    lafan = small_collection("lafan", args.lafan)
    lch = small_collection("lch", args.lch, args.lch_packed)
    selected = {**group_small(lafan, args.seed), **group_small(lch, args.seed)}
    takes = lafan + lch
    actors, captures = {}, defaultdict(list)
    for line in args.bones_manifest.read_text().splitlines():
        row = json.loads(line)
        raw = Path(row["source_path"])
        base = canonical_name(raw.stem.split("__")[0])
        actor = "bones/" + row["actor"]
        capture = f"bones/{raw.parent.name}/{base}"
        t = Take("bones", raw, actor, capture, row["category"], Path(row["reference_path"]), row)
        split = {"library": "train", "validation": "validation", "holdout": "holdout"}[row["split"]]
        if actors.setdefault(actor, split) != split:
            raise ValueError("Upstream BONES manifest mixes one performer across splits")
        takes.append(t)
        captures[capture].append((t.key, split))
    for path in sorted(args.original.glob("*.npz")):
        name = canonical_name(path.stem)
        match = re.fullmatch(r"(\d{6})-(.+)__(A\d+)", name)
        if match:
            date, base, actor = match.groups()
            capture = f"bones/{date}/{base}"
            group = f"bones/{actor}"
            # The previous BONES library used this actor-only SHA256 split.
            bucket = hashlib.sha256(actor.encode()).digest()[0]
            prior = "train" if bucket < 205 else "validation" if bucket < 230 else "holdout"
            split = actors.get(group, prior)
        else:
            actor = re.fullmatch(r"(.+)-\d+", name)
            if not actor:
                raise ValueError(f"Unrecognized actor/take naming for original example {name!r}")
            group = "original/" + actor.group(1)
            capture = "original/" + name
            split = "train"
        t = Take("original", path, group, capture, "original", path)
        takes.append(t)
        captures[capture].append((t.key, split))
    for group, items in captures.items():
        votes = Counter(split for _, split in items)
        owner = max(votes, key=lambda s: (votes[s], ident(f"{args.seed}:{group}:{s}")))
        selected.update((key, split) for key, split in items if split == owner)
    if len({t.key for t in takes}) != len(takes):
        raise ValueError("Duplicate source identity in input inventory")
    return takes, selected


def as_rotation(quat):
    return Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4))


def lafan_arrays(raw):
    if float(raw["frequency"]) != FPS or list(raw["split_points"]) != [0, len(raw["qpos"])]:
        raise ValueError("Expected one complete 50 Hz take; never differentiate across internal splits")
    raw_joints, raw_bodies = raw["joint_names"].tolist(), raw["body_names"].tolist()
    if raw_joints[0] != "root" or set(raw_joints[1:]) != set(JOINTS) or len(raw_joints) != 30:
        raise ValueError("Unexpected floating-base G1 joint layout")
    joint_ids = [raw_joints.index(j) - 1 for j in JOINTS]
    body_ids = [raw_bodies.index(b) for b in BODIES]
    qpos = raw["qpos"]
    joints = qpos[:, 7:][:, joint_ids]
    pos = raw["xpos"][:, body_ids]
    quat = raw["xquat"][:, body_ids]
    np.testing.assert_allclose(pos[:, 0], qpos[:, :3], atol=1e-5)
    np.testing.assert_allclose(np.abs((quat[:, 0] * qpos[:, 3:7]).sum(-1)), 1, atol=1e-5)
    # Isaac's body_lin_vel_w is the velocity of the link COM. MuJoCo's
    # cvel lives at the root subtree COM, even for distal links.
    offset = np.broadcast_to(raw["body_ipos"][body_ids], pos.shape)
    com = pos + as_rotation(quat).apply(offset.reshape(-1, 3)).reshape(pos.shape)
    angular_step = (as_rotation(quat[1:]) * as_rotation(quat[:-1]).inv()).as_rotvec()
    angular_step = angular_step.reshape(-1, 30, 3) * FPS
    angular = np.concatenate((angular_step[:1], (angular_step[:-1] + angular_step[1:]) / 2,
                              angular_step[-1:]))
    return {"joint_pos": joints, "joint_vel": np.gradient(joints, axis=0) * FPS,
            "body_pos_w": pos, "body_quat_w": quat, "body_lin_vel_w": np.gradient(com, axis=0) * FPS,
            "body_ang_vel_w": angular}


def load_take(t):
    metadata = {}
    if t.source == "lafan":
        with np.load(t.path) as raw:
            arrays = lafan_arrays(raw)
        return arrays, {"input_fps": FPS, "conversion": "50 Hz retarget, named MuJoCo to Isaac BFS, per-take COM derivatives"}
    with np.load(t.native) as f:
        if float(f["fps"]) != FPS:
            raise ValueError("Unexpected native FPS")
        arrays = {k: f[k] for k in FIELDS}
        if t.source == "bones":
            if f["joint_names"].tolist() != list(JOINTS) or f["body_names"].tolist() != list(BODIES):
                raise ValueError("BONES native array labels disagree with G1")
            meta = t.entry
            rate, start, end = int(meta["source_fps"]), int(meta["start_frame"]), int(meta["end_frame"])
            frames = f["source_frame"]
            if rate != 120 or int(meta["reference_fps"]) != FPS or len(frames) != len(arrays["joint_pos"]):
                raise ValueError("BONES source-frame metadata mismatch")
            np.testing.assert_allclose(frames, start + np.arange(len(frames)) * rate / FPS, atol=1e-7)
            with t.path.open() as stream:
                header = next(csv.reader(stream))
                rows = np.loadtxt(stream, delimiter=",", ndmin=2)
            if not 0 <= start < end <= len(rows) or not start <= frames[0] <= frames[-1] < end:
                raise ValueError("Invalid source range")
            times = rows[:, header.index("Frame")]
            np.testing.assert_array_equal(times, np.arange(len(rows)))
            selectors = np.linspace(0, len(frames) - 1, min(len(frames), 7), dtype=int)
            clock = frames[selectors]
            xyz = np.stack([np.interp(clock, times, rows[:, header.index("root_translate" + axis)])
                            for axis in "XYZ"], axis=1) / 100
            xyz[:, :2] -= xyz[0, :2].copy()
            np.testing.assert_allclose(arrays["body_pos_w"][selectors, 0], xyz, atol=2e-5)
            joints = np.stack([np.interp(clock, times, rows[:, header.index(j + "_dof")])
                               for j in JOINTS], axis=1)
            np.testing.assert_allclose(arrays["joint_pos"][selectors], np.deg2rad(joints), atol=1e-5)
            rotation = Rotation.from_euler("xyz", rows[:, [header.index("root_rotate" + a) for a in "XYZ"]], degrees=True)
            angle = (Slerp(times, rotation)(clock) * as_rotation(arrays["body_quat_w"][selectors, 0]).inv()).magnitude()
            if max(angle) > 1e-4:
                raise ValueError("BONES world quaternion does not match raw CSV")
            metadata = {"source_fps": rate, "source_frames": [start, end],
                        "conversion": "named 120 Hz CSV to native 50 Hz; 7-point raw pose check",
                        "source_contact": meta.get("metrics", {})}
    if t.source == "lch":
        with np.load(t.path) as raw:
            n = len(arrays["joint_pos"])
            if float(raw["fps"]) != FPS or len(raw["dof_pos"]) not in (n, n + 1):
                raise ValueError("LCH cache does not match corrected 50 Hz input")
            idx, orig = raw["lch_index"][:n], raw["orig_t"][:n]
            if not (np.diff(idx) == 1).all() or not np.allclose(np.diff(orig), 3., atol=.03):
                raise ValueError("LCH source-frame mapping is discontinuous")
            np.testing.assert_allclose(arrays["joint_pos"], raw["dof_pos"][:n, BFS], atol=.002)
            dot = (arrays["body_quat_w"][:, 0] * raw["root_rot"][:n, [3, 0, 1, 2]]).sum(-1)
            np.testing.assert_allclose(abs(dot), 1, atol=2e-5)
            offset = arrays["body_pos_w"][0, 0, :2] - raw["root_pos"][0, :2]
            arrays["body_pos_w"][..., :2] -= offset
            np.testing.assert_allclose(arrays["body_pos_w"][:, 0], raw["root_pos"][:n], atol=.001)
            metadata = {"source_fps": 150, "retarget_fps": FPS, "lch_index_first": int(idx[0]),
                        "original_frame_first": float(orig[0]), "original_frame_last": float(orig[-1]),
                        "conversion": "paired Isaac cache; MJCF XYZW pose and raw clock verified at every frame"}
    return arrays, metadata or {"source_fps": FPS, "conversion": "original ScaleTrack tracking format"}


def quality(arrays):
    n = len(arrays["joint_pos"])
    for k, shape in FIELDS.items():
        values = arrays[k]
        if values.shape != (n, *shape) or not np.isfinite(values).all() or n < 3:
            raise ValueError(f"Invalid motion field {k}")
    quat = arrays["body_quat_w"]
    if np.abs(np.linalg.norm(quat, axis=-1) - 1).max() > 1e-4:
        raise ValueError("Non-unit orientation")
    pos = arrays["body_pos_w"]
    # Same fixed material point on both ankle frames. These are kinematic
    # ground-proximity estimates; actual slip is measured in the simulator.
    feet = pos[:, [18, 19]] + as_rotation(quat[:, [18, 19]]).apply([.04525, 0, -.08339]).reshape(n, 2, 3)
    near = feet[..., 2] < np.quantile(feet[..., 2], .10, axis=0) + .035
    planar = np.linalg.norm(np.gradient(feet[..., :2], axis=0) * FPS, axis=-1)
    pelvis = pos[:, 0, :2]
    speed = np.linalg.norm(np.gradient(pelvis, axis=0) * FPS, axis=-1)
    end = float(speed[-min(8, n):].mean())
    peak = int(np.argmax(speed))
    post_peak = np.flatnonzero(speed[peak:] > .18)
    brake = None
    if end < .15 and speed[peak] > .4 and len(post_peak):
        stop_index = min(n - 1, peak + post_peak[-1] + 1)
        brake = float(np.linalg.norm(np.diff(pelvis[peak:stop_index+1], axis=0), axis=-1).sum())
    return {"speed_start_mps": float(speed[:min(8,n)].mean()), "speed_end_mps": end,
            "speed_p95_mps": float(np.quantile(speed, .95)), "braking_path_m": brake,
            "near_ground_left_right": near.mean(0).tolist(),
            "near_ground_foot_speed_p90_mps": float(np.quantile(planar[near], .90)),
            "pelvis_z_range_m": [float(pos[:, 0, 2].min()), float(pos[:, 0, 2].max())]}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_checksums(directory, rows):
    # Standard sha256sum format, relative to the dataset root on either host.
    paths = [f"{s}.yaml" for s in SPLITS] + ["clips.jsonl", "excluded.jsonl", "summary.json"]
    with (directory / "SHA256SUMS").open("w") as stream:
        for name in paths:
            print(f"{file_hash(directory / name)}  {name}", file=stream)
        for row in sorted(rows, key=lambda r: r["key"]):
            print(f"{row['sha256']}  clips/{row['key']}.npz", file=stream)


def build(args, out):
    takes, selected = inventory(args)
    (out / "clips").mkdir()
    manifests = {s: {} for s in SPLITS}
    fingerprints, actors, captures, rows, excluded = {}, {}, {}, [], []
    for t in takes:
        if t.key not in selected:
            excluded.append({"key": t.key, "source": t.source, "input": t.path.name, "actor": t.actor,
                             "take": t.capture, "reason": "overlapping performer/capture folds"})
            continue
        split = selected[t.key]
        arrays, meta = load_take(t)
        metrics = quality(arrays)
        # Re-encoders and different scene origins may have the same joint track.
        # Eliminate redundant tracks even when an input label changed.
        fp = hashlib.sha256(np.asarray(arrays["joint_pos"], dtype="<f4").tobytes()).hexdigest()
        if fp in fingerprints:
            excluded.append({"key": t.key, "source": t.source, "input": t.path.name, "actor": t.actor,
                             "take": t.capture, "reason": "identical joint track", "duplicate_of": fingerprints[fp]})
            continue
        fingerprints[fp] = t.key
        if actors.setdefault(t.actor, split) != split or captures.setdefault(t.capture, split) != split:
            raise ValueError("Performer or recording leaked between splits")
        dest = out / "clips" / (t.key + ".npz")
        if t.source in ("bones", "original"):
            shutil.copyfile(t.native, dest)
        else:
            np.savez_compressed(dest, fps=FPS, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
        digest = file_hash(dest)
        manifests[split][t.key] = str(args.mount_prefix / "clips" / dest.name)
        rows.append({"key": t.key, "source": t.source, "take": t.capture, "actor": t.actor,
                     "input": t.path.name, "input_sha256": file_hash(t.path), "split": split,
                     "category": t.category, "frames": len(arrays["joint_pos"]), "fps": FPS,
                     "bytes": dest.stat().st_size, "sha256": digest, "quality": metrics, **meta})
    counts = Counter((r["source"], r["split"]) for r in rows)
    for s, manifest in manifests.items():
        if any(counts[source, s] == 0 for source in ("lafan", "lch", "bones")):
            raise ValueError(f"Split {s} lacks a required source")
        (out / f"{s}.yaml").write_text(yaml.safe_dump(manifest, sort_keys=True))
    for name, records in (("clips.jsonl", rows), ("excluded.jsonl", excluded)):
        with (out / name).open("w") as stream:
            for record in records:
                print(json.dumps(record, sort_keys=True, allow_nan=False), file=stream)
    write_json(out / "summary.json", {
        "seed": args.seed, "joint_order": JOINTS, "body_order": BODIES, "fps": FPS, "quaternion": "WXYZ",
        "coordinates": "world, z up, metres, radians; linear velocities at link COM",
        "contact_audit": "fixed foot material point near its clip-specific floor; kinematic proxy only",
        "braking_path_m": "pelvis XY path from peak native planar speed to final movement into the settling tail",
        "excluded": dict(Counter(r["reason"] for r in excluded)),
        "counts": {s: {src: counts[src,s] for src in ("lafan","lch","bones","original")} for s in SPLITS},
        "frames_by_split": {s: sum(r["frames"] for r in rows if r["split"] == s) for s in SPLITS},
        "run_stops": {s: sum(r["split"] == s and r["category"] == "run_stop" and
                             r["quality"]["braking_path_m"] is not None and
                             r["quality"]["speed_p95_mps"] > 1.45 for r in rows) for s in SPLITS},
        "clip_bytes": sum(r["bytes"] for r in rows), "manifest_sha256": file_hash(out / "clips.jsonl"),
    })
    write_checksums(out, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lafan", "lch", "lch-packed", "bones-manifest", "original", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mount-prefix", type=Path, default=Path(".switch_local/data"))
    p.add_argument("--seed", type=int, default=42107)
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists; build a new version in an empty destination")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as stage:
        staged = Path(stage) / "assembled"
        staged.mkdir()
        build(args, staged)
        if args.output.exists():
            raise FileExistsError(args.output)
        staged.rename(args.output)
    print((args.output / "summary.json").read_text())


if __name__ == "__main__":
    main()
