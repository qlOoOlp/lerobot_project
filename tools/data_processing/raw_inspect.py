"""Inspect episode data (Hz / dims / consistency / pose-jump) before/around conversion.

Single-file tool, format-agnostic core. Reports per-stream Hz, shapes, value ranges,
cross-modal alignment, per-dataset fps distribution, and pose teleports.

Formats (`--format`):
  - record3d_h5     : raw UMI `episode_*.h5` (pre-conversion preflight)
  - lerobot_dataset : an existing LeRobotDataset (repo_id or local root) — e.g. metaworld;
                      schema/dim/range/pose-jump sanity on already-converted data

    python tools/data_processing/raw_inspect.py --raw-root <ft_data dir> --target-fps 30
    python tools/data_processing/raw_inspect.py --raw-root <ds root> --format lerobot_dataset --per-episode

Needs: numpy, pillow (+ h5py for record3d_h5, + lerobot for lerobot_dataset).

To support another format: write one `iter_<fmt>(root, **opts) -> Iterator[Episode]`
and add it to ITERATORS. Analysis/report is format-agnostic. Pose-jump auto-handles
xyz+rpy (6D) and xyz+rot6d (>=9D, canonical) pose streams.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ANCHOR = "rgb"  # stream used as the episode's clock/reference

# pose "teleport" thresholds (ported from find_pose_jumps.py; calibrated on
# move260626: clean episodes peak ~0.006 m/step, known-bad ~0.20 m/step).
POSE_JUMP_DEFAULTS = {"pos": 0.05, "ratio": 20.0, "rot": 10.0}


# ─────────────────────────────── data model ─────────────────────────────── #
@dataclass
class Stream:
    name: str
    kind: str                       # "image" | "vector" | "scalar"
    count: int
    shape: tuple[int, ...]
    dtype: str
    timestamps: np.ndarray | None = None
    values: np.ndarray | None = None   # full array for vector/scalar (value range)
    sample: np.ndarray | None = None   # one frame for images (shape/range)


@dataclass
class Episode:
    episode_id: str
    streams: list[Stream] = field(default_factory=list)


# ─────────────────────────── loader: record3d h5 ────────────────────────── #
_EP_RE = re.compile(r"episode_(\d+)")
_DEFAULT_DEPTH_SHAPE = (256, 192)


def _ep_num(episode_id: str) -> int:
    m = _EP_RE.search(episode_id)
    return int(m.group(1)) if m else -1


def find_episodes(raw_root: str | Path) -> list[Path]:
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"raw-root does not exist: {root}")
    return sorted(root.glob("episode_*.h5"),
                  key=lambda p: (int(m.group(1)) if (m := _EP_RE.search(p.stem)) else -1, p.name))


def _to_bytes(item: Any) -> bytes:
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, np.void):
        return bytes(item)
    return item.tobytes() if hasattr(item, "tobytes") else bytes(item)


def load_record3d_h5(path: str | Path, use_depth: bool = True,
                     pose_source: str = "pose_relative") -> Episode:
    import h5py
    from PIL import Image

    path = Path(path)
    pose_key = pose_source.removeprefix("record3d/")
    streams: list[Stream] = []

    with h5py.File(path, "r") as h:
        if "record3d/rgb/frames" in h:
            frames, ts = h["record3d/rgb/frames"], np.asarray(h["record3d/rgb/timestamp"][:], float)
            n = int(min(len(frames), len(ts)))
            img = np.asarray(Image.open(io.BytesIO(_to_bytes(frames[0]))).convert("RGB")) if n else None
            streams.append(Stream("rgb", "image", n,
                                  tuple(img.shape) if img is not None else (),
                                  str(img.dtype) if img is not None else "uint8",
                                  timestamps=ts[:n], sample=img))

        if use_depth and "record3d/depth/frames" in h:
            dframes, dts = h["record3d/depth/frames"], np.asarray(h["record3d/depth/timestamp"][:], float)
            n = int(min(len(dframes), len(dts)))
            a = h["record3d/depth/frames"].attrs
            shape = (int(a["height"]), int(a["width"])) if "height" in a and "width" in a else _DEFAULT_DEPTH_SHAPE
            buf = np.frombuffer(_to_bytes(dframes[0]), np.float32) if n else np.empty(0, np.float32)
            dsample = buf.reshape(shape) if buf.size == shape[0] * shape[1] else None
            streams.append(Stream("depth", "image", n, shape, "float32", timestamps=dts[:n], sample=dsample))

        base = f"record3d/{pose_key}"
        if f"{base}/xyzrpy" in h:
            xyz = np.asarray(h[f"{base}/xyzrpy"][:], float)
            pts = np.asarray(h[f"{base}/timestamp"][:], float)
            n = int(min(len(xyz), len(pts)))
            xyz = xyz[:n].reshape(n, -1)
            streams.append(Stream(f"pose:{pose_key}", "vector", n, (xyz.shape[1] if n else 0,),
                                  str(xyz.dtype), timestamps=pts[:n], values=xyz))

        if "gripper/state/value" in h:
            gv = np.asarray(h["gripper/state/value"][:], float)
            gts = np.asarray(h["gripper/state/timestamp"][:], float)
            n = int(min(len(gv), len(gts)))
            gv = gv[:n].reshape(n, -1)
            dim = gv.shape[1] if n else 0
            streams.append(Stream("gripper", "scalar" if dim == 1 else "vector", n, (dim,),
                                  str(gv.dtype), timestamps=gts[:n], values=gv))

    return Episode(path.stem, streams)


# ───────────────────── adapter: existing LeRobotDataset ─────────────────── #
def iter_record3d_h5(root, use_depth=True, pose_source="pose_relative", max_episodes=None, **_):
    for p in find_episodes(root)[:max_episodes]:
        yield load_record3d_h5(p, use_depth=use_depth, pose_source=pose_source)


def iter_lerobot_dataset(root, max_episodes=None, **_):
    """Inspect an existing LeRobotDataset (repo_id or local root) — e.g. metaworld.

    Reads numeric columns from the parquet-backed hf_dataset (no video decode).
    Pose-jump is enabled when observation.state's axes start with x,y,z (canonical).
    Timestamps are index/fps (regular) so fps/non-mono are trivially clean; useful
    for schema/dim/value-range/pose-jump sanity on already-converted datasets.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = str(root)
    ds = (LeRobotDataset("local/inspect", root=root)
          if (Path(root) / "meta").exists() else LeRobotDataset(root))
    feats = ds.meta.features
    hf = ds.hf_dataset.with_format("numpy")
    cols = set(hf.column_names)
    ep_idx = np.asarray(hf["episode_index"]).reshape(-1)
    ts = np.asarray(hf["timestamp"], np.float64).reshape(-1) if "timestamp" in cols else None
    img_keys = [k for k in feats if k.startswith("observation.image")]
    vec_keys = [k for k in ("observation.state", "action") if k in cols]
    vals = {k: np.asarray(hf[k]) for k in vec_keys}
    axes = (feats.get("observation.state", {}).get("names") or {})
    axes = axes.get("axes", []) if isinstance(axes, dict) else []
    pose_ok = list(axes[:3]) == ["x", "y", "z"]

    for e in sorted({int(x) for x in ep_idx})[:max_episodes]:
        m = ep_idx == e
        n = int(m.sum())
        ets = ts[m] if ts is not None else None
        streams = [Stream(k.split(".")[-1], "image", n, tuple(feats[k]["shape"]),
                          feats[k]["dtype"], timestamps=ets) for k in img_keys]
        for vk in vec_keys:
            v = vals[vk][m].reshape(n, -1)
            streams.append(Stream(vk.split(".")[-1], "vector", n, (v.shape[1],),
                                  "float32", timestamps=ets, values=v))
        if pose_ok:
            sv = vals["observation.state"][m].reshape(n, -1)
            streams.append(Stream("pose:state", "vector", n, (sv.shape[1],),
                                  "float32", timestamps=ets, values=sv))
        yield Episode(f"episode_{e:06d}", streams)


ITERATORS: dict[str, Callable[..., object]] = {
    "record3d_h5": iter_record3d_h5,
    "lerobot_dataset": iter_lerobot_dataset,
}


# ───────────────────────────────── analysis ─────────────────────────────── #
def _ts_stats(ts: np.ndarray | None) -> dict | None:
    if ts is None:
        return None
    ts = np.asarray(ts, float).reshape(-1)
    if ts.size < 2:
        return {"count": int(ts.size), "fps": None}
    dt = np.diff(ts)
    med = float(np.median(dt))
    return {"count": int(ts.size), "span": float(ts[-1] - ts[0]),
            "dt_mean": float(dt.mean()), "dt_std": float(dt.std()),
            "dt_min": float(dt.min()), "dt_max": float(dt.max()),
            "fps": float(1 / dt.mean()) if dt.mean() > 0 else None,
            "monotonic": bool(np.all(dt > 0)),
            "n_gaps": int(np.sum(dt > 1.8 * med)) if med > 0 else 0}


def _value_stats(s: Stream) -> dict | None:
    if s.kind in ("vector", "scalar") and s.values is not None and s.count:
        v = np.asarray(s.values, float).reshape(s.count, -1)
        return {"min": [round(x, 6) for x in v.min(0)], "max": [round(x, 6) for x in v.max(0)]}
    if s.kind == "image" and s.sample is not None and s.sample.size:
        v = np.asarray(s.sample, float)
        return {"min": float(v.min()), "max": float(v.max()), "mean": round(float(v.mean()), 3)}
    return None


def _max_nn_gap(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.sort(np.asarray(b, float).reshape(-1))
    if a.size == 0 or b.size == 0:
        return float("nan")
    idx = np.clip(np.searchsorted(b, a), 1, b.size - 1)
    nn = np.where(np.abs(a - b[idx - 1]) <= np.abs(b[idx] - a), b[idx - 1], b[idx])
    return float(np.max(np.abs(a - nn)))


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """(N,3) roll/pitch/yaw -> (N,3,3) rotation matrices (Rz@Ry@Rx, matches find_pose_jumps)."""
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    n = len(rpy)
    R = np.empty((n, 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sp, cp * sr, cp * cr
    return R


def _rot6d_to_R(d6: np.ndarray) -> np.ndarray:
    """(N,6) rot6d (first two rotation-matrix columns) -> (N,3,3) via Gram-Schmidt."""
    a1, a2 = d6[:, :3], d6[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-12)
    a2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # columns


def _pose_jump_stats(values: np.ndarray | None) -> dict | None:
    """Per-step position/rotation jump stats for a pose stream.

    Auto-detects rotation by dim: 6D = xyz+rpy (record3d), >=9D = xyz+rot6d(+…)
    (canonical). Flags "teleports" — a single step anomalously large in position
    (abs or vs median) or rotation.
    """
    if values is None:
        return None
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 6 or len(x) < 3:
        return None
    dpos = np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1)  # (N-1,)
    R = _rot6d_to_R(x[:, 3:9]) if x.shape[1] >= 9 else _rpy_to_R(x[:, 3:6])
    tr = np.sum(R[:-1] * R[1:], axis=(1, 2))  # == trace(R[i].T @ R[i+1])
    drot = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
    med = float(np.median(dpos)) if dpos.size else 0.0
    return {
        "dpos_max": round(float(dpos.max()), 5),
        "dpos_frame": int(dpos.argmax() + 1),
        "dpos_med": round(med, 6),
        # raw (unrounded) so threshold comparison matches find_pose_jumps exactly;
        # rounding happens only at display time.
        "ratio": (float(dpos.max()) / med if med > 1e-9 else None),
        "drot_max": round(float(drot.max()), 2),
        "drot_frame": int(drot.argmax() + 1),
    }


def _jump_reasons(pj: dict | None, thr: dict) -> list[str]:
    if not pj:
        return []
    reasons = []
    if pj["dpos_max"] > thr["pos"]:
        reasons.append(f"dpos={pj['dpos_max']:.3f}m@{pj['dpos_frame']}")
    if pj["ratio"] is not None and pj["ratio"] > thr["ratio"]:
        reasons.append(f"ratio={pj['ratio']:.0f}x")
    if pj["drot_max"] > thr["rot"]:
        reasons.append(f"drot={pj['drot_max']:.1f}deg@{pj['drot_frame']}")
    return reasons


def _pick_anchor(streams: list[Stream]) -> str | None:
    """Reference/clock stream: prefer ANCHOR name, else first image, else first stream."""
    names = [s.name for s in streams]
    if ANCHOR in names:
        return ANCHOR
    img = next((s.name for s in streams if s.kind == "image"), None)
    return img or (names[0] if names else None)


def analyze_episode(ep: Episode) -> dict:
    streams = [{"name": s.name, "kind": s.kind, "count": s.count, "shape": list(s.shape),
                "dtype": s.dtype, "time": _ts_stats(s.timestamps), "value": _value_stats(s)}
               for s in ep.streams]
    by_name = {s.name: s for s in ep.streams}
    anchor_name = _pick_anchor(ep.streams)
    anchor = by_name.get(anchor_name)
    cross = {}
    if anchor is not None and anchor.timestamps is not None:
        for s in ep.streams:
            if s.name == anchor_name or s.timestamps is None:
                continue
            gap = _max_nn_gap(anchor.timestamps, s.timestamps)
            cross[s.name] = {"gap": None if gap != gap else round(gap, 6),
                             "co_sampled": bool(len(s.timestamps) == len(anchor.timestamps)
                                                and gap == gap and gap < 1e-4)}
    fps = next((s["time"]["fps"] for s in streams if s["name"] == anchor_name and s["time"]), None)
    pose = next((s for s in ep.streams if s.name.startswith("pose:") and s.values is not None), None)
    pose_jump = _pose_jump_stats(pose.values) if pose is not None else None
    return {"episode_id": ep.episode_id, "anchor": anchor_name, "fps": fps, "streams": streams,
            "cross": cross, "pose_jump": pose_jump}


def analyze_dataset(eps: list[dict], target_fps: float | None = None, tol: float = 0.1,
                    jump: dict | None = None) -> dict:
    jump = jump or POSE_JUMP_DEFAULTS
    n = len(eps)
    total = sum(next((s["count"] for s in e["streams"] if s["name"] == e.get("anchor")), 0) for e in eps)
    fps_dist = dict(sorted(Counter(round(e["fps"]) for e in eps if e["fps"]).items()))
    warnings, outliers = [], []
    if target_fps:
        outliers = [{"episode": e["episode_id"], "fps": round(e["fps"], 2) if e["fps"] else None}
                    for e in eps if e["fps"] is None or abs(e["fps"] - target_fps) > tol * target_fps]
        if outliers:
            warnings.append(f"target_fps={target_fps}: {len(outliers)}/{n} episodes deviate "
                            f">{tol:.0%} — mislabel risk if all converted at {target_fps}fps.")
    # dim/dtype consistency per stream
    dims: dict[str, Counter] = {}
    for e in eps:
        for s in e["streams"]:
            dims.setdefault(s["name"], Counter())[(tuple(s["shape"]), s["dtype"])] += 1
    dim_consistency = {name: {"consistent": len(c) == 1,
                              "variants": [{"shape": list(sh), "dtype": dt, "n": k} for (sh, dt), k in c.items()]}
                       for name, c in dims.items()}
    for name, dc in dim_consistency.items():
        if not dc["consistent"]:
            warnings.append(f"stream '{name}': inconsistent shape/dtype ({len(dc['variants'])} variants).")
    # per-stream episode lists for timestamp issues
    nonmono_by: dict[str, list[int]] = {}
    gaps_by: dict[str, list[int]] = {}
    for e in eps:
        for s in e["streams"]:
            t = s["time"]
            if not t:
                continue
            if t.get("monotonic") is False:
                nonmono_by.setdefault(s["name"], []).append(_ep_num(e["episode_id"]))
            if t.get("n_gaps"):
                gaps_by.setdefault(s["name"], []).append(_ep_num(e["episode_id"]))
    nonmono_all = sorted({x for v in nonmono_by.values() for x in v})
    gaps_all = sorted({x for v in gaps_by.values() for x in v})
    if nonmono_all:
        warnings.append(f"non-monotonic timestamps in {len(nonmono_all)} episode(s).")
    if gaps_all:
        warnings.append(f"timestamp gaps (dropped frames) in {len(gaps_all)} episode(s).")

    # pose "teleport" jumps
    jumpy = [{"episode": e["episode_id"], "reasons": r}
             for e in eps if (r := _jump_reasons(e.get("pose_jump"), jump))]
    if jumpy:
        warnings.append(f"pose JUMP (teleport) in {len(jumpy)} episode(s) "
                        f"[>{jump['pos']}m OR >{jump['ratio']:.0f}x OR >{jump['rot']}deg].")

    # warning -> (stream ->) affected episode numbers
    issues = {
        "fps_outliers": sorted(_ep_num(o["episode"]) for o in outliers),
        "non_monotonic": {k: sorted(v) for k, v in nonmono_by.items()},
        "gaps": {k: sorted(v) for k, v in gaps_by.items()},
        "pose_jump": sorted(_ep_num(j["episode"]) for j in jumpy),
    }

    return {"n_episodes": n, "total_frames": total, "fps_distribution": fps_dist,
            "outliers": outliers, "dim_consistency": dim_consistency,
            "jumpy_episodes": jumpy, "jump_thresholds": jump,
            "issues": issues, "warnings": warnings}


# ───────────────────────────────── report ───────────────────────────────── #
def _fmt_time(t: dict | None) -> str:
    if not t or t.get("fps") is None:
        return f"n={t['count'] if t else 0} (fps n/a)"
    flags = []
    if t["monotonic"] is False:
        flags.append("NON-MONOTONIC")
    if t["n_gaps"]:
        flags.append(f"{t['n_gaps']}gaps")
    tail = ("  [" + ",".join(flags) + "]") if flags else ""
    return (f"n={t['count']:>5}  {t['fps']:6.2f}fps  dt={t['dt_mean']*1000:6.2f}±{t['dt_std']*1000:.1f}ms  "
            f"{t['span']:.1f}s{tail}")


def _fmt_value(v: dict | None) -> str:
    if not v:
        return ""
    if "mean" in v:  # image
        return f"range[{v['min']:.0f}..{v['max']:.0f}] mean={v['mean']}"
    mn, mx = v["min"], v["max"]
    return "dims " + " ".join(f"{a:g}..{b:g}" for a, b in zip(mn, mx)) if len(mn) <= 4 \
        else f"dims[{len(mn)}] {min(mn):g}..{max(mx):g}"


def _fmt_align(c: dict) -> str:
    if c["co_sampled"]:
        return "co-sampled"
    return f"gap<={c['gap']*1000:.1f}ms" if c["gap"] is not None else "n/a"


def print_episode(ep: dict) -> None:
    print(f"-- {ep['episode_id']}  fps={round(ep['fps'],2) if ep['fps'] else None}")
    for s in ep["streams"]:
        shape = "x".join(map(str, s["shape"])) or "scalar"
        print(f"   {s['name']:<18} {s['kind']:<7} {shape:<12} {s['dtype']:<8} {_fmt_time(s['time'])}")
        vt = _fmt_value(s["value"])
        if vt:
            print(f"   {'':<47}{vt}")
    if ep["cross"]:
        al = ", ".join(f"{k}:{_fmt_align(c)}" for k, c in ep["cross"].items())
        print(f"   align vs {ep['anchor']}: {al}")
    pj = ep.get("pose_jump")
    if pj:
        ratio = f"{pj['ratio']:.0f}x" if pj["ratio"] is not None else "n/a"
        print(f"   pose-jump: dpos_max={pj['dpos_max']:.3f}m@{pj['dpos_frame']}  "
              f"ratio={ratio}(med={pj['dpos_med']*1000:.1f}mm)  "
              f"drot_max={pj['drot_max']:.1f}deg@{pj['drot_frame']}")


def print_dataset(ds: dict, target_fps: float | None) -> None:
    bar = "=" * 72
    print(bar, "RAW DATASET INSPECTION", bar, sep="\n")
    print(f"episodes     : {ds['n_episodes']}")
    print(f"total frames : {ds['total_frames']} (anchor stream)")
    if ds["fps_distribution"]:
        print("fps distrib. : " + "  ".join(f"{f}fps×{c}" for f, c in ds["fps_distribution"].items()))
        if len(ds["fps_distribution"]) > 1:
            print(f"               ⚠ MIXED rate ({len(ds['fps_distribution'])} distinct fps)")
    if target_fps is not None:
        n_out = len(ds["outliers"])
        print(f"target fps   : {target_fps}  →  {'OK ✓' if not n_out else f'{n_out} OUTLIERS ⚠'}")
        for o in ds["outliers"][:12]:
            print(f"                 - {o['episode']}: {o['fps']}fps")
        if n_out > 12:
            print(f"                 ... (+{n_out - 12} more)")
    jumpy = ds.get("jumpy_episodes", [])
    if jumpy:
        thr = ds["jump_thresholds"]
        print(f"pose JUMP     : {len(jumpy)} episode(s) ⚠  "
              f"[>{thr['pos']}m OR >{thr['ratio']:.0f}x OR >{thr['rot']}deg]")
        for j in jumpy[:12]:
            print(f"                 - {j['episode']}: {', '.join(j['reasons'])}")
        if len(jumpy) > 12:
            print(f"                 ... (+{len(jumpy) - 12} more)")
    print("-" * 72, "dimension consistency:", sep="\n")
    for name, dc in ds["dim_consistency"].items():
        if dc["consistent"]:
            v = dc["variants"][0]
            print(f"   {name:<18} ✓ {'x'.join(map(str, v['shape'])) or 'scalar'} {v['dtype']}")
        else:
            print(f"   {name:<18} ⚠ INCONSISTENT: " +
                  " | ".join(f"{'x'.join(map(str, v['shape']))} {v['dtype']}({v['n']})" for v in dc["variants"]))
    print("-" * 72)
    print("WARNINGS:" if ds["warnings"] else "no warnings ✓")
    for w in ds["warnings"]:
        print(f"   ⚠ {w}")
    print(bar)


def print_issues(ds: dict) -> None:
    """Per-warning, per-stream affected episode NUMBERS (for building skip lists)."""
    iss = ds.get("issues")
    if not iss:
        return
    print("-" * 72, "ISSUES — 에피소드 번호 (warning별 / 스트림별)", sep="\n")
    if iss["fps_outliers"]:
        print(f"   fps_outliers ({len(iss['fps_outliers'])}): {iss['fps_outliers']}")
    if iss["pose_jump"]:
        print(f"   pose_jump    ({len(iss['pose_jump'])}): {iss['pose_jump']}")
    if any(iss["non_monotonic"].values()):
        print("   non_monotonic:")
        for name, lst in iss["non_monotonic"].items():
            print(f"      {name:<18} ({len(lst)}): {lst}")
    if any(iss["gaps"].values()):
        print("   gaps:")
        for name, lst in iss["gaps"].items():
            print(f"      {name:<18} ({len(lst)}): {lst}")
    print("=" * 72)


# ─────────────────────────────────── cli ────────────────────────────────── #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect raw episodes before dataset conversion.")
    ap.add_argument("--raw-root", required=True, help="record3d: episode_*.h5 dir | lerobot_dataset: repo_id or dataset root")
    ap.add_argument("--format", default="record3d_h5", choices=sorted(ITERATORS))
    ap.add_argument("--pose-source", default="pose_relative")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--target-fps", type=float, default=None, help="Flag episodes deviating from this fps.")
    ap.add_argument("--fps-tol", type=float, default=0.1)
    ap.add_argument("--pos-thresh", type=float, default=POSE_JUMP_DEFAULTS["pos"], help="pose jump: abs position step [m].")
    ap.add_argument("--ratio-thresh", type=float, default=POSE_JUMP_DEFAULTS["ratio"], help="pose jump: max/median step ratio.")
    ap.add_argument("--rot-thresh", type=float, default=POSE_JUMP_DEFAULTS["rot"], help="pose jump: rotation step [deg].")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--per-episode", action="store_true")
    ap.add_argument("--out-dir", default="outputs/data_processing/raw_inspect",
                    help="Auto-save <name>_<ts>.log/.json here (default: outputs/data_processing/raw_inspect).")
    ap.add_argument("--no-save", action="store_true", help="Do not write the log/json files.")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if target-fps outliers exist.")
    args = ap.parse_args(argv)

    print(f"Scanning {args.raw_root} (format={args.format}) ...", file=sys.stderr)
    it = ITERATORS[args.format](
        args.raw_root, use_depth=not args.no_depth, pose_source=args.pose_source,
        max_episodes=args.max_episodes,
    )
    eps = []
    for i, ep in enumerate(it):
        eps.append(analyze_episode(ep))
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1} episodes", file=sys.stderr)
    if not eps:
        print(f"No episodes found under {args.raw_root} (format={args.format})", file=sys.stderr)
        return 2

    jump = {"pos": args.pos_thresh, "ratio": args.ratio_thresh, "rot": args.rot_thresh}
    ds = analyze_dataset(eps, target_fps=args.target_fps, tol=args.fps_tol, jump=jump)
    # build the report once (captured), then echo to stdout + save to file
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if args.per_episode:
            for ep in eps:
                print_episode(ep)
            print()
        print_dataset(ds, args.target_fps)
        print_issues(ds)
    report = buf.getvalue()
    print(report, end="")

    if not args.no_save:
        outdir = Path(args.out_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        name = "_".join(Path(args.raw_root).resolve().parts[-2:]).strip("_") or "raw"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = outdir / f"{name}_{ts}"
        stem.with_suffix(".log").write_text(report)
        stem.with_suffix(".json").write_text(json.dumps({"dataset": ds, "episodes": eps}, indent=2))
        print(f"[saved] {stem}.log  +  {stem}.json", file=sys.stderr)

    return 1 if (args.strict and ds["outliers"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
