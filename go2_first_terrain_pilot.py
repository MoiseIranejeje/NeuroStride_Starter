r"""NeuroStride Go2: first gentle-uphill terrain comparison.

Run on Windows from NeuroStride_Starter:
  .\.venv\Scripts\python.exe go2_first_terrain_pilot.py --seeds 5601 --slopes 0 2 --seconds 8

SNN V5 controls Go2 independently (plus the documented classical heading
supervisor). Third-party ANN is a separate baseline. Both use the same seed.
This tests a PLANAR SLOPE, not rocks/bumps or successful rough-terrain training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np
import torch

import go2_heading_benchmark as heading
import go2_v5_upright_sweep as upright
from go2_v5_steering import WEIGHTS, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "go2_terrain_pilot"
CONTROLLERS = ("snn_v5", "snn_v5_bias", "ann_teacher")


def build_terrain(slope_deg: float):
    """Rotate the existing Go2 floor (a plane) once per fresh model."""
    model = mm.load("unitree_go2")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise RuntimeError("Expected a named 'floor' plane in Go2 scene")
    # Positive degrees mean uphill in the robot's +X direction.
    theta = -math.radians(slope_deg)
    model.geom_quat[gid, :] = (
        math.cos(theta / 2), 0, math.sin(theta / 2), 0
    )
    # Rebuild geom-derived constants before any trial; use a fresh model per slope.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def replay_metrics(path: Path, slope_deg: float):
    with np.load(path, allow_pickle=False) as saved:
        qpos = saved["qpos"].copy()
        times = saved["sim_time"].copy()
    if qpos.ndim != 2 or qpos.shape[1] != 19:
        raise ValueError(f"Unexpected Go2 qpos in {path}")
    angles = np.array([upright.roll_pitch_rad(q[3:7]) for q in qpos])
    angles = np.rad2deg(angles)
    # Height relative to the sloping floor directly under the torso XY.
    # This is not ground-contact clearance of the robot's feet.
    floor_z = np.tan(np.deg2rad(slope_deg)) * qpos[:, 0]
    local_torso_height = qpos[:, 2] - floor_z
    return {
        "roll_rms_deg": float(np.sqrt(np.mean(angles[:, 0] ** 2))),
        "pitch_rms_deg": float(np.sqrt(np.mean(angles[:, 1] ** 2))),
        "min_torso_above_plane_m": float(np.min(local_torso_height)),
        "recorded_frames": len(times),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[5601])
    ap.add_argument("--slopes", type=float, nargs="+", default=[0.0, 2.0])
    ap.add_argument("--modes", nargs="+", choices=CONTROLLERS,
                    default=list(CONTROLLERS))
    ap.add_argument("--command", type=float, default=0.45)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()
    if not (0 < args.seconds <= 12 and 0 < args.command <= 1):
        ap.error("Use 0 < seconds <= 12 and 0 < command <= 1")
    if any(abs(s) > 3 for s in args.slopes):
        ap.error("First terrain pilot is limited to slopes within +/-3 degrees")
    if len(args.seeds) > 8 or len(args.slopes) > 5:
        ap.error("Too many conditions for a first pilot")
    if args.viewer and len(args.seeds) * len(args.slopes) * len(args.modes) > 1:
        ap.error("Use --viewer with ONE seed, ONE slope, and ONE mode")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    # Redirect temporary replays so earlier experiment files are never overwritten.
    upright.OUT = OUT / "scratch_snn"
    heading.RESULT_DIR = OUT / "scratch_ann"

    snn = None
    if any(m.startswith("snn") for m in args.modes):
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"V5 checkpoint missing: {WEIGHTS}")
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        if not str(ckpt.get("source", "")).startswith("V5 learned"):
            print("WARNING: checkpoint may be V4 fallback, not a learned V5")
        net, stats = make_model(ckpt)
        net.eval()
        snn = net, stats

    teacher = heading.load_teacher() if "ann_teacher" in args.modes else None
    rows = []
    for slope in args.slopes:
        model = build_terrain(slope)
        for seed in args.seeds:
            for mode in args.modes:
                print(f"Terrain {slope:+.1f} deg | seed {seed} | {mode}", flush=True)
                if mode == "ann_teacher":
                    result = heading.run(
                        model, "ann_feedback", None, teacher,
                        args.command, seed, args.seconds, .9, .2,
                        viewer=args.viewer, record=True,
                    )
                    original = (heading.RESULT_DIR /
                                f"ann_feedback_cmd{args.command:.2f}_seed{seed}.npz")
                    sim_s, x, y = (result["sim_s"], result["forward_m"],
                                   result["sideways_m"])
                    max_yaw = result["max_abs_yaw_deg"]
                    ended = result["terminated"]
                else:
                    bias = -0.02 if mode == "snn_v5_bias" else 0.0
                    result = upright.run(
                        model, snn[0], snn[1], seed, bias,
                        args.command, args.seconds, show_viewer=args.viewer,
                    )
                    original = upright.replay_path(seed, bias, args.command)
                    sim_s, x, y = (result["simulated_s"], result["forward_m"],
                                   result["final_sideways_m"])
                    max_yaw = result["max_abs_yaw_deg"]
                    ended = result["terminated"]

                if not original.is_file():
                    raise FileNotFoundError(f"No replay saved: {original}")
                label = f"{mode}_slope{slope:+.1f}_seed{seed}"
                saved = OUT / f"{label}.npz"
                shutil.copy2(original, saved)
                posture = replay_metrics(saved, slope)
                # Use a local ground-height reference on slopes; do not equate
                # world Z with standing clearance when going uphill/downhill.
                passed = bool(
                    ended == "completed requested duration"
                    and sim_s >= args.seconds - .002
                    and x >= 1.5
                    and abs(y) <= .25
                    and max_yaw <= 15
                    and posture["min_torso_above_plane_m"] >= .18
                )
                row = dict(controller=mode, slope_deg=slope, seed=seed,
                           sim_s=sim_s, forward_m=x, sideways_m=y,
                           max_abs_yaw_deg=max_yaw, termination=ended,
                           passes_pilot=passed, replay=str(saved), **posture)
                rows.append(row)
                print(f"  {sim_s:.2f}s X={x:+.3f}m Y={y:+.3f}m "
                      f"peak_yaw={max_yaw:.2f}deg "
                      f"pitch_rms={posture['pitch_rms_deg']:.2f}deg "
                      f"pass={passed}", flush=True)

    csv_path = OUT / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    agg = {}
    for slope in args.slopes:
        for mode in args.modes:
            subset = [r for r in rows if r["slope_deg"] == slope
                      and r["controller"] == mode]
            agg[f"slope_{slope:+.1f}_{mode}"] = {
                "runs": len(subset),
                "passes": sum(r["passes_pilot"] for r in subset),
                "mean_forward_m": float(np.mean([r["forward_m"] for r in subset])),
                "mean_peak_yaw_deg": float(np.mean([
                    r["max_abs_yaw_deg"] for r in subset])),
            }
    (OUT / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print("\n=== FIRST GO2 TERRAIN PILOT ===")
    print(json.dumps(agg, indent=2))
    print("Episode CSV:", csv_path)
    print("This is a mild plane slope, NOT an uneven-rocky-terrain result.")


if __name__ == "__main__":
    main()
