r"""NeuroStride: aim for <=5 degrees peak yaw without retraining the walking SNN.

Stages:
  1. sweep: DIAGNOSTIC classical yaw-control tuning on development starts only.
  2. distill: train a new spiking yaw network to reproduce the selected control law.
  3. test: compare original yaw SNN, new yaw SNN, and a classical diagnostic.

This is classical-to-SNN supervised distillation. It is not SNN PPO, does not
measure hardware power, and does not guarantee the five-degree goal.

Run from NeuroStride_Starter in Windows VS Code PowerShell:
  .\.venv\Scripts\python.exe go2_yaw5_fasttrack.py sweep --seeds 6501 6502 6801
  .\.venv\Scripts\python.exe go2_yaw5_fasttrack.py distill --epochs 35
  .\.venv\Scripts\python.exe go2_yaw5_fasttrack.py test --seeds 6501 6502 6801

Held-out validation, ONLY after selecting and freezing the controller:
  .\.venv\Scripts\python.exe go2_yaw5_fasttrack.py test --seeds 6901 6902 6903 6904 6905 6906 6907 6908 6909 6910

Reuses go2_snn_yaw_distillation.run(), keeping identical MuJoCo physics,
locomotion policy, PD motor dynamics, and task metrics. Original checkpoints
and previous experiment CSVs are NOT overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

import go2_snn_yaw_distillation as old

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "go2_yaw5_fasttrack"
SELECTION = OUT / "selected_control.json"
NEW_WEIGHTS = OUT / "yaw5_spiking_controller.pt"
ORIGINAL_WEIGHTS = ROOT / "results" / "go2_snn_yaw_distillation" / "spiking_yaw_controller.pt"
GOAL_DEG = 5.0

# Diagnostic candidates, NOT final SNNs. Positive bias is a testable
# feedforward compensation for the negative heading drift seen in earlier logs.
# We deliberately test a bounded set, not an unlimited parameter search.
CANDIDATES = [
    ("C00", 0.9, 0.20, 0.00),  # previous classical baseline
    ("C01", 2.0, 0.20, 0.00),
    ("C02", 4.0, 0.20, 0.00),
    ("C03", 6.0, 0.20, 0.00),
    ("C04", 2.0, 0.20, 0.04),
    ("C05", 4.0, 0.20, 0.04),
    ("C06", 6.0, 0.20, 0.04),
    ("C07", 4.0, 0.20, 0.08),
    ("C08", 4.0, 0.30, 0.04),
    ("C09", 6.0, 0.30, 0.04),
    ("C10", 6.0, 0.30, 0.08),
]


def reference_command(error_degrees: Any, gain: float, cap: float, bias: float):
    """Diagnostic/teacher only. Must NOT run in new SNN evaluation."""
    e = np.asarray(error_degrees, dtype=np.float32)
    return np.clip(bias - gain * np.deg2rad(e), -cap, cap).astype(np.float32)


def neural_command(net, error_degrees):
    """SNN-only yaw command, with no classical calculation at test time.

    Unlike the old neutral-subtracted actor, this model learns the entire
    bounded yaw command, including any nonzero feedforward bias at e=0.
    """
    if isinstance(error_degrees, torch.Tensor):
        e = error_degrees.to(dtype=torch.float32).reshape(-1, 1)
    else:
        e = torch.as_tensor(error_degrees, dtype=torch.float32).reshape(-1, 1)
    raw, spike_rate = net((e / 20.0).clamp(-3.0, 3.0))
    command = (old.YAW_LIMIT * raw[:, 0]).clamp(-old.YAW_LIMIT, old.YAW_LIMIT)
    return command, spike_rate


def prepare_base():
    if not old.ss.BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f"Frozen V6 locomotion checkpoint missing: {old.ss.BASE_WEIGHTS}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base = old.ss.make_model(torch.load(old.ss.BASE_WEIGHTS, map_location="cpu", weights_only=True))
    base[0].eval()
    return base


def read_selection():
    if not SELECTION.is_file():
        raise FileNotFoundError(f"No controller selected. Run 'sweep' first: {SELECTION}")
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    gain = float(sel["gain"])
    cap = float(sel["cap_rad_s"])
    bias = float(sel["bias_rad_s"])
    if not (0.5 <= gain <= 8 and 0 < cap <= 0.3 and 0 <= bias <= 0.1):
        raise ValueError("Saved yaw-controller selection outside tested bounds")
    return sel


def run_episode(model, base, seed, seconds, speed, kind, gain=0.9, cap=0.2, bias=0.0,
                yaw_net=None, viewer=False):
    """Run a full actual simulation using the original, verified Go2 loop."""
    if kind == "diagnostic":
        # Classical reference runs ONLY for discovery/comparison.
        # This is intentionally separated from new SNN-only execution.
        old.classical_teacher = lambda e: reference_command(e, gain, cap, bias)
        internal_mode = "snn_classic"
    elif kind == "original":
        old.YAW_LIMIT = 0.2
        old.infer_yaw = old._original_infer_yaw  # saved during setup
        internal_mode = "snn_spiking_yaw"
    elif kind == "distilled":
        old.YAW_LIMIT = cap
        old.infer_yaw = neural_command
        internal_mode = "snn_spiking_yaw"
    elif kind == "zero":
        internal_mode = "snn_zero"
    else:
        raise ValueError(f"Unknown mode: {kind}")

    result = old.run(model, internal_mode, seed, speed, seconds, base,
                     yaw_net if kind in ("original", "distilled") else None,
                     None, 0.0, viewer)
    # The upstream run() uses fixed file names. Move each replay immediately,
    # preventing later candidates from silently destroying evidence.
    replay = Path(result["replay"])
    dst = OUT / "replays" / f"{kind}_kp{gain:g}_cap{cap:.2f}_bias{bias:.2f}_seed{seed}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = dst.with_name(f"{dst.stem}__archived_{stamp}.npz")
        dst.rename(archived)
    shutil.move(str(replay), str(dst))
    result.update(mode=kind, replay=str(dst))
    return result


def save_rows(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sweep(seeds, seconds, speed):
    from go2_first_terrain_pilot import build_terrain
    base = prepare_base()
    model = build_terrain(0.0)
    old.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    # Save previous globals to avoid accidental changes if called from code.
    original_classical = old.classical_teacher
    rows, ranked = [], []
    try:
        for cid, gain, cap, bias in CANDIDATES:
            group = []
            for seed in seeds:
                r = run_episode(model, base, seed, seconds, speed, "diagnostic", gain, cap, bias)
                rows.append(dict(candidate=cid, gain=gain, cap_rad_s=cap,
                                 bias_rad_s=bias, **r))
                group.append(r)
                print(f"{cid} kp={gain:g} cap={cap:.2f} bias={bias:+.2f} "
                      f"seed={seed}: X={r['forward_m']:.3f} m, "
                      f"yaw={r['max_abs_yaw_deg']:.2f}°, "
                      f"height={r['minimum_local_height_m']:.3f}m, "
                      f"stop={r['stop']}", flush=True)
            valid = all((r["stop"] == "completed" and r["time_s"] >= seconds - 0.002
                         and r["forward_m"] >= 1.5 and abs(r["sideways_m"]) <= 0.25
                         and r["minimum_local_height_m"] >= 0.18) for r in group)
            worst_yaw = float(max(r["max_abs_yaw_deg"] for r in group))
            mean_yaw = float(np.mean([r["max_abs_yaw_deg"] for r in group]))
            max_y = float(max(abs(r["sideways_m"]) for r in group))
            ranked.append(dict(candidate=cid, gain=gain, cap_rad_s=cap,
                               bias_rad_s=bias, valid=valid,
                               goal_5deg_all=bool(valid and worst_yaw <= GOAL_DEG),
                               worst_yaw_deg=worst_yaw, mean_yaw_deg=mean_yaw,
                               worst_abs_sideways_m=max_y,
                               passes_15deg=sum(bool(r["pass_strict"]) for r in group)))
    finally:
        old.classical_teacher = original_classical
    save_rows(rows, OUT / "sweep_episodes.csv")
    ranked.sort(key=lambda x: (not x["valid"], x["worst_yaw_deg"],
                               x["mean_yaw_deg"], x["worst_abs_sideways_m"]))
    (OUT / "sweep_ranking.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    good = [r for r in ranked if r["valid"]]
    if not good:
        raise RuntimeError("No candidate preserved walking and basic stability. Keep the old yaw SNN.")
    best = good[0]
    choice = dict(best, development_seeds=seeds,
                  training_target="analytic classical-to-spiking supervised distillation",
                  note="Selection on exposed development starts, not independent validation")
    SELECTION.write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print("\nSELECTED P+BIAS DIAGNOSTIC:", json.dumps(choice, indent=2), flush=True)
    if best["worst_yaw_deg"] > GOAL_DEG:
        print("IMPORTANT: No candidate reached <=5° on every dev seed. "
              "This is still a promising diagnostic; do not claim 5° success.", flush=True)
    print("Saved", OUT / "sweep_episodes.csv", flush=True)
    print("Saved", OUT / "sweep_ranking.json", flush=True)


def distill(epochs, lr, batch):
    selected = read_selection()
    gain, cap, bias = selected["gain"], selected["cap_rad_s"], selected["bias_rad_s"]
    old.OUT = OUT
    old.WEIGHTS = NEW_WEIGHTS
    old.YAW_LIMIT = cap
    old.classical_teacher = lambda e: reference_command(e, gain, cap, bias)
    old.infer_yaw = neural_command
    print(f"Training yaw LIF on selected diagnostic law: "
          f"kp={gain}, cap={cap}, bias={bias}", flush=True)
    old.train(epochs=epochs, lr=lr, batch_size=batch)
    cp = torch.load(NEW_WEIGHTS, map_location="cpu", weights_only=True)
    cp["law"] = "clip(bias_rad_s - gain*heading_error_rad, -cap_rad_s, cap_rad_s)"
    cp["gain"] = float(gain)
    cp["bias_rad_s"] = float(bias)
    cp["cap_rad_s"] = float(cap)
    cp["origin"] = "supervised spiking approximation of a tuned classical diagnostic"
    cp["not_end_to_end_rl"] = True
    torch.save(cp, NEW_WEIGHTS)
    net = old.make_yaw_net()
    net.load_state_dict(cp["state_dict"])
    net.eval()
    with torch.inference_mode():
        e = torch.tensor([-20., -10., -5., 0., 5., 10., 20.])
        pred, _ = neural_command(net, e)
        target = reference_command(e.numpy(), gain, cap, bias)
    print("\nHeading angle -> spiking yaw command vs target:")
    for angle, actual, expected in zip(e.tolist(), pred.tolist(), target.tolist()):
        print(f"  {angle:+5.0f}°: SNN {actual:+.4f} rad/s; target {expected:+.4f}")
    print("Checkpoint:", NEW_WEIGHTS, flush=True)


def test(seeds, seconds, speed, modes, viewer):
    from go2_first_terrain_pilot import build_terrain
    sel = read_selection()
    gain, cap, bias = sel["gain"], sel["cap_rad_s"], sel["bias_rad_s"]
    if viewer and len(seeds) * len(modes) != 1:
        raise ValueError("--viewer requires one seed and one mode")
    if not ORIGINAL_WEIGHTS.is_file():
        raise FileNotFoundError(f"Original spiking yaw checkpoint missing: {ORIGINAL_WEIGHTS}")
    if "distilled" in modes and not NEW_WEIGHTS.is_file():
        raise FileNotFoundError(f"New yaw checkpoint missing: {NEW_WEIGHTS}. Run distill first.")
    base = prepare_base()
    model = build_terrain(0.0)
    old.OUT = OUT
    old._original_infer_yaw = old.infer_yaw
    old.classical_teacher = lambda e: reference_command(e, gain, cap, bias)
    nets = {}
    for key, weights in (("original", ORIGINAL_WEIGHTS), ("distilled", NEW_WEIGHTS)):
        if key in modes:
            net = old.make_yaw_net()
            cp = torch.load(weights, map_location="cpu", weights_only=True)
            net.load_state_dict(cp["state_dict"])
            net.eval()
            nets[key] = net
    rows = []
    for seed in seeds:
        for kind in modes:
            r = run_episode(model, base, seed, seconds, speed, kind, gain, cap, bias,
                            yaw_net=nets.get(kind), viewer=viewer)
            rows.append(dict(mode=kind, seed=seed, gain=gain, cap_rad_s=cap,
                             bias_rad_s=bias,
                             time_s=r["time_s"], forward_m=r["forward_m"],
                             sideways_m=r["sideways_m"],
                             max_abs_yaw_deg=r["max_abs_yaw_deg"],
                             pitch_rms_deg=r["pitch_rms_deg"],
                             roll_rms_deg=r["roll_rms_deg"],
                             minimum_local_height_m=r["minimum_local_height_m"],
                             pass_15deg=r["pass_strict"],
                             pass_5deg=bool(r["pass_strict"] and r["max_abs_yaw_deg"] <= GOAL_DEG),
                             stop=r["stop"],
                             wall_inference_ms=r["wall_inference_ms"],
                             replay=r["replay"]))
            print(f"{kind:<10} seed={seed}: {r['time_s']:.2f}s "
                  f"X={r['forward_m']:.3f}m yaw={r['max_abs_yaw_deg']:.2f}° "
                  f"PASS15={r['pass_strict']} PASS5={rows[-1]['pass_5deg']}", flush=True)
    output = OUT / "evaluation" / "episodes.csv"
    save_rows(rows, output)
    summary = {}
    for kind in modes:
        group = [r for r in rows if r["mode"] == kind]
        summary[kind] = dict(n=len(group), pass_15=sum(r["pass_15deg"] for r in group),
                             pass_5=sum(r["pass_5deg"] for r in group),
                             mean_peak_yaw_deg=float(np.mean([r["max_abs_yaw_deg"] for r in group])),
                             worst_peak_yaw_deg=float(max(r["max_abs_yaw_deg"] for r in group)),
                             mean_forward_m=float(np.mean([r["forward_m"] for r in group])))
    (OUT / "evaluation" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n", json.dumps(summary, indent=2), flush=True)
    print("CSV:", output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="job", required=True)
    s = sub.add_parser("sweep", help="Classical diagnostic sweep; no new network yet")
    s.add_argument("--seeds", type=int, nargs="+", default=[6501, 6502, 6801])
    s.add_argument("--seconds", type=float, default=8.0)
    s.add_argument("--command", type=float, default=0.45)
    d = sub.add_parser("distill", help="Train a separate yaw LIF to mimic the best diagnostic")
    d.add_argument("--epochs", type=int, default=35)
    d.add_argument("--lr", type=float, default=0.003)
    d.add_argument("--batch", type=int, default=256)
    t = sub.add_parser("test", help="Evaluate autonomous SNN yaw against frozen baselines")
    t.add_argument("--seeds", type=int, nargs="+", default=[6501, 6502, 6801])
    t.add_argument("--seconds", type=float, default=8.0)
    t.add_argument("--command", type=float, default=0.45)
    t.add_argument("--modes", nargs="+", choices=["original", "distilled", "diagnostic", "zero"],
                   default=["original", "distilled", "diagnostic"])
    t.add_argument("--viewer", action="store_true")
    args = p.parse_args()
    if args.job == "sweep":
        if not 1 <= len(set(args.seeds)) == len(args.seeds) <= 10:
            p.error("Use 1..10 distinct development seeds")
        if not 1 <= args.seconds <= 12 or not 0.1 <= args.command <= 0.7:
            p.error("Use seconds 1..12, forward command 0.1..0.7 m/s")
        sweep(args.seeds, args.seconds, args.command)
    elif args.job == "distill":
        if not 1 <= args.epochs <= 150 or not 0 < args.lr <= 0.01:
            p.error("Epochs 1..150 and learning rate 0..0.01 required")
        if not 32 <= args.batch <= 1024:
            p.error("batch must be between 32 and 1024")
        distill(args.epochs, args.lr, args.batch)
    else:
        if not 1 <= len(set(args.seeds)) == len(args.seeds) <= 20:
            p.error("Use 1..20 distinct seeds")
        if not 1 <= args.seconds <= 12 or not 0.1 <= args.command <= 0.7:
            p.error("Use seconds 1..12, forward command 0.1..0.7 m/s")
        test(args.seeds, args.seconds, args.command, args.modes, args.viewer)


if __name__ == "__main__":
    main()
