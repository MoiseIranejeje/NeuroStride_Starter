r"""NeuroStride Go2: matched slope expert + on-policy corrective dataset for V6.

Windows / VS Code PowerShell:
  .\.venv\Scripts\python.exe go2_collect_slope_v6.py --seeds 5701 5702 --slopes 0 2 --seconds 8 --command 0.45

TWO distinct data sources:
  expert:  ANN teacher drives the simulator and labels its visited states.
  student: V5 SNN drives the simulator; ANN teacher LABELS those states but never acts.

The source ANN is the independently verified pretrained walk-these-ways controller,
not a trained NeuroStride ANN. V5 is a spiking student plus classical heading PD.
The output is a TRAINING/CORRECTION dataset; it does NOT train V6 or prove terrain
adaptation. Keep the later final test seeds/terrain angles separate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

from go2_first_terrain_pilot import build_terrain
from go2_heading_benchmark import heading_rad, wrap_pi, load_teacher
from go2_v5_steering import WEIGHTS, make_model
from train_go2_snn_v3 import features_v3_online
from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, DECIMATION, DEFAULT_JOINT_POS,
    EXPECTED_ACTUATORS, HISTORY_LEN, KP, KD, OBS_DIM, build_obs,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "go2_slope_v6"
HEIGHT_LIMIT = 0.18  # torso height over local sloped ground


def roll_pitch_degrees(wxyz):
    w, x, y, z = map(float, wxyz)
    r = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    p = math.asin(float(np.clip(2 * (w*y - z*x), -1.0, 1.0)))
    return math.degrees(r), math.degrees(p)


def one_run(model, seed, slope, mode, seconds, forward_speed, teacher, student, episode_index):
    """Roll out exactly one controller. Always ask ANN for labels at its visited states."""
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home < 0:
        raise RuntimeError("Missing Go2 home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.normal(0.0, 0.012, 12)
    data.qpos[2] += float(rng.uniform(-0.005, 0.005))
    mujoco.mj_forward(model, data)

    if model.nu != 12 or [model.actuator(i).name for i in range(12)] != EXPECTED_ACTUATORS:
        raise RuntimeError("Go2 actuator mapping changed")
    if abs(model.opt.timestep - .002) > 1e-9:
        raise RuntimeError("Unexpected physics timestep")

    start_xy = data.qpos[:2].copy()
    yaw0 = heading_rad(data.qpos[3:7])
    cmd = BASE_COMMAND.copy()
    cmd[0] = forward_speed
    hist = deque((np.zeros(OBS_DIM, np.float32) for _ in range(HISTORY_LEN)), maxlen=HISTORY_LEN)
    prev = np.zeros(12, np.float64)
    last = np.zeros(12, np.float64)
    target = DEFAULT_JOINT_POS.copy()
    phase = 0.0
    torque_lo = model.actuator_ctrlrange[:, 0]
    torque_hi = model.actuator_ctrlrange[:, 1]
    t_body, t_adapt = teacher
    if mode == "student":
        s_net, (mx, sx, my, sy) = student

    samples = {
        "obs": [], "history": [], "teacher_action": [], "executed_action": [],
        "student_action": [], "episode": [], "seed": [], "slope_deg": [],
        "source": [], "sim_time": [], "command_x": [], "command_yaw": [],
        "yaw_error_deg": [], "base_xyz": [], "torso_above_plane_m": [],
    }
    full_state = {"qpos": [], "qvel": [], "time": []}
    peak_yaw = 0.0
    roll_sq = pitch_sq = 0.0
    posture_steps = 0
    worst_height = float(data.qpos[2] - math.tan(math.radians(slope)) * data.qpos[0])
    stop = "completed"

    def record_state():
        if full_state["time"] and float(data.time) == full_state["time"][-1]:
            return
        full_state["qpos"].append(data.qpos.copy())
        full_state["qvel"].append(data.qvel.copy())
        full_state["time"].append(float(data.time))

    record_state()
    frames = int(round(seconds / model.opt.timestep))
    for step in range(frames):
        if step % DECIMATION == 0:
            error = wrap_pi(heading_rad(data.qpos[3:7]) - yaw0)
            cmd[2] = float(np.clip(-0.9 * error, -0.2, 0.2))
            phase = (phase + cmd[4] * model.opt.timestep * DECIMATION) % 1.0
            obs = build_obs(data, cmd, prev, last, phase)
            hist.append(obs)
            history = np.stack(hist).astype(np.float32)
            flat = torch.from_numpy(history.reshape(1, -1).copy())

            with torch.inference_mode():
                latent = t_adapt(flat)
                expert_raw = t_body(torch.cat([flat, latent], dim=1)).squeeze(0).cpu().numpy()
                if mode == "student":
                    feat = features_v3_online(history)
                    normalized = np.clip((feat - mx) / sx, -5, 5).astype(np.float32)
                    pred, _ = s_net(torch.from_numpy(normalized[None]))
                    student_raw = pred.squeeze(0).cpu().numpy() * sy + my
                else:
                    student_raw = np.full(12, np.nan, np.float32)

            if expert_raw.shape != (12,) or not np.all(np.isfinite(expert_raw)) or np.max(np.abs(expert_raw)) > 12:
                stop = "invalid expert output"; break
            if mode == "student" and (student_raw.shape != (12,) or not np.isfinite(student_raw).all() or np.max(np.abs(student_raw)) > 12):
                stop = "invalid SNN output"; break

            label = np.clip(expert_raw, -6, 6).astype(np.float32)
            action = label.copy() if mode == "expert" else np.clip(student_raw, -6, 6).astype(np.float32)
            clearance = float(data.qpos[2] - math.tan(math.radians(slope)) * data.qpos[0])

            new_sample = {
                "obs": obs.copy(), "history": history.reshape(-1).copy(),
                "teacher_action": label, "executed_action": action,
                "student_action": student_raw.astype(np.float32),
                "episode": int(episode_index), "seed": int(seed),
                "slope_deg": float(slope), "source": mode,
                "sim_time": float(data.time), "command_x": float(cmd[0]),
                "command_yaw": float(cmd[2]), "yaw_error_deg": math.degrees(error),
                "base_xyz": data.qpos[:3].copy(), "torso_above_plane_m": clearance,
            }
            for key, value in new_sample.items():
                samples[key].append(value)

            prev, last = last.copy(), action.astype(np.float64)
            target = DEFAULT_JOINT_POS + last * ACTION_SCALE

        torques = KP * (target - data.qpos[7:19]) - KD * data.qvel[6:18]
        data.ctrl[:] = np.clip(torques, torque_lo, torque_hi)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            stop = "nonfinite state"; break
        r, p = roll_pitch_degrees(data.qpos[3:7])
        roll_sq += r*r
        pitch_sq += p*p
        posture_steps += 1
        yaw_error = abs(math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - yaw0)))
        peak_yaw = max(peak_yaw, yaw_error)
        clearance = float(data.qpos[2] - math.tan(math.radians(slope)) * data.qpos[0])
        worst_height = min(worst_height, clearance)
        if step % DECIMATION == 0:
            record_state()
        if clearance < HEIGHT_LIMIT:
            stop = "torso below local clearance threshold"; break
    record_state()
    dx, dy = map(float, data.qpos[:2] - start_xy)
    full_time = float(data.time)
    pass_task = bool(stop == "completed" and full_time >= seconds - .002
                     and dx >= 1.5 and abs(dy) <= .25 and peak_yaw <= 15
                     and worst_height >= HEIGHT_LIMIT)
    report = {
        "episode": episode_index, "source": mode, "seed": seed, "slope_deg": slope,
        "requested_seconds": seconds, "sim_seconds": round(full_time, 4),
        "samples": len(samples["obs"]), "forward_m": round(dx, 5),
        "sideways_m": round(dy, 5), "max_abs_yaw_deg": round(peak_yaw, 4),
        "pitch_rms_deg": round(math.sqrt(pitch_sq / max(posture_steps,1)),4),
        "roll_rms_deg": round(math.sqrt(roll_sq / max(posture_steps,1)),4),
        "min_torso_above_plane_m": round(worst_height, 5),
        "termination": stop, "passes_strict_task": pass_task,
    }
    return samples, full_state, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[5701,5702])
    parser.add_argument("--slopes", nargs="+", type=float, default=[0.0, 2.0])
    parser.add_argument("--modes", nargs="+", choices=("expert","student"), default=["expert","student"])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--command", type=float, default=0.45)
    args = parser.parse_args()
    if not (0 < args.seconds <= 12 and 0 < args.command <= 1):
        parser.error("Use 0<seconds<=12 and 0<command<=1")
    if len(args.seeds) > 6 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 1–6 unique seeds")
    if len(args.slopes) > 5 or not args.slopes or any(abs(s)>3 for s in args.slopes):
        parser.error("Use at most 5 mild slopes within +/-3 degrees")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"No V5 weights: {WEIGHTS}")
    checkpoint = torch.load(WEIGHTS,map_location="cpu",weights_only=True)
    student = make_model(checkpoint)
    student[0].eval()
    teacher = load_teacher() # checksum-verifies third-party TorchScript weights
    OUT.mkdir(parents=True,exist_ok=True)

    all_samples = None
    summaries = []
    eid = 0
    for slope in args.slopes:
        model = build_terrain(slope)
        for seed in args.seeds:
            for mode in args.modes:
                print(f"Slope {slope:+.1f}° | seed {seed} | {mode} controlled",flush=True)
                samples, states, report = one_run(
                    model,seed,slope,mode,args.seconds,args.command,teacher,student,eid)
                if not report["samples"]:
                    raise RuntimeError(f"No data from episode {eid}: {report['termination']}")
                if all_samples is None:
                    all_samples = {k:[] for k in samples}
                for key in samples:
                    all_samples[key].extend(samples[key])
                slug = f"{mode}_slope{slope:+.1f}_seed{seed}"
                np.savez_compressed(OUT/f"{slug}_states.npz",
                    qpos=np.asarray(states["qpos"],np.float64),
                    qvel=np.asarray(states["qvel"],np.float64),
                    sim_time=np.asarray(states["time"],np.float64))
                summaries.append(report)
                eid += 1
                print(f"  t={report['sim_seconds']:.2f}s X={report['forward_m']:+.3f}m "
                      f"maxYaw={report['max_abs_yaw_deg']:.2f}° "
                      f"labelled={report['samples']} strictPass={report['passes_strict_task']}",flush=True)

    data = {}
    for key, values in all_samples.items():
        if key == "source": data[key] = np.asarray(values,dtype="U8")
        elif key in ("episode","seed"): data[key] = np.asarray(values,dtype=np.int32)
        else: data[key] = np.asarray(values,dtype=np.float32)
    n = len(data["obs"])
    if data["obs"].shape != (n,70) or data["history"].shape != (n,2100) or data["teacher_action"].shape != (n,12):
        raise RuntimeError("Dataset arrays have unexpected shapes")
    expected_last = data["history"].reshape(n,30,70)[:,-1,:]
    if not np.allclose(expected_last,data["obs"],atol=1e-6):
        raise RuntimeError("History alignment check failed")

    np.savez_compressed(OUT/"slope_steering_teacher_and_dagger.npz",**data)
    with (OUT/"episodes.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    meta = {
        "purpose":"Supervised terrain training labels and DAgger for future SNN V6",
        "teacher_source":"THIRD-PARTY verified pretrained walk-these-ways ANN",
        "student_source":str(checkpoint.get("source","unidentified V5 checkpoint")),
        "sampling_rate_hz":1/(model.opt.timestep*DECIMATION),
        "steering_feedback":"External gain0.9, yaw rate clipped to +/-0.2 rad/s",
        "seeds":args.seeds,"slopes_deg":args.slopes,"modes":args.modes,
        "total_samples":n,"counts":{v:int(np.sum(data["source"]==v)) for v in args.modes},
        "labels":"At each state: ANN 12 joint-position offset labels, clipped +/-6",
        "student_action":"NaN when source is expert, observed SNN action when source is student",
        "limitations":"Teacher-labelled data, not trained V6 or terrain success; training/test must split by episode and seed",
    }
    (OUT/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("\n=== SLOPE TRAINING DATA READY (NO V6 TRAINING YET) ===",flush=True)
    print(json.dumps(meta,indent=2),flush=True)
    print("Saved:",OUT/"slope_steering_teacher_and_dagger.npz",flush=True)
    print("Saved:",OUT/"episodes.csv",flush=True)
    print("Saved:",OUT/"metadata.json",flush=True)

if __name__ == "__main__":
    main()
