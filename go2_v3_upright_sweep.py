
"""NeuroStride Go2: upright and straight walking experiment.

Uses the trained SNN V3 checkpoint.
alpha=1.0: original SNN.
alpha<1.0: SNN with classical motor-action smoothing.

Measures posture, heading, movement, survival, and saves replays.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import mujoco_menagerie as mm
import numpy as np
import torch

from go2_compare_v3 import load_actor
from train_go2_snn_v3 import features_v3_online
from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, DECIMATION,
    DEFAULT_JOINT_POS, EXPECTED_ACTUATORS,
    HISTORY_LEN, KP, KD, LOW_HEIGHT_THRESHOLD,
    OBS_DIM, build_obs,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "go2_upright_sweep"


def euler_deg_from_wxyz(quaternion):
    w, x, y, z = (float(v) for v in quaternion)

    roll = math.atan2(
        2 * (w*x + y*z),
        1 - 2 * (x*x + y*y)
    )

    pitch = math.asin(
        max(-1.0, min(1.0, 2 * (w*y - z*x)))
    )

    yaw = math.atan2(
        2 * (w*z + x*y),
        1 - 2 * (y*y + z*z)
    )

    return tuple(
        math.degrees(v) for v in (roll, pitch, yaw)
    )


def one_rollout(
    net, stats, model, alpha, speed,
    seed, seconds, viewer=False, record=True
):
    data = mujoco.MjData(model)

    home = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )

    if home < 0:
        raise ValueError("Go2 model missing home keyframe")

    mujoco.mj_resetDataKeyframe(model, data, home)

    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30

    rng = np.random.default_rng(seed)

    data.qpos[7:19] += rng.normal(
        0.0, 0.012, size=12
    )

    data.qpos[2] += float(
        rng.uniform(-0.005, 0.005)
    )

    mujoco.mj_forward(model, data)

    origin = data.qpos[:3].copy()

    _, _, start_yaw = euler_deg_from_wxyz(
        data.qpos[3:7]
    )

    command = BASE_COMMAND.copy()
    command[0] = speed

    history = deque(
        (
            np.zeros(OBS_DIM, dtype=np.float32)
            for _ in range(HISTORY_LEN)
        ),
        maxlen=HISTORY_LEN,
    )

    prev = np.zeros(12, dtype=np.float64)
    last = np.zeros(12, dtype=np.float64)

    targets = DEFAULT_JOINT_POS.copy()
    phase = 0.0

    lower = model.actuator_ctrlrange[:, 0]
    upper = model.actuator_ctrlrange[:, 1]

    mu_x, sd_x, mu_y, sd_y = stats

    max_frames = int(
        round(seconds / model.opt.timestep)
    )

    stop = "completed requested duration"

    min_height = float(data.qpos[2])
    max_side = 0.0
    cumulative_side = 0.0
    planar_path = 0.0

    last_xy = data.qpos[:2].copy()

    rolls = []
    pitches = []
    yaws = []
    spikes = []

    torques_clipped = 0
    loop_steps = 0
    decisions = 0

    saved = {
        "qpos": [],
        "qvel": [],
        "sim_time": [],
        "joint_torque": [],
    }

    def snapshot():
        current_time = float(data.time)

        if (
            saved["sim_time"]
            and saved["sim_time"][-1] == current_time
        ):
            return

        saved["qpos"].append(data.qpos.copy())
        saved["qvel"].append(data.qvel.copy())
        saved["sim_time"].append(current_time)
        saved["joint_torque"].append(data.ctrl.copy())

    if record:
        snapshot()

    viewer_cm = (
        mujoco.viewer.launch_passive(model, data)
        if viewer
        else contextlib.nullcontext(None)
    )

    wall_last = time.monotonic()

    with viewer_cm as view:

        for step in range(max_frames):

            if viewer and not view.is_running():
                stop = "viewer closed"
                break

            if step % DECIMATION == 0:

                phase = (
                    phase
                    + command[4]
                    * model.opt.timestep
                    * DECIMATION
                ) % 1.0

                obs = build_obs(
                    data, command, prev, last, phase
                )

                history.append(obs)

                features = features_v3_online(
                    np.stack(history).astype(np.float32)
                )

                x = np.clip(
                    (features - mu_x) / sd_x,
                    -5, 5
                ).astype(np.float32)

                with torch.inference_mode():
                    out, spike_activity = net(
                        torch.from_numpy(x[None])
                    )

                raw = (
                    out.squeeze(0).cpu().numpy()
                    * sd_y + mu_y
                )

                if (
                    raw.shape != (12,)
                    or not np.isfinite(raw).all()
                ):
                    stop = "invalid neural action"
                    break

                if np.max(np.abs(raw)) > 8:
                    stop = "neural output guard exceeded +/-8"
                    break

                raw = np.clip(
                    raw, -6.0, 6.0
                ).astype(np.float64)

                # alpha=1: original V3.
                # alpha<1: smoothed motor command.
                applied = (
                    alpha * raw
                    + (1.0 - alpha) * last
                )

                prev = last.copy()
                last = applied.copy()

                targets = (
                    DEFAULT_JOINT_POS
                    + applied * ACTION_SCALE
                )

                spikes.append(float(spike_activity))
                decisions += 1

            torque = (
                KP * (targets - data.qpos[7:19])
                - KD * data.qvel[6:18]
            )

            torques_clipped += int(
                np.any(
                    (torque < lower) | (torque > upper)
                )
            )

            data.ctrl[:] = np.clip(
                torque, lower, upper
            )

            mujoco.mj_step(model, data)
            loop_steps += 1

            if (
                not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
            ):
                stop = "nonfinite MuJoCo state"
                break

            min_height = min(
                min_height, float(data.qpos[2])
            )

            new_xy = data.qpos[:2].copy()

            planar_path += float(
                np.linalg.norm(new_xy - last_xy)
            )

            cumulative_side += abs(
                float(new_xy[1] - last_xy[1])
            )

            last_xy = new_xy

            max_side = max(
                max_side,
                abs(float(new_xy[1] - origin[1]))
            )

            roll, pitch, yaw = euler_deg_from_wxyz(
                data.qpos[3:7]
            )

            rolls.append(roll)
            pitches.append(pitch)
            yaws.append(yaw)

            if step % DECIMATION == 0:

                if record:
                    snapshot()

                if viewer and view.is_running():
                    view.cam.lookat[:] = data.qpos[:3]
                    view.sync()

                    now = time.monotonic()

                    time.sleep(
                        max(
                            0.0,
                            model.opt.timestep * DECIMATION
                            - (now - wall_last)
                        )
                    )

                    wall_last = time.monotonic()

            if float(data.qpos[2]) < LOW_HEIGHT_THRESHOLD:
                stop = "crossed 0.18m torso height threshold"
                break

        if record:
            snapshot()

        if viewer and view.is_running():
            view.cam.lookat[:] = data.qpos[:3]
            view.sync()

    dur = float(data.time)

    dx, dy = (
        data.qpos[:2] - origin[:2]
    ).tolist()

    rolls = np.asarray(rolls, dtype=float)
    pitches = np.asarray(pitches, dtype=float)
    yaws = np.asarray(yaws, dtype=float)

    yaw_relative = (
        np.unwrap(np.deg2rad(yaws))
        - math.radians(start_yaw)
    ) * 180.0 / math.pi

    roll_rms = (
        math.sqrt(float(np.mean(rolls**2)))
        if len(rolls) else float("nan")
    )

    pitch_rms = (
        math.sqrt(float(np.mean(pitches**2)))
        if len(pitches) else float("nan")
    )

    tilt_rms = (
        math.sqrt(
            float(np.mean(rolls**2 + pitches**2))
        )
        if len(rolls) else float("nan")
    )

    peak_tilt = (
        float(
            np.max(
                np.sqrt(rolls**2 + pitches**2)
            )
        )
        if len(rolls) else float("nan")
    )

    final_yaw = (
        float(yaw_relative[-1])
        if len(yaw_relative) else 0.0
    )

    max_yaw = (
        float(np.max(np.abs(yaw_relative)))
        if len(yaw_relative) else 0.0
    )

    duration_ok = (
        stop == "completed requested duration"
        and dur >= seconds - 0.002
    )

    pass_orig = bool(
        duration_ok
        and dx >= 1.5
        and abs(dy) <= 0.5
        and min_height >= LOW_HEIGHT_THRESHOLD
    )

    upright_straight_target = bool(
        pass_orig
        and tilt_rms <= 10
        and max_yaw <= 15
        and max_side <= 0.25
    )

    row = {
        "controller": (
            "V3_SNN_only"
            if alpha == 1.0
            else "V3_SNN_plus_action_filter"
        ),
        "alpha": float(alpha),
        "seed": int(seed),
        "command_mps": float(speed),
        "duration_s": round(dur, 4),
        "forward_m": round(dx, 5),
        "sideways_m": round(dy, 5),
        "minimum_height_m": round(min_height, 5),
        "mean_forward_mps": (
            round(dx / dur, 5) if dur else 0.0
        ),
        "max_abs_sideways_m": round(max_side, 5),
        "cumulative_abs_sideways_m": round(
            cumulative_side, 5
        ),
        "forward_path_efficiency": round(
            dx / max(planar_path, 1e-9), 5
        ),
        "roll_rms_deg": round(roll_rms, 4),
        "pitch_rms_deg": round(pitch_rms, 4),
        "tilt_rms_deg": round(tilt_rms, 4),
        "peak_tilt_deg": round(peak_tilt, 4),
        "final_heading_error_deg": round(
            final_yaw, 4
        ),
        "max_heading_error_deg": round(max_yaw, 4),
        "fraction_steps_tilt_gt15deg": round(
            float(
                np.mean(
                    (rolls**2 + pitches**2) > 15.0**2
                )
            ) if len(rolls) else 0.0,
            4
        ),
        "fraction_physics_steps_with_torque_clip": round(
            torques_clipped / max(loop_steps, 1), 4
        ),
        "snn_decisions": int(decisions),
        "mean_hidden_spike_rate": (
            round(float(np.mean(spikes)), 5)
            if spikes else 0.0
        ),
        "termination": stop,
        "passed_original_8s_task": pass_orig,
        "passed_experimental_upright_straight_target":
            upright_straight_target,
    }

    if record:
        OUT.mkdir(parents=True, exist_ok=True)

        path = OUT / (
            f"v3_alpha{alpha:.2f}"
            f"_cmd{speed:.2f}_seed{seed}.npz"
        )

        np.savez_compressed(
            path,
            qpos=np.stack(saved["qpos"]).astype(np.float64),
            qvel=np.stack(saved["qvel"]).astype(np.float64),
            sim_time=np.asarray(
                saved["sim_time"], dtype=np.float64
            ),
            joint_torque=np.stack(
                saved["joint_torque"]
            ).astype(np.float32),
            summary_json=np.array(json.dumps(row)),
        )

        print("Saved replay:", path, flush=True)

    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[3101, 3102, 3103]
    )

    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=[1.0, 0.8, 0.6]
    )

    parser.add_argument(
        "--command", type=float, default=0.45
    )

    parser.add_argument(
        "--seconds", type=float, default=8.0
    )

    parser.add_argument(
        "--viewer", action="store_true"
    )

    parser.add_argument(
        "--no-record", action="store_true"
    )

    args = parser.parse_args()

    if (
        not 1 <= args.seconds <= 30
        or not 0 <= args.command <= 1
    ):
        parser.error("Invalid duration or speed")

    if not 1 <= len(args.seeds) <= 20:
        parser.error("Maximum 20 seeds")

    if not 1 <= len(args.alphas) <= 5:
        parser.error("Maximum 5 smoothing levels")

    if any(not 0 < a <= 1 for a in args.alphas):
        parser.error("Alpha must be in (0,1]")

    if (
        len(set(args.seeds)) != len(args.seeds)
        or len(set(args.alphas)) != len(args.alphas)
    ):
        parser.error("Seeds and alphas must be unique")

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    snn, stats = load_actor("v3")

    model = mm.load("unitree_go2")

    if (
        model.nu != 12
        or abs(model.opt.timestep - 0.002) > 1e-8
        or [
            model.actuator(i).name
            for i in range(model.nu)
        ] != EXPECTED_ACTUATORS
    ):
        raise ValueError(
            "Go2 model or actuator mapping mismatch"
        )

    rows = []

    for seed in args.seeds:
        for alpha in args.alphas:

            print(
                f"GO2 posture audit: "
                f"seed={seed} alpha={alpha:.2f} "
                f"command={args.command:.2f}",
                flush=True,
            )

            row = one_rollout(
                snn, stats, model, alpha,
                args.command, seed, args.seconds,
                viewer=args.viewer,
                record=not args.no_record,
            )

            rows.append(row)

            print(
                json.dumps(row, indent=2),
                flush=True,
            )

    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / "episodes.csv").open(
        "w", newline="", encoding="utf8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}

    for alpha in args.alphas:

        sample = [
            r for r in rows
            if r["alpha"] == alpha
        ]

        aggregate[f"alpha_{alpha:.2f}"] = {
            "runs": len(sample),
            "original_task_pass_count": int(
                sum(
                    r["passed_original_8s_task"]
                    for r in sample
                )
            ),
            "experimental_upright_straight_pass_count":
                int(
                    sum(
                        r[
                            "passed_experimental_upright_straight_target"
                        ]
                        for r in sample
                    )
                ),
            "mean_survival_s": float(
                np.mean([
                    r["duration_s"] for r in sample
                ])
            ),
            "mean_forward_m": float(
                np.mean([
                    r["forward_m"] for r in sample
                ])
            ),
            "mean_abs_sideways_m": float(
                np.mean([
                    abs(r["sideways_m"])
                    for r in sample
                ])
            ),
            "mean_tilt_rms_deg": float(
                np.mean([
                    r["tilt_rms_deg"] for r in sample
                ])
            ),
            "mean_abs_final_yaw_deg": float(
                np.mean([
                    abs(r["final_heading_error_deg"])
                    for r in sample
                ])
            ),
        }

    (OUT / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf8",
    )

    print(
        "\n=== NEUROSTRIDE UPRIGHT / STRAIGHTNESS ABLATION ==="
    )
    print(json.dumps(aggregate, indent=2))
    print("Episode CSV:", OUT / "episodes.csv")
    print("Full-state trajectories:", OUT)


if __name__ == "__main__":
    main()
