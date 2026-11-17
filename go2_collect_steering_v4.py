
"""Collect Go2 ANN steering demonstrations for SNN V4."""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np
import torch

from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, DECIMATION,
    DEFAULT_JOINT_POS, EXPECTED_ACTUATORS,
    HISTORY_LEN, KP, KD, LOW_HEIGHT_THRESHOLD,
    OBS_DIM, POLICY_DIR, build_obs,
    verify_checkpoints,
)

OUT = Path(__file__).resolve().parent / "results" / "go2_steering_v4"


def yaw_angle(q):
    w, x, y, z = map(float, q)
    return math.atan2(
        2 * (w*z + x*y),
        1 - 2 * (y*y + z*z)
    )


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[4101, 4102]
    )
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--forward", type=float, default=0.45)
    args = parser.parse_args()

    if not (0 < args.seconds <= 20 and 0 < args.forward <= 1):
        parser.error("Invalid duration or forward speed")

    verify_checkpoints()
    torch.set_num_threads(min(4, torch.get_num_threads()))

    body = torch.jit.load(
        str(POLICY_DIR / "body_latest.jit"),
        map_location="cpu"
    ).eval()

    adapt = torch.jit.load(
        str(POLICY_DIR / "adaptation_module_latest.jit"),
        map_location="cpu"
    ).eval()

    model = mm.load("unitree_go2")

    names = [
        model.actuator(i).name
        for i in range(model.nu)
    ]

    if (
        names != EXPECTED_ACTUATORS
        or abs(model.opt.timestep - 0.002) > 1e-8
    ):
        raise ValueError("Unexpected Go2 model configuration")

    home = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )

    if home < 0:
        raise ValueError("Go2 home keyframe missing")

    programs = [
        ("straight", 0.0),
        ("turn_left", +0.15),
        ("turn_right", -0.15),
        ("heading_hold", None),
    ]

    keys = (
        "obs", "history", "teacher_action",
        "teacher_action_raw", "episode", "seed",
        "mode", "command_x", "command_yaw",
        "yaw_error_rad", "base_xyz", "sim_time"
    )

    collected = {key: [] for key in keys}
    reports = []

    steps_per_episode = round(
        args.seconds / model.opt.timestep
    )

    for seed in args.seeds:
        for program, requested_yaw in programs:

            episode_id = len(reports)
            data = mujoco.MjData(model)

            mujoco.mj_resetDataKeyframe(
                model, data, home
            )

            data.qpos[7:19] = DEFAULT_JOINT_POS
            data.qpos[2] = 0.30

            rng = np.random.default_rng(seed)

            data.qpos[7:19] += rng.normal(
                0, 0.012, size=12
            )

            data.qpos[2] += float(
                rng.uniform(-0.005, 0.005)
            )

            mujoco.mj_forward(model, data)

            initial_xy = data.qpos[:2].copy()
            initial_yaw = yaw_angle(data.qpos[3:7])

            command = BASE_COMMAND.copy()
            command[0] = args.forward

            history = deque(
                (
                    np.zeros(OBS_DIM, dtype=np.float32)
                    for _ in range(HISTORY_LEN)
                ),
                maxlen=HISTORY_LEN
            )

            previous = np.zeros(12)
            last = np.zeros(12)
            target = DEFAULT_JOINT_POS.copy()

            phase = 0.0
            min_height = float(data.qpos[2])
            max_yaw = 0.0
            sample_count = 0
            termination = "completed"

            for step in range(steps_per_episode):

                if step % DECIMATION == 0:

                    yaw_error = wrap(
                        yaw_angle(data.qpos[3:7])
                        - initial_yaw
                    )

                    if requested_yaw is None:
                        command[2] = float(np.clip(
                            -0.9 * yaw_error,
                            -0.2, 0.2
                        ))
                    else:
                        command[2] = requested_yaw

                    phase = (
                        phase
                        + command[4]
                        * model.opt.timestep
                        * DECIMATION
                    ) % 1.0

                    obs = build_obs(
                        data, command,
                        previous, last, phase
                    )

                    history.append(obs)

                    sequence = np.stack(
                        history
                    ).astype(np.float32).reshape(1, -1)

                    with torch.inference_mode():
                        h = torch.from_numpy(sequence)
                        latent = adapt(h)

                        raw = body(
                            torch.cat([h, latent], dim=1)
                        ).squeeze(0).numpy()

                    if (
                        raw.shape != (12,)
                        or not np.isfinite(raw).all()
                    ):
                        termination = "invalid teacher action"
                        break

                    if np.max(np.abs(raw)) > 12:
                        termination = "action guard exceeded"
                        break

                    applied = np.clip(
                        raw, -6.0, 6.0
                    ).astype(np.float64)

                    collected["obs"].append(obs)
                    collected["history"].append(sequence[0])
                    collected["teacher_action"].append(applied)
                    collected["teacher_action_raw"].append(raw)
                    collected["episode"].append(episode_id)
                    collected["seed"].append(seed)
                    collected["mode"].append(program)
                    collected["command_x"].append(command[0])
                    collected["command_yaw"].append(command[2])
                    collected["yaw_error_rad"].append(yaw_error)
                    collected["base_xyz"].append(data.qpos[:3].copy())
                    collected["sim_time"].append(float(data.time))

                    sample_count += 1

                    previous = last.copy()
                    last = applied.copy()

                    target = (
                        DEFAULT_JOINT_POS
                        + applied * ACTION_SCALE
                    )

                torque = (
                    KP * (target - data.qpos[7:19])
                    - KD * data.qvel[6:18]
                )

                data.ctrl[:] = np.clip(
                    torque,
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1]
                )

                mujoco.mj_step(model, data)

                if not (
                    np.isfinite(data.qpos).all()
                    and np.isfinite(data.qvel).all()
                ):
                    termination = "nonfinite simulator state"
                    break

                min_height = min(
                    min_height,
                    float(data.qpos[2])
                )

                max_yaw = max(
                    max_yaw,
                    abs(math.degrees(wrap(
                        yaw_angle(data.qpos[3:7])
                        - initial_yaw
                    )))
                )

                if data.qpos[2] < LOW_HEIGHT_THRESHOLD:
                    termination = "low torso height"
                    break

            dx, dy = data.qpos[:2] - initial_xy

            final_yaw = math.degrees(wrap(
                yaw_angle(data.qpos[3:7])
                - initial_yaw
            ))

            result = {
                "episode": episode_id,
                "seed": seed,
                "mode": program,
                "requested_yaw_rate": (
                    requested_yaw
                    if requested_yaw is not None
                    else "feedback"
                ),
                "simulated_seconds": round(float(data.time), 3),
                "samples": sample_count,
                "forward_m": round(float(dx), 4),
                "sideways_m": round(float(dy), 4),
                "final_yaw_deg": round(final_yaw, 3),
                "max_abs_yaw_deg": round(max_yaw, 3),
                "min_torso_height_m": round(min_height, 4),
                "termination": termination,
            }

            reports.append(result)

            print(
                f"Seed {seed} | {program:12s} | "
                f"{data.time:.2f}s | "
                f"forward={dx:+.2f}m | "
                f"yaw={final_yaw:+.1f}deg | "
                f"min_height={min_height:.3f}m | "
                f"samples={sample_count} | "
                f"{termination}",
                flush=True
            )

    if not collected["obs"]:
        raise RuntimeError("No teacher samples collected")

    OUT.mkdir(parents=True, exist_ok=True)

    arrays = {}

    for key, values in collected.items():
        if key == "mode":
            arrays[key] = np.asarray(values, dtype="U16")
        elif key in ("episode", "seed"):
            arrays[key] = np.asarray(values, dtype=np.int32)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)

    np.savez_compressed(
        OUT / "steering_teacher.npz",
        **arrays
    )

    with (OUT / "episodes.csv").open(
        "w", newline="", encoding="utf8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=reports[0].keys()
        )
        writer.writeheader()
        writer.writerows(reports)

    metadata = {
        "source": "Third-party pretrained ANN teacher",
        "purpose": "Training data for SNN V4 steering",
        "samples": len(collected["obs"]),
        "seeds": args.seeds,
        "forward_mps": args.forward,
        "modes": [program for program, _ in programs],
        "action_semantics": (
            "12 joint-position offsets, clipped to +/-6"
        ),
        "limitations": (
            "Flat terrain only; SNN V4 not yet trained"
        ),
    }

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf8"
    )

    print("\n=== GO2 STEERING TEACHER DATA ===")
    print("Samples:", len(collected["obs"]))
    print("Output:", OUT)
    print("Saved: steering_teacher.npz, episodes.csv, metadata.json")
    print("ANN teaching data only — SNN V4 not yet trained.")


if __name__ == "__main__":
    main()
