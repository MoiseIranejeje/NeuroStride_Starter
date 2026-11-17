"""NeuroStride Go2 | headless ANN-teacher demonstration collector.

This script collects expert observation histories + motor action targets for
later SNN imitation. It does not train an SNN or demonstrate uneven terrain.

Place this file in the root of NeuroStride_Starter and execute with its .venv.
It uses the verified checkpoint files downloaded from the reference repo.
"""
from __future__ import annotations

from collections import deque
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
POLICY_DIR = (ROOT / "go2_cpu_reference" / "rl-locomotion-learning-main" /
              "stage2-go2-mujoco-inference" / "checkpoints")
RESULTS = ROOT / "results" / "go2_teacher"
EXPECTED_HASHES = {
    "body_latest.jit": "7b6e604e2147742a89ef50d91e7ee501023331b2589d1c3143a9d2ba858db7b5",
    "adaptation_module_latest.jit": "0e091f829dcfbedd4ccca6752863e1e2feca105f79da07d04e3545b8815dcc13",
}

# Match reference author's 06_run_policy.py, not the Menagerie 'home' joint pose.
DEFAULT_JOINT_POS = np.array([
    0.1, 0.8, -1.5,
   -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
   -0.1, 1.0, -1.5,
], dtype=np.float64)

# Reference gait commands: [forward, lateral, yaw, height, gait_Hz, phase,
# offset, bound, duty, swing_height, pitch, roll, stance_width, length, aux].
BASE_COMMAND = np.array([
    0.5, 0, 0, 0, 2.0, 0.5, 0, 0, 0.5, 0.06, 0, 0, 0, 0, 0,
], dtype=np.float64)
COMMAND_SCALES = np.array([
    2.0, 2.0, 0.25, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.15,
    0.3, 0.3, 1.0, 1.0, 1.0,
], dtype=np.float64)
ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float64)
KP, KD = 25.0, 0.6
DECIMATION, OBS_DIM, HISTORY_LEN = 10, 70, 30
SIM_SECONDS_PER_EPISODE = 12.0
# Each run starts from the same pose and uses a different commanded velocity.
# This is a first pilot dataset, NOT a robust multi-seed evaluation.
FORWARD_COMMANDS_MPS = (0.3, 0.4, 0.5, 0.6)
LOW_HEIGHT_THRESHOLD = 0.18
EXPECTED_ACTUATORS = [
    f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
]


def verify_checkpoints() -> None:
    for filename, expected in EXPECTED_HASHES.items():
        path = POLICY_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing pretrained teacher file: {path}")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != expected:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        print(f"Verified teacher: {filename}")


def inverse_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Exact projected-gravity transform used by the reference teacher.
    w, u = q[0], q[1:]
    return (v * (2 * w**2 - 1) - np.cross(u, v) * (2 * w)
            + u * (2 * (u @ v)))


def build_obs(data: mujoco.MjData, command: np.ndarray,
              previous_action: np.ndarray, last_action: np.ndarray,
              phase: float) -> np.ndarray:
    gravity = inverse_rotate(data.qpos[3:7], np.array([0., 0., -1.]))
    gait_phase, gait_offset, gait_bound = command[5:8]
    foot_phases = np.array([
        phase + gait_phase + gait_offset + gait_bound,
        phase + gait_offset,
        phase + gait_bound,
        phase + gait_phase,
    ]) % 1.0
    clock = np.sin(2 * np.pi * foot_phases)
    obs = np.concatenate([
        gravity,
        command * COMMAND_SCALES,
        data.qpos[7:19] - DEFAULT_JOINT_POS,
        data.qvel[6:18] * 0.05,
        last_action,
        previous_action,
        clock,
    ]).astype(np.float32)
    if obs.shape != (OBS_DIM,) or not np.all(np.isfinite(obs)):
        raise ValueError("Malformed Go2 70-d observation")
    return obs


def main() -> None:
    verify_checkpoints()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    body = torch.jit.load(str(POLICY_DIR / "body_latest.jit"), map_location="cpu")
    adaptation = torch.jit.load(str(POLICY_DIR / "adaptation_module_latest.jit"), map_location="cpu")
    body.eval()
    adaptation.eval()

    model = mm.load("unitree_go2")
    if model.nu != 12 or abs(model.opt.timestep - 0.002) > 1e-8:
        raise ValueError(f"Expected 12 motors and dt=0.002; got nu={model.nu}, dt={model.opt.timestep}")
    names = [model.actuator(i).name for i in range(model.nu)]
    if names != EXPECTED_ACTUATORS:
        raise ValueError(f"Unexpected motor order: {names}")
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home < 0:
        raise ValueError("MuJoCo Go2 model missing home keyframe")
    data = mujoco.MjData(model)
    torque_lower = model.actuator_ctrlrange[:, 0]
    torque_upper = model.actuator_ctrlrange[:, 1]

    features, histories, targets, latents = [], [], [], []
    episode_ids, command_xs, sim_times, positions = [], [], [], []
    episode_reports = []
    total_steps = int(SIM_SECONDS_PER_EPISODE / model.opt.timestep)

    for ep, vx_command in enumerate(FORWARD_COMMANDS_MPS):
        mujoco.mj_resetDataKeyframe(model, data, home)
        data.qpos[7:19] = DEFAULT_JOINT_POS
        data.qpos[2] = 0.30
        mujoco.mj_forward(model, data)
        start_xy = data.qpos[:2].copy()
        command = BASE_COMMAND.copy()
        command[0] = vx_command
        hist = deque((np.zeros(OBS_DIM, dtype=np.float32) for _ in range(HISTORY_LEN)),
                     maxlen=HISTORY_LEN)
        previous = np.zeros(12, dtype=np.float64)
        last = np.zeros(12, dtype=np.float64)
        joint_targets = DEFAULT_JOINT_POS.copy()
        phase = 0.0
        min_height = float(data.qpos[2])
        samples_at_start = len(targets)
        terminated_low = False

        for step in range(total_steps):
            if step % DECIMATION == 0:
                phase = (phase + command[4] * model.opt.timestep * DECIMATION) % 1.0
                observation = build_obs(data, command, previous, last, phase)
                hist.append(observation)
                sequence = np.concatenate(hist).astype(np.float32)

                with torch.no_grad():
                    history_tensor = torch.from_numpy(sequence[None, :])
                    latent_tensor = adaptation(history_tensor)
                    action_tensor = body(torch.cat([history_tensor, latent_tensor], dim=1))
                    latent = latent_tensor.squeeze(0).numpy().astype(np.float32)
                    action = action_tensor.squeeze(0).numpy().astype(np.float32)
                if latent.shape != (2,) or action.shape != (12,):
                    raise ValueError("Pretrained policy returned wrong output shape")
                if not np.isfinite(action).all():
                    raise ValueError("Pretrained policy produced non-finite action")

                # Dataset: predict joint-target offsets, not motor torque.
                features.append(observation)
                histories.append(sequence)
                targets.append(action)
                latents.append(latent)
                episode_ids.append(ep)
                command_xs.append(vx_command)
                sim_times.append(float(data.time))
                positions.append(data.qpos[:3].copy())

                previous = last.copy()
                last = action.astype(np.float64)
                joint_targets = DEFAULT_JOINT_POS + last * ACTION_SCALE

            q, qd = data.qpos[7:19], data.qvel[6:18]
            torque = KP * (joint_targets - q) - KD * qd
            data.ctrl[:] = np.clip(torque, torque_lower, torque_upper)
            mujoco.mj_step(model, data)
            min_height = min(min_height, float(data.qpos[2]))
            if not np.isfinite(data.qpos).all():
                raise RuntimeError(f"Non-finite MuJoCo state during episode {ep}")
            if data.qpos[2] < LOW_HEIGHT_THRESHOLD:
                terminated_low = True
                break

        dx, dy = data.qpos[:2] - start_xy
        report = {
            "episode": ep,
            "commanded_forward_mps": vx_command,
            "simulated_seconds": round(float(data.time), 3),
            "samples": len(targets) - samples_at_start,
            "forward_m": round(float(dx), 4),
            "sideways_m": round(float(dy), 4),
            "mean_forward_mps": round(float(dx / max(data.time, 1e-8)), 4),
            "minimum_torso_height_m": round(min_height, 4),
            "height_threshold_crossed": terminated_low,
        }
        episode_reports.append(report)
        print(f"Episode {ep+1}/{len(FORWARD_COMMANDS_MPS)}: "
              f"command={vx_command:.2f} m/s, forward={dx:.2f}m, "
              f"sideways={dy:.2f}m, samples={report['samples']}, "
              f"low_height={terminated_low}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = RESULTS / "go2_teacher_pilot.npz"
    np.savez_compressed(
        dataset,
        obs=np.stack(features).astype(np.float32),           # [N,70]
        history=np.stack(histories).astype(np.float32),      # [N,2100]
        teacher_action=np.stack(targets).astype(np.float32), # [N,12]
        teacher_latent=np.stack(latents).astype(np.float32), # [N,2], diagnostic only
        episode=np.array(episode_ids, dtype=np.int16),
        command_x=np.array(command_xs, dtype=np.float32),
        sim_time=np.array(sim_times, dtype=np.float32),
        base_xyz=np.stack(positions).astype(np.float32),
    )
    with (RESULTS / "go2_teacher_pilot_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_reports[0].keys())
        writer.writeheader()
        writer.writerows(episode_reports)
    metadata = {
        "goal": "ANN teacher dataset for later SNN behavior cloning; not SNN training",
        "model": "MuJoCo Menagerie unitree_go2",
        "pretrained_teacher": "Ashfak-Kausik/rl-locomotion-learning v1.0-policy",
        "checkpoint_sha256": EXPECTED_HASHES,
        "policy_hz": 50,
        "physics_hz": 500,
        "action_semantics": "12 raw ANN joint position offsets (before per-joint scales and PD)",
        "dataset_shapes": {"obs": [len(features), 70], "history": [len(features), 2100], "teacher_action": [len(features), 12]},
        "limitations": "One fixed start pose per command, no uneven terrain, no independent validation seeds",
    }
    (RESULTS / "go2_teacher_pilot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n=== TEACHER DATASET COLLECTED ===")
    print(f"Observations: {len(features)} x 70")
    print(f"Observation histories: {len(features)} x 2100")
    print(f"Teacher actions: {len(features)} x 12")
    print(f"Saved dataset: {dataset}")
    print(f"Saved episode reports: {RESULTS / 'go2_teacher_pilot_episodes.csv'}")
    print("NOTE: These are ANN-generated training examples, not a trained SNN.")


if __name__ == "__main__":
    main()
