"""Evaluate the trained NeuroStride Go2 SNN acting ALONE in MuJoCo.

Uses no ANN teacher during control. The teacher's checkpoint is NOT loaded.
It reproduces MuJoCo/observation conventions from go2_collect_teacher.py.

Place beside train_go2_snn_v1.py + go2_collect_teacher.py in NeuroStride_Starter.
Run: .\\.venv\\Scripts\\python.exe go2_snn_closed_loop.py
Optional: .\\.venv\\Scripts\\python.exe go2_snn_closed_loop.py --viewer

Expected output is an experimental result, never a guaranteed walk.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import mujoco_menagerie as mm
import numpy as np
import torch

from train_go2_snn_v1 import Go2SpikingActor
from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, DECIMATION, DEFAULT_JOINT_POS,
    EXPECTED_ACTUATORS, HISTORY_LEN, KP, KD, LOW_HEIGHT_THRESHOLD,
    OBS_DIM, build_obs,
)

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'results' / 'go2_snn_v1' / 'go2_snn_v1_best.pt'
RESULT_DIR = ROOT / 'results' / 'go2_snn_v1'


def load_snn():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f'Missing SNN weights: {WEIGHTS}. Train SNN or extract supplied checkpoint first.')
    ckpt = torch.load(str(WEIGHTS), map_location='cpu', weights_only=True)
    config = ckpt['config']
    model = Go2SpikingActor(
        feature_dim=int(config['feature_dim']),
        population=int(config['population']),
        hidden=int(config['hidden']),
        steps=int(config['steps']),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    mu_x = ckpt['feature_mean'].cpu().numpy()
    std_x = ckpt['feature_std'].cpu().numpy()
    mu_y = ckpt['target_mean'].cpu().numpy()
    std_y = ckpt['target_std'].cpu().numpy()
    if mu_x.shape != (140,) or std_x.shape != (140,) or mu_y.shape != (12,) or std_y.shape != (12,):
        raise ValueError('Unexpected SNN checkpoint feature/action statistics')
    return model, mu_x, std_x, mu_y, std_y, ckpt


def evaluate(vx_command: float, duration_seconds: float, render: bool):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    snn, mu_x, std_x, mu_y, std_y, ckpt = load_snn()

    model = mm.load('unitree_go2')
    if model.nu != 12:
        raise ValueError(f'Expected 12 actuators, found {model.nu}')
    names = [model.actuator(i).name for i in range(model.nu)]
    if names != EXPECTED_ACTUATORS:
        raise ValueError(f'Unexpected actuator order: {names}')
    if abs(model.opt.timestep - 0.002) > 1e-8:
        raise ValueError(f'Expected 0.002 second MuJoCo timestep, got {model.opt.timestep}')
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key < 0:
        raise ValueError('Go2 home keyframe missing')
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    mujoco.mj_forward(model, data)
    start_xyz = data.qpos[:3].copy()
    command = BASE_COMMAND.copy()
    command[0] = vx_command
    history = deque((np.zeros(OBS_DIM, dtype=np.float32) for _ in range(HISTORY_LEN)), maxlen=HISTORY_LEN)
    prev = np.zeros(12, dtype=np.float64)
    last = np.zeros(12, dtype=np.float64)
    target_joints = DEFAULT_JOINT_POS.copy()
    phase = 0.0
    min_height = float(data.qpos[2])
    max_abs_y = 0.0
    total_abs_lateral = 0.0
    last_y = float(data.qpos[1])
    counts = 0
    spike_rates = []
    term_reason = 'completed full duration'
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    total_steps = int(duration_seconds / model.opt.timestep)

    print(f'Go2 SNN ONLY | command {vx_command:.2f} m/s | duration {duration_seconds:.1f}s | viewer={render}')
    print(f'Trained SNN checkpoint selected at epoch {ckpt["epoch"]}; no ANN teacher loaded')
    cm = mujoco.viewer.launch_passive(model, data) if render else contextlib.nullcontext(None)
    with cm as viewer:
        for step in range(total_steps):
            if render and not viewer.is_running():
                term_reason = 'viewer closed by user'
                break
            if step % DECIMATION == 0:
                phase = (phase + command[4] * model.opt.timestep * DECIMATION) % 1.0
                obs = build_obs(data, command, prev, last, phase)
                history.append(obs)
                context = np.mean(np.stack(history, axis=0), axis=0)
                x = np.concatenate([obs, context]).astype(np.float32)
                x_norm = np.clip((x - mu_x) / std_x, -5.0, 5.0)
                with torch.inference_mode():
                    prediction, spike_rate = snn(torch.from_numpy(x_norm[None]))
                    action = prediction.squeeze(0).cpu().numpy() * std_y + mu_y
                if not np.isfinite(action).all():
                    term_reason = 'non-finite SNN action'
                    break
                # Guard unrealistic raw joint targets; does not substitute
                # for training a stable controller.
                if float(np.max(np.abs(action))) > 8.0:
                    term_reason = 'SNN raw action exceeded +/-8 safety guard'
                    break
                action = np.clip(action, -6.0, 6.0)
                spike_rates.append(float(spike_rate))
                counts += 1
                prev = last.copy()
                last = action.astype(np.float64)
                target_joints = DEFAULT_JOINT_POS + last * ACTION_SCALE

            q = data.qpos[7:19]
            qd = data.qvel[6:18]
            data.ctrl[:] = np.clip(KP * (target_joints - q) - KD * qd, low, high)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                term_reason = 'non-finite MuJoCo state'
                break
            z = float(data.qpos[2])
            min_height = min(min_height, z)
            y = float(data.qpos[1])
            total_abs_lateral += abs(y - last_y)
            last_y = y
            max_abs_y = max(max_abs_y, abs(y - start_xyz[1]))
            if render and step % DECIMATION == 0:
                viewer.sync()
                time.sleep(model.opt.timestep * DECIMATION)
            if z < LOW_HEIGHT_THRESHOLD:
                term_reason = f'fell below {LOW_HEIGHT_THRESHOLD:.2f} m torso threshold'
                break
        if render and viewer.is_running():
            viewer.sync()

    delta = data.qpos[:3] - start_xyz
    summary = {
        'controller': 'NeuroStride Go2 SNN v1 — behavior cloned',
        'command_forward_mps': float(vx_command),
        'sim_seconds': float(data.time),
        'forward_m': float(delta[0]),
        'sideways_m': float(delta[1]),
        'mean_forward_mps': float(delta[0] / max(data.time, 1e-8)),
        'minimum_torso_height_m': float(min_height),
        'final_torso_height_m': float(data.qpos[2]),
        'maximum_abs_sideways_offset_m': float(max_abs_y),
        'cumulative_abs_lateral_travel_m': float(total_abs_lateral),
        'snn_inferences': counts,
        'mean_hidden_spike_rate': float(np.mean(spike_rates)) if spike_rates else 0.0,
        'termination': term_reason,
        'success_criteria': '8 simulated seconds, >=1.5m forward, abs final side <=0.5m, torso >=0.18m',
    }
    success = (data.time >= 8 - 0.001 and delta[0] >= 1.5 and abs(delta[1]) <= 0.5
               and min_height >= LOW_HEIGHT_THRESHOLD and term_reason == 'completed full duration')
    summary['pilot_success'] = bool(success) if duration_seconds >= 8 else None
    RESULT_DIR.mkdir(exist_ok=True, parents=True)
    out_name = f'closed_loop_cmd_{vx_command:.2f}.json'
    (RESULT_DIR / out_name).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n=== REAL SNN CLOSED-LOOP GO2 EVALUATION ===')
    print(json.dumps(summary, indent=2))
    print('Saved result:', RESULT_DIR / out_name)
    print('This is an SNN driving Go2; no ANN policy is consulted during rollout.')
    print('Any failure is a real learning signal, not a training-script error by itself.')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--command', type=float, default=0.5)
    p.add_argument('--seconds', type=float, default=8.0)
    p.add_argument('--viewer', action='store_true')
    args = p.parse_args()
    if not 0.0 <= args.command <= 1.0:
        p.error('command must be between 0 and 1 m/s for this pilot')
    if not 0.5 <= args.seconds <= 60.0:
        p.error('duration must be in [0.5,60] simulated seconds')
    evaluate(args.command, args.seconds, args.viewer)

if __name__ == '__main__':
    main()
