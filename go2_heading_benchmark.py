"""NeuroStride: paired ANN vs SNN heading-feedback experiment on flat Go2.

The teacher ANN weights are external (walk-these-ways). The V3 SNN is our
behavior-cloned spiking student. *_feedback variants add an EXTERNAL classical
heading supervisor; they are NOT newly trained SNNs.

Example (Windows, project root):
  .\\.venv\\Scripts\\python.exe go2_heading_benchmark.py --seeds 3101 --seconds 8
  .\\.venv\\Scripts\\python.exe go2_heading_benchmark.py --seeds 3101 3102 3103 --seconds 8
  .\\.venv\\Scripts\\python.exe go2_heading_benchmark.py --modes snn_feedback --seeds 3101 --viewer
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
    ACTION_SCALE, BASE_COMMAND, DECIMATION, DEFAULT_JOINT_POS,
    EXPECTED_ACTUATORS, HISTORY_LEN, KP, KD, LOW_HEIGHT_THRESHOLD,
    OBS_DIM, POLICY_DIR, build_obs, verify_checkpoints,
)

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results' / 'go2_heading_benchmark'
MODES = ('snn_zero', 'snn_feedback', 'ann_zero', 'ann_feedback')


def heading_rad(wxyz):
    """World yaw angle from MuJoCo floating-base quaternion (w,x,y,z)."""
    w, x, y, z = map(float, wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y*y + z*z))


def wrap_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def load_teacher():
    verify_checkpoints()   # official-release uploader SHA-256 values
    body = torch.jit.load(str(POLICY_DIR / 'body_latest.jit'), map_location='cpu')
    adap = torch.jit.load(str(POLICY_DIR / 'adaptation_module_latest.jit'), map_location='cpu')
    body.eval(); adap.eval()
    return body, adap


def run(model, mode, snn, teacher, speed, seed, seconds, gain, max_rate,
        viewer=False, record=True):
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home < 0:
        raise ValueError('Go2 model missing home keyframe')
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.normal(0.0, 0.012, size=12)
    data.qpos[2] += float(rng.uniform(-0.005, 0.005))
    mujoco.mj_forward(model, data)

    origin = data.qpos[:3].copy()
    yaw0 = heading_rad(data.qpos[3:7])
    cmd = BASE_COMMAND.copy()
    cmd[0] = speed
    history = deque((np.zeros(OBS_DIM, np.float32) for _ in range(HISTORY_LEN)),
                    maxlen=HISTORY_LEN)
    prev = np.zeros(12, np.float64)
    last = np.zeros(12, np.float64)
    target = DEFAULT_JOINT_POS.copy()
    phase = 0.0
    lower, upper = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    stop = 'completed requested duration'
    min_height = float(data.qpos[2]); max_y = 0.0; total_y = 0.0
    prev_xy = data.qpos[:2].copy(); path_m = 0.0
    max_heading = 0.0; mean_heading_accum = 0.0; sample_n = 0
    num_actions = 0; activity_sum = 0.0
    frames = {'qpos': [], 'qvel': [], 'sim_time': [], 'joint_torque': [], 'yaw_cmd': []}

    def snapshot():
        t = float(data.time)
        if frames['sim_time'] and t == frames['sim_time'][-1]:
            return
        frames['qpos'].append(data.qpos.copy())
        frames['qvel'].append(data.qvel.copy())
        frames['sim_time'].append(t)
        frames['joint_torque'].append(data.ctrl.copy())
        frames['yaw_cmd'].append(float(cmd[2]))

    if record: snapshot()
    mu_x = sd_x = mu_y = sd_y = None
    if mode.startswith('snn'):
        net, stats = snn
        mu_x, sd_x, mu_y, sd_y = stats
    elif teacher is None:
        raise RuntimeError('ANN teacher is not available')
    manager = (mujoco.viewer.launch_passive(model, data) if viewer
               else contextlib.nullcontext(None))
    with manager as view:
        for step in range(int(round(seconds / model.opt.timestep))):
            if viewer and not view.is_running():
                stop = 'viewer closed'; break
            if step % DECIMATION == 0:
                error = wrap_pi(heading_rad(data.qpos[3:7]) - yaw0)
                # In Go2's yaw convention, positive yaw command turns CCW.
                cmd[2] = float(np.clip(-gain * error, -max_rate, max_rate)) if mode.endswith('feedback') else 0.0
                phase = (phase + cmd[4] * model.opt.timestep * DECIMATION) % 1.0
                obs = build_obs(data, cmd, prev, last, phase)
                history.append(obs)
                hist = np.stack(history).astype(np.float32)
                with torch.inference_mode():
                    if mode.startswith('snn'):
                        feat = features_v3_online(hist)
                        x = np.clip((feat - mu_x) / sd_x, -5.0, 5.0).astype(np.float32)
                        out, spike_rate = net(torch.from_numpy(x[None]))
                        raw = out.squeeze(0).cpu().numpy() * sd_y + mu_y
                        activity_sum += float(spike_rate)
                    else:
                        seq = torch.from_numpy(hist.reshape(1, -1))
                        body, adapt = teacher
                        latent = adapt(seq)
                        raw = body(torch.cat([seq, latent], dim=1)).squeeze(0).cpu().numpy()
                if raw.shape != (12,) or not np.isfinite(raw).all():
                    stop = 'invalid policy output'; break
                if np.max(np.abs(raw)) > 12.0:
                    stop = 'action guard exceeded +/-12'; break
                action = np.clip(raw, -6.0, 6.0).astype(np.float64)
                prev, last = last.copy(), action.copy()
                target = DEFAULT_JOINT_POS + action * ACTION_SCALE
                num_actions += 1
            torques = KP * (target - data.qpos[7:19]) - KD * data.qvel[6:18]
            data.ctrl[:] = np.clip(torques, lower, upper)
            mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                stop = 'nonfinite MuJoCo state'; break
            min_height = min(min_height, float(data.qpos[2]))
            xy = data.qpos[:2].copy()
            path_m += float(np.linalg.norm(xy - prev_xy))
            total_y += abs(float(xy[1] - prev_xy[1]))
            max_y = max(max_y, abs(float(xy[1] - origin[1])))
            prev_xy = xy
            err = abs(math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - yaw0)))
            max_heading = max(max_heading, err)
            mean_heading_accum += err; sample_n += 1
            if step % DECIMATION == 0:
                if record: snapshot()
                if viewer and view.is_running():
                    view.cam.lookat[:] = data.qpos[:3]
                    view.sync()
                    time.sleep(model.opt.timestep * DECIMATION)
            if data.qpos[2] < LOW_HEIGHT_THRESHOLD:
                stop = 'crossed torso height 0.18m'; break
        if record: snapshot()
        if viewer and view.is_running():
            view.cam.lookat[:] = data.qpos[:3]
            view.sync()

    dx = float(data.qpos[0] - origin[0]); dy = float(data.qpos[1] - origin[1])
    duration = float(data.time)
    final_yaw = math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - yaw0))
    full = stop == 'completed requested duration' and duration >= seconds - .002
    row = {
        'mode': mode, 'seed': seed, 'command_forward_mps': speed,
        'requested_s': seconds, 'sim_s': round(duration, 4),
        'forward_m': round(dx, 5), 'sideways_m': round(dy, 5),
        'mean_forward_mps': round(dx / max(duration, 1e-9), 5),
        'minimum_torso_height_m': round(min_height, 5),
        'max_abs_sideways_m': round(max_y, 5),
        'cumulative_abs_lateral_m': round(total_y, 5),
        'forward_path_efficiency': round(dx / max(path_m, 1e-9), 5),
        'final_yaw_deg': round(final_yaw, 4),
        'max_abs_yaw_deg': round(max_heading, 4),
        'mean_abs_yaw_deg': round(mean_heading_accum / max(sample_n, 1), 4),
        'decisions': num_actions,
        'mean_hidden_spike_rate': round(activity_sum / max(num_actions, 1), 5) if mode.startswith('snn') else None,
        'terminated': stop,
        'passed_flat_8s': bool(full and dx >= 1.5 and abs(dy) <= .5 and min_height >= .18),
        'passed_straight_heading': bool(full and dx >= 1.5 and abs(dy) <= .25 and max_heading <= 15 and min_height >= .18),
        'external_heading_supervisor': bool(mode.endswith('feedback')),
        'policy_source': 'trained NeuroStride SNN V3' if mode.startswith('snn') else 'third-party pretrained ANN',
    }
    if record:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        dest = RESULT_DIR / f'{mode}_cmd{speed:.2f}_seed{seed}.npz'
        np.savez_compressed(dest,
            qpos=np.asarray(frames['qpos'], np.float64),
            qvel=np.asarray(frames['qvel'], np.float64),
            sim_time=np.asarray(frames['sim_time'], np.float64),
            joint_torque=np.asarray(frames['joint_torque'], np.float32),
            yaw_cmd=np.asarray(frames['yaw_cmd'], np.float32),
            summary_json=np.array(json.dumps(row)),
        )
        print('Saved full-state replay:', dest, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[3101])
    ap.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    ap.add_argument('--command', type=float, default=.45)
    ap.add_argument('--seconds', type=float, default=8.)
    ap.add_argument('--gain', type=float, default=.9, help='yaw correction rad/s per rad of heading error')
    ap.add_argument('--max-yaw-rate', type=float, default=.20, help='abs yaw rate command cap (rad/s)')
    ap.add_argument('--viewer', action='store_true', help='opens viewer for each selected mode (Windows local)')
    ap.add_argument('--no-record', action='store_true')
    a = ap.parse_args()
    if not 0 < a.seconds <= 60 or not 0 <= a.command <= 1.0:
        ap.error('seconds 0..60 and command 0..1 expected')
    if a.gain < 0 or not 0 <= a.max_yaw_rate <= .5:
        ap.error('gain>=0 and max-yaw-rate 0..0.5 expected')
    if len(a.seeds) > 20 or len(a.modes) > 4 or len(set(a.seeds)) != len(a.seeds):
        ap.error('maximum 20 distinct seeds, 4 modes')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    snn = load_actor('v3') if any(m.startswith('snn') for m in a.modes) else None
    teacher = load_teacher() if any(m.startswith('ann') for m in a.modes) else None
    mj = mm.load('unitree_go2')
    names = [mj.actuator(i).name for i in range(mj.nu)]
    if names != EXPECTED_ACTUATORS or abs(mj.opt.timestep - .002) > 1e-9:
        raise RuntimeError(f'Unexpected Go2 actuator configuration: {names}, dt={mj.opt.timestep}')
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        for mode in a.modes:
            print(f'\nRunning {mode} seed={seed}, forward={a.command}m/s, horizon={a.seconds}s', flush=True)
            r = run(mj, mode, snn, teacher, a.command, seed, a.seconds,
                    a.gain, a.max_yaw_rate, viewer=a.viewer, record=not a.no_record)
            print(json.dumps(r, indent=2), flush=True)
            rows.append(r)
    with (RESULT_DIR/'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    aggr = {}
    for mode in a.modes:
        sample = [r for r in rows if r['mode'] == mode]
        aggr[mode] = {
            'episodes': len(sample), 'passed_flat_8s': sum(int(r['passed_flat_8s']) for r in sample),
            'passed_straight_heading': sum(int(r['passed_straight_heading']) for r in sample),
            'mean_forward_m': round(float(np.mean([r['forward_m'] for r in sample])), 5),
            'mean_sim_s': round(float(np.mean([r['sim_s'] for r in sample])), 5),
            'mean_abs_final_yaw_deg': round(float(np.mean([abs(r['final_yaw_deg']) for r in sample])), 4),
            'mean_max_abs_yaw_deg': round(float(np.mean([r['max_abs_yaw_deg'] for r in sample])), 4),
            'mean_abs_sideways_m': round(float(np.mean([abs(r['sideways_m']) for r in sample])), 5),
        }
    result = {
        'conditions': {'command_mps': a.command, 'requested_seconds': a.seconds,
                       'seeds': a.seeds, 'gain': a.gain, 'max_yaw_rate': a.max_yaw_rate},
        'definitions': {'straight': '8s + forward>=1.5m + abs final side<=0.25m + max abs yaw<=15deg + min z>=.18m',
                        'feedback': 'EXTERNAL classical yaw-rate feedback: commanded yaw=-gain*heading_error, clipped'},
        'results': aggr,
    }
    (RESULT_DIR/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\n=== HEADING BENCHMARK SUMMARY ===\n', json.dumps(result, indent=2))
    print('Files:', RESULT_DIR/'episodes.csv', RESULT_DIR/'summary.json')

if __name__ == '__main__':
    main()
