r"""NeuroStride Go2 V5: tiny reversible stance-bias ablation for upright walking.

This is an engineering diagnostic, NOT SNN training. The V5 checkpoint and
its external heading supervisor remain frozen. A symmetrical fore/aft thigh
angle offset is applied AFTER the neural action to test body pitch.

Windows PowerShell:
  .\.venv\Scripts\python.exe go2_v5_upright_sweep.py --seeds 5301 --offsets 0 -0.02 0.02
  .\.venv\Scripts\python.exe go2_v5_upright_sweep.py --seeds 5301 5302 5303 --offsets 0 -0.02 0.02
  .\.venv\Scripts\python.exe go2_v5_upright_sweep.py --plot-only --seed 5301
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

from go2_v5_steering import WEIGHTS, make_model
from go2_heading_benchmark import heading_rad, wrap_pi
from train_go2_snn_v3 import features_v3_online
from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, DECIMATION, DEFAULT_JOINT_POS,
    EXPECTED_ACTUATORS, HISTORY_LEN, KP, KD, LOW_HEIGHT_THRESHOLD,
    OBS_DIM, build_obs,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "go2_snn_v5" / "upright_sweep"


def roll_pitch_rad(q):
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(np.clip(2 * (w*y - z*x), -1, 1))
    return roll, pitch


def replay_path(seed, bias, command):
    return OUT / f"v5_pitchbias{bias:+.3f}_cmd{command:.2f}_seed{seed}.npz"


def run(model, net, stats, seed, bias, speed, seconds, show_viewer=False):
    """Matched MuJoCo run; modifying exactly 4 thigh targets only."""
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home < 0:
        raise RuntimeError('Go2 home keyframe missing')
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.normal(0, 0.012, 12)
    data.qpos[2] += float(rng.uniform(-0.005, 0.005))
    mujoco.mj_forward(model, data)

    names = [model.actuator(i).name for i in range(model.nu)]
    expected = list(EXPECTED_ACTUATORS)
    if names != expected:
        raise RuntimeError(f'Actuator mismatch: {names}')
    # Positive bias: +bias to the two FRONT thighs; -bias to REAR thighs.
    # We do NOT presume which sign reduces pitch, hence symmetric sweep.
    thigh_bias = np.zeros(12, np.float64)
    thigh_bias[1] = thigh_bias[4] = float(bias)
    thigh_bias[7] = thigh_bias[10] = -float(bias)

    origin = data.qpos[:3].copy()
    yaw0 = heading_rad(data.qpos[3:7])
    command = BASE_COMMAND.copy()
    command[0] = speed
    past = deque((np.zeros(OBS_DIM, np.float32) for _ in range(HISTORY_LEN)), maxlen=HISTORY_LEN)
    prev = np.zeros(12, np.float64)
    last = np.zeros(12, np.float64)
    target = DEFAULT_JOINT_POS.copy()
    phase = 0.
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    mx, sx, my, sy = stats
    n_decisions = 0
    min_height = float(data.qpos[2])
    clipped_steps = 0
    stop = 'completed requested duration'
    records = {'qpos': [], 'qvel': [], 'sim_time': [], 'joint_torque': [], 'yaw_cmd': []}
    rollvals, pitchvals, yawvals = [], [], []
    max_yaw = 0.

    def snapshot():
        stamp = float(data.time)
        if records['sim_time'] and stamp == records['sim_time'][-1]:
            return
        records['qpos'].append(data.qpos.copy())
        records['qvel'].append(data.qvel.copy())
        records['sim_time'].append(stamp)
        records['joint_torque'].append(data.ctrl.copy())
        records['yaw_cmd'].append(float(command[2]))

    snapshot()
    cm = mujoco.viewer.launch_passive(model, data) if show_viewer else contextlib.nullcontext(None)
    with cm as view:
        for step in range(int(round(seconds / model.opt.timestep))):
            if show_viewer and not view.is_running():
                stop = 'viewer closed'
                break
            if step % DECIMATION == 0:
                yaw_err = wrap_pi(heading_rad(data.qpos[3:7]) - yaw0)
                command[2] = float(np.clip(-0.9 * yaw_err, -0.2, 0.2))
                phase = (phase + command[4] * model.opt.timestep * DECIMATION) % 1.0
                obs = build_obs(data, command, prev, last, phase)
                past.append(obs)
                feat = features_v3_online(np.stack(past).astype(np.float32))
                x = np.clip((feat - mx) / sx, -5., 5.).astype(np.float32)
                with torch.inference_mode():
                    pred, _ = net(torch.from_numpy(x[None]))
                raw = pred.squeeze(0).cpu().numpy() * sy + my
                if raw.shape != (12,) or not np.isfinite(raw).all():
                    stop = 'invalid network command'
                    break
                if np.max(np.abs(raw)) > 12:
                    stop = 'network action safety guard exceeded'
                    break
                action = np.clip(raw, -6., 6.).astype(np.float64)
                prev, last = last.copy(), action.copy()
                # Identical neural command to V5 baseline; postprocessing
                # applies only a small fore/aft thigh position reference bias.
                target = DEFAULT_JOINT_POS + action * ACTION_SCALE + thigh_bias
                n_decisions += 1

            tau_raw = KP * (target - data.qpos[7:19]) - KD * data.qvel[6:18]
            clipped_steps += int(np.any((tau_raw < low) | (tau_raw > high)))
            data.ctrl[:] = np.clip(tau_raw, low, high)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop = 'nonfinite simulator state'
                break
            min_height = min(min_height, float(data.qpos[2]))
            r, p = roll_pitch_rad(data.qpos[3:7])
            yaw = math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - yaw0))
            rollvals.append(math.degrees(r))
            pitchvals.append(math.degrees(p))
            yawvals.append(yaw)
            max_yaw = max(max_yaw, abs(yaw))
            if step % DECIMATION == 0:
                snapshot()
                if show_viewer and view.is_running():
                    view.cam.lookat[:] = data.qpos[:3]
                    view.sync()
                    time.sleep(model.opt.timestep * DECIMATION)
            if data.qpos[2] < LOW_HEIGHT_THRESHOLD:
                stop = 'torso below 0.18 m'
                break
        snapshot()
        if show_viewer and view.is_running():
            view.sync()

    dx, dy = (data.qpos[:2] - origin[:2]).tolist()
    t = float(data.time)
    rollvals = np.array(rollvals)
    pitchvals = np.array(pitchvals)
    success = (stop == 'completed requested duration' and t >= seconds - .002 and
               dx >= 1.5 and abs(dy) <= .25 and max_yaw <= 15 and min_height >= .18)
    row = dict(
        seed=int(seed), pitch_bias_rad=float(bias), command_mps=float(speed),
        simulated_s=round(t, 4), forward_m=round(float(dx), 5),
        final_sideways_m=round(float(dy), 5),
        min_height_m=round(min_height, 5),
        roll_rms_deg=round(float(np.sqrt(np.mean(rollvals**2))), 4) if len(rollvals) else None,
        pitch_rms_deg=round(float(np.sqrt(np.mean(pitchvals**2))), 4) if len(pitchvals) else None,
        mean_pitch_deg=round(float(np.mean(pitchvals)), 4) if len(pitchvals) else None,
        combined_tilt_rms_deg=round(float(np.sqrt(np.mean(rollvals**2+pitchvals**2))), 4) if len(rollvals) else None,
        max_abs_yaw_deg=round(max_yaw, 4), final_yaw_deg=round(yawvals[-1], 4) if yawvals else None,
        fraction_motor_clipped=round(clipped_steps/max(len(rollvals), 1), 5),
        decisions=n_decisions, terminated=stop,
        passes_straight_task=bool(success),
    )
    path = replay_path(seed, bias, speed)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
        qpos=np.stack(records['qpos']), qvel=np.stack(records['qvel']),
        sim_time=np.array(records['sim_time']),
        joint_torque=np.stack(records['joint_torque']).astype(np.float32),
        yaw_cmd=np.array(records['yaw_cmd'], np.float32),
        summary_json=np.array(json.dumps(row)))
    return row


def make_plot(seed, speed, biases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    found = 0
    for bias in biases:
        path = replay_path(seed, bias, speed)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as arr:
            q = arr['qpos']; times = arr['sim_time']
        rr, pp, yy = [], [], []
        for quat in q[:,3:7]:
            r,p=roll_pitch_rad(quat);rr.append(math.degrees(r));pp.append(math.degrees(p))
            yy.append(heading_rad(quat))
        yaw = (np.unwrap(yy)-float(yy[0]))*180/math.pi
        label=f'bias {bias:+.3f} rad ({times[-1]:.2f}s)'
        axes[0,0].plot(q[:,0]-q[0,0],q[:,1]-q[0,1],label=label)
        axes[0,1].plot(times,pp,label=label)
        axes[1,0].plot(times,rr,label=label)
        axes[1,1].plot(times,yaw,label=label)
        found+=1
    if not found:raise RuntimeError('No matching replay files to plot')
    labels=[('X/Y trajectory','Forward X (m)','Sideways Y (m)'),
            ('Torso pitch','Seconds','Pitch (degrees)'),
            ('Torso roll','Seconds','Roll (degrees)'),
            ('Heading drift','Seconds','Yaw error (degrees)')]
    for a,(title,x,y) in zip(axes.flat,labels):
        a.set(title=title,xlabel=x,ylabel=y)
        a.grid(alpha=.25);a.legend(fontsize='small')
    axes[1,1].axhline(-15,c='gray',ls=':')
    axes[1,1].axhline(15,c='gray',ls=':')
    dest=OUT/f'v5_posture_seed{seed}.png'
    fig.savefig(dest,dpi=160)
    plt.close(fig)
    print('Saved posture graph:',dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[5301])
    p.add_argument('--offsets',type=float,nargs='+',default=[0.,-.02,.02])
    p.add_argument('--command',type=float,default=.45)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--seed',type=int,default=5301,help='Plot seed for --plot-only')
    args=p.parse_args()
    if not(1<=args.seconds<=20 and 0<=args.command<=1):p.error('Unsupported speed or duration')
    if any(abs(b)>.05 for b in args.offsets):p.error('Safety bound: offsets must be within +/-0.05 rad')
    if len(args.seeds)>10 or len(args.offsets)>7:p.error('Please use <=10 seeds and <=7 offsets')
    if args.plot_only:
        make_plot(args.seed,args.command,args.offsets)
        return
    if not WEIGHTS.is_file():raise FileNotFoundError(f'Missing trained V5 checkpoint: {WEIGHTS}')
    checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    if not str(checkpoint.get('source','')).startswith('V5 learned'):
        print('WARNING: saved V5 checkpoint indicates V4 fallback, not trained V5')
    net,stats=make_model(checkpoint)
    net.eval()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    model=mm.load('unitree_go2')
    if abs(model.opt.timestep-.002)>1e-8 or model.nu!=12:
        raise ValueError('Go2 scene does not match 500Hz / 12-motor spec')
    rows=[]
    for seed in args.seeds:
        for bias in args.offsets:
            print(f'V5 upright sweep seed={seed}, front/rear thigh bias={bias:+.3f}rad',flush=True)
            row=run(model,net,stats,seed,bias,args.command,args.seconds,show_viewer=args.viewer)
            print(f"  {row['simulated_s']:.2f}s X={row['forward_m']:+.3f}m "
                  f"pitch RMS={row['pitch_rms_deg']:.2f}deg "
                  f"roll RMS={row['roll_rms_deg']:.2f}deg "
                  f"max yaw={row['max_abs_yaw_deg']:.2f}deg "
                  f"straight={row['passes_straight_task']}",flush=True)
            rows.append(row)
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'episodes.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    agg={}
    for bias in args.offsets:
        rs=[r for r in rows if r['pitch_bias_rad']==bias]
        agg[f'bias_{bias:+.3f}']={
            'episodes':len(rs),
            'strict_straight_passes':int(sum(r['passes_straight_task'] for r in rs)),
            'mean_forward_m':float(np.mean([r['forward_m'] for r in rs])),
            'mean_roll_rms_deg':float(np.mean([r['roll_rms_deg'] for r in rs])),
            'mean_pitch_rms_deg':float(np.mean([r['pitch_rms_deg'] for r in rs])),
            'mean_max_abs_yaw_deg':float(np.mean([r['max_abs_yaw_deg'] for r in rs])),
        }
    (OUT/'summary.json').write_text(json.dumps(agg,indent=2),encoding='utf8')
    print('\n=== V5 UPRIGHT / STANCE BIAS ABLATION ===')
    print(json.dumps(agg,indent=2))
    print('CSV:',OUT/'episodes.csv')
    for seed in args.seeds:
        make_plot(seed,args.command,args.offsets)

if __name__=='__main__':
    main()
