"""Visual training-preview, recording and EXACT state-sequence playback for Go2.

This uses the identical Menagerie Go2 physics/observation/action conventions
used in go2_collect_teacher.py and go2_snn_closed_loop.py.

Examples from NeuroStride_Starter:
  python go2_visual_replay.py record --controller snn --viewer
  python go2_visual_replay.py record --controller teacher --viewer --seconds 10
  python go2_visual_replay.py record --controller snn --checkpoint results/go2_snn_v2/go2_snn_v2_best.pt --viewer
  python go2_visual_replay.py play --file results/go2_replays/your_recording.npz

"record" runs a NEW rollout, stores simulator qpos/qvel at 50fps and shows a
live robot when --viewer is used. "play" reproduces those saved qpos/qvel
frames exactly in a viewer (no policy calls, no physics stepping). Without
recorded qpos/qvel historical DAgger episodes cannot be reconstructed exactly.
"""
from __future__ import annotations

import argparse
import contextlib
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
    OBS_DIM, POLICY_DIR, build_obs, verify_checkpoints,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'go2_replays'
DEFAULT_SNN = ROOT/'results'/'go2_snn_v1'/'go2_snn_v1_best.pt'


def load_go2():
    model = mm.load('unitree_go2')
    assert model.nu == 12, f'Expected 12 Go2 actuators, received {model.nu}'
    actual = [model.actuator(i).name for i in range(model.nu)]
    if actual != EXPECTED_ACTUATORS:
        raise ValueError(f'Incorrect Go2 actuator mapping: {actual}')
    if abs(model.opt.timestep - .002) > 1e-8:
        raise ValueError(f'Unexpected Go2 sim timestep: {model.opt.timestep}')
    data = mujoco.MjData(model)
    return model, data


def load_snn(checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'SNN checkpoint missing: {checkpoint_path}')
    c = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = c['config']
    snn = Go2SpikingActor(
        feature_dim=int(config['feature_dim']),
        population=int(config['population']),
        hidden=int(config['hidden']),
        steps=int(config['steps']),
    )
    snn.load_state_dict(c['model_state'])
    snn.eval()
    mu_x = c['feature_mean'].numpy()
    std_x = c['feature_std'].numpy()
    mu_y = c['target_mean'].numpy()
    std_y = c['target_std'].numpy()
    return snn, mu_x, std_x, mu_y, std_y, c


def show_camera(viewer, data):
    # Keep Go2 in frame while it walks. User may still rotate the camera.
    viewer.cam.lookat[:] = np.asarray(data.qpos[:3], dtype=float)
    viewer.sync()


def record(controller, checkpoint, command_speed, seconds, viewer_enabled, save,
           run_label=None):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, data = load_go2()
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key < 0:
        raise ValueError('Go2 home keyframe not available')
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[7:19] = DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    mujoco.mj_forward(model, data)
    start_xyz = data.qpos[:3].copy()
    command = BASE_COMMAND.copy()
    command[0] = command_speed
    history = deque((np.zeros(OBS_DIM, dtype=np.float32) for _ in range(HISTORY_LEN)),
                    maxlen=HISTORY_LEN)
    prev = np.zeros(12, dtype=np.float64)
    last = np.zeros(12, dtype=np.float64)
    targets = DEFAULT_JOINT_POS.copy()
    phase = 0.
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    spikes = []
    n_inferences = 0

    if controller == 'teacher':
        verify_checkpoints()
        body = torch.jit.load(str(POLICY_DIR/'body_latest.jit'), map_location='cpu')
        adapt = torch.jit.load(str(POLICY_DIR/'adaptation_module_latest.jit'), map_location='cpu')
        body.eval()
        adapt.eval()
        desc = 'pretrained external ANN teacher, NOT NeuroStride SNN'
    else:
        path = Path(checkpoint) if checkpoint else DEFAULT_SNN
        if not path.is_absolute():
            path = ROOT/path
        snn, mu_x, sd_x, mu_y, sd_y, c = load_snn(path)
        desc = f'SNN checkpoint {path.name} (epoch {c["epoch"]}), NO ANN loaded'

    samples = {'qpos': [], 'qvel': [], 'sim_time': [], 'joint_torque': []}
    def snapshot():
        samples['qpos'].append(data.qpos.copy())
        samples['qvel'].append(data.qvel.copy())
        samples['sim_time'].append(float(data.time))
        samples['joint_torque'].append(data.ctrl.copy())
    snapshot()

    stop = 'completed full duration'
    zmin = float(data.qpos[2])
    y0 = float(data.qpos[1])
    last_y = y0
    cum_side = 0.
    max_y = 0.
    steps = int(round(seconds/model.opt.timestep))
    print(f'Go2 LIVE ROLLOUT: {desc}; command={command_speed:.2f}m/s, target {seconds:.2f}s')
    print('Viewer:', viewer_enabled, '| recording:', save, '| not training weight updates')
    cm = mujoco.viewer.launch_passive(model, data) if viewer_enabled else contextlib.nullcontext(None)
    with cm as viewer:
        for step in range(steps):
            if viewer_enabled and not viewer.is_running():
                stop = 'viewer closed'
                break
            if step % DECIMATION == 0:
                phase = (phase+command[4]*model.opt.timestep*DECIMATION)%1.
                obs = build_obs(data, command, prev, last, phase)
                history.append(obs)
                with torch.inference_mode():
                    if controller == 'teacher':
                        h = torch.from_numpy(np.concatenate(history).astype(np.float32)[None])
                        latent = adapt(h)
                        raw = body(torch.cat([h, latent], dim=1)).squeeze(0).cpu().numpy()
                    else:
                        feat = np.concatenate([obs,np.mean(np.stack(history), axis=0)])
                        x = np.clip((feat-mu_x)/sd_x, -5, 5).astype(np.float32)
                        out, rate = snn(torch.from_numpy(x[None]))
                        raw = out.squeeze(0).cpu().numpy()*sd_y+mu_y
                        spikes.append(float(rate))
                if raw.shape!=(12,) or not np.isfinite(raw).all():
                    stop='invalid neural action'
                    break
                if controller == 'snn' and np.abs(raw).max()>8:
                    stop='SNN exceeded raw-action guard +/-8'
                    break
                action = np.clip(raw,-6.,6.)
                prev = last.copy()
                last = action.copy()
                targets = DEFAULT_JOINT_POS + last*ACTION_SCALE
                n_inferences+=1

            data.ctrl[:] = np.clip(KP*(targets-data.qpos[7:19])-KD*data.qvel[6:18],low,high)
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop='nonfinite MuJoCo state'
                break
            zmin = min(zmin, float(data.qpos[2]))
            y = float(data.qpos[1])
            max_y = max(max_y,abs(y-y0))
            cum_side+=abs(y-last_y)
            last_y=y
            if step % DECIMATION == 0:
                snapshot()
                if viewer_enabled:
                    show_camera(viewer,data)
                    # Replay live near real-time; on slower PCs a viewer may
                    # run slower. Simulation time still comes from MuJoCo.
                    time.sleep(model.opt.timestep*DECIMATION)
            if zmin<LOW_HEIGHT_THRESHOLD:
                stop=f'crossed low-height threshold ({LOW_HEIGHT_THRESHOLD:.2f}m)'
                break
        # Save final state even when termination occurs between policy frames.
        snapshot()
        if viewer_enabled and viewer.is_running():
            show_camera(viewer,data)
    delta = data.qpos[:3]-start_xyz
    summary = {
        'controller': 'ANN teacher' if controller=='teacher' else 'NeuroStride SNN',
        'checkpoint': None if controller=='teacher' else str(path),
        'command_forward_mps': float(command_speed),
        'sim_seconds': float(data.time),
        'forward_m': float(delta[0]),
        'sideways_m': float(delta[1]),
        'min_height_m': float(zmin),
        'final_height_m': float(data.qpos[2]),
        'max_abs_sideways_m': float(max_y),
        'cumulative_abs_sideways_m': float(cum_side),
        'neural_policy_decisions': n_inferences,
        'mean_hidden_spike_rate': float(np.mean(spikes)) if spikes else None,
        'termination': stop,
        'passed_requested_duration': stop=='completed full duration' and data.time>=seconds-.001,
    }
    print('\n=== GO2 LIVE ROLLOUT RESULT ===')
    print(json.dumps(summary,indent=2))
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        safe_label = ''.join(ch for ch in (run_label or 'trial') if ch.isalnum() or ch in '-_')[:30]
        name = f'{controller}_{safe_label}_cmd{command_speed:.2f}_{int(round(data.time*1000))}ms.npz'
        dest=OUT/name
        # Exact joint state samples, no policy weights in recording.
        np.savez_compressed(dest,
            qpos=np.stack(samples['qpos']).astype(np.float64),
            qvel=np.stack(samples['qvel']).astype(np.float64),
            sim_time=np.array(samples['sim_time'],dtype=np.float64),
            joint_torque=np.stack(samples['joint_torque']).astype(np.float32),
            summary_json=np.array(json.dumps(summary)),
            )
        print('RECORDED:',dest)
        (OUT/(dest.stem+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
        print('Replay using: python go2_visual_replay.py play --file "'+str(dest)+'"')
    return summary


def play(trajectory, speed, loop=False):
    if not trajectory.is_file():
        raise FileNotFoundError(f'No saved episode recording: {trajectory}')
    with np.load(trajectory, allow_pickle=False) as z:
        qpos=np.asarray(z['qpos'],dtype=np.float64)
        qvel=np.asarray(z['qvel'],dtype=np.float64)
        times=np.asarray(z['sim_time'],dtype=np.float64)
        print('Saved episode metadata:',str(z['summary_json'].item())[:2500])
    model,data=load_go2()
    if qpos.shape[1:] != (model.nq,) or qvel.shape[1:] != (model.nv,) or len(qpos)!=len(qvel) or len(times)!=len(qpos):
        raise ValueError(f'Recording dimensions mismatch expected model nq={model.nq}, nv={model.nv}')
    with mujoco.viewer.launch_passive(model,data) as viewer:
        print(f'Playing exact recorded states: {len(times)} frames, speed {speed}x, loop={loop}')
        while viewer.is_running():
            for i,t in enumerate(times):
                if not viewer.is_running():
                    break
                data.qpos[:]=qpos[i]
                data.qvel[:]=qvel[i]
                data.time=float(t)
                mujoco.mj_forward(model,data)
                show_camera(viewer,data)
                if i+1<len(times):
                    elapsed=float(times[i+1]-t)
                    time.sleep(max(0,min(elapsed/speed,0.2)))
            if not loop:
                break
        print('Playback finished. This is a recorded state sequence, NOT a new AI evaluation.')


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='mode',required=True)
    rec=sub.add_parser('record',help='Simulate ANN or SNN and optionally save exact state trajectory')
    rec.add_argument('--controller',choices=['snn','teacher'],default='snn')
    rec.add_argument('--checkpoint',default=None,help='Defaults to SNN V1; use V2 .pt for V2')
    rec.add_argument('--command',type=float,default=0.5)
    rec.add_argument('--seconds',type=float,default=8.0)
    rec.add_argument('--viewer',action='store_true')
    rec.add_argument('--no-save',action='store_true')
    rec.add_argument('--label',default=None)
    rep=sub.add_parser('play',help='Play back saved MuJoCo states WITHOUT running any controller')
    rep.add_argument('--file',required=True)
    rep.add_argument('--speed',type=float,default=1.0)
    rep.add_argument('--loop',action='store_true',help='Replay continuously until viewer closes')
    args=p.parse_args()
    if args.mode=='record':
        if not (0<=args.command<=1.0 and .2<=args.seconds<=120.):
            p.error('command must be 0..1 and seconds .2..120')
        record(args.controller,args.checkpoint,args.command,args.seconds,args.viewer,not args.no_save,args.label)
    else:
        if not .1<=args.speed<=4.0:
            p.error('speed must be 0.1..4.0')
        file=Path(args.file)
        if not file.is_absolute():
            file=ROOT/file
        play(file,args.speed,args.loop)


if __name__=='__main__':
    main()
