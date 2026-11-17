"""Paired, independent 8-second GO2 SNN V2-vs-V3 closed-loop evaluation.

No ANN teacher is loaded, nor used in the robot's control loop.
Both policies start from the identical per-seed randomized pose and receive the
identical command. Each run is an actual closed-loop MuJoCo simulation.

Examples (Bash, Windows PowerShell; from NeuroStride_Starter root):
    python go2_compare_v3.py --commands 0.45 --seeds 3101 3102 3103 --seconds 8
    python go2_compare_v3.py --commands 0.45 --seeds 3101 --viewer --record

Output CSV and exact full-state .npz recordings are written under
results/go2_snn_v3/evaluation. To replay a saved recording:
    python go2_visual_replay.py play --file results/go2_snn_v3/evaluation/<filename>.npz
Or render an MP4 in headless Codespaces:
    MUJOCO_GL=osmesa python go2_render_mp4.py --file <filename.npz>
This file is simulation evaluation, NOT offline imitation score or training.
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
from train_go2_snn_v3 import features_v3_online
from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, DECIMATION, DEFAULT_JOINT_POS,
    EXPECTED_ACTUATORS, HISTORY_LEN, KP, KD, LOW_HEIGHT_THRESHOLD,
    OBS_DIM, build_obs,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/go2_snn_v3/evaluation'
MODELS = {
    'v2': ROOT / 'results/go2_snn_v2/go2_snn_v2_best.pt',
    'v3': ROOT / 'results/go2_snn_v3/go2_snn_v3_best.pt',
}


def load_actor(version):
    path = MODELS[version]
    if not path.is_file():
        raise FileNotFoundError(f'Missing {version} policy: {path}')
    c = torch.load(path, map_location='cpu', weights_only=True)
    cfg = c['config']
    required_dim = 140 if version == 'v2' else 350
    if int(cfg['feature_dim']) != required_dim:
        raise ValueError(f'Expected {version} {required_dim}-dim policy, got {cfg["feature_dim"]}')
    model = Go2SpikingActor(feature_dim=required_dim,
                            population=int(cfg['population']),
                            hidden=int(cfg['hidden']),steps=int(cfg['steps']))
    model.load_state_dict(c['model_state'])
    model.eval()
    stats = [np.asarray(c[k].cpu().numpy(), dtype=np.float32) for k in
             ['feature_mean','feature_std','target_mean','target_std']]
    if not all(np.isfinite(q).all() for q in stats):
        raise ValueError('Nonfinite checkpoint normalization')
    return model,stats


def eval_once(version, model, stats, mj_model, cmd_speed, seed, seconds,
              viewer=False, record=True):
    data = mujoco.MjData(mj_model)
    home=mujoco.mj_name2id(mj_model,mujoco.mjtObj.mjOBJ_KEY,'home')
    if home<0: raise ValueError('Go2 scene has no home keyframe')
    mujoco.mj_resetDataKeyframe(mj_model,data,home)
    data.qpos[7:19]=DEFAULT_JOINT_POS
    data.qpos[2]=0.30
    # Paired disturbance: same deterministic seed/starting state for V2/V3.
    rng=np.random.default_rng(seed)
    data.qpos[7:19]+=rng.normal(0.0,0.012,size=12)
    data.qpos[2]+=float(rng.uniform(-0.005,0.005))
    mujoco.mj_forward(mj_model,data)
    origin=data.qpos[:3].copy()
    zmin=float(data.qpos[2]); max_side=0.;cum_side=0.;last_y=float(data.qpos[1])
    command=BASE_COMMAND.copy();command[0]=cmd_speed
    hist=deque((np.zeros(OBS_DIM,np.float32) for _ in range(HISTORY_LEN)),
               maxlen=HISTORY_LEN)
    prev=np.zeros(12,np.float64);last=np.zeros(12,np.float64)
    targets=DEFAULT_JOINT_POS.copy();phase=0.
    lo=mj_model.actuator_ctrlrange[:,0];hi=mj_model.actuator_ctrlrange[:,1]
    mu_x,sd_x,mu_y,sd_y=stats
    steps=int(round(seconds/mj_model.opt.timestep))
    stop='completed requested duration'
    spike_rates=[];decision_count=0
    frames={'qpos':[],'qvel':[],'sim_time':[],'joint_torque':[]}
    def snapshot():
        if frames['sim_time'] and float(data.time)==frames['sim_time'][-1]: return
        frames['qpos'].append(data.qpos.copy());frames['qvel'].append(data.qvel.copy())
        frames['sim_time'].append(float(data.time));frames['joint_torque'].append(data.ctrl.copy())
    if record:snapshot()
    manager = mujoco.viewer.launch_passive(mj_model,data) if viewer else contextlib.nullcontext(None)
    with manager as view:
        for step in range(steps):
            if viewer and not view.is_running():
                stop='viewer closed';break
            if step % DECIMATION == 0:
                phase = (phase+command[4]*mj_model.opt.timestep*DECIMATION)%1.
                obs=build_obs(data,command,prev,last,phase)
                hist.append(obs)
                stack=np.stack(hist).astype(np.float32)
                if version=='v3':
                    features=features_v3_online(stack)
                else:
                    features=np.concatenate([obs,stack.mean(axis=0)]).astype(np.float32)
                x=np.clip((features-mu_x)/sd_x,-5,5).astype(np.float32)
                with torch.inference_mode():
                    out,activity=model(torch.from_numpy(x[None]))
                raw=(out.squeeze(0).cpu().numpy()*sd_y+mu_y)
                if raw.shape!=(12,) or not np.isfinite(raw).all():
                    stop='invalid SNN action';break
                if np.max(np.abs(raw))>8.0:
                    stop='raw action exceeded safety guard +/-8';break
                action=np.clip(raw,-6.,6.).astype(np.float64)
                prev=last.copy();last=action.copy()
                targets=DEFAULT_JOINT_POS+action*ACTION_SCALE
                decision_count+=1
                spike_rates.append(float(activity))
            data.ctrl[:]=np.clip(KP*(targets-data.qpos[7:19])-KD*data.qvel[6:18],lo,hi)
            mujoco.mj_step(mj_model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop='nonfinite simulator state';break
            zmin=min(zmin,float(data.qpos[2]))
            delta_y=float(data.qpos[1]-origin[1])
            max_side=max(max_side,abs(delta_y))
            cum_side+=abs(float(data.qpos[1])-last_y)
            last_y=float(data.qpos[1])
            if step % DECIMATION == 0:
                if record:snapshot()
                if viewer and view.is_running():
                    view.cam.lookat[:]=data.qpos[:3]
                    view.sync()
                    time.sleep(mj_model.opt.timestep*DECIMATION)
            if zmin<LOW_HEIGHT_THRESHOLD:
                stop='crossed 0.18m torso height threshold';break
        if record:snapshot()
        if viewer and view.is_running():
            view.cam.lookat[:]=data.qpos[:3];view.sync()
    dx=float(data.qpos[0]-origin[0]);dy=float(data.qpos[1]-origin[1]);dur=float(data.time)
    completed=(stop=='completed requested duration' and dur>=seconds-.002)
    passed=completed and dx>=1.5 and abs(dy)<=0.5 and zmin>=LOW_HEIGHT_THRESHOLD
    row={
        'controller':version.upper(), 'command_mps':cmd_speed,'seed':seed,
        'sim_seconds':round(dur,4), 'forward_m':round(dx,5),
        'sideways_m':round(dy,5),'mean_forward_mps':round(dx/dur,5) if dur>0 else 0.,
        'minimum_torso_height_m':round(zmin,5),
        'max_abs_sideways_m':round(max_side,5),
        'cumulative_abs_sideways_m':round(cum_side,5),
        'snn_decisions':decision_count,
        'hidden_spike_rate':round(float(np.mean(spike_rates)),5) if spike_rates else 0,
        'termination':stop,'pass_8s_task':bool(passed),
    }
    if record:
        OUT.mkdir(parents=True,exist_ok=True)
        path=OUT/f'{version}_cmd{cmd_speed:.2f}_seed{seed}.npz'
        summary={**row,'source':'independent SNN closed-loop MuJoCo recording; no ANN teacher loaded'}
        np.savez_compressed(path,
            qpos=np.stack(frames['qpos']).astype(np.float64),
            qvel=np.stack(frames['qvel']).astype(np.float64),
            sim_time=np.array(frames['sim_time'],dtype=np.float64),
            joint_torque=np.stack(frames['joint_torque']).astype(np.float32),
            summary_json=np.array(json.dumps(summary)),
        )
        print('Saved exact replay:',path)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commands',type=float,nargs='+',default=[0.45],
                        help='Forward commands in m/s, default 0.45')
    parser.add_argument('--seeds',type=int,nargs='+',default=[3101,3102,3103])
    parser.add_argument('--seconds',type=float,default=8.)
    parser.add_argument('--viewer',action='store_true',help='Show only V3 runs in desktop MuJoCo viewer')
    parser.add_argument('--no-record',action='store_true',help='Skip qpos/qvel replay files')
    args=parser.parse_args()
    if any(not 0.<=c<=1.0 for c in args.commands):parser.error('commands must be in [0,1]')
    if not 1.<=args.seconds<=120.:parser.error('seconds must be in [1,120]')
    if len(args.seeds)>20 or len(args.commands)>10:parser.error('limit one run to <=20 seeds and <=10 commands')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    models={v:load_actor(v) for v in ['v2','v3']}
    mj_model=mm.load('unitree_go2')
    actual=[mj_model.actuator(i).name for i in range(mj_model.nu)]
    if actual!=EXPECTED_ACTUATORS or abs(mj_model.opt.timestep-.002)>1e-8:
        raise ValueError(f'Unexpected MuJoCo Go2 model actuator mapping or timestep {actual}')
    rows=[]
    for command in args.commands:
        for seed in args.seeds:
            for ver in ['v2','v3']:
                net,stats=models[ver]
                print(f'Now running {ver.upper()} teacher-FREE, command={command}, seed={seed}, horizon={args.seconds}s',flush=True)
                r=eval_once(ver,net,stats,mj_model,command,seed,args.seconds,
                           viewer=args.viewer and ver=='v3',record=not args.no_record)
                rows.append(r)
                print(json.dumps(r,indent=2),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'paired_results.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics={}
    for ver in ['v2','v3']:
        subset=[r for r in rows if r['controller']==ver.upper()]
        metrics[ver]={'episodes':len(subset),
          'pass_count':sum(r['pass_8s_task'] for r in subset),
          'mean_survival_s':float(np.mean([r['sim_seconds'] for r in subset])),
          'mean_forward_m':float(np.mean([r['forward_m'] for r in subset])),
          'mean_abs_side_m':float(np.mean([abs(r['sideways_m']) for r in subset]))}
    (OUT/'paired_summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    print('\n=== INDEPENDENT GO2 SNN PAIRED COMPARISON ===');print(json.dumps(metrics,indent=2))
    print('Raw evidence:', OUT/'paired_results.csv')
    print('These seeds have now been exposed; choose NEW seeds for future evaluation after further tuning.')

if __name__=='__main__':main()
