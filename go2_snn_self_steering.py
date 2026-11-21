r'''NeuroStride: autonomous SNN heading corrections for MuJoCo Unitree Go2.

Purpose: REMOVE the external classical heading supervisor from the SNN test.
Two spiking networks alone determine locomotion joint-action offsets and steering
joint-action corrections; the common torque-limited PD motor interface remains.

TRAIN (teacher queried OFFLINE ONLY for counterfactual steering action deltas):
 .\.venv\Scripts\python.exe go2_snn_self_steering.py train --epochs 16
TEST (ANN is NOT LOADED with SNN-only modes):
 .\.venv\Scripts\python.exe go2_snn_self_steering.py test --modes snn_zero snn_heading --seeds 6401 6402 --slopes 0 2
LATER controlled baseline comparison:
 .\.venv\Scripts\python.exe go2_snn_self_steering.py test --modes snn_zero snn_heading snn_classic ann_zero ann_classic --seeds 6403 6404 --slopes 0 2

The spiking steering head learns ANN action differences at identical stored
states with different yaw commands; no ANN intervention is permitted at test.
Not PPO, not end-to-end robot RL, and not evidence for physical energy savings.
'''
from __future__ import annotations
import os
import argparse
import contextlib
import csv
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from go2_v5_steering import make_model
from train_go2_snn_v1 import Go2SpikingActor
from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, COMMAND_SCALES, DECIMATION,
    DEFAULT_JOINT_POS, EXPECTED_ACTUATORS, HISTORY_LEN, KP, KD, OBS_DIM, build_obs,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/go2_slope_v6/slope_steering_teacher_and_dagger.npz'
BASE_WEIGHTS = ROOT / 'results/go2_snn_v6/go2_snn_v6_best.pt'
OUT = ROOT / 'results/go2_snn_self_steering'
HEAD_WEIGHTS = OUT / 'spiking_steering_head.pt'
ACTION_DELTA_SCALE = 0.25  # output scaling in teacher action units, NOT joint radians
YAW_ERRORS_DEG = np.asarray([-27., -15., -7., 0., 7., 15., 27.], np.float32)
RANDOM_SEED = 20261121


def local_features(h):
    '''Exactly the existing V3 five-part 350-dimensional temporal feature order.'''
    if h.ndim != 3 or h.shape[1:] != (30, 70):
        raise ValueError(f'Expected batch of [30,70] histories; got {h.shape}')
    return np.concatenate((h[:, -1], h.mean(axis=1), h[:, -2], h[:, -6], h[:, -16]), axis=1).astype(np.float32)


def heading_input(x350, errors_deg):
    if x350.shape != (len(errors_deg), 350):
        raise ValueError('Invalid head features')
    return np.concatenate((x350, (np.asarray(errors_deg, np.float32)/20.0)[:, None]),axis=1).astype(np.float32)


def make_head():
    return Go2SpikingActor(feature_dim=351, population=3, hidden=96, steps=8)


def head_delta(head, x, e):
    '''Output joint action DELTA, with exactly zero output at zero heading error.

    All gait and steering inference comes from learned spiking models: no
    proportional or derivative heading controller executes in this branch.
    '''
    inp = torch.cat((x, e[:, None]/20.), dim=1)
    zero = torch.cat((x, torch.zeros_like(e[:, None])), dim=1)
    out, rate = head(inp)
    neutral, _ = head(zero)
    return (out-neutral)*ACTION_DELTA_SCALE, rate


def query_teacher(hist, yaw_cmds, batch=160):
    '''Counterfactual expert actions on the SAME robot state and observation history.'''
    from go2_heading_benchmark import load_teacher
    expert, adaptation = load_teacher()  # explicitly TRAIN ONLY
    n, k = yaw_cmds.shape
    result = np.empty((n, k, 12), np.float32)
    for lo in range(0, n, max(1,batch//k)):
        hi = min(n, lo + max(1,batch//k))
        histories = np.repeat(hist[lo:hi, None, :, :], k, axis=1)
        histories[:, :, -1, 5] = yaw_cmds[lo:hi] * float(COMMAND_SCALES[2])
        states = torch.from_numpy(histories.reshape(-1,2100).copy())
        with torch.inference_mode():
            latent = adaptation(states)
            raw = expert(torch.cat((states, latent), dim=1))
        predicted = raw.detach().cpu().numpy().reshape(-1,k,12)
        if not np.all(np.isfinite(predicted)) or np.max(np.abs(predicted)) > 12:
            raise RuntimeError('Expert produced abnormal yaw-probe actions')
        result[lo:hi] = np.clip(predicted, -6., 6.)
    return result


def train(epochs, lr, batch, limit):
    if not DATA.is_file():
        raise FileNotFoundError(f'Run go2_collect_slope_v6.py first: {DATA}')
    if not BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f'Train V6 before this experiment: {BASE_WEIGHTS}')
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    ckpt = torch.load(BASE_WEIGHTS,map_location='cpu',weights_only=True)
    _, (mx,sx,_,_) = make_model(ckpt)
    with np.load(DATA,allow_pickle=False) as d:
        hist = np.asarray(d['history'],np.float32).reshape(-1,30,70)
        seeds = np.asarray(d['seed'],np.int64)
        sources = np.asarray(d['source']).astype(str)
        slopes = np.asarray(d['slope_deg'],np.float32)
    if set(np.unique(seeds)) != {5701,5702} or not {'student','expert'}.issubset(set(sources)):
        raise ValueError('Expected collector episodes: seeds 5701,5702 and expert/student modes')
    if not np.any((seeds==5701)&(slopes>1.5)&(sources=='student')):
        raise ValueError('No student-visited uphill training states')
    # Subsample temporally correlated frames, retaining all episode groups.
    select = np.arange(0,len(hist),max(1,int(limit)),dtype=np.int64)
    hist, seeds, sources, slopes = hist[select].copy(),seeds[select],sources[select],slopes[select]
    # The new controller sends yaw=0 to the BASE locomotion SNN.
    # We zero the history's yaw-command column to match this input convention.
    hist[:,:,5]=0.0
    x350 = np.clip((local_features(hist)-mx)/sx, -5, 5).astype(np.float32)
    e = np.broadcast_to(YAW_ERRORS_DEG, (len(hist),len(YAW_ERRORS_DEG)))
    # The mapping from heading error to corrective turn rate is used for
    # offline labels ONLY; no gain/clipping supervisor is used in SNN test.
    wanted = np.clip(-1.15 * np.deg2rad(e), -0.24, +0.24).astype(np.float32)
    # More headroom than previous 0.2rad/s feedback, up to 0.24.
    neutral = np.zeros((len(hist),1),np.float32)
    labels = query_teacher(hist,np.concatenate((neutral,wanted),axis=1))
    delta = labels[:,1:]-labels[:,:1]
    if not np.isfinite(delta).all():
        raise RuntimeError('Nonfinite steering corrections')
    print('Training data: states',len(hist),'teacher delta mean abs',
          round(float(np.mean(np.abs(delta))),5),'max abs',round(float(np.max(np.abs(delta))),5),flush=True)
    if np.mean(np.abs(delta)>0.5)>.01:
        raise RuntimeError('More than 1% of teacher action corrections exceed 0.5; investigate labels')
    X = np.repeat(x350[:,None,:],len(YAW_ERRORS_DEG),axis=1).reshape(-1,350)
    E = e.reshape(-1).astype(np.float32)
    Y = delta.reshape(-1,12).astype(np.float32)
    S = np.repeat(seeds,len(YAW_ERRORS_DEG))
    W = np.repeat(np.where(sources=='student',2.0,1.0)*np.where(slopes>1.5,1.75,1.0),len(YAW_ERRORS_DEG)).astype(np.float32)
    train_idx = np.where(S==5701)[0]
    valid_idx = np.where(S==5702)[0]
    xt=torch.from_numpy(X); et=torch.from_numpy(E); yt=torch.from_numpy(Y); wt=torch.from_numpy(W)
    head=make_head()
    head.readout.weight.data.zero_()
    head.readout.bias.data.zero_()
    optimizer = torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-5)
    rng=np.random.default_rng(RANDOM_SEED)
    OUT.mkdir(parents=True,exist_ok=True)
    history=[]
    best=float('inf')
    best_epoch=0
    no_correction_rmse=float(np.sqrt(np.mean(Y[valid_idx]**2)))
    for epoch in range(1,epochs+1):
        head.train(); losses=[]
        for indices in np.array_split(rng.permutation(train_idx),max(1,math.ceil(len(train_idx)/batch))):
            pred,activity=head_delta(head,xt[indices],et[indices])
            per=(pred-yt[indices]).square().mean(dim=1)
            loss=(per*wt[indices]).sum()/wt[indices].sum()+1e-6*activity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval(); acc=0.; count=0
        with torch.inference_mode():
            for lo in range(0,len(valid_idx),128):
                j=valid_idx[lo:lo+128]
                pred,_=head_delta(head,xt[j],et[j])
                err=pred.numpy()-Y[j]
                acc+=float(np.square(err,dtype=np.float64).sum());count+=err.size
        rmse=math.sqrt(acc/max(1,count))
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'held_seed5702_delta_rmse':rmse,
             'zero_correction_rmse':no_correction_rmse}
        history.append(rec)
        if rmse < best:
            best=rmse;best_epoch=epoch
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                        'epoch':epoch,'base_checkpoint':str(BASE_WEIGHTS),
                        'training_seed':5701,'selection_seed':5702,
                        'provenance':'learned spiking joint-action steering correction; teacher offline only',
                        'input_dim':351,'delta_scale':ACTION_DELTA_SCALE},HEAD_WEIGHTS)
        if epoch==1 or epoch%4==0 or epoch==epochs:
            print(f'epoch {epoch:02d} train {rec["train_loss"]:.6f} '
                  f'held delta RMSE {rmse:.5f} vs zero {no_correction_rmse:.5f}',flush=True)
    with (OUT/'training_history.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    meta={'training':'offline teacher counterfactual joint-action-delta imitation; NOT RL',
          'training_seed':5701,'selection_seed':5702,'states':len(hist),
          'training_pairs':int(len(train_idx)),'validation_pairs':int(len(valid_idx)),
          'best_epoch':best_epoch,'best_held_delta_rmse':best,
          'no_correction_rmse':no_correction_rmse,
          'weights':str(HEAD_WEIGHTS),
          'notice':'Only an offline diagnostic. Straight walking must be shown in held-out MuJoCo rollouts.'}
    (OUT/'training_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print(json.dumps(meta,indent=2),flush=True)
    if best>=no_correction_rmse:
        print('WARNING: head failed to beat zero-correction offline baseline; do NOT claim improvement.',flush=True)


def run(model, mode, seed, speed, seconds, base, heading_net, expert, slope, viewer):
    import mujoco
    from go2_heading_benchmark import heading_rad, wrap_pi
    from go2_first_terrain_pilot import replay_metrics
    data=mujoco.MjData(model)
    key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home')
    if key<0:raise RuntimeError('Go2 home keyframe missing')
    mujoco.mj_resetDataKeyframe(model,data,key)
    data.qpos[7:19]=DEFAULT_JOINT_POS
    data.qpos[2]=0.30
    rng=np.random.default_rng(seed)
    data.qpos[7:19]+=rng.normal(0,.012,12)
    data.qpos[2]+=float(rng.uniform(-.005,.005))
    mujoco.mj_forward(model,data)
    if (model.nu!=12 or abs(model.opt.timestep-.002)>1e-9 or
        [model.actuator(i).name for i in range(12)]!=EXPECTED_ACTUATORS):
        raise RuntimeError('Unexpected Go2 motors or physics step')
    origin=data.qpos[:2].copy()
    initial_yaw=heading_rad(data.qpos[3:7])
    cmd=BASE_COMMAND.copy();cmd[0]=speed
    h=deque((np.zeros(OBS_DIM,np.float32) for _ in range(HISTORY_LEN)),maxlen=HISTORY_LEN)
    prev=np.zeros(12,np.float64);last=prev.copy()
    target=DEFAULT_JOINT_POS.copy(); phase=0.
    lows=model.actuator_ctrlrange[:,0];highs=model.actuator_ctrlrange[:,1]
    stop='completed';peak=0.;min_height=10.;mean_spikes=0.;n_decisions=0
    wall_action_s=0.; snapshots={'qpos':[],'qvel':[],'sim_time':[],'joint_torque':[],'yaw_cmd':[]}

    def snap():
        if snapshots['sim_time'] and float(data.time)==snapshots['sim_time'][-1]:return
        snapshots['qpos'].append(data.qpos.copy());snapshots['qvel'].append(data.qvel.copy())
        snapshots['sim_time'].append(float(data.time));snapshots['joint_torque'].append(data.ctrl.copy())
        snapshots['yaw_cmd'].append(float(cmd[2]))
    snap()
    cm=contextlib.nullcontext(None)
    if viewer:
        import mujoco.viewer
        cm=mujoco.viewer.launch_passive(model,data)
    with cm as view:
        for t in range(int(round(seconds/model.opt.timestep))):
            if viewer and not view.is_running():stop='viewer closed';break
            if t%DECIMATION==0:
                yaw_error=math.degrees(wrap_pi(heading_rad(data.qpos[3:7])-initial_yaw))
                if mode in ('snn_classic','ann_classic'):
                    cmd[2]=float(np.clip(-.9*math.radians(yaw_error),-.2,.2))
                else:
                    cmd[2]=0.0
                phase=(phase+cmd[4]*model.opt.timestep*DECIMATION)%1.0
                obs=build_obs(data,cmd,prev,last,phase)
                h.append(obs)
                history=np.stack(h).astype(np.float32)
                start=time.perf_counter()
                with torch.inference_mode():
                    if mode.startswith('snn'):
                        net,stats=base
                        mx,sx,my,sy=stats
                        feat=local_features(history[None])[0]
                        standardized=np.clip((feat-mx)/sx,-5,5).astype(np.float32)
                        neural_action,rate=net(torch.from_numpy(standardized[None]))
                        raw=neural_action.squeeze(0).numpy()*sy+my
                        mean_spikes+=float(rate)
                        if mode=='snn_heading':
                            heading_input_tensor=torch.from_numpy(standardized[None])
                            correction,heading_activity=head_delta(
                                heading_net,heading_input_tensor,
                                torch.tensor([yaw_error],dtype=torch.float32))
                            gain = float(os.environ.get("GO2_STEERING_GAIN", "1.0"))
                            if not (0.0 <= gain <= 4.0):
                                raise ValueError("GO2_STEERING_GAIN must be between 0 and 4")
                            raw = raw + gain * correction.squeeze(0).numpy()
                            mean_spikes+=float(heading_activity)
                    else:
                        body,adapt=expert
                        f=torch.from_numpy(history.reshape(1,-1).copy())
                        latent=adapt(f)
                        raw=body(torch.cat((f,latent),dim=1)).squeeze(0).numpy()
                wall_action_s+=time.perf_counter()-start
                if raw.shape!=(12,) or not np.all(np.isfinite(raw)) or np.max(np.abs(raw))>12:
                    stop='invalid motor action';break
                action=np.clip(raw,-6,6).astype(np.float64)
                prev,last=last.copy(),action.copy()
                target=DEFAULT_JOINT_POS+last*ACTION_SCALE
                n_decisions+=1
            data.ctrl[:]=np.clip(KP*(target-data.qpos[7:19])-KD*data.qvel[6:18],lows,highs)
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop='nonfinite state';break
            z=float(data.qpos[2]-math.tan(math.radians(slope))*data.qpos[0])
            min_height=min(min_height,z)
            peak=max(peak,abs(math.degrees(wrap_pi(heading_rad(data.qpos[3:7])-initial_yaw))))
            if t%DECIMATION==0:
                snap()
                if viewer and view.is_running():
                    view.cam.lookat[:]=data.qpos[:3]
                    view.sync()
                    time.sleep(model.opt.timestep*DECIMATION)
            if z<.18:stop='torso below local slope height threshold';break
        snap()
    out=OUT/'evaluation';out.mkdir(parents=True,exist_ok=True)
    path=out/f'{mode}_slope{slope:+.1f}_seed{seed}.npz'
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in snapshots.items()})
    dx,dy=map(float,data.qpos[:2]-origin)
    passed=bool(stop=='completed' and data.time>=seconds-.002 and dx>=1.5
                and abs(dy)<=.25 and peak<=15 and min_height>=.18)
    posture=replay_metrics(path,slope)
    return {'mode':mode,'slope_deg':float(slope),'seed':int(seed),
            'time_s':round(float(data.time),4),'forward_m':round(dx,5),
            'sideways_m':round(dy,5),'max_abs_yaw_deg':round(peak,4),
            'pitch_rms_deg':round(float(posture['pitch_rms_deg']),4),
            'roll_rms_deg':round(float(posture['roll_rms_deg']),4),
            'minimum_local_height_m':round(min_height,5),
            'pass_strict':passed,'stop':stop,'policy_inference_ms_per_decision':round(1000*wall_action_s/max(1,n_decisions),3),
            'mean_hidden_activity':round(mean_spikes/max(1,n_decisions),4) if mode.startswith('snn') else None,
            'external_heading_supervisor':mode in ('snn_classic','ann_classic'),
            'ann_used_during_snn_rollout':False if mode.startswith('snn') else None,
            'replay':str(path)}


def test(modes,seeds,slopes,speed,seconds,viewer):
    from go2_first_terrain_pilot import build_terrain
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not BASE_WEIGHTS.is_file():raise FileNotFoundError(f'Missing trained V6: {BASE_WEIGHTS}')
    base=make_model(torch.load(BASE_WEIGHTS,map_location='cpu',weights_only=True))
    base[0].eval()
    head=None
    if 'snn_heading' in modes:
        if not HEAD_WEIGHTS.is_file():raise FileNotFoundError(f'Train heading SNN first: {HEAD_WEIGHTS}')
        head=make_head()
        head.load_state_dict(torch.load(HEAD_WEIGHTS,map_location='cpu',weights_only=True)['state_dict'])
        head.eval()
    teacher=None
    if any(m.startswith('ann') for m in modes):
        from go2_heading_benchmark import load_teacher
        teacher=load_teacher()
    rows=[]
    for slope in slopes:
        world=build_terrain(slope)
        for seed in seeds:
            for mode in modes:
                print(f'Go2 TEST {mode} seed={seed} slope={slope:+.1f}° (no teacher for SNN)',flush=True)
                result=run(world,mode,seed,speed,seconds,base,head,teacher,slope,viewer)
                rows.append(result)
                print('  ',result['time_s'],'s, X',result['forward_m'],'m, peak yaw',
                      result['max_abs_yaw_deg'],'deg, PASS',result['pass_strict'],flush=True)
    path=OUT/'evaluation/episodes.csv'
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for mode in modes:
        for slope in slopes:
            g=[x for x in rows if x['mode']==mode and x['slope_deg']==slope]
            summary[f'{mode}_slope{slope:+.1f}']={'n':len(g),'passes':sum(x['pass_strict'] for x in g),
                'mean_forward_m':float(np.mean([x['forward_m'] for x in g])),
                'mean_peak_yaw_deg':float(np.mean([x['max_abs_yaw_deg'] for x in g])),
                'mean_inference_ms':float(np.mean([x['policy_inference_ms_per_decision'] for x in g]))}
    (OUT/'evaluation/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',path,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='job',required=True)
    tr=sub.add_parser('train')
    tr.add_argument('--epochs',type=int,default=16)
    tr.add_argument('--lr',type=float,default=.0006)
    tr.add_argument('--batch',type=int,default=64)
    tr.add_argument('--stride',type=int,default=4,help='Use every Nth stored training state to keep CPU usage reasonable')
    te=sub.add_parser('test')
    te.add_argument('--modes',nargs='+',choices=('snn_zero','snn_heading','snn_classic','ann_zero','ann_classic'),
                    default=['snn_zero','snn_heading'])
    te.add_argument('--seeds',nargs='+',type=int,default=[6401,6402])
    te.add_argument('--slopes',nargs='+',type=float,default=[0.,2.])
    te.add_argument('--command',type=float,default=.45)
    te.add_argument('--seconds',type=float,default=8.)
    te.add_argument('--viewer',action='store_true')
    a=ap.parse_args()
    if a.job=='train':
        if not 1<=a.epochs<=80 or not 0<a.lr<=.003 or not 16<=a.batch<=256 or not 1<=a.stride<=20:
            ap.error('Out-of-range training arguments')
        train(a.epochs,a.lr,a.batch,a.stride)
    else:
        if not a.seeds or len(a.seeds)>12 or len(set(a.seeds))!=len(a.seeds):ap.error('Choose <=12 unique seeds')
        if not a.slopes or len(a.slopes)>6 or any(abs(x)>3 for x in a.slopes):ap.error('Choose <=6 slopes within +/-3°')
        if not 1<=a.seconds<=12 or not .1<=a.command<=.7:ap.error('Use 1..12s and 0.1..0.7m/s')
        if a.viewer and len(a.seeds)*len(a.slopes)*len(a.modes)!=1:ap.error('--viewer only one run')
        test(a.modes,a.seeds,a.slopes,a.command,a.seconds,a.viewer)

if __name__=='__main__':main()
