r'''NeuroStride: on-policy corrective training for the EXISTING spiking steering head.

Put this single file beside go2_snn_self_steering.py in NeuroStride_Starter.
The original V6 walking SNN and original steering-head checkpoint are unchanged.

Windows PowerShell (separate commands):
 .\.venv\Scripts\python.exe go2_snn_heading_onpolicy.py collect --seeds 6701 6702 6703 --gains 1 2
 .\.venv\Scripts\python.exe go2_snn_heading_onpolicy.py train --epochs 12
 .\.venv\Scripts\python.exe go2_snn_heading_onpolicy.py test --seeds 6501 6502 --slopes 0
 # ONLY after diagnosing on 6501/6502: final untouched flat-ground seeds
 .\.venv\Scripts\python.exe go2_snn_heading_onpolicy.py test --seeds 6801 6802 6803 6804 6805 6806 6807 6808 6809 6810 --slopes 0

What this experiment is: expert-labelled, on-policy DATA AGGREGATION / supervised
fine-tuning of a LIF spiking controller, not reinforcement learning or robot hardware.
During student rollouts there is NO ANN query and NO classical yaw supervisor.
The pretrained ANN is queried only AFTER collected trajectories are complete.
The target teacher yaw-rate mapping is used ONLY to generate offline labels.
A lower offline loss does not establish successful autonomous walking.
'''
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import go2_snn_self_steering as ss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'go2_snn_heading_onpolicy'
DATA = OUT / 'collected' / 'student_heading_states.npz'
ADAPTED_HEAD = OUT / 'adapted_spiking_steering_head.pt'


def load_frozen():
    if not ss.BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f'Frozen V6 locomotion checkpoint missing: {ss.BASE_WEIGHTS}')
    if not ss.HEAD_WEIGHTS.is_file():
        raise FileNotFoundError(f'Original steering head missing: {ss.HEAD_WEIGHTS}')
    base = ss.make_model(torch.load(ss.BASE_WEIGHTS, map_location='cpu', weights_only=True))
    base[0].eval()
    head = ss.make_head()
    old_ckpt = torch.load(ss.HEAD_WEIGHTS, map_location='cpu', weights_only=True)
    head.load_state_dict(old_ckpt['state_dict'])
    head.eval()
    return base, head


def episode_onpolicy(base, head, seed, slope, gain, seconds, speed):
    '''Capture exact obs histories in ss.run without copying or altering its physics.'''
    from collections import deque
    from go2_heading_benchmark import heading_rad, wrap_pi
    from go2_first_terrain_pilot import build_terrain

    if gain != 1 and not any('GO2_STEERING_GAIN' in str(x)
                             for x in ss.run.__code__.co_consts):
        raise RuntimeError('Your go2_snn_self_steering.py lacks the gain patch. '
                           'Use --gains 1, or add the gain edit from our last experiment.')
    world = build_terrain(slope)
    original = ss.build_obs
    old_out = ss.OUT
    old_gain = os.environ.get('GO2_STEERING_GAIN')
    buffer = deque((np.zeros(70, np.float32) for _ in range(30)), maxlen=30)
    histories, errors, times = [], [], []
    initial_yaw = None

    def observe(data, cmd, prev, last, phase):
        nonlocal initial_yaw
        obs = original(data, cmd, prev, last, phase)
        if initial_yaw is None:
            initial_yaw = heading_rad(data.qpos[3:7])
        err = math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - initial_yaw))
        buffer.append(obs.copy())
        histories.append(np.stack(buffer).astype(np.float32))
        errors.append(float(err))
        times.append(float(data.time))
        return obs

    try:
        ss.build_obs = observe
        ss.OUT = OUT / 'collected' / f'g{gain:g}'
        os.environ['GO2_STEERING_GAIN'] = f'{gain:g}'
        result = ss.run(world, 'snn_heading', seed, speed, seconds,
                        base, head, None, slope, viewer=False)
    finally:
        ss.build_obs = original
        ss.OUT = old_out
        if old_gain is None:
            os.environ.pop('GO2_STEERING_GAIN', None)
        else:
            os.environ['GO2_STEERING_GAIN'] = old_gain

    if not histories or abs(times[0]) > 1e-8:
        raise RuntimeError('Observation recorder was not synchronized with physics')
    n = len(histories)
    if not (len(errors) == len(times) == n):
        raise RuntimeError('History/yaw sampling mismatch')
    return (np.stack(histories), np.asarray(errors, np.float32),
            np.asarray(times, np.float32), result)


def collect(seeds, slopes, gains, seconds, speed, stride):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base, head = load_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    hist_parts, error_parts, seed_parts, slope_parts, gain_parts = [], [], [], [], []
    reports = []
    for slope in slopes:
        for seed in seeds:
            for gain in gains:
                print(f'SNN-only collecting: seed={seed}, slope={slope:+.1f}, gain={gain:g}', flush=True)
                h, e, t, result = episode_onpolicy(base, head, seed, slope, gain, seconds, speed)
                keep = np.arange(0, len(h), stride)
                hist_parts.append(h[keep])
                error_parts.append(e[keep])
                seed_parts.append(np.full(len(keep), seed, np.int32))
                slope_parts.append(np.full(len(keep), slope, np.float32))
                gain_parts.append(np.full(len(keep), gain, np.float32))
                reports.append({**result, 'collection_gain':float(gain),
                                'recorded_policy_decisions':len(h), 'kept_states':len(keep)})
                print(f"  {result['time_s']:.2f}s forward={result['forward_m']:.3f}m "
                      f"max_yaw={result['max_abs_yaw_deg']:.2f}° "
                      f"saved_states={len(keep)} pass={result['pass_strict']}", flush=True)
    H = np.concatenate(hist_parts).astype(np.float32)
    E = np.concatenate(error_parts).astype(np.float32)
    S = np.concatenate(seed_parts)
    slopes_array = np.concatenate(slope_parts)
    gains_array = np.concatenate(gain_parts)
    # Crucial: ANN is loaded only after ALL SNN rollouts have completed.
    # Request expert yaw=0 and expert yaw=correction on the exact same states.
    perturb = np.stack((E, -E,
                        np.clip(E-5., -30., 30.),
                        np.clip(E+5., -30., 30.)), axis=1).astype(np.float32)
    yaw_cmds = np.clip(-1.15*np.deg2rad(perturb), -.24, +.24)
    commands = np.concatenate((np.zeros((len(H),1),np.float32), yaw_cmds), axis=1)
    print(f'All SNN rollouts finished. Querying offline ANN teacher on {len(H)} stored states ...',flush=True)
    expert = ss.query_teacher(H, commands, batch=100)
    deltas = expert[:,1:,:] - expert[:,0:1,:]
    if deltas.shape != (len(H),4,12) or not np.isfinite(deltas).all():
        raise RuntimeError('Expert counterfactual steering labels invalid')
    np.savez_compressed(DATA, history=H, measured_heading_error_deg=E,
                        error_probes_deg=perturb, expert_action_deltas=deltas,
                        seed=S, slope_deg=slopes_array, collection_gain=gains_array)
    with (DATA.parent/'episodes.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=reports[0].keys())
        writer.writeheader();writer.writerows(reports)
    meta={'label_source':'Third-party ANN expert queried only after SNN-only rollouts',
          'student_modes':'SNN V6 locomotion + frozen SNN steering head; no external heading feedback',
          'source_head':str(ss.HEAD_WEIGHTS),'source_v6':str(ss.BASE_WEIGHTS),
          'seeds':seeds,'slopes_deg':slopes,'collection_gains':gains,
          'seconds':seconds,'command_mps':speed,'sample_stride_decisions':stride,
          'states':len(H),'teacher_counterfactual_pairs':int(deltas.shape[0]*deltas.shape[1]),
          'mean_abs_action_delta':float(np.mean(np.abs(deltas))),
          'warning':'Dataset is policy-visitation supervised correction, not end-to-end SNN RL.'}
    (DATA.parent/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('DATASET COMPLETE:',str(DATA),flush=True)
    print(json.dumps(meta,indent=2),flush=True)


def train(epochs, lr, batch, target_scale):
    if not DATA.exists():
        raise FileNotFoundError(f'Collect on-policy states first: {DATA}')
    torch.manual_seed(20261121)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base, head = load_frozen()
    del base
    frozen = copy.deepcopy(head).eval()
    with np.load(DATA,allow_pickle=False) as d:
        hist = d['history'].astype(np.float32)
        e = d['error_probes_deg'].astype(np.float32)
        delta = d['expert_action_deltas'].astype(np.float32)
        seeds = d['seed'].astype(np.int32)
        slope = d['slope_deg'].astype(np.float32)
    if hist.ndim != 3 or hist.shape[1:] != (30,70):
        raise ValueError(f'Unexpected dataset history shape {hist.shape}')
    n=len(hist)
    if e.shape!=(n,4) or delta.shape!=(n,4,12):
        raise ValueError('Unexpected heading probe or action label dimensions')
    if len(set(seeds.tolist()))<2:
        raise ValueError('Need at least two distinct collection seeds for independent offline selection')
    all_seeds=sorted(set(seeds.tolist()))
    validation_seed=all_seeds[-1]
    train_seeds=all_seeds[:-1]
    ckpt=torch.load(ss.BASE_WEIGHTS,map_location='cpu',weights_only=True)
    _,(mx,sx,_,_)=ss.make_model(ckpt)
    h_zero=hist.copy()
    h_zero[:,:,5]=0.0
    x=ss.local_features(h_zero)
    x=np.clip((x-mx)/sx,-5.,5.).astype(np.float32)
    X=np.repeat(x[:,None,:],4,axis=1).reshape(-1,350)
    E=e.reshape(-1)
    Y=(target_scale*delta).reshape(-1,12)
    SEED=np.repeat(seeds,4)
    W=np.repeat(1.+np.clip(np.abs(e).mean(axis=1)/12.,0.,2.),4)
    W=W*np.where(np.repeat(slope,4)>1.5,1.25,1.0)
    ix_train=np.where(SEED!=validation_seed)[0]
    ix_valid=np.where(SEED==validation_seed)[0]
    if not len(ix_train) or not len(ix_valid):
        raise RuntimeError('Empty train or held-seed validation split')
    xt=torch.from_numpy(X);et=torch.from_numpy(E);yt=torch.from_numpy(Y);wt=torch.from_numpy(W.astype(np.float32))
    rng=np.random.default_rng(20261121)
    def offline_rmse(network,indices):
        network.eval();sum_sq=0.;count=0
        with torch.inference_mode():
            for j in np.array_split(indices,max(1,math.ceil(len(indices)/256))):
                pred,_=ss.head_delta(network,xt[j],et[j])
                err=pred.numpy()-Y[j]
                sum_sq+=float(np.sum(np.square(err,dtype=np.float64)))
                count+=err.size
        return math.sqrt(sum_sq/max(1,count))
    original_rmse=offline_rmse(frozen,ix_valid)
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-5)
    best=original_rmse
    best_epoch=0
    best_state={k:v.detach().cpu().clone() for k,v in frozen.state_dict().items()}
    history=[]
    print(f'On-policy training pairs={len(ix_train)} selection_pairs={len(ix_valid)} '
          f'held_seed={validation_seed}, ORIGINAL head RMSE={original_rmse:.6f}',flush=True)
    for epoch in range(1,epochs+1):
        head.train();losses=[]
        for j in np.array_split(rng.permutation(ix_train),max(1,math.ceil(len(ix_train)/batch))):
            pred,rate=ss.head_delta(head,xt[j],et[j])
            with torch.inference_mode():
                anchor,_=ss.head_delta(frozen,xt[j],et[j])
            mse=(pred-yt[j]).square().mean(dim=1)
            anchor_penalty=(pred-anchor).square().mean()
            loss=(mse*wt[j]).sum()/wt[j].sum()+0.035*anchor_penalty+1e-6*rate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        rmse=offline_rmse(head,ix_valid)
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),
             'held_seed':validation_seed,'held_delta_rmse':rmse,
             'original_head_rmse':original_rmse}
        history.append(rec)
        if rmse+1e-7<best:
            best=rmse;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        if epoch==1 or epoch%3==0 or epoch==epochs:
            print(f'epoch={epoch:02d}/{epochs} loss={rec["train_loss"]:.6f} '
                  f'held RMSE={rmse:.6f} original={original_rmse:.6f}',flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'best_epoch':best_epoch,
                'original_head':str(ss.HEAD_WEIGHTS),
                'frozen_base':str(ss.BASE_WEIGHTS),
                'training_seeds':train_seeds,'selection_seed':validation_seed,
                'teacher_label_target_scale':float(target_scale),
                'offline_improved':bool(best_epoch>0),
                'provenance':'On-policy supervised spiking steering head; ANN offline only'},ADAPTED_HEAD)
    with (OUT/'training_history.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    summary={'training':'on-policy ANN action-delta imitation, not RL',
             'states':n,'pairs':len(X),'train_seeds':train_seeds,
             'selection_seed':validation_seed,'baseline_head_selection_rmse':original_rmse,
             'selected_head_selection_rmse':best,'best_epoch':best_epoch,
             'offline_improved':bool(best_epoch>0),'target_scale':target_scale,
             'checkpoint':str(ADAPTED_HEAD),
             'warning':'Only actual new-seed MuJoCo rollouts can prove improved straightness.'}
    (OUT/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


def test(seeds, slopes, speed, seconds, modes, viewer):
    from go2_first_terrain_pilot import build_terrain
    if not ADAPTED_HEAD.is_file():
        raise FileNotFoundError(f'Train on-policy head first: {ADAPTED_HEAD}')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    base, original = load_frozen()
    candidate=ss.make_head()
    candidate.load_state_dict(torch.load(ADAPTED_HEAD,map_location='cpu',weights_only=True)['state_dict'])
    candidate.eval()
    OUT.mkdir(parents=True,exist_ok=True)
    old_gain=os.environ.get('GO2_STEERING_GAIN')
    old_out=ss.OUT
    os.environ['GO2_STEERING_GAIN']='1.0'
    rows=[]
    try:
        for slope in slopes:
            world=build_terrain(slope)
            for seed in seeds:
                for variant in modes:
                    ss.OUT=OUT/'evaluation'/variant
                    selected=original if variant=='original' else candidate
                    runtime_mode='snn_classic' if variant=='classic' else 'snn_heading'
                    print(f'TEST {variant} seed={seed} slope={slope:+.1f}°, '
                          f'ANN online=False; classic supervisor={variant=="classic"}',flush=True)
                    result=ss.run(world,runtime_mode,seed,speed,seconds,
                                  base,selected,None,slope,viewer)
                    rows.append({**result,'variant':variant})
                    print(f"  {result['time_s']:.2f}s X={result['forward_m']:.3f}m "
                          f"yaw={result['max_abs_yaw_deg']:.2f}° "
                          f"pass={result['pass_strict']}",flush=True)
    finally:
        ss.OUT=old_out
        if old_gain is None:os.environ.pop('GO2_STEERING_GAIN',None)
        else:os.environ['GO2_STEERING_GAIN']=old_gain
    out=OUT/'evaluation';out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'paired_results.csv'
    with csv_path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summary={}
    for variant in modes:
        for slope in slopes:
            g=[r for r in rows if r['variant']==variant and r['slope_deg']==slope]
            summary[f'{variant}_slope{slope:+.1f}']={
                'runs':len(g),'passes':sum(r['pass_strict'] for r in g),
                'mean_forward_m':float(np.mean([r['forward_m'] for r in g])),
                'mean_max_abs_yaw_deg':float(np.mean([r['max_abs_yaw_deg'] for r in g])),
                'external_heading_supervisor':variant=='classic'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    print('DATA SAVED:',csv_path,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='job',required=True)
    co=sub.add_parser('collect',help='SNN visits states, ANN only labels afterwards')
    co.add_argument('--seeds',nargs='+',type=int,default=[6701,6702,6703])
    co.add_argument('--slopes',nargs='+',type=float,default=[0.])
    co.add_argument('--gains',nargs='+',type=float,default=[1.,2.])
    co.add_argument('--seconds',type=float,default=8.)
    co.add_argument('--command',type=float,default=.45)
    co.add_argument('--stride',type=int,default=2)
    tr=sub.add_parser('train')
    tr.add_argument('--epochs',type=int,default=12)
    tr.add_argument('--lr',type=float,default=.00025)
    tr.add_argument('--batch',type=int,default=96)
    tr.add_argument('--target-scale',type=float,default=1.5,
                    help='Experimentally strengthen on-policy teacher deltas (NOT a runtime heading law)')
    te=sub.add_parser('test')
    te.add_argument('--seeds',nargs='+',type=int,default=[6501,6502])
    te.add_argument('--slopes',nargs='+',type=float,default=[0.])
    te.add_argument('--command',type=float,default=.45)
    te.add_argument('--seconds',type=float,default=8.)
    te.add_argument('--modes',nargs='+',choices=['original','adapted','classic'],
                    default=['original','adapted','classic'])
    te.add_argument('--viewer',action='store_true')
    a=ap.parse_args()
    if a.job=='collect':
        if len(set(a.seeds))<2 or len(a.seeds)>8 or len(set(a.seeds))!=len(a.seeds):
            ap.error('Collection requires 2–8 unique seeds (latest seed held out from updates)')
        if not a.gains or len(a.gains)>3 or len(set(a.gains))!=len(a.gains) or any(g<.25 or g>4 for g in a.gains):
            ap.error('Use 1–3 gains between .25 and 4')
        if a.stride<1 or a.stride>12:ap.error('Use a state stride in 1..12')
        if not a.slopes or len(set(a.slopes))!=len(a.slopes) or not (1<=a.seconds<=12) or not (.1<=a.command<=.7) or any(abs(s)>3 for s in a.slopes):
            ap.error('Seconds 1..12, command .1..0.7, unique slopes within +/-3 degrees')
        collect(a.seeds,a.slopes,a.gains,a.seconds,a.command,a.stride)
    elif a.job=='train':
        if not (1<=a.epochs<=50) or not (0<a.lr<=.002) or not (16<=a.batch<=256) or not (.5<=a.target_scale<=4):
            ap.error('Training arguments out of bounds')
        train(a.epochs,a.lr,a.batch,a.target_scale)
    else:
        if not a.seeds or len(a.seeds)>15 or len(set(a.seeds))!=len(a.seeds):
            ap.error('Test requires 1–15 unique seeds')
        if not a.slopes or len(a.slopes)>5 or any(abs(s)>3 for s in a.slopes):
            ap.error('Test slopes +/-3° and max 5')
        if not (1<=a.seconds<=12) or not (.1<=a.command<=.7):
            ap.error('seconds 1..12; command .1..0.7')
        if a.viewer and len(a.seeds)*len(a.slopes)*len(a.modes)!=1:
            ap.error('--viewer only with one seed, slope and mode')
        test(a.seeds,a.slopes,a.command,a.seconds,a.modes,a.viewer)

if __name__=='__main__':main()
