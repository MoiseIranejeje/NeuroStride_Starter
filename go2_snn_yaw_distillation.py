"""NeuroStride — spiking yaw-command controller distilled from a classical reference.

Goal: retain frozen V6 SNN walking network. Replace the RUN-TIME classical
    yaw-rate supervisor with a learned LIF SNN taking current heading error
    and outputting a yaw-rate command. This is SUPERVISED DISTILLATION of a
    classical reference function, not end-to-end RL or original learned control.

This experiment tests whether the same motor-interface mechanism that made
    snn_classic pass can run through a spiking network alone.

PowerShell in NeuroStride_Starter:
    .\\.venv\\Scripts\\python.exe go2_snn_yaw_distillation.py train --epochs 35
    .\\.venv\\Scripts\\python.exe go2_snn_yaw_distillation.py test --modes snn_zero snn_spiking_yaw snn_classic --seeds 6501 6502 --slopes 0 --seconds 8

Fresh-start gate (ONLY after development comparison):
    .\\.venv\\Scripts\\python.exe go2_snn_yaw_distillation.py test --modes snn_spiking_yaw snn_classic ann_classic --seeds 6801 6802 6803 6804 6805 6806 6807 6808 6809 6810 --slopes 0 --seconds 8

Spiking yaw controller: a 1-input, 2-hidden-LIF-layer, analog-output network.
SNN-only test modes never load/query the pretrained ANN. Both SNN-only and
assisted ANN variants retain low-level torque-limited PD motor controllers.
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

import numpy as np
import torch

import go2_snn_self_steering as ss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'go2_snn_yaw_distillation'
WEIGHTS = OUT / 'spiking_yaw_controller.pt'
YAW_LIMIT = 0.2  # rad/s; matches proven classical baseline
SEED = 20261121


def make_yaw_net():
    # Reuses exact spiking actor code: population input encoding, 2 hidden
    # discrete LIF layers, surrogate gradients and a continuous readout.
    from train_go2_snn_v1 import Go2SpikingActor
    return Go2SpikingActor(feature_dim=1, population=5, hidden=64, steps=12)


def infer_yaw(net, error_deg):
    """Return normalized learned yaw-rate action and mean LIF spike rate."""
    if isinstance(error_deg, torch.Tensor):
        error = error_deg.to(dtype=torch.float32).reshape(-1, 1)
    else:
        error = torch.as_tensor(error_deg, dtype=torch.float32).reshape(-1, 1)
    raw, spikes = net((error / 20.0).clamp(-3.0, 3.0))
    # Subtract the same network's neutral-state response. This makes the
    # learned controller command EXACTLY zero when heading error is zero,
    # avoiding a potentially cumulative steering bias from readout offsets.
    neutral, _ = net(torch.zeros_like(error))
    cmd = (YAW_LIMIT * (raw[:, 0] - neutral[:, 0])).clamp(-YAW_LIMIT, YAW_LIMIT)
    return cmd, spikes


def classical_teacher(error_degrees):
    """ONLY used in training and explicitly classical comparator episodes."""
    return np.clip(-0.9 * np.deg2rad(error_degrees), -YAW_LIMIT, YAW_LIMIT).astype(np.float32)


def train(epochs, lr, batch_size):
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(SEED)
    net = make_yaw_net()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    # Synthetic *state errors*, no Go2 training episodes, by design:
    # exact bounded classical yaw-reference controller is the supervised label.
    angles = np.linspace(-40.0, 40.0, 4001, dtype=np.float32)
    target = classical_teacher(angles)
    held = (np.arange(len(angles)) % 5 == 0)
    idx_train = np.flatnonzero(~held)
    idx_valid = np.flatnonzero(held)
    x = torch.from_numpy(angles)
    y = torch.from_numpy(target)
    best_rmse = float('inf')
    best_epoch = 0
    history = []
    OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        net.train(); loss_values = []
        batches = np.array_split(rng.permutation(idx_train), math.ceil(len(idx_train)/batch_size))
        for indices in batches:
            pred, spike_rate = infer_yaw(net, x[indices])
            loss = torch.mean((pred-y[indices])**2) + 1e-6 * spike_rate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            loss_values.append(float(loss.detach()))
        net.eval()
        with torch.inference_mode():
            pred, rate = infer_yaw(net, x[idx_valid])
            err = pred - y[idx_valid]
            rmse = float(torch.sqrt(torch.mean(err**2)))
            max_abs_err = float(err.abs().max())
        history.append({'epoch': epoch, 'train_loss': float(np.mean(loss_values)),
                        'held_error_rad_s_rmse': rmse, 'held_max_error_rad_s': max_abs_err,
                        'mean_spike_rate': float(rate)})
        if rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch
            torch.save({'state_dict': {k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                        'epoch': epoch, 'reference': 'clip(-0.9 * heading_error_rad, -0.2, +0.2)',
                        'learning': 'supervised classical heading law distillation; NOT RL',
                        'heading_error_scale_deg': 20.0,
                        'command_limit_rad_s': YAW_LIMIT}, WEIGHTS)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'epoch {epoch:03d}/{epochs} yaw RMSE={rmse:.5f} rad/s '
                  f'max_error={max_abs_err:.5f} rad/s', flush=True)
    with (OUT / 'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    summary = {'reference': 'classical bounded proportional yaw supervisor, OFFLINE labels only',
               'training': 'supervised classical-to-spiking distillation, NOT RL',
               'best_epoch': best_epoch, 'best_rmse_rad_s': best_rmse,
               'required_next': 'actual closed-loop test on Go2 with matched control baselines',
               'checkpoint': str(WEIGHTS)}
    (OUT / 'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if best_rmse >= 0.02:
        print('WARNING: large yaw imitation RMSE. Do not assume closed-loop parity.',flush=True)


def run(model, mode, seed, speed, seconds, base, yaw_net, teacher, slope, viewer):
    import mujoco
    from go2_heading_benchmark import heading_rad, wrap_pi
    from go2_first_terrain_pilot import replay_metrics
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home < 0:
        raise RuntimeError('Go2 model home keyframe missing')
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[7:19] = ss.DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.normal(0, 0.012, 12)
    data.qpos[2] += float(rng.uniform(-0.005, 0.005))
    mujoco.mj_forward(model, data)
    if (model.nu != 12 or abs(model.opt.timestep-0.002)>1e-9 or
            [model.actuator(i).name for i in range(12)] != ss.EXPECTED_ACTUATORS):
        raise RuntimeError('Unexpected Go2 actuator configuration')
    origin = data.qpos[:2].copy()
    yaw_start = heading_rad(data.qpos[3:7])
    cmd = ss.BASE_COMMAND.copy()
    cmd[0] = speed
    history = deque((np.zeros(ss.OBS_DIM,np.float32) for _ in range(ss.HISTORY_LEN)),
                    maxlen=ss.HISTORY_LEN)
    prev = np.zeros(12,np.float64)
    last = prev.copy()
    target = ss.DEFAULT_JOINT_POS.copy()
    phase = 0.0
    lows = model.actuator_ctrlrange[:,0]
    highs = model.actuator_ctrlrange[:,1]
    stop='completed'
    peak=0.0
    min_z=10.0
    total_spike_rate=0.0
    n_decisions=0
    timing=0.0
    snapshots={'qpos':[],'qvel':[],'sim_time':[],'joint_torque':[],'yaw_cmd':[]}

    def snap():
        t=float(data.time)
        if snapshots['sim_time'] and t==snapshots['sim_time'][-1]:
            return
        snapshots['qpos'].append(data.qpos.copy())
        snapshots['qvel'].append(data.qvel.copy())
        snapshots['sim_time'].append(t)
        snapshots['joint_torque'].append(data.ctrl.copy())
        snapshots['yaw_cmd'].append(float(cmd[2]))

    snap()
    cm=contextlib.nullcontext(None)
    if viewer:
        import mujoco.viewer
        cm=mujoco.viewer.launch_passive(model,data)
    with cm as view:
        for step in range(int(round(seconds/model.opt.timestep))):
            if viewer and not view.is_running():
                stop='viewer closed';break
            if step % ss.DECIMATION == 0:
                heading_err_deg=math.degrees(wrap_pi(heading_rad(data.qpos[3:7])-yaw_start))
                # The learned command enters the SAME yaw-command interface as
                # the successful classical feedback controller. No classical
                # yaw formula runs here in snn_spiking_yaw mode.
                tic=time.perf_counter()
                if mode in ('snn_spiking_yaw','ann_shared_spiking_yaw'):
                    with torch.inference_mode():
                        yaw_rate,spikes=infer_yaw(yaw_net,np.array([heading_err_deg],np.float32))
                    cmd[2]=float(yaw_rate.item())
                    total_spike_rate+=float(spikes)
                elif mode in ('snn_classic','ann_classic'):
                    cmd[2]=float(classical_teacher(np.asarray([heading_err_deg]))[0])
                else:
                    cmd[2]=0.0
                phase=(phase+cmd[4]*model.opt.timestep*ss.DECIMATION)%1.0
                obs=ss.build_obs(data,cmd,prev,last,phase)
                history.append(obs)
                h=np.stack(history).astype(np.float32)
                with torch.inference_mode():
                    if mode.startswith('snn'):
                        actor,stats=base
                        mx,sx,my,sy=stats
                        feat=ss.local_features(h[None])[0]
                        x=np.clip((feat-mx)/sx,-5,5).astype(np.float32)
                        raw,spikes=actor(torch.from_numpy(x[None]))
                        action=raw.squeeze(0).numpy()*sy+my
                        total_spike_rate+=float(spikes)
                    else:
                        body,adapt=teacher
                        st=torch.from_numpy(h.reshape(1,-1).copy())
                        latent=adapt(st)
                        action=body(torch.cat((st,latent),dim=1)).squeeze(0).numpy()
                timing+=time.perf_counter()-tic
                if action.shape !=(12,) or not np.all(np.isfinite(action)) or np.max(np.abs(action))>12:
                    stop='invalid motor action';break
                action=np.clip(action,-6,6).astype(np.float64)
                prev,last=last.copy(),action.copy()
                target=ss.DEFAULT_JOINT_POS+last*ss.ACTION_SCALE
                n_decisions+=1
            data.ctrl[:]=np.clip(ss.KP*(target-data.qpos[7:19])-ss.KD*data.qvel[6:18],lows,highs)
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop='nonfinite state';break
            z=float(data.qpos[2]-math.tan(math.radians(slope))*data.qpos[0])
            min_z=min(min_z,z)
            peak=max(peak,abs(math.degrees(wrap_pi(heading_rad(data.qpos[3:7])-yaw_start))))
            if step % ss.DECIMATION == 0:
                snap()
                if viewer and view.is_running():
                    view.cam.lookat[:]=data.qpos[:3]
                    view.sync()
                    time.sleep(model.opt.timestep*ss.DECIMATION)
            if z<0.18:
                stop='torso below local slope height threshold';break
        snap()
    out=OUT/'evaluation';out.mkdir(parents=True,exist_ok=True)
    path=out/f'{mode}_slope{slope:+.1f}_seed{seed}.npz'
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in snapshots.items()})
    dx,dy=map(float,data.qpos[:2]-origin)
    pass_strict=bool(stop=='completed' and data.time>=seconds-0.002 and dx>=1.5 and
                     abs(dy)<=0.25 and peak<=15 and min_z>=0.18)
    posture=replay_metrics(path,slope)
    return {'mode':mode,'seed':int(seed),'slope_deg':float(slope),
            'time_s':round(float(data.time),4),'forward_m':round(dx,5),
            'sideways_m':round(dy,5),'max_abs_yaw_deg':round(peak,4),
            'pitch_rms_deg':round(float(posture['pitch_rms_deg']),4),
            'roll_rms_deg':round(float(posture['roll_rms_deg']),4),
            'minimum_local_height_m':round(min_z,5),'pass_strict':pass_strict,
            'stop':stop,'mean_spike_rate':round(total_spike_rate/max(n_decisions,1),4) if
            (mode.startswith('snn') or 'spiking_yaw' in mode) else None,
            'wall_inference_ms':round(1000*timing/max(1,n_decisions),3),
            'external_classic_supervisor': mode.endswith('classic'),
            'expert_ann_during_runtime': mode.startswith('ann'),
            'replay':str(path)}


def test(modes,seeds,slopes,speed,seconds,viewer):
    from go2_first_terrain_pilot import build_terrain
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not ss.BASE_WEIGHTS.is_file():
        raise FileNotFoundError(f'Frozen V6 SNN missing: {ss.BASE_WEIGHTS}')
    base=ss.make_model(torch.load(ss.BASE_WEIGHTS,map_location='cpu',weights_only=True))
    base[0].eval()
    yaw_net=None
    if any('spiking_yaw' in m for m in modes):
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f'Train yaw SNN first: {WEIGHTS}')
        yaw_net=make_yaw_net()
        cp=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
        yaw_net.load_state_dict(cp['state_dict'])
        yaw_net.eval()
    teacher=None
    if any(m.startswith('ann') for m in modes):
        from go2_heading_benchmark import load_teacher
        teacher=load_teacher()
    rows=[]
    for slope in slopes:
        world=build_terrain(slope)
        for seed in seeds:
            for mode in modes:
                print(f'Go2 {mode} seed={seed} slope={slope:+.1f}°',flush=True)
                r=run(world,mode,seed,speed,seconds,base,yaw_net,teacher,slope,viewer)
                rows.append(r)
                print(f'  {r["time_s"]:.2f}s X={r["forward_m"]:.3f}m '
                      f'Y={r["sideways_m"]:+.3f}m peak yaw={r["max_abs_yaw_deg"]:.2f}° '
                      f'PASS={r["pass_strict"]}',flush=True)
    with (OUT/'evaluation'/'episodes.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary={}
    for mode in modes:
        for slope in slopes:
            grp=[r for r in rows if r['mode']==mode and r['slope_deg']==slope]
            summary[f'{mode}_slope{slope:+.1f}']={
                'episodes':len(grp),'passes':sum(r['pass_strict'] for r in grp),
                'mean_forward_m':float(np.mean([r['forward_m'] for r in grp])),
                'mean_peak_yaw_deg':float(np.mean([r['max_abs_yaw_deg'] for r in grp]))}
    (OUT/'evaluation'/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',OUT/'evaluation'/'episodes.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='job',required=True)
    tr=sub.add_parser('train')
    tr.add_argument('--epochs',type=int,default=35)
    tr.add_argument('--lr',type=float,default=0.003)
    tr.add_argument('--batch',type=int,default=256)
    te=sub.add_parser('test')
    te.add_argument('--modes',nargs='+',choices=('snn_zero','snn_spiking_yaw','snn_classic','ann_zero','ann_classic','ann_shared_spiking_yaw'),default=['snn_zero','snn_spiking_yaw','snn_classic'])
    te.add_argument('--seeds',type=int,nargs='+',default=[6501,6502])
    te.add_argument('--slopes',type=float,nargs='+',default=[0.])
    te.add_argument('--command',type=float,default=.45)
    te.add_argument('--seconds',type=float,default=8.)
    te.add_argument('--viewer',action='store_true')
    args=p.parse_args()
    if args.job=='train':
        if not (1<=args.epochs<=150 and 0<args.lr<=.01 and 32<=args.batch<=1024):
            p.error('Training arguments out of range')
        train(args.epochs,args.lr,args.batch)
    else:
        if not args.seeds or len(args.seeds)>20 or len(set(args.seeds))!=len(args.seeds):
            p.error('Choose 1..20 unique seeds')
        if not args.slopes or any(abs(s)>3 for s in args.slopes):
            p.error('Slopes must be within ±3°')
        if args.viewer and len(args.seeds)*len(args.slopes)*len(args.modes)!=1:
            p.error('--viewer requires exactly one model/slope/seed')
        if not 1<=args.seconds<=12 or not .1<=args.command<=.7:
            p.error('Use 1–12s and 0.1–0.7m/s')
        test(args.modes,args.seeds,args.slopes,args.command,args.seconds,args.viewer)

if __name__=='__main__':
    main()
