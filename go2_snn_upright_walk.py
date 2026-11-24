"""NeuroStride: genuine spiking torso-height feedback with frozen SNN walking.

Scientific scope:
- Frozen V6 locomotion SNN + frozen yaw5 SNN provide joint actions/steering.
- 'fixed16'/'fixed24' use conventional fixed height offsets as diagnostics.
- 'neural' adds a NEW LIF SNN that reads measured body height and vertical
  velocity and issues a bounded knee-extension command at every policy step.
- The new height SNN is SUPERVISED-DISTILLED from an analytic feedback teacher,
  NOT end-to-end reinforcement learning or proof of biological computing.
- Conventional torque-limited PD remains the low-level actuator controller.
- Never overwrites the existing locomotion or yaw checkpoints.

In NeuroStride_Starter (Windows PowerShell):
  .\\.venv\\Scripts\\python.exe go2_snn_upright_walk.py test --modes baseline fixed16 fixed24 --seeds 7601 7602 7603
  .\\.venv\\Scripts\\python.exe go2_snn_upright_walk.py train --epochs 25
  .\\.venv\\Scripts\\python.exe go2_snn_upright_walk.py test --modes baseline fixed16 fixed24 neural --seeds 7601 7602 7603
  .\\.venv\\Scripts\\python.exe go2_snn_upright_walk.py test --modes baseline fixed16 fixed24 neural --seeds 7701 7702 7703 7704 7705 7706 7707 7708 7709 7710 --tag holdout

Report ALL policies, including failures; don't tune on the 7701-7710 holdout.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from collections import deque
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'go2_snn_upright_walk'
POSTURE_CHECKPOINT = OUT / 'spiking_height_controller.pt'
STANDING_REFERENCE_ROOT_Z = 0.3158  # 0.24 rad STATIC experiment, NOT overall robot height
MANUFACTURER_STANDING_HEIGHT = 0.4000  # measured via mesh top, NOT base qpos[2]
MAX_KNEE_EXTENSION = 0.24
EXTENSION_MAX_CHANGE_PER_STEP = 0.010  # rad at each 50-Hz policy decision
MODES = ('baseline', 'fixed16', 'fixed24', 'neural')


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + x.abs()).square()


class SpikingHeightController(nn.Module):
    """Two input features -> two hidden LIF layers -> analog extension readout.

    Each decision encodes normalised torso-height error and vertical velocity
    with deterministic binary pulse-density spikes; low-level PD stays classic.
    """
    def __init__(self, hidden=32, steps=8, population=5):
        super().__init__()
        self.hidden = hidden
        self.steps = steps
        self.population = population
        self.enc = nn.Linear(2 * population, hidden)
        self.rec = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, 1)
        self.register_buffer('centers', torch.linspace(-2.0, 2.0, population))
        self.register_buffer('phases', torch.remainder(torch.arange(2 * population).float() * 0.61803398875, 1))
        nn.init.constant_(self.enc.bias, 0.35)
        nn.init.constant_(self.rec.bias, 0.35)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError(f'Expected [batch,2] height and velocity features, got {tuple(features.shape)}')
        z = features.float().clamp(-4, 4)
        rates = torch.exp(-0.5 * ((z.unsqueeze(-1) - self.centers) / 1.4).square()).flatten(1)
        rates = rates.clamp(0, 1)
        v1 = z.new_zeros((z.shape[0], self.hidden))
        v2 = z.new_zeros((z.shape[0], self.hidden))
        count = z.new_zeros((z.shape[0], self.hidden))
        count1 = z.new_zeros((z.shape[0], self.hidden))
        for t in range(self.steps):
            inp = (torch.floor(self.phases + (t + 1) * rates) -
                   torch.floor(self.phases + t * rates)).clamp(0, 1)
            mem1 = .7 * v1 + self.enc(inp)
            spike1 = SurrogateSpike.apply(mem1 - .8)
            v1 = mem1 - .8 * spike1.detach()
            mem2 = .7 * v2 + self.rec(spike1)
            spike2 = SurrogateSpike.apply(mem2 - .8)
            v2 = mem2 - .8 * spike2.detach()
            count += spike2
            count1 += spike1
        decoded = self.readout(count / self.steps).squeeze(-1)
        extension = MAX_KNEE_EXTENSION * torch.sigmoid(decoded)
        spike_rate = .5 * (count.mean() + count1.mean()) / self.steps
        return extension, spike_rate


def teacher_extension(root_z, vertical_velocity, target_root_z=STANDING_REFERENCE_ROOT_Z):
    """Analytic OFFLINE labels only; NEVER called by the 'neural' rollout."""
    h = np.asarray(root_z, dtype=np.float32)
    v = np.asarray(vertical_velocity, dtype=np.float32)
    desired = .20 + 1.5 * (float(target_root_z) - h) - .12 * v
    return np.clip(desired, .02, MAX_KNEE_EXTENSION).astype(np.float32)


def height_features(root_z, vertical_velocity, target_root_z):
    return torch.tensor([[(target_root_z - float(root_z)) / .05,
                          float(vertical_velocity) / .30]], dtype=torch.float32)


def _save_csv(rows, path):
    if not rows:
        raise ValueError('Refusing to write empty experiment CSV')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def train(epochs, lr, batch_size, target_root_z):
    """Train only a NEW height LIF head, offline analytic teacher supervision."""
    torch.manual_seed(20261124)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(20261124)
    # Deliberate synthetic height/velocity coverage, including crouch and bounce.
    h = rng.uniform(.22, .39, 2400).astype(np.float32)
    v = rng.uniform(-.65, .65, 2400).astype(np.float32)
    x = np.stack(((target_root_z - h) / .05, v / .30), axis=1).astype(np.float32)
    y = teacher_extension(h, v, target_root_z)
    perm = rng.permutation(len(h))
    train_ids, valid_ids = perm[:1920], perm[1920:]
    inputs = torch.from_numpy(x)
    labels = torch.from_numpy(y)
    net = SpikingHeightController()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    best = float('inf')
    history = []
    OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(train_ids)
        for ii in np.array_split(order, math.ceil(len(order) / batch_size)):
            pred, rate = net(inputs[ii])
            loss = (pred - labels[ii]).square().mean() + 1e-6 * rate
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.inference_mode():
            pred, rate = net(inputs[valid_ids])
            diff = pred - labels[valid_ids]
            rmse = float(torch.sqrt(diff.square().mean()))
            max_error = float(diff.abs().max())
        history.append({'epoch': epoch, 'held_rmse_extension_rad': rmse,
                        'held_max_error_extension_rad': max_error,
                        'spike_rate_per_step': float(rate)})
        if rmse < best:
            best = rmse
            torch.save({'state_dict': net.state_dict(), 'target_root_z': float(target_root_z),
                        'max_extension_rad': MAX_KNEE_EXTENSION,
                        'training': 'supervised imitation of analytic root-height/vertical-speed feedback, NOT reinforcement learning'},
                       POSTURE_CHECKPOINT)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'Epoch {epoch:03d}: validation extension RMSE {rmse:.5f} rad, max err {max_error:.5f} rad', flush=True)
    _save_csv(history, OUT / 'training_history.csv')
    _save_json({'best_held_rmse_extension_rad': best, 'model': str(POSTURE_CHECKPOINT),
                'training': 'analytic feedback to SNN supervised distillation, not RL',
                'note': 'Offline error does not show Go2 gait stability: run paired MuJoCo test'},
               OUT / 'training_summary.json')
    if best > .035:
        print('WARNING: poor offline match (>0.035 rad). Do not claim successful control without a closed-loop test.', flush=True)


def load_frozen(height_required):
    import go2_snn_self_steering as ss
    import go2_snn_yaw_distillation as original_yaw
    import go2_yaw5_fasttrack as yaw5
    for path in (ss.BASE_WEIGHTS, yaw5.NEW_WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(f'Frozen checkpoint missing: {path}. Do not retrain old policy to fix path.')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base = ss.make_model(torch.load(ss.BASE_WEIGHTS, map_location='cpu', weights_only=True))
    base[0].eval()
    cp = torch.load(yaw5.NEW_WEIGHTS, map_location='cpu', weights_only=True)
    yaw_net = original_yaw.make_yaw_net()
    yaw_net.load_state_dict(cp['state_dict'])
    yaw_net.eval()
    yaw_cap = float(cp['cap_rad_s'] if 'cap_rad_s' in cp else cp['command_limit_rad_s'])
    if not (0.05 <= yaw_cap <= .30):
        raise ValueError(f'Unexpected yaw cap {yaw_cap}')
    # CRITICAL: yaw5.neural_command reads this module-level limit.
    # Preserve the proven +/-.30 rad/s command, not the older default .20.
    original_yaw.YAW_LIMIT = yaw_cap
    height_net, target_height = None, None
    if height_required:
        if not POSTURE_CHECKPOINT.is_file():
            raise FileNotFoundError('New posture SNN weights missing. Run train first.')
        hp = torch.load(POSTURE_CHECKPOINT, map_location='cpu', weights_only=True)
        height_net = SpikingHeightController()
        height_net.load_state_dict(hp['state_dict'])
        height_net.eval()
        target_height = float(hp['target_root_z'])
    return ss, yaw5, base, yaw_net, yaw_cap, height_net, target_height


def extension_offsets(extension):
    """Identical target geometry as static calibration; four thigh/knee pairs."""
    ext = float(np.clip(extension, 0, MAX_KNEE_EXTENSION))
    offset = np.zeros(12, dtype=np.float64)
    for j in (0, 3, 6, 9):
        offset[j+1] = -.5 * ext
        offset[j+2] = ext
    return offset


def roll_pitch_yaw(quat):
    w, x, y, z = map(float, quat)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(float(np.clip(2 * (w*y - z*x), -1, 1)))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return math.degrees(roll), math.degrees(pitch), yaw


def rollout(mode, model, seed, command, seconds, frozen, tag, viewer=False):
    import mujoco
    from go2_standing_height_calibration import visual_bounds
    from mujoco import viewer as mujoco_viewer
    ss, yaw5, base, yaw_net, yaw_cap, height_net, target_height = frozen
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home < 0:
        raise RuntimeError('Go2 home keyframe missing')
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[7:19] = ss.DEFAULT_JOINT_POS
    data.qpos[2] = .30
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.normal(0, .012, 12)
    data.qpos[2] += float(rng.uniform(-.005, .005))
    mujoco.mj_forward(model, data)
    if model.nu != 12 or abs(float(model.opt.timestep) - .002) > 1e-9:
        raise RuntimeError('Unexpected Go2 actuator count or simulation step')
    names = [model.actuator(i).name for i in range(12)]
    if names != list(ss.EXPECTED_ACTUATORS):
        raise RuntimeError('Go2 actuator order is different from trained policy')

    if mode == 'neural' and height_net is None:
        raise RuntimeError('Load trained height SNN for neural mode')
    start = data.qpos[:3].copy()
    yaw_start = roll_pitch_yaw(data.qpos[3:7])[2]
    command_vec = ss.BASE_COMMAND.copy()
    command_vec[0] = command
    history = deque((np.zeros(ss.OBS_DIM, np.float32) for _ in range(ss.HISTORY_LEN)), maxlen=ss.HISTORY_LEN)
    prev_action = np.zeros(12, np.float64)
    last_action = np.zeros(12, np.float64)
    target = ss.DEFAULT_JOINT_POS.copy()
    phase = 0.0
    current_extension = 0.0
    last_height_spike_rate = 0.0
    torque_lo, torque_hi = model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1]
    clip_steps = nsteps = nd = 0
    pitch_sq = roll_sq = 0.0
    peak_yaw = peak_cross = 0.0
    min_root_z = float(data.qpos[2])
    stop = 'completed'
    visual_heights = []
    root_heights = []
    raw_extensions = []
    snapshot = {k: [] for k in ('qpos','qvel','sim_time','joint_torque','yaw_cmd',
                               'knee_extension_rad','visual_top_height_m','roll_deg','pitch_deg','cross_track_m')}

    def cross_track():
        dx, dy = data.qpos[:2] - start[:2]
        return float(-math.sin(yaw_start) * dx + math.cos(yaw_start) * dy)

    def snap(top_height=np.nan):
        r,p,_ = roll_pitch_yaw(data.qpos[3:7])
        if snapshot['sim_time'] and abs(data.time - snapshot['sim_time'][-1]) < 1e-12:
            return
        snapshot['qpos'].append(data.qpos.copy())
        snapshot['qvel'].append(data.qvel.copy())
        snapshot['sim_time'].append(float(data.time))
        snapshot['joint_torque'].append(data.ctrl.copy())
        snapshot['yaw_cmd'].append(float(command_vec[2]))
        snapshot['knee_extension_rad'].append(float(current_extension))
        snapshot['visual_top_height_m'].append(float(top_height))
        snapshot['roll_deg'].append(r)
        snapshot['pitch_deg'].append(p)
        snapshot['cross_track_m'].append(cross_track())

    snap()
    manager = mujoco_viewer.launch_passive(model,data) if viewer else contextlib.nullcontext(None)
    with manager as view:
        for step in range(int(round(seconds/model.opt.timestep))):
            if viewer and not view.is_running():
                stop = 'viewer closed'
                break
            if step % ss.DECIMATION == 0:
                yaw = roll_pitch_yaw(data.qpos[3:7])[2]
                err = math.degrees(math.atan2(math.sin(yaw - yaw_start), math.cos(yaw-yaw_start)))
                with torch.inference_mode():
                    yc, _ = yaw5.neural_command(yaw_net, [err])
                    command_vec[2] = float(np.clip(yc[0].item(), -yaw_cap, yaw_cap))
                    if mode == 'neural':
                        feat = height_features(float(data.qpos[2]),float(data.qvel[2]),target_height)
                        ext, rate = height_net(feat)
                        desired_extension = float(ext.item())
                        last_height_spike_rate = float(rate)
                    else:
                        desired_extension = {'baseline':0.,'fixed16':.16,'fixed24':.24}[mode]
                # A gradual physical target transition for ALL modes: avoids a jump at t=0.
                change = float(np.clip(desired_extension-current_extension,
                                       -EXTENSION_MAX_CHANGE_PER_STEP, EXTENSION_MAX_CHANGE_PER_STEP))
                current_extension = float(np.clip(current_extension+change, 0, MAX_KNEE_EXTENSION))
                phase = (phase + command_vec[4]*model.opt.timestep*ss.DECIMATION)%1.0
                obs = ss.build_obs(data,command_vec,prev_action,last_action,phase)
                history.append(obs)
                h = np.stack(history).astype(np.float32)
                actor,stats = base
                mx,sx,my,sy = stats
                x350 = ss.local_features(h[None])[0]
                inp = np.clip((x350-mx)/sx,-5,5).astype(np.float32)
                with torch.inference_mode():
                    raw,_ = actor(torch.from_numpy(inp[None]))
                action = raw.squeeze(0).cpu().numpy()*sy+my
                if action.shape != (12,) or not np.isfinite(action).all() or np.max(np.abs(action))>12:
                    stop = 'invalid SNN motor output'
                    break
                action = np.clip(action,-6,6).astype(np.float64)
                prev_action,last_action = last_action.copy(),action.copy()
                target = ss.DEFAULT_JOINT_POS + last_action*ss.ACTION_SCALE + extension_offsets(current_extension)
                nd += 1

            torque = ss.KP*(target-data.qpos[7:19])-ss.KD*data.qvel[6:18]
            clip_steps += int(np.any((torque<torque_lo)|(torque>torque_hi)))
            data.ctrl[:] = np.clip(torque,torque_lo,torque_hi)
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop='nonfinite simulation state';break
            nsteps += 1
            r,p,y = roll_pitch_yaw(data.qpos[3:7])
            pitch_sq += p*p
            roll_sq += r*r
            min_root_z = min(min_root_z,float(data.qpos[2]))
            peak_yaw=max(peak_yaw,abs(math.degrees(math.atan2(math.sin(y-yaw_start), math.cos(y-yaw_start)))))
            peak_cross=max(peak_cross,abs(cross_track()))
            if min_root_z < .18:
                stop='low torso / possible fall';break
            if step % ss.DECIMATION == 0:
                measured_top = np.nan
                if step % 50 == 0 and data.time >= .75:
                    # Sample REAL rendered geometry in the exact simulation.
                    mujoco.mj_forward(model,data)
                    _, upper_bounds=visual_bounds(model,data)
                    measured_top=float(upper_bounds[2])
                    visual_heights.append(measured_top)
                    root_heights.append(float(data.qpos[2]))
                    raw_extensions.append(current_extension)
                snap(measured_top)
                if viewer and view.is_running():
                    view.cam.lookat[:] = data.qpos[:3]
                    view.sync()
                    time.sleep(model.opt.timestep*ss.DECIMATION)
        snap()
        if viewer and view.is_running():
            view.sync()

    dx,dy=map(float,data.qpos[:2]-start[:2])
    full=bool(stop=='completed' and float(data.time)>=seconds-.002)
    pass15=bool(full and dx>=1.5 and abs(dy)<=.25 and peak_yaw<=15 and min_root_z>=.18)
    row={
        'mode':mode,'seed':int(seed),'command_mps':float(command),'duration_s':round(float(data.time),4),
        'forward_m':round(dx,5),'sideways_m':round(dy,5),'peak_cross_track_m':round(peak_cross,5),
        'max_abs_yaw_deg':round(peak_yaw,4),'pitch_rms_deg':round(math.sqrt(pitch_sq/max(nsteps,1)),4),
        'roll_rms_deg':round(math.sqrt(roll_sq/max(nsteps,1)),4),'minimum_root_height_m':round(min_root_z,5),
        'mean_root_height_after_075s_m':round(float(np.mean(root_heights)),5) if root_heights else None,
        'mean_visual_top_height_after_075s_m':round(float(np.mean(visual_heights)),5) if visual_heights else None,
        'visual_top_height_std_m':round(float(np.std(visual_heights)),5) if visual_heights else None,
        'mean_abs_top_height_error_from_40cm_m':round(float(np.mean(np.abs(np.array(visual_heights)-.40))),5) if visual_heights else None,
        'mean_extension_after_075s_rad':round(float(np.mean(raw_extensions)),4) if raw_extensions else None,
        'height_neural_spike_rate_last':round(last_height_spike_rate,5),
        'torque_clip_fraction':round(clip_steps/max(nsteps,1),5),
        'pass_15deg':pass15,'pass_10deg':bool(pass15 and peak_yaw<=10),'pass_5deg':bool(pass15 and peak_yaw<=5),
        'stop':stop,
    }
    output=OUT/tag/'replays'/f'{mode}_v{command:.2f}_seed{seed}.npz'
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output.rename(output.with_name(output.stem+'__archived_'+stamp+'.npz'))
    np.savez_compressed(output, **{k:np.asarray(v) for k,v in snapshot.items()},summary_json=np.asarray(json.dumps(row)))
    row['replay']=str(output)
    return row


def test(args):
    from go2_first_terrain_pilot import build_terrain
    torch.set_num_threads(min(4, torch.get_num_threads()))
    frozen=load_frozen('neural' in args.modes)
    model=build_terrain(0.0)
    rows=[]
    for speed in args.speeds:
        for seed in args.seeds:
            for mode in args.modes:
                r=rollout(mode,model,seed,speed,args.seconds,frozen,args.tag,viewer=args.viewer)
                rows.append(r)
                top=r['mean_visual_top_height_after_075s_m']
                print(f'{mode:8s} seed={seed} v={speed:.2f} | '
                      f't={r["duration_s"]:.2f}s X={r["forward_m"]:.3f}m '
                      f'top={top if top is not None else float("nan"):.3f}m '
                      f'pitchRMS={r["pitch_rms_deg"]:.2f}° rollRMS={r["roll_rms_deg"]:.2f}° '
                      f'yaw={r["max_abs_yaw_deg"]:.2f}° Y={r["sideways_m"]:+.3f}m '
                      f'PASS10={r["pass_10deg"]} stop={r["stop"]}',flush=True)
    out=OUT/args.tag
    _save_csv(rows,out/'episodes.csv')
    summary=[]
    for speed in args.speeds:
        for mode in args.modes:
            g=[r for r in rows if r['mode']==mode and r['command_mps']==speed]
            def avg(k):
                values = [float(r[k]) for r in g if r[k] is not None]
                return round(float(np.mean(values)),5) if values else None
            summary.append({'mode':mode,'speed_mps':speed,'n':len(g),
                'pass_10_count':sum(r['pass_10deg'] for r in g),
                'pass_15_count':sum(r['pass_15deg'] for r in g),
                'mean_visual_top_height_m':avg('mean_visual_top_height_after_075s_m'),
                'mean_abs_top_error_m':avg('mean_abs_top_height_error_from_40cm_m'),
                'mean_pitch_rms_deg':avg('pitch_rms_deg'),'mean_roll_rms_deg':avg('roll_rms_deg'),
                'mean_peak_yaw_deg':avg('max_abs_yaw_deg'),
                'worst_peak_yaw_deg':max(r['max_abs_yaw_deg'] for r in g),
                'mean_abs_final_y_m':round(float(np.mean([abs(r['sideways_m']) for r in g])),5),
                'mean_peak_cross_track_m':avg('peak_cross_track_m'),
                'mean_torque_clip_fraction':avg('torque_clip_fraction'),
                'mean_forward_m':avg('forward_m')})
    _save_json(summary,out/'summary.json')
    import go2_snn_self_steering as ss
    import go2_yaw5_fasttrack as yaw5
    manifest={'gait':{'path':str(ss.BASE_WEIGHTS),'sha256':_hash(ss.BASE_WEIGHTS)},
              'yaw':{'path':str(yaw5.NEW_WEIGHTS),'sha256':_hash(yaw5.NEW_WEIGHTS)},
              'standing_static_reference':{'root_z_m':STANDING_REFERENCE_ROOT_Z,'mesh_top_m':.4038},
              'neural_height_checkpoint':({'path':str(POSTURE_CHECKPOINT),'sha256':_hash(POSTURE_CHECKPOINT)}
                                           if 'neural' in args.modes else None),
              'scope':'flat MuJoCo, frozen V6 gait and spiking yaw, static extension or analytic-distilled spiking height head, low-level PD remains'}
    _save_json(manifest,out/'manifest.json')
    print('Saved CSV:',out/'episodes.csv',flush=True)
    print('Saved summary:',out/'summary.json',flush=True)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('train',help='Train new spiking height head from offline feedback labels')
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--target-root-z',type=float,default=STANDING_REFERENCE_ROOT_Z)
    t=sub.add_parser('test',help='Paired Go2 walking experiments using frozen gait and yaw SNNs')
    t.add_argument('--modes',nargs='+',choices=MODES,default=['baseline','fixed16','fixed24'])
    t.add_argument('--seeds',nargs='+',type=int,default=[7601,7602,7603])
    t.add_argument('--speeds',nargs='+',type=float,default=[.45])
    t.add_argument('--seconds',type=float,default=8.0)
    t.add_argument('--tag',default='development')
    t.add_argument('--viewer',action='store_true')
    args=ap.parse_args()
    if args.stage=='train':
        if not (1<=args.epochs<=100 and 1e-5<=args.lr<=.02 and 16<=args.batch<=512 and .27<=args.target_root_z<=.34):
            ap.error('Training args out of supported range')
        train(args.epochs,args.lr,args.batch,args.target_root_z)
    else:
        if not (1<=args.seconds<=12) or any(not .25<=v<=.65 for v in args.speeds):
            ap.error('Test seconds 1..12; speeds 0.25..0.65 m/s')
        if not args.seeds or len(args.seeds)>15 or len(set(args.seeds))!=len(args.seeds):
            ap.error('Use one to fifteen unique seeds')
        if not args.tag.replace('_','').replace('-','').isalnum() or len(args.tag)>40:
            ap.error('Tag must be simple alphanumeric/underscore/hyphen, length <=40')
        if args.viewer and len(args.modes)*len(args.seeds)*len(args.speeds)!=1:
            ap.error('--viewer requires exactly one policy, seed and speed')
        test(args)


if __name__=='__main__':
    main()
