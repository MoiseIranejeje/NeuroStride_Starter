r"""NeuroStride V5: same-state steering distillation + paired MuJoCo evaluation.

Train on verified third-party ANN labels; evaluate SNN alone (ANN not queried
inside SNN rollouts). This is supervised SNN, not SNN PPO/RL.

Windows VS Code examples:
  .\.venv\Scripts\python.exe go2_v5_steering.py train --epochs 30
  .\.venv\Scripts\python.exe go2_v5_steering.py test --seeds 5301 5302 5303
  .\.venv\Scripts\python.exe go2_v5_steering.py test --modes snn_v5_feedback --seeds 5301 --viewer
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_go2_snn_v1 import Go2SpikingActor
from train_go2_snn_v3 import features_v3
from go2_collect_teacher import COMMAND_SCALES, POLICY_DIR, verify_checkpoints

ROOT = Path(__file__).resolve().parent
V4 = ROOT / 'results/go2_snn_v4/go2_snn_v4_best.pt'
STEERING = ROOT / 'results/go2_steering_v4/steering_teacher.npz'
WALKING = ROOT / 'results/go2_teacher/go2_teacher_pilot.npz'
DAGGER = ROOT / 'results/go2_dagger_v2/dagger_rollouts.npz'
OUT = ROOT / 'results/go2_snn_v5'
WEIGHTS = OUT / 'go2_snn_v5_best.pt'
YAWS = np.array([-0.20, 0.00, 0.20], dtype=np.float32)
SEED = 20261117


def make_model(checkpoint):
    c = checkpoint['config']
    if int(c['feature_dim']) != 350:
        raise ValueError('Expected 350-feature V3/V4 SNN')
    model = Go2SpikingActor(
        feature_dim=350, population=int(c['population']),
        hidden=int(c['hidden']), steps=int(c['steps'])
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    stats = [checkpoint[k].detach().cpu().numpy().astype(np.float32)
             for k in ('feature_mean','feature_std','target_mean','target_std')]
    return model, stats


def counterfactual_inputs(features, history):
    """Change only the LATEST yaw command at exactly the same robot state.

    The 70-d observation is gravity(3) followed by 15 scaled commands.
    Thus yaw command sits at index 3+2=5. Its scale is 0.25.
    History features (350d) are [current,mean,lag1,lag5,lag15].
    """
    if features.ndim != 2 or features.shape[1] != 350 or history.shape[1:] != (30,70):
        raise ValueError('Malformed steering feature/history shapes')
    yaw_scaled = YAWS * float(COMMAND_SCALES[2])
    old_yaw = history[:, -1, 5].copy()
    x = np.repeat(features[:,None,:], 3, axis=1)
    x[:,:,5] = yaw_scaled[None,:]
    x[:,:,75] += (yaw_scaled[None,:] - old_yaw[:,None]) / 30.0
    return x.astype(np.float32)


def label_same_state_steering(history, teacher_body, teacher_adapt, batch_size=96):
    """ANN labels for yaw=-.2,0,+.2 on each identical recorded history."""
    labels = np.empty((len(history), 3, 12), np.float32)
    for lo in range(0, len(history), batch_size):
        hi = min(lo + batch_size, len(history))
        triple = np.repeat(history[lo:hi,None,:,:], 3, axis=1)
        triple[:,:,-1,5] = YAWS[None,:] * float(COMMAND_SCALES[2])
        flat = torch.from_numpy(triple.reshape(-1, 2100).copy())
        with torch.inference_mode():
            latent = teacher_adapt(flat)
            predictions = teacher_body(torch.cat([flat,latent], dim=1))
        raw = predictions.detach().cpu().numpy().reshape(-1,3,12)
        if not np.isfinite(raw).all() or np.max(np.abs(raw)) > 12:
            raise ValueError('Unexpected teacher action in yaw counterfactual labels')
        labels[lo:hi] = np.clip(raw,-6,6)
        if (lo // batch_size) % 10 == 0:
            print(f'  Counterfactual expert labels: {hi}/{len(history)}', flush=True)
    return labels


def load_old_data():
    with np.load(WALKING, allow_pickle=False) as f:
        w_x,w_y,w_episode = features_v3(f)
    with np.load(DAGGER, allow_pickle=False) as f:
        d_x,d_y,d_episode = features_v3(f)
    # Previous episode selections retained; old action labels clipped to legal range.
    x = np.concatenate([w_x[w_episode!=2],d_x[d_episode!=5]]).astype(np.float32)
    y = np.clip(np.concatenate([w_y[w_episode!=2],d_y[d_episode!=5]]),-6,6)
    diagnostic_x = w_x[w_episode == 2].astype(np.float32)
    diagnostic_y = np.clip(w_y[w_episode == 2], -6, 6).astype(np.float32)
    return x, y.astype(np.float32), diagnostic_x, diagnostic_y


def train(epochs,lr):
    for p in (V4,STEERING,WALKING,DAGGER):
        if not p.is_file(): raise FileNotFoundError(f'Missing input: {p}')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    verify_checkpoints()
    teacher_body = torch.jit.load(str(POLICY_DIR/'body_latest.jit'),map_location='cpu').eval()
    teacher_adapt = torch.jit.load(str(POLICY_DIR/'adaptation_module_latest.jit'),map_location='cpu').eval()
    with np.load(STEERING,allow_pickle=False) as f:
        x_steer,_,_=features_v3(f)
        history=f['history'].astype(np.float32).reshape(-1,30,70)
        ids=f['seed'].copy()
        assert len(x_steer)==3200 and set(np.unique(ids))=={4101,4102}
    x_cf=counterfactual_inputs(x_steer,history)
    print('Generating counterfactual expert actions at SAME robot states...')
    y_cf=label_same_state_steering(history,teacher_body,teacher_adapt)
    del teacher_body,teacher_adapt
    ckpt=torch.load(V4,map_location='cpu',weights_only=True)
    net, (mx,sx,my,sy)=make_model(ckpt)
    x_old,y_old,x_diagnostic,y_diagnostic=load_old_data()
    # Whole seed 4102 reserved for checkpoint SELECTION, not final test.
    train_ids=np.where(ids==4101)[0]; val_ids=np.where(ids==4102)[0]
    normx=lambda a: np.clip((a-mx)/sx,-5,5).astype(np.float32)
    normy=lambda a: ((a-my)/sy).astype(np.float32)
    x_tr=torch.from_numpy(normx(x_cf[train_ids]))
    y_tr=torch.from_numpy(normy(y_cf[train_ids]))
    x_val=torch.from_numpy(normx(x_cf[val_ids]))
    y_val=torch.from_numpy(y_cf[val_ids])
    old_x=torch.from_numpy(normx(x_old)); old_y=torch.from_numpy(normy(y_old))
    old_diag_x=torch.from_numpy(normx(x_diagnostic))
    old_diag_y=torch.from_numpy(y_diagnostic)
    sy_t=torch.from_numpy(sy); my_t=torch.from_numpy(my)
    rng=np.random.default_rng(SEED)
    OUT.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    best=float('inf'); history_rows=[]
    def measure(epoch):
        net.eval()
        with torch.inference_mode():
            p,_=net(x_val.reshape(-1,350))
            pred=(p.reshape(-1,3,12)*sy_t+my_t)
            action_rmse=float((pred-y_val).square().mean().sqrt())
            d_pred=pred[:,2]-pred[:,0]
            d_label=y_val[:,2]-y_val[:,0]
            delta_rmse=float((d_pred-d_label).square().mean().sqrt())
            cos=nn.functional.cosine_similarity(d_pred,d_label,dim=1)
            mean_cosine=float(cos.mean())
            old_predictions=[]
            for j in range(0,len(old_diag_x),128):
                old_p,_=net(old_diag_x[j:j+128])
                old_predictions.append(old_p*sy_t+my_t)
            old_diag_rmse=float((torch.cat(old_predictions)-old_diag_y).square().mean().sqrt())
        # Select for steering change accuracy while monitoring action fidelity.
        score=delta_rmse+0.25*action_rmse+1.0*max(0.0,old_diag_rmse-0.20)
        return dict(epoch=epoch,teacher_action_rmse=action_rmse,
                    steering_delta_rmse=delta_rmse,steering_delta_cosine=mean_cosine,
                    old_sample_action_rmse=old_diag_rmse,selection_score=score)
    start=measure(0)
    best=start['selection_score']
    best_row=start
    # Keep a copy of V4 as the safe no-regression baseline.
    # Only replace it when a later epoch wins on the selection objective.
    torch.save({**ckpt, 'epoch': 0, 'source': 'V4 unchanged: no selected V5 improvement yet'}, WEIGHTS)
    print('V4 starting held-steering delta RMSE',round(start['steering_delta_rmse'],5),
          'delta cosine',round(start['steering_delta_cosine'],4))
    if not np.isfinite(start['selection_score']):raise ValueError('Invalid baseline metrics')
    n=len(x_tr)
    for epoch in range(1,epochs+1):
        net.train(); losses=[]
        order=rng.permutation(n)
        for lo in range(0,n,64):
            take=order[lo:lo+64]
            xb=x_tr[take]; yb=y_tr[take]
            oldtake=rng.integers(0,len(old_x),size=min(64,len(take)))
            optimizer.zero_grad(set_to_none=True)
            pred,spikes=net(xb.reshape(-1,350))
            pred=pred.reshape(-1,3,12)
            base=nn.functional.smooth_l1_loss(pred,yb,beta=.5)
            delta=(pred[:,2]-pred[:,0])-(yb[:,2]-yb[:,0])
            steer=delta.square().mean()
            old_pred,_=net(old_x[oldtake])
            retain=nn.functional.smooth_l1_loss(old_pred,old_y[oldtake],beta=.5)
            loss=base+5.0*steer+0.75*retain+0.0001*spikes
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row=measure(epoch);row['train_loss']=float(np.mean(losses))
        history_rows.append(row)
        if np.isfinite(row['selection_score']) and row['selection_score']<best:
            best=row['selection_score']
            best_row=row
            torch.save({
                'model_state':{k:v.cpu().clone() for k,v in net.state_dict().items()},
                'feature_mean':ckpt['feature_mean'].clone(),
                'feature_std':ckpt['feature_std'].clone(),
                'target_mean':ckpt['target_mean'].clone(),
                'target_std':ckpt['target_std'].clone(),
                'config':{**ckpt['config'],'v5_training':'same-state ANN yaw-difference distillation',
                          'steering_train_seed':4101,'steering_validation_seed':4102},
                'epoch':epoch,
                'source':'V5 learned counterfactual steering-distillation checkpoint',
            },WEIGHTS)
        if epoch==1 or epoch%5==0 or epoch==epochs:
            print(f'Epoch {epoch:02d} | loss={row["train_loss"]:.4f} '
                  f'| yaw-delta RMSE={row["steering_delta_rmse"]:.4f} '
                  f'| cosine={row["steering_delta_cosine"]:.3f} '
                  f'| action RMSE={row["teacher_action_rmse"]:.4f}',flush=True)
    if not WEIGHTS.is_file():raise RuntimeError('No trained checkpoint was saved')
    if best_row['epoch'] == 0:
        print('WARNING: no V5 epoch beat V4 selection baseline; saved model is V4 fallback.')
    with (OUT/'training_history.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(history_rows[0]));w.writeheader();w.writerows(history_rows)
    report=dict(training='SNN supervised counterfactual yaw distillation, not RL',
                train_teacher_states=len(train_ids),validation_teacher_states=len(val_ids),
                yaw_probes_rad_s=YAWS.tolist(),baseline=start,best=best_row,
                warning='Selection seed 4102 is correlated with training; require independent MuJoCo closed-loop tests')
    (OUT/'training_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('\nSaved selected checkpoint (V5, or V4 fallback if no gain):',WEIGHTS)
    print(json.dumps(report,indent=2))


def test(args):
    """Run paired, teacher-free SNN simulations in real MuJoCo."""
    import mujoco_menagerie as mm
    import go2_heading_benchmark as hb
    from go2_compare_v3 import load_actor
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if not WEIGHTS.is_file():raise FileNotFoundError(f'Train V5 first: {WEIGHTS}')
    c=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    v5=make_model(c)
    v5_is_trained = str(c.get('source','')).startswith('V5 learned')
    if not v5_is_trained:
        print('WARNING: V5 did not beat the offline selection score; the saved checkpoint is unchanged V4.')
    v3=load_actor('v3')
    v4c=torch.load(V4,map_location='cpu',weights_only=True)
    v4=make_model(v4c)
    teacher=hb.load_teacher() if 'ann_feedback' in args.modes else None
    model=mm.load('unitree_go2')
    if model.nu!=12 or abs(model.opt.timestep-.002)>1e-8:
        raise ValueError('Go2 motor count or simulation timestep mismatch')
    folder=OUT/'evaluation';folder.mkdir(parents=True,exist_ok=True)
    hb.RESULT_DIR=folder
    mapping={'snn_v3_feedback':v3,'snn_v4_feedback':v4,
             'snn_v5_zero':v5,'snn_v5_feedback':v5,
             'ann_feedback':None}
    records=[]
    for seed in args.seeds:
        for mode in args.modes:
            print(f'Go2 {mode} seed={seed} forward={args.command} m/s',flush=True)
            row=hb.run(model,mode,mapping[mode],teacher,args.command,seed,args.seconds,
                       gain=.9,max_rate=.2,viewer=args.viewer and mode.startswith('snn_v5'),record=True)
            row['policy_source']=('V5 newly trained SNN' if v5_is_trained else 'V4 unchanged fallback') if mode.startswith('snn_v5') else (
                'V4 trained SNN' if mode.startswith('snn_v4') else (
                'V3 trained SNN' if mode.startswith('snn_v3') else 'Third-party ANN'))
            path=folder/f'{mode}_cmd{args.command:.2f}_seed{seed}.npz'
            with np.load(path,allow_pickle=False) as f:
                vals={k:f[k].copy() for k in f.files}
            vals['summary_json']=np.array(json.dumps(row))
            np.savez_compressed(path,**vals)
            records.append(row)
            print(f'  {row["sim_s"]:.3f}s X={row["forward_m"]:.3f}m '
                  f'max|yaw|={row["max_abs_yaw_deg"]:.2f}° '
                  f'straight={row["passed_straight_heading"]}',flush=True)
    with (folder/'paired_results.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary={mode:{
        'n':sum(r['mode']==mode for r in records),
        'passes':sum(bool(r['passed_straight_heading']) for r in records if r['mode']==mode),
        'mean_forward_m':float(np.mean([r['forward_m'] for r in records if r['mode']==mode])),
        'mean_peak_yaw_deg':float(np.mean([r['max_abs_yaw_deg'] for r in records if r['mode']==mode])),
        'mean_survival_s':float(np.mean([r['sim_s'] for r in records if r['mode']==mode]))
    } for mode in args.modes}
    (folder/'paired_summary.json').write_text(json.dumps({
        'seeds':args.seeds,'forward_cmd':args.command,
        'horizon':args.seconds,'results':summary,
        'note':'Models run closed-loop; ANN is a separate baseline, no teacher in SNN rollouts.'
    },indent=2),encoding='utf8')
    print('\n=== ACTUAL V5 vs V3/V4/ANN ON GO2 ===')
    print(json.dumps(summary,indent=2))
    print('Saved:',folder)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='job',required=True)
    p=sub.add_parser('train',help='Teacher counterfactual steering labels + SNN V5 fit')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--lr',type=float,default=0.00015)
    p=sub.add_parser('test',help='Paired closed-loop flat Go2 evaluation')
    p.add_argument('--seeds',nargs='+',type=int,default=[5301,5302,5303])
    p.add_argument('--command',type=float,default=.45)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--modes',nargs='+',choices=('snn_v3_feedback','snn_v4_feedback',
                    'snn_v5_zero','snn_v5_feedback','ann_feedback'),
                    default=['snn_v3_feedback','snn_v4_feedback','snn_v5_zero',
                             'snn_v5_feedback','ann_feedback'])
    p.add_argument('--viewer',action='store_true')
    a=parser.parse_args()
    if a.job=='train':
        if not (1<=a.epochs<=100 and 0<a.lr<=.001):parser.error('Invalid training settings')
        train(a.epochs,a.lr)
    else:
        if not a.seeds or len(a.seeds)>20 or len(set(a.seeds))!=len(a.seeds):parser.error('Invalid seeds')
        if not (1<=a.seconds<=30 and 0<=a.command<=1):parser.error('Invalid command/horizon')
        test(a)


if __name__=='__main__':main()
