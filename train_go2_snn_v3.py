"""NeuroStride Go2 SNN V3: time-aware LIF behavior cloning + DAgger fine-tuning.

Requires authentic results/go2_teacher/go2_teacher_pilot.npz and
results/go2_dagger_v2/dagger_rollouts.npz and V2 best checkpoint.

Innovation relative to V2: preserve the SAME first 140 features, append
70-dimensional observations lagged by 1, 5 and 15 policy frames to retain
motion phase and accelerations. Inflate the original LIF input layer with
ZERO weights for new inputs; at epoch zero it is numerically V2 itself.

This is not PPO/RL. It is supervised SNN teacher imitation on recorded data.
Even good action-prediction loss does not prove stable locomotion.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_go2_snn_v1 import Go2SpikingActor, features_from_npz

ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / 'results/go2_teacher/go2_teacher_pilot.npz'
DAGGER = ROOT / 'results/go2_dagger_v2/dagger_rollouts.npz'
V2 = ROOT / 'results/go2_snn_v2/go2_snn_v2_best.pt'
OUTPUT = ROOT / 'results/go2_snn_v3'
SEED = 20261116
# A specific DAgger trajectory reserved for *V3 model selection*.
# V2 used it in training, so it is NOT a model-comparison independent test.
VAL_DAGGER_EPISODE = 5
FEATURE_ORDER = ['current_70', 'history_mean_70', 'previous_1_70', 'previous_5_70', 'previous_15_70']


def features_v3(data):
    x2, y, ep = features_from_npz(data)  # validates obs/history alignment and finite values
    frames = np.asarray(data['history'], dtype=np.float32).reshape(-1, 30, 70)
    x3 = np.concatenate([x2, frames[:, -2], frames[:, -6], frames[:, -16]], axis=1)
    if x3.shape[1] != 350:
        raise RuntimeError('Unexpected V3 feature count')
    return x3.astype(np.float32), y, ep


def features_v3_online(frames):
    """frames: [30,70], current observation is frames[-1]."""
    if frames.shape != (30, 70):
        raise ValueError(f'Expected [30,70] observation history, got {frames.shape}')
    return np.concatenate([frames[-1], frames.mean(axis=0), frames[-2], frames[-6], frames[-16]]).astype(np.float32)


def v3_from_v2(v2_ckpt, x3_train):
    c = v2_ckpt['config']
    if int(c['feature_dim']) != 140:
        raise ValueError('Expected source V2 checkpoint with exactly 140 features')
    model = Go2SpikingActor(feature_dim=350, population=int(c['population']),
                             hidden=int(c['hidden']), steps=int(c['steps']))
    old = Go2SpikingActor(feature_dim=140, population=int(c['population']),
                           hidden=int(c['hidden']), steps=int(c['steps']))
    old.load_state_dict(v2_ckpt['model_state'])
    with torch.no_grad():
        model.encoder.weight.zero_()
        # First 140 features and their deterministic population phases are
        # UNCHANGED. This guarantees exact V2 inference before fine-tuning.
        model.encoder.weight[:, :140*int(c['population'])].copy_(old.encoder.weight)
        model.encoder.bias.copy_(old.encoder.bias)
        model.recurrent.load_state_dict(old.recurrent.state_dict())
        model.readout.load_state_dict(old.readout.state_dict())
    mu_x = np.concatenate([v2_ckpt['feature_mean'].cpu().numpy(), x3_train[:, 140:].mean(axis=0)])
    sd_x = np.concatenate([v2_ckpt['feature_std'].cpu().numpy(), np.maximum(x3_train[:, 140:].std(axis=0), 0.05)])
    mu_y = v2_ckpt['target_mean'].cpu().numpy().copy()
    sd_y = v2_ckpt['target_std'].cpu().numpy().copy()
    if mu_x.shape != (350,):
        raise ValueError('Normalization dimension mismatch')
    # Internal proof: in all circumstances V2==warm-started V3 for equivalent
    # original 140 features despite new extra time-ordered history features.
    probe = np.random.default_rng(SEED).normal(size=(4,350)).astype(np.float32)
    norm2 = np.clip((probe[:, :140]-mu_x[:140])/sd_x[:140],-5,5)
    norm3 = np.clip((probe-mu_x)/sd_x,-5,5)
    with torch.inference_mode():
        old_pred,_=old(torch.from_numpy(norm2.astype('float32')))
        new_pred,_=model(torch.from_numpy(norm3.astype('float32')))
    maxdiff=float(torch.max(torch.abs(old_pred-new_pred)))
    if maxdiff > 1e-5:
        raise RuntimeError(f'V2→V3 network inflation failed, max action error {maxdiff}')
    print(f'Warm-start exactness verified: max V2/V3 action diff = {maxdiff:.8g}')
    return model,mu_x.astype('float32'),sd_x.astype('float32'),mu_y.astype('float32'),sd_y.astype('float32')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def train(epochs=40, lr=0.00015, batch_size=128):
    if any(not p.exists() for p in [TEACHER,DAGGER,V2]):
        raise FileNotFoundError('Requires the three original input files in results/go2_teacher, results/go2_dagger_v2, results/go2_snn_v2')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    with np.load(TEACHER,allow_pickle=False) as z: xb,yb,eb=features_v3(z)
    with np.load(DAGGER,allow_pickle=False) as z: xd,yd,ed=features_v3(z)
    old_ckpt=torch.load(V2,map_location='cpu',weights_only=True)
    # Keep original expert validation command 0.50 out of supervised updates.
    bc_train=(eb!=2)
    da_train=(ed!=VAL_DAGGER_EPISODE)
    da_val=(ed==VAL_DAGGER_EPISODE)
    if bc_train.sum()!=1800 or da_train.sum()<500 or da_val.sum()<100:
        raise ValueError('Unexpected episode sizes/splits: dataset may have changed')
    xtrain=np.concatenate([xb[bc_train],xd[da_train]])
    ytrain=np.concatenate([yb[bc_train],np.clip(yd[da_train],-6,6)])
    weight=np.concatenate([np.ones(bc_train.sum(),np.float32),np.full(da_train.sum(),2.0,np.float32)])
    model,mu_x,sd_x,mu_y,sd_y=v3_from_v2(old_ckpt,xtrain)
    normx=lambda a: torch.from_numpy(np.clip((a-mu_x)/sd_x,-5,5).astype('float32'))
    normy=lambda a: torch.from_numpy(((a-mu_y)/sd_y).astype('float32'))
    loader=DataLoader(TensorDataset(normx(xtrain),normy(ytrain),torch.from_numpy(weight)),
                      batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    val_x=normx(xd[da_val]); val_y=torch.from_numpy(np.clip(yd[da_val],-6,6).astype('float32'))
    # Diagnostic old teacher state .50, no weight updates. Previously used for
    # V1 selection; NOT an untouched test and not a V3 success criterion.
    diag_x=normx(xb[~bc_train]); diag_y=torch.from_numpy(yb[~bc_train])
    out=OUTPUT;out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    best_score=float('inf');history=[]
    def check(epoch,train_loss):
        model.eval()
        with torch.inference_mode():
            pred,activity=model(val_x)
            pred=pred*torch.from_numpy(sd_y)+torch.from_numpy(mu_y)
            dagger_rmse=float((pred-val_y).square().mean().sqrt())
            p2,_=model(diag_x)
            p2=p2*torch.from_numpy(sd_y)+torch.from_numpy(mu_y)
            normal_rmse=float((p2-diag_y).square().mean().sqrt())
        # Balance recovery improvement and normal-teacher imitation. Both are
        # model-selection diagnostics, not evidence of closed-loop walking.
        score=dagger_rmse+2.0*max(0.,normal_rmse-.13)
        row={'epoch':epoch,'train_weighted_loss':train_loss,
             'val_dagger_ep5_rmse_clipped':dagger_rmse,'reused_expert_050_rmse':normal_rmse,
             'selection_score':score,'hidden_spike_rate':float(activity)}
        history.append(row)
        return row
    baseline=check(0,float('nan'))
    print('V3 = 350 time-aware features; 2 LIF layers, 128 neurons each, 8 spike steps')
    print(f'Train original ANN frames {bc_train.sum()}, DAgger frames {da_train.sum()}, validation DAgger episode5 {da_val.sum()}')
    print('Epoch 0 = functionally exact V2 checkpoint')
    print(f'V2 starting val_dagger5={baseline["val_dagger_ep5_rmse_clipped"]:.4f}, old_050={baseline["reused_expert_050_rmse"]:.4f}')
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for x,y,w in loader:
            opt.zero_grad(set_to_none=True)
            p,spikes=model(x)
            per=nn.functional.smooth_l1_loss(p,y,reduction='none',beta=.5).mean(1)
            loss=(w*per).sum()/w.sum()+.0001*spikes
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
            losses.append(float(loss.detach()))
        row=check(epoch,float(np.mean(losses)))
        if row['selection_score'] < best_score:
            best_score=row['selection_score'];best_epoch=epoch; best_row=row.copy()
            torch.save({
                'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'feature_mean':torch.from_numpy(mu_x.copy()),'feature_std':torch.from_numpy(sd_x.copy()),
                'target_mean':torch.from_numpy(mu_y.copy()),'target_std':torch.from_numpy(sd_y.copy()),
                'config':{'feature_dim':350,'population':int(old_ckpt['config']['population']),
                          'hidden':int(old_ckpt['config']['hidden']),'steps':int(old_ckpt['config']['steps']),
                          'feature_order':FEATURE_ORDER,'lag_frames':[1,5,15],
                          'seed':SEED,'validation_dagger_episode':VAL_DAGGER_EPISODE},
                'epoch':epoch,'source':'V2 weight inflation + time-aware supervised DAgger fine-tuning',
                'val_dagger_ep5_rmse_clipped':row['val_dagger_ep5_rmse_clipped'],
                'reused_expert_050_rmse':row['reused_expert_050_rmse'],
            },out/'go2_snn_v3_best.pt')
        if epoch==1 or epoch%5==0 or epoch==epochs:
            print(f'epoch {epoch:02d} trainloss {row["train_weighted_loss"]:.4f} DAggerRMSE {row["val_dagger_ep5_rmse_clipped"]:.4f} '
                  f'old50RMSE {row["reused_expert_050_rmse"]:.4f} spike_rate {row["hidden_spike_rate"]:.3f}',flush=True)
    with (out/'training_history.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    summary={'source_checkpoint':'results/go2_snn_v2/go2_snn_v2_best.pt',
      'experiment':'V3 new temporal features current+mean+lags 1/5/15, zero-initialized new weights',
      'training':'surrogate-gradient LIF supervised imitation with teacher demonstrations and DAgger corrections',
      'training_samples_expert':int(bc_train.sum()),'training_samples_dagger':int(da_train.sum()),
      'validation_dagger_episode':int(VAL_DAGGER_EPISODE),'validation_dagger_samples':int(da_val.sum()),
      'v2_before_training_dagger_ep5_rmse':baseline['val_dagger_ep5_rmse_clipped'],
      'v2_before_training_old_050_rmse':baseline['reused_expert_050_rmse'],
      'selected_epoch':int(best_epoch),'v3_best_dagger_ep5_rmse':best_row['val_dagger_ep5_rmse_clipped'],
      'v3_best_old_050_rmse':best_row['reused_expert_050_rmse'],
      'selection_score':best_row['selection_score'],
      'source_hashes_sha256':{'teacher':sha256(TEACHER),'dagger':sha256(DAGGER),'v2':sha256(V2)},
      'limitations':'Not an independent closed-loop test. Validation DAgger episode5 was previously used in V2 training; the old .5 teacher set was used to select V1. Must run real V2 vs V3 MuJoCo rollouts on new starts/commands before concluding improvement.'}
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('\n=== V3 OFFLINE TRAINING SUMMARY ===');print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--lr',type=float,default=.00015)
    a=p.parse_args()
    if not(1<=a.epochs<=200): p.error('epochs must be between 1 and 200')
    train(a.epochs,a.lr)

if __name__=='__main__': main()
