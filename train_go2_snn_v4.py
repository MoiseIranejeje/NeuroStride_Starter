"""NeuroStride Go2 V4: fine-tune spiking V3 on real ANN steering demonstrations.

- Preserves V3 architecture and normalization: 350 input features, 2x128 LIF.
- Adds teacher yaw left/right/heading-hold behavior; retains older walking + DAgger.
- Training is supervised behavior cloning, NOT reinforcement learning.
- Holdout is a related teacher seed for CHECKPOINT SELECTION, not final test.
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
from torch.utils.data import DataLoader, TensorDataset

from train_go2_snn_v1 import Go2SpikingActor
from train_go2_snn_v3 import features_v3

ROOT = Path(__file__).resolve().parent
V3 = ROOT / 'results/go2_snn_v3/go2_snn_v3_best.pt'
STEERING = ROOT / 'results/go2_steering_v4/steering_teacher.npz'
WALKING = ROOT / 'results/go2_teacher/go2_teacher_pilot.npz'
DAGGER = ROOT / 'results/go2_dagger_v2/dagger_rollouts.npz'
OUTPUT = ROOT / 'results/go2_snn_v4'
SEED = 20261117


def read_examples(path):
    with np.load(path, allow_pickle=False) as f:
        x, y, episode = features_v3(f)
        extras = {k: f[k].copy() for k in ('seed', 'mode') if k in f.files}
    return x, np.clip(y, -6.0, 6.0), episode, extras


def run(epochs=40, lr=0.0001, batch_size=128):
    for path in (V3, STEERING, WALKING, DAGGER):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required input: {path}')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    xs, ys, _, metadata = read_examples(STEERING)
    xw, yw, epw, _ = read_examples(WALKING)
    xd, yd, epd, _ = read_examples(DAGGER)
    seed = metadata.get('seed')
    mode = metadata.get('mode')
    if seed is None or mode is None or xs.shape[1] != 350:
        raise ValueError('Expected V4 steering dataset with 350 input features and seed/mode metadata')
    steer_train = seed == 4101
    steer_val = seed == 4102
    if steer_train.sum() != 1600 or steer_val.sum() != 1600:
        raise ValueError('Expected four 400-sample runs per steering seed')
    if set(np.unique(mode)) != {'straight', 'turn_left', 'turn_right', 'heading_hold'}:
        raise ValueError('Expected all four steering behaviors')

    # Keep old forward walking skills and SNN-state recovery examples.
    # Prior SNN V3 used the original teacher command=0.50 as a diagnostic;
    # exclude it from new updates too. It is NOT an independent test.
    keep_walk = epw != 2
    keep_dagger = epd != 5
    xtrain = np.concatenate((xs[steer_train], xw[keep_walk], xd[keep_dagger]))
    ytrain = np.concatenate((ys[steer_train], yw[keep_walk], yd[keep_dagger]))
    weights = np.concatenate((
        np.full(steer_train.sum(), 2.5, np.float32),
        np.ones(keep_walk.sum(), np.float32),
        np.full(keep_dagger.sum(), 1.5, np.float32),
    ))

    ckpt = torch.load(V3, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    if int(cfg['feature_dim']) != 350:
        raise ValueError('Not a 350-feature V3 checkpoint')
    net = Go2SpikingActor(
        feature_dim=350,
        population=int(cfg['population']),
        hidden=int(cfg['hidden']),
        steps=int(cfg['steps']),
    )
    net.load_state_dict(ckpt['model_state'])
    mu_x = ckpt['feature_mean'].numpy().copy()
    sd_x = ckpt['feature_std'].numpy().copy()
    mu_y = ckpt['target_mean'].numpy().copy()
    sd_y = ckpt['target_std'].numpy().copy()

    def norm_x(a):
        return torch.from_numpy(np.clip((a - mu_x) / sd_x, -5, 5).astype(np.float32))

    def norm_y(a):
        return torch.from_numpy(((a - mu_y) / sd_y).astype(np.float32))

    train_ds = TensorDataset(norm_x(xtrain), norm_y(ytrain), torch.from_numpy(weights))
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_x = norm_x(xs[steer_val])
    val_y = torch.from_numpy(ys[steer_val])
    old_x = norm_x(xw[~keep_walk])
    old_y = torch.from_numpy(yw[~keep_walk])
    scale = torch.from_numpy(sd_y)
    offset = torch.from_numpy(mu_y)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    history = []
    best_score = float('inf')
    best_epoch = None

    def measure(epoch, train_loss):
        net.eval()
        with torch.inference_mode():
            pred, activity = net(val_x)
            predicted = pred * scale + offset
            rmse = float((predicted - val_y).square().mean().sqrt())
            # Diagnostics on all four steering behaviors, not just straight.
            by_mode = {}
            for kind in ('straight', 'turn_left', 'turn_right', 'heading_hold'):
                m = torch.from_numpy((mode[steer_val] == kind))
                by_mode[kind] = float((predicted[m] - val_y[m]).square().mean().sqrt())
            old_pred, _ = net(old_x)
            old_rmse = float(((old_pred * scale + offset) - old_y).square().mean().sqrt())
        # Protect old straight-walking action imitation against collapse.
        score = rmse + 2.0 * max(0., old_rmse - 0.15)
        row = dict(epoch=epoch, train_weighted_loss=train_loss,
                   val_steering_4102_rmse=rmse,
                   old_walking_050_rmse=old_rmse,
                   selection_score=score, hidden_spike_rate=float(activity))
        row.update({f'val_{key}_rmse': value for key, value in by_mode.items()})
        history.append(row)
        return row

    baseline = measure(0, float('nan'))
    print(f'Input: {len(xtrain)} training samples = {int(steer_train.sum())} steering + '
          f'{int(keep_walk.sum())} older walking + {int(keep_dagger.sum())} DAgger')
    print('Validation: 1600 teacher steps from related seed 4102 (NOT independent test).')
    print(f'V3 baseline steering RMSE: {baseline["val_steering_4102_rmse"]:.5f}, '
          f'old-walking diagnostic: {baseline["old_walking_050_rmse"]:.5f}')

    def save_best(row):
        torch.save({
            'model_state': {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            'feature_mean': ckpt['feature_mean'].clone(),
            'feature_std': ckpt['feature_std'].clone(),
            'target_mean': ckpt['target_mean'].clone(),
            'target_std': ckpt['target_std'].clone(),
            'config': {**cfg, 'seed': SEED, 'v4_training': 'supervised steering BC + walking + DAgger',
                       'steering_train_seed': 4101, 'steering_validation_seed': 4102},
            'epoch': row['epoch'], 'source': 'V3 frozen initialization, fine-tuned V4',
            'val_steering_4102_rmse': row['val_steering_4102_rmse'],
            'old_walking_050_rmse': row['old_walking_050_rmse'],
        }, OUTPUT / 'go2_snn_v4_best.pt')

    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for xb, yb, weight in loader:
            opt.zero_grad(set_to_none=True)
            pred, spikes = net(xb)
            per = nn.functional.smooth_l1_loss(pred, yb, reduction='none', beta=0.5).mean(1)
            loss = (per * weight).sum() / weight.sum() + 0.0001 * spikes
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach()))
        row = measure(epoch, float(np.mean(losses)))
        if row['selection_score'] < best_score:
            best_score, best_epoch = row['selection_score'], epoch
            save_best(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'Epoch {epoch:02d} | train {row["train_weighted_loss"]:.4f} | '
                  f'steering RMSE {row["val_steering_4102_rmse"]:.5f} | '
                  f'old-walking RMSE {row["old_walking_050_rmse"]:.5f}', flush=True)

    with (OUTPUT / 'training_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    best = next(row for row in history if row['epoch'] == best_epoch)
    summary = {
        'training_type': 'Supervised SNN imitation/DAgger, NOT PPO',
        'checkpoint_source': str(V3.relative_to(ROOT)),
        'training_samples': len(xtrain),
        'steering_train_seed': 4101, 'steering_validation_seed': 4102,
        'validation_samples': int(steer_val.sum()),
        'epoch_selected': best_epoch,
        'v3_start_val_steering_rmse': baseline['val_steering_4102_rmse'],
        'v4_best_val_steering_rmse': best['val_steering_4102_rmse'],
        'v3_start_old_walking_rmse': baseline['old_walking_050_rmse'],
        'v4_best_old_walking_rmse': best['old_walking_050_rmse'],
        'per_behavior_held_seed_metrics': {k: v for k, v in best.items() if k.startswith('val_')},
        'note': 'Selection seed 4102 is very similar to training seed 4101; offline RMSE is not closed-loop ability.',
    }
    (OUTPUT / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print('\n=== SNN V4 OFFLINE STEERING TRAINING ===')
    print(json.dumps(summary, indent=2))
    print('Next: evaluate V3 and V4 teacher-free on new MuJoCo starts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 0 < args.lr <= 0.001:
        parser.error('epochs should be 1..100 and learning rate in (0,0.001]')
    run(args.epochs, args.lr)
