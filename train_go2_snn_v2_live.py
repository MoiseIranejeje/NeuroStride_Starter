r"""NeuroStride: DAgger fine-tuning of *the same spiking* Go2 actor.

This is supervised imitation learning (behavior cloning + on-policy expert labels),
NOT PPO and NOT an end-to-end differentiable physics simulation. The optional
--live mode pauses weight training every N epochs to run the current spiking
checkpoint independently in MuJoCo's live 3-D viewer, then resumes training.

Run from NeuroStride_Starter (keeping this alongside v1 sources and data):
  .\.venv\Scripts\python.exe train_go2_snn_v2_live.py
  .\.venv\Scripts\python.exe train_go2_snn_v2_live.py --live --preview-every 10

Dependencies: torch numpy, existing train_go2_snn_v1.py; and for --live,
muJoCo + mujoco_menagerie + go2_visual_replay.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_go2_snn_v1 import Go2SpikingActor, features_from_npz

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'go2_snn_v2_live'  # live reruns never overwrite supplied v2 checkpoint
V1 = ROOT / 'results' / 'go2_snn_v1' / 'go2_snn_v1_best.pt'
TEACHER = ROOT / 'results' / 'go2_teacher' / 'go2_teacher_pilot.npz'
DAGGER = ROOT / 'results' / 'go2_dagger_v2' / 'dagger_rollouts.npz'
SEED = 20261114


def load_data():
    missing = [str(p) for p in (V1, TEACHER, DAGGER) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing required input(s):\n' + '\n'.join(missing))
    with np.load(TEACHER, allow_pickle=False) as z:
        x_bc, y_bc, ep_bc = features_from_npz(z)
    with np.load(DAGGER, allow_pickle=False) as z:
        x_da, y_da, ep_da = features_from_npz(z)
    if not (set(np.unique(ep_bc)) == {0, 1, 2, 3} and set(np.unique(ep_da)) == set(range(6))):
        raise ValueError('Expected 4 expert and 6 DAgger episodes')
    if len(x_da) < 100:
        raise ValueError('DAgger data unexpectedly short')
    return x_bc, y_bc, ep_bc, x_da, y_da, ep_da


def checkpoint(model, original, epoch, val_dagger, original_val, path):
    # Normalize exactly as v1. Never infer normalization from a val/test split.
    payload = {
        'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'feature_mean': original['feature_mean'].clone(),
        'feature_std': original['feature_std'].clone(),
        'target_mean': original['target_mean'].clone(),
        'target_std': original['target_std'].clone(),
        'config': dict(original['config']),
        'epoch': epoch,
        'source': 'NeuroStride Go2 SNN V2 trained on V1 weights + verified ANN and DAgger labels',
        'dagger_heldout_episode': 3,
        'dagger_validation_rmse_clipped_actions': val_dagger,
        'old_command_050_diagnostic_rmse': original_val,
        'target_clipping_for_teacher_recovery': [-6.0, 6.0],
    }
    torch.save(payload, path)


def run_training(epochs: int, live: bool, preview_every: int):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    x_bc, y_bc, ep_bc, x_da, y_da, ep_da = load_data()
    original = torch.load(V1, map_location='cpu', weights_only=True)
    cfg = original['config']
    model = Go2SpikingActor(
        feature_dim=int(cfg['feature_dim']), population=int(cfg['population']),
        hidden=int(cfg['hidden']), steps=int(cfg['steps']))
    model.load_state_dict(original['model_state'])
    mu_x = original['feature_mean'].cpu().numpy()
    sd_x = original['feature_std'].cpu().numpy()
    mu_y = original['target_mean'].cpu().numpy()
    sd_y = original['target_std'].cpu().numpy()

    # Hold out entire DAgger episode 3 for selecting V2 checkpoints.
    # Original expert episode 2 (0.5 m/s) was already used for V1 checkpoint
    # selection, so its error is only a reused diagnostic, NOT independent test.
    ib = (ep_bc != 2)
    idt = (ep_da != 3)
    idv = (ep_da == 3)
    # Teacher sometimes requests offsets > safety policy's +/-6. Match the
    # actually executable command target, retaining other teacher labels.
    y_da_clip = np.clip(y_da, -6, 6)
    x_train = np.concatenate([x_bc[ib], x_da[idt]])
    y_train = np.concatenate([y_bc[ib], y_da_clip[idt]])
    w_train = np.concatenate([
        np.ones(int(np.sum(ib)), np.float32),
        np.full(int(np.sum(idt)), 2.0, np.float32),
    ])
    x_val = x_da[idv]
    y_val = y_da_clip[idv]
    x_diag = x_bc[~ib]
    y_diag = y_bc[~ib]
    normalize_x = lambda x: np.clip((x-mu_x)/sd_x, -5, 5).astype(np.float32)
    normalize_y = lambda y: ((y-mu_y)/sd_y).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(normalize_x(x_train)),
        torch.from_numpy(normalize_y(y_train)),
        torch.from_numpy(w_train))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_dataset, shuffle=True, batch_size=128, generator=generator)
    val_x = torch.from_numpy(normalize_x(x_val))
    diag_x = torch.from_numpy(normalize_x(x_diag))
    val_targets = torch.from_numpy(y_val)
    diag_targets = torch.from_numpy(y_diag)
    sd_t = torch.from_numpy(sd_y)
    mu_t = torch.from_numpy(mu_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.00005)
    OUT.mkdir(parents=True, exist_ok=True)
    best_rmse = float('inf')
    records = []
    initial_stats = None
    print(f'Go2 SNN V2 | train original ANN frames={int(ib.sum())}, '
          f'DAgger frames={int(idt.sum())}, held-out DAgger episode 3={int(idv.sum())}')
    print('Training the spiking actor via supervised DAgger; --live shows independent checkpoint rollouts')
    for epoch in range(epochs+1):
        if epoch:
            model.train()
            batch_losses = []
            for bx, by, bw in loader:
                optimizer.zero_grad(set_to_none=True)
                out, spikes = model(bx)
                # Smooth L1 is more robust to difficult recovery target outliers.
                per_item = nn.functional.smooth_l1_loss(out, by, reduction='none', beta=0.5).mean(1)
                loss = (per_item*bw).sum()/bw.sum() + 0.0001*spikes
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                batch_losses.append(float(loss.detach()))
        else:
            batch_losses = [float('nan')]

        model.eval()
        with torch.inference_mode():
            val_pred, val_spike = model(val_x)
            val_pred = val_pred * sd_t + mu_t
            val_rmse = float((val_pred-val_targets).square().mean().sqrt())
            diag_pred, diag_spikes = model(diag_x)
            diag_pred = diag_pred * sd_t + mu_t
            diag_rmse = float((diag_pred-diag_targets).square().mean().sqrt())
        row = {'epoch': epoch,
               'weighted_train_loss': float(np.mean(batch_losses)),
               'heldout_dagger_ep3_clipped_action_rmse': val_rmse,
               'reused_command_050_teacher_rmse': diag_rmse,
               'mean_val_spike_rate': float(val_spike)}
        records.append(row)
        if epoch == 0:
            initial_stats = row.copy()
            print(f'baseline V1 | DAgger validation RMSE {val_rmse:.4f}, '
                  f'original 0.50 command diagnostic RMSE {diag_rmse:.4f}')
        if epoch > 0 and val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            checkpoint(model, original, epoch, val_rmse, diag_rmse,
                       OUT/'go2_snn_v2_best.pt')
        if epoch and (epoch % 5 == 0 or epoch == epochs):
            print(f'epoch {epoch:02d} | dagger val RMSE {val_rmse:.4f} | '
                  f'old command .50 RMSE {diag_rmse:.4f} | '
                  f'spike rate {float(val_spike):.3f}', flush=True)
        if live and epoch and epoch % preview_every == 0:
            # Viewer shows current weights; this file is intentionally
            # different from the selected best, for transparent progress.
            current = OUT/'go2_snn_v2_current.pt'
            checkpoint(model, original, epoch, val_rmse, diag_rmse, current)
            print(f'\nLIVE ROLLOUT PREVIEW — epoch {epoch}. This pauses weight updates. '
                  'It is NOT the training episode used for gradients.\n', flush=True)
            subprocess.run([sys.executable, str(ROOT/'go2_visual_replay.py'),
                            'record', '--checkpoint', str(current), '--controller', 'snn',
                            '--command', '0.50', '--seconds', '5', '--viewer', '--no-save'],
                            check=True)
    with (OUT/'training_history.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    summary = {
        'method': 'Supervised DAgger correction; 2-layer LIF SNN, continuous action decoder',
        'trained_from_checkpoint': str(V1),
        'num_train_ANN_teacher_frames': int(ib.sum()),
        'num_train_DAgger_frames': int(idt.sum()),
        'num_validation_DAgger_frames': int(idv.sum()),
        'validation_DAgger_episode': 3,
        'best_epoch': best_epoch,
        'initial_V1_DAgger_val_rmse_clipped': initial_stats['heldout_dagger_ep3_clipped_action_rmse'],
        'best_V2_DAgger_val_rmse_clipped': best_rmse,
        'initial_V1_old_command_050_rmse': initial_stats['reused_command_050_teacher_rmse'],
        'last_V2_old_command_050_rmse': records[-1]['reused_command_050_teacher_rmse'],
        'selection_caveat': 'DAgger episode 3 used for model selection; old command 0.5 used previously for v1 checkpoint selection; neither is a clean final untouched test.',
        'critical_next_step': 'Closed-loop rollouts without ANN teacher; stability may improve or deteriorate despite offline RMSE.'
    }
    (OUT/'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print('=== V2 OFFLINE TRAINING FINISHED ===')
    print(json.dumps(summary, indent=2))
    print('Next: run go2_visual_replay.py record --controller snn --checkpoint '
          'results/go2_snn_v2/go2_snn_v2_best.pt --viewer')


def main():
    global OUT
    p=argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--live', action='store_true', help='Periodically show current SNN checkpoint in live MuJoCo')
    p.add_argument('--preview-every', type=int, default=10)
    p.add_argument('--output-dir', default='results/go2_snn_v2_live',help='Relative folder for your new training run')
    args=p.parse_args()
    if args.epochs < 1 or args.epochs > 200:
        p.error('epochs must be 1..200')
    OUT = ROOT/args.output_dir
    if not OUT.resolve().is_relative_to(ROOT.resolve()):
        p.error('output-dir must be inside NeuroStride_Starter')
    if args.preview_every < 1:
        p.error('preview-every must be >=1')
    run_training(args.epochs, args.live, args.preview_every)


if __name__=='__main__':
    main()
