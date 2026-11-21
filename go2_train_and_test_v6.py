r"""NeuroStride Go2 V6: terrain DAgger fine-tuning + matched MuJoCo test.

Run from Windows VS Code, NeuroStride_Starter root:
  .\.venv\Scripts\python.exe go2_train_and_test_v6.py train --epochs 30
  .\.venv\Scripts\python.exe go2_train_and_test_v6.py test --seeds 5801 5802 --slopes 0 2

The pretrained ANN labels terrain states; only the SNN drives during SNN tests.
This is supervised spiking-network fine-tuning, NOT end-to-end SNN RL.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from go2_v5_steering import WEIGHTS as V5_WEIGHTS, make_model
from train_go2_snn_v3 import features_v3

ROOT = Path(__file__).resolve().parent
TERRAIN_DATA = ROOT / 'results/go2_slope_v6/slope_steering_teacher_and_dagger.npz'
STEERING_DATA = ROOT / 'results/go2_steering_v4/steering_teacher.npz'
WALKING_DATA = ROOT / 'results/go2_teacher/go2_teacher_pilot.npz'
DAGGER_DATA = ROOT / 'results/go2_dagger_v2/dagger_rollouts.npz'
OUT = ROOT / 'results/go2_snn_v6'
V6_WEIGHTS = OUT / 'go2_snn_v6_best.pt'
SEED = 20261119


def load_examples(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required dataset missing: {path}')
    with np.load(path, allow_pickle=False) as data:
        x, y, episode = features_v3(data)
        extras = {key: data[key].copy() for key in ('seed', 'source', 'slope_deg') if key in data.files}
    if x.ndim != 2 or x.shape[1] != 350 or y.shape != (len(x), 12):
        raise ValueError(f'Unexpected feature/target shape from {path}: {x.shape}, {y.shape}')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'Nonfinite features or teacher targets in {path}')
    return x, np.clip(y, -6, 6).astype(np.float32), episode, extras


def teacher_state_measure(net, x, y, mx, sx, my, sy, batch=128):
    """Raw 12-action RMSE for teacher labels at specified visited robot states."""
    if not len(x):
        return float('nan')
    net.eval()
    error, count = 0.0, 0
    with torch.inference_mode():
        for a in range(0, len(x), batch):
            xb = torch.from_numpy(np.clip((x[a:a+batch]-mx)/sx, -5, 5).astype(np.float32))
            prediction, _ = net(xb)
            actual = prediction.cpu().numpy() * sy + my
            difference = actual - y[a:a+batch]
            error += float(np.square(difference, dtype=np.float64).sum())
            count += difference.size
    return float(np.sqrt(error / count))


def train(epochs: int, lr: float, batch: int):
    if not V5_WEIGHTS.is_file():
        raise FileNotFoundError(f'V5 pretrained SNN missing: {V5_WEIGHTS}')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    xt, yt, _, meta = load_examples(TERRAIN_DATA)
    required = ('seed', 'source', 'slope_deg')
    if not all(k in meta for k in required):
        raise ValueError('Terrain dataset missing seed/source/slope_deg fields')
    seed, source, slope = (meta[k] for k in required)
    if not set(np.unique(seed)).issuperset({5701,5702}):
        raise ValueError('Expected terrain data from seeds 5701 (train) and 5702 (selection)')
    if not {'expert', 'student'}.issubset(set(np.unique(source))):
        raise ValueError('Need both expert and student-visited terrain examples')
    train_mask = seed == 5701
    valid_mask = seed == 5702
    if not np.any(train_mask & (source == 'student') & (slope > 1.5)):
        raise ValueError('No uphill V5-visited corrective examples in training set')
    if not np.any(valid_mask & (source == 'student') & (slope > 1.5)):
        raise ValueError('No uphill V5-visited validation examples')
    assert not np.any(train_mask & valid_mask)

    xs, ys, _, ms = load_examples(STEERING_DATA)
    if 'seed' not in ms:
        raise ValueError('Steering data missing original start seed')
    xw, yw, ew, _ = load_examples(WALKING_DATA)
    xd, yd, ed, _ = load_examples(DAGGER_DATA)

    # Train on one entire terrain seed; never randomly mix frames from seed 5702.
    # Retain prior straight gait and turning examples to limit forgetting.
    msteer = ms['seed'] == 4101
    mwalk = ew != 2       # Existing 0.50 m/s diagnostic remains outside updates.
    mdagger = ed != 5     # Preserve prior DAgger selection episode outside updates.
    xtrain = np.concatenate((xt[train_mask], xs[msteer], xw[mwalk], xd[mdagger]), axis=0)
    ytrain = np.concatenate((yt[train_mask], ys[msteer], yw[mwalk], yd[mdagger]), axis=0)
    terrain_weights = np.where(
        slope[train_mask] > 1.5,
        np.where(source[train_mask] == 'student', 5.0, 3.0),
        np.where(source[train_mask] == 'student', 1.5, 1.0)
    ).astype(np.float32)
    weights = np.concatenate((
        terrain_weights,
        np.full(msteer.sum(), 1.25, np.float32),
        np.full(mwalk.sum(), 0.7, np.float32),
        np.full(mdagger.sum(), 1.0, np.float32),
    ))
    assert len(xtrain) == len(ytrain) == len(weights)
    if not np.all(np.isfinite(weights)):
        raise ValueError('Nonfinite training sample weights')

    ckpt = torch.load(V5_WEIGHTS, map_location='cpu', weights_only=True)
    net, stats = make_model(ckpt)
    mx, sx, my, sy = stats
    if xtrain.shape[1] != len(mx) or ytrain.shape[1] != len(my):
        raise RuntimeError('V5 normalization dimensions do not match dataset')
    nx = np.clip((xtrain-mx)/sx, -5, 5).astype(np.float32)
    ny = ((ytrain-my)/sy).astype(np.float32)
    X = torch.from_numpy(nx)
    Y = torch.from_numpy(ny)
    W = torch.from_numpy(weights)

    xv, yv = xt[valid_mask], yt[valid_mask]
    sv, pv = source[valid_mask], slope[valid_mask]
    subsets = {
        'uphill_student': (sv == 'student') & (pv > 1.5),
        'uphill_expert': (sv == 'expert') & (pv > 1.5),
        'flat_student': (sv == 'student') & (np.abs(pv) < 0.1),
        'flat_expert': (sv == 'expert') & (np.abs(pv) < 0.1),
    }
    for name, mask in subsets.items():
        if not np.any(mask):
            raise ValueError(f'Missing validation group {name}')
    # Previously exposed diagnostic: not an untouched final test.
    xold, yold = xw[~mwalk], yw[~mwalk]

    OUT.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    rows = []

    def measure(epoch, train_loss):
        m = {key + '_rmse': teacher_state_measure(net, xv[mask], yv[mask], mx, sx, my, sy)
             for key, mask in subsets.items()}
        m['old_walking_rmse'] = teacher_state_measure(net, xold, yold, mx, sx, my, sy)
        # Prioritize V5-visited slope states, while protecting flat gait skill.
        m['selection_score'] = (
            m['uphill_student_rmse']
            + 0.35 * m['uphill_expert_rmse']
            + 0.35 * m['flat_student_rmse']
            + 0.15 * m['flat_expert_rmse']
            + 2.0 * max(0.0, m['old_walking_rmse'] - baseline_old - 0.04)
        )
        return {'epoch': int(epoch), 'train_loss': float(train_loss), **m}

    baseline_old = teacher_state_measure(net, xold, yold, mx, sx, my, sy)
    base = measure(0, 0.0)  # Epoch zero: loss was not optimized or measured
    best = base
    # The V5 model is protected: an epoch-zero fallback if V6 does not improve.
    fallback = dict(ckpt)
    fallback['source'] = 'V5 unchanged fallback: no V6 selection improvement'
    fallback['epoch'] = 0
    torch.save(fallback, V6_WEIGHTS)
    print(f'TRAIN examples={len(xtrain)} terrain={int(train_mask.sum())} '
          f'old_steer={int(msteer.sum())} old_walk={int(mwalk.sum())} '
          f'old_dagger={int(mdagger.sum())}', flush=True)
    print(f'SELECTION examples={int(valid_mask.sum())}, seed=5702 (related start, NOT a final test)', flush=True)
    print(f'V5 baseline uphill SNN-state RMSE: {base["uphill_student_rmse"]:.5f} '
          f'flat SNN-state RMSE: {base["flat_student_rmse"]:.5f}', flush=True)
    rows.append(base)
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch):
            indices = order[start:start + batch]
            pred, spikes = net(X[indices])
            per = nn.functional.smooth_l1_loss(pred, Y[indices], beta=0.5, reduction='none').mean(1)
            loss = (per * W[indices]).sum() / W[indices].sum() + 1e-4 * spikes
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = measure(epoch, float(np.mean(losses)))
        rows.append(row)
        if np.isfinite(row['selection_score']) and row['selection_score'] < best['selection_score']:
            best = row
            torch.save({
                'model_state': {k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                'feature_mean': ckpt['feature_mean'].detach().cpu().clone(),
                'feature_std': ckpt['feature_std'].detach().cpu().clone(),
                'target_mean': ckpt['target_mean'].detach().cpu().clone(),
                'target_std': ckpt['target_std'].detach().cpu().clone(),
                'config': {**ckpt['config'], 'v6_training': 'supervised slope expert+on-policy DAgger',
                           'terrain_train_seed': 5701, 'terrain_selection_seed': 5702},
                'epoch': epoch,
                'source': 'V6 learned on ANN-labelled uphill and flat examples',
            }, V6_WEIGHTS)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'Epoch {epoch:02d} | uphill student RMSE {row["uphill_student_rmse"]:.5f} '
                  f'| flat student {row["flat_student_rmse"]:.5f} '
                  f'| old gait {row["old_walking_rmse"]:.5f} '
                  f'| score {row["selection_score"]:.5f}', flush=True)

    with (OUT/'training_history.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'training': 'Supervised SNN terrain adaptation (expert + on-policy DAgger), not PPO/RL',
        'train_terrain_seed': 5701,
        'checkpoint_selection_seed': 5702,
        'selection_warning': 'Only 2 nearly-identical seeds; validation is correlated, not independent robustness evidence',
        'train_examples': len(xtrain),
        'selection_examples': int(valid_mask.sum()),
        'baseline_v5': base,
        'best_selected': best,
        'improved_offline': best['epoch'] > 0,
        'checkpoint': str(V6_WEIGHTS),
        'next': 'Run paired V5 vs V6 vs ANN on NEW simulation seeds and grades',
    }
    (OUT/'training_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('\n=== V6 TRAINING RESULT (offline imitation, NOT robot success) ===')
    print(json.dumps(report,indent=2),flush=True)


def test(seeds, slopes, command, seconds, modes, viewer):
    import go2_heading_benchmark as heading
    import go2_v5_upright_sweep as upright
    from go2_first_terrain_pilot import build_terrain, replay_metrics

    if not V5_WEIGHTS.is_file():
        raise FileNotFoundError(f'Missing frozen V5: {V5_WEIGHTS}')
    if 'v6' in modes and not V6_WEIGHTS.is_file():
        raise FileNotFoundError(f'Train V6 first: {V6_WEIGHTS}')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    v5 = make_model(torch.load(V5_WEIGHTS, map_location='cpu', weights_only=True))
    v6 = make_model(torch.load(V6_WEIGHTS, map_location='cpu', weights_only=True)) if 'v6' in modes else None
    if v6 is not None:
        content = torch.load(V6_WEIGHTS, map_location='cpu', weights_only=True)
        if not str(content.get('source','')).startswith('V6 learned'):
            print('WARNING: V6 selected checkpoint is unchanged V5 fallback.',flush=True)
    teacher = heading.load_teacher() if 'ann' in modes else None
    folder = OUT/'evaluation'
    folder.mkdir(parents=True,exist_ok=True)
    upright.OUT = folder/'temporary_snn_replays'
    heading.RESULT_DIR = folder/'temporary_ann_replays'
    results = []
    for slope in slopes:
        model = build_terrain(slope)
        for seed in seeds:
            for mode in modes:
                print(f'REAL Go2 rollout: {mode}, seed={seed}, slope={slope:+.1f}°, command={command:.2f}',flush=True)
                if mode == 'ann':
                    r = heading.run(model,'ann_feedback',None,teacher,command,seed,seconds,
                                    gain=.9,max_rate=.2,viewer=viewer,record=True)
                    recorded = heading.RESULT_DIR/f'ann_feedback_cmd{command:.2f}_seed{seed}.npz'
                    sim_s, x, y, max_yaw, stop = (r['sim_s'],r['forward_m'],r['sideways_m'],
                                                   r['max_abs_yaw_deg'],r['terminated'])
                else:
                    net,stats = v5 if mode == 'v5' else v6
                    r = upright.run(model,net,stats,seed,0.0,command,seconds,show_viewer=viewer)
                    recorded = upright.replay_path(seed,0.0,command)
                    sim_s, x, y, max_yaw, stop = (r['simulated_s'],r['forward_m'],r['final_sideways_m'],
                                                   r['max_abs_yaw_deg'],r['terminated'])
                if not recorded.is_file():
                    raise FileNotFoundError(f'Replay was not saved: {recorded}')
                outpath = folder/f'{mode}_slope{slope:+.1f}_seed{seed}.npz'
                shutil.copy2(recorded,outpath)
                posture = replay_metrics(outpath,slope)
                passed = bool(stop == 'completed requested duration'
                              and sim_s >= seconds - 0.002
                              and x >= 1.5 and abs(y) <= 0.25
                              and max_yaw <= 15 and posture['min_torso_above_plane_m'] >= 0.18)
                entry = {'model':mode,'slope_deg':slope,'seed':seed,'duration_s':sim_s,
                         'forward_m':x,'sideways_m':y,'peak_yaw_deg':max_yaw,
                         'min_height_above_plane_m':posture['min_torso_above_plane_m'],
                         'pitch_rms_deg':posture['pitch_rms_deg'],
                         'passed_strict':passed,'stop':stop,
                         'replay':str(outpath)}
                results.append(entry)
                print(f'  {sim_s:.2f}s, X={x:+.3f} m, Y={y:+.3f} m, max yaw={max_yaw:.2f}°, PASS={passed}',flush=True)
    if not results:
        raise RuntimeError('No evaluation results')
    path = folder/'paired_results.csv'
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {}
    for slope in slopes:
        for mode in modes:
            group = [r for r in results if r['slope_deg']==slope and r['model']==mode]
            summary[f'{mode}_slope{slope:+.1f}'] = {
                'episodes':len(group),
                'strict_passes':sum(r['passed_strict'] for r in group),
                'mean_forward_m':float(np.mean([r['forward_m'] for r in group])),
                'mean_peak_yaw_deg':float(np.mean([r['peak_yaw_deg'] for r in group])),
                'mean_duration_s':float(np.mean([r['duration_s'] for r in group])),
            }
    report = {'seeds':seeds,'slopes_deg':slopes,'command_mps':command,
              'seconds':seconds,'results':summary,
              'note':'V5 and V6 each drive independently, same outer yaw supervisor; ANN is third-party teacher.'}
    (folder/'paired_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('\n=== REAL V5 vs V6 vs ANN TERRAIN COMPARISON ===')
    print(json.dumps(report,indent=2))
    print('Saved',path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='task',required=True)
    pt=sub.add_parser('train',help='Fine-tune V5 into V6 using real slope labels')
    pt.add_argument('--epochs',type=int,default=30)
    pt.add_argument('--lr',type=float,default=0.0001)
    pt.add_argument('--batch',type=int,default=96)
    pe=sub.add_parser('test',help='Matched independent Go2 rollouts for V5,V6,ANN')
    pe.add_argument('--seeds',nargs='+',type=int,default=[5801,5802])
    pe.add_argument('--slopes',nargs='+',type=float,default=[0.,2.])
    pe.add_argument('--command',type=float,default=0.45)
    pe.add_argument('--seconds',type=float,default=8.0)
    pe.add_argument('--modes',nargs='+',choices=('v5','v6','ann'),default=['v5','v6','ann'])
    pe.add_argument('--viewer',action='store_true')
    args=parser.parse_args()
    if args.task == 'train':
        if not 1<=args.epochs<=100 or not 0<args.lr<=0.001 or not 16<=args.batch<=512:
            parser.error('Invalid training settings')
        train(args.epochs,args.lr,args.batch)
    else:
        if not args.seeds or len(set(args.seeds))!=len(args.seeds) or len(args.seeds)>10:
            parser.error('Choose 1-10 unique seeds')
        if not args.slopes or len(args.slopes)>5 or any(abs(s)>3 for s in args.slopes):
            parser.error('Slope must lie within +/-3 degrees')
        if not (1<=args.seconds<=15 and 0<args.command<=1):
            parser.error('Invalid speed/duration')
        if args.viewer and len(args.seeds)*len(args.slopes)*len(args.modes) != 1:
            parser.error('--viewer requires one seed, slope, and model')
        test(args.seeds,args.slopes,args.command,args.seconds,args.modes,args.viewer)


if __name__=='__main__':
    main()
