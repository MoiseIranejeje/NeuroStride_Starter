"""Compare frozen SNN V3, trained SNN V4, and ANN teacher on Go2 heading.

Must run inside user's working NeuroStride project with MuJoCo installed.
The ANN is never consulted in either SNN-controlled rollout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco_menagerie as mm
import numpy as np
import torch

import go2_heading_benchmark as hb
from go2_compare_v3 import load_actor
from train_go2_snn_v1 import Go2SpikingActor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/go2_snn_v4/evaluation'
CHECKPOINT = ROOT / 'results/go2_snn_v4/go2_snn_v4_best.pt'
MODES = ('snn_v3_zero', 'snn_v4_zero', 'snn_v3_feedback',
         'snn_v4_feedback', 'ann_feedback')


def load_v4():
    c = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    config = c['config']
    if int(config['feature_dim']) != 350:
        raise ValueError('Expected 350-dimensional V4 SNN checkpoint')
    net = Go2SpikingActor(feature_dim=350, population=int(config['population']),
                          hidden=int(config['hidden']), steps=int(config['steps']))
    net.load_state_dict(c['model_state'])
    net.eval()
    stats = [c[k].cpu().numpy().astype(np.float32) for k in
             ('feature_mean', 'feature_std', 'target_mean', 'target_std')]
    return net, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[4201, 4202, 4203])
    p.add_argument('--command', type=float, default=0.45)
    p.add_argument('--seconds', type=float, default=8.0)
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--viewer', action='store_true',
                   help='Show MuJoCo viewer for selected V4 runs only (desktop)')
    p.add_argument('--no-record', action='store_true')
    a = p.parse_args()
    if not 0 <= a.command <= 1 or not 1 <= a.seconds <= 30:
        p.error('Command or duration outside supported range')
    if not a.seeds or len(a.seeds) > 20:
        p.error('Provide between 1 and 20 seeds')

    torch.set_num_threads(min(4, torch.get_num_threads()))
    v3, v4 = load_actor('v3'), load_v4()
    teacher = hb.load_teacher() if 'ann_feedback' in a.modes else None
    model = mm.load('unitree_go2')
    if model.nu != 12 or abs(model.opt.timestep - 0.002) > 1e-8:
        raise RuntimeError('Unexpected Go2 model actuator count or physics timestep')
    hb.RESULT_DIR = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        for mode in a.modes:
            controller = v3 if mode.startswith('snn_v3') else (
                v4 if mode.startswith('snn_v4') else None)
            print(f'Go2 {mode}, seed {seed}, speed {a.command:.2f}m/s', flush=True)
            result = hb.run(
                model, mode, controller, teacher, a.command, seed,
                a.seconds, gain=0.9, max_rate=0.20,
                viewer=a.viewer and mode.startswith('snn_v4'),
                record=not a.no_record,
            )
            result['policy_source'] = (
                'NeuroStride SNN V4, supervised steering fine-tuning'
                if mode.startswith('snn_v4') else
                'NeuroStride SNN V3, frozen baseline'
                if mode.startswith('snn_v3') else
                'Third-party pretrained ANN teacher')
            rows.append(result)
            # Repair the older benchmark's hardcoded V3 metadata in full-state
            # replay files. The controller weights already used were correct.
            if not a.no_record:
                path = OUT / f'{mode}_cmd{a.command:.2f}_seed{seed}.npz'
                with np.load(path, allow_pickle=False) as z:
                    payload = {k: z[k].copy() for k in z.files}
                payload['summary_json'] = np.array(json.dumps(result))
                np.savez_compressed(path, **payload)
            print(f'  {result["sim_s"]:.2f}s, X={result["forward_m"]:+.3f}m, '
                  f'Y={result["sideways_m"]:+.3f}m, '
                  f'maxYaw={result["max_abs_yaw_deg"]:.2f}deg, '
                  f'straight={result["passed_straight_heading"]}', flush=True)

    with (OUT / 'paired_results.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    grouped = {}
    for mode in a.modes:
        run = [r for r in rows if r['mode'] == mode]
        grouped[mode] = {
            'episodes': len(run),
            'straight_passes': int(sum(r['passed_straight_heading'] for r in run)),
            'flat_passes': int(sum(r['passed_flat_8s'] for r in run)),
            'mean_forward_m': float(np.mean([r['forward_m'] for r in run])),
            'mean_max_abs_yaw_deg': float(np.mean([r['max_abs_yaw_deg'] for r in run])),
            'mean_abs_sideways_m': float(np.mean([abs(r['sideways_m']) for r in run])),
            'mean_survival_s': float(np.mean([r['sim_s'] for r in run])),
        }
    report = {'conditions': {'seeds': a.seeds, 'command_mps': a.command,
                             'duration_s': a.seconds, 'yaw_feedback_gain': 0.9,
                             'yaw_limit_rad_s': 0.2},
              'results': grouped,
              'caveat': 'These seeds become development seeds after inspection; reserve NEW seeds for a final test.'}
    (OUT / 'paired_summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('\n=== V3 vs V4 vs ANN (actual MuJoCo) ===')
    print(json.dumps(report, indent=2))
    print('Saved data + full-state replay files in', OUT)


if __name__ == '__main__':
    main()
