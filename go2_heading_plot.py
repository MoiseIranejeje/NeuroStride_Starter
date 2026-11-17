"""Plots actual recorded ANN/SNN Go2 heading benchmark trajectories."""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / 'results' / 'go2_heading_benchmark'
MODES = ('snn_zero', 'snn_feedback', 'ann_zero', 'ann_feedback')


def heading(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=3101)
    p.add_argument('--command', type=float, default=.45)
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    a = p.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    n = 0
    for mode in a.modes:
        path = OUT / f'{mode}_cmd{a.command:.2f}_seed{a.seed}.npz'
        if not path.is_file():
            print('Missing recording:', path)
            continue
        with np.load(path, allow_pickle=False) as r:
            q, t = r['qpos'].copy(), r['sim_time'].copy()
            cmd = r['yaw_cmd'].copy()
        xy = q[:, :2] - q[0, :2]
        h = np.unwrap(heading(q[:, 3:7]))
        hd = np.rad2deg(h - h[0]); n += 1
        axes[0, 0].plot(xy[:, 0], xy[:, 1], label=f'{mode} ({t[-1]:.1f}s)')
        axes[0, 1].plot(t, hd, label=mode)
        axes[1, 0].plot(t, q[:, 2], label=mode)
        axes[1, 1].plot(t, cmd, label=mode)
    if not n: raise SystemExit('No recordings found. Run go2_heading_benchmark.py first.')
    axes[0, 0].set(xlabel='Forward X (m)', ylabel='Side Y (m)', title='Actual robot path')
    axes[0, 0].axis('equal')
    axes[0, 1].set(xlabel='Time (s)', ylabel='Yaw relative to start (deg)', title='Heading drift')
    axes[0, 1].axhline(15, color='gray', linestyle=':')
    axes[0, 1].axhline(-15, color='gray', linestyle=':')
    axes[1, 0].set(xlabel='Time (s)', ylabel='Torso Z (m)', title='Standing height')
    axes[1, 0].axhline(.18, color='red', linestyle=':', label='stop threshold')
    axes[1, 1].set(xlabel='Time (s)', ylabel='Requested yaw rate (rad/s)', title='Heading feedback commands')
    for ax in axes.flat:
        ax.grid(alpha=.28); ax.legend(fontsize=8)
    fig.suptitle(f'NeuroStride — matched Go2 ANN vs SNN heading test; seed {a.seed}, forward cmd {a.command:.2f} m/s')
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT / f'heading_seed{a.seed}.png'
    fig.savefig(file, dpi=160)
    plt.close(fig)
    print('Saved real measurement plot:', file)


if __name__ == '__main__':
    main()
