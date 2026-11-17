"""Render an actual MuJoCo state recording into an MP4, without a desktop viewer.

Usage in GitHub Codespaces (after dependencies are installed):
  MUJOCO_GL=osmesa python go2_render_mp4.py --file results/go2_replays/<real-file>.npz
  MUJOCO_GL=osmesa python go2_render_mp4.py --file results/go2_replays/<real-file>.npz --fps 25

This re-renders saved qpos/qvel frames. It does NOT invent movement or train a policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco_menagerie as mm
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error('fps must be between 1 and 60')
    if args.width < 128 or args.height < 128:
        parser.error('width and height must be at least 128')
    path = args.file if args.file.is_absolute() else ROOT / args.file
    if not path.is_file():
        raise FileNotFoundError(f'No recorded trajectory at {path}')

    with np.load(path, allow_pickle=False) as z:
        qpos = np.asarray(z['qpos'], dtype=np.float64)
        qvel = np.asarray(z['qvel'], dtype=np.float64)
        times = np.asarray(z['sim_time'], dtype=np.float64)
        summary = json.loads(str(z['summary_json'].item()))
    if len(times) < 2 or np.any(np.diff(times) < 0):
        raise ValueError('Recording has too few frames or nonmonotonic time')

    model = mm.load('unitree_go2')
    data = mujoco.MjData(model)
    if qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError('Recorded qpos/qvel do not match this Go2 model')

    output = args.output or ROOT / 'results' / 'go2_videos' / f'{path.stem}.mp4'
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    print('Recording:', path)
    print('Original sim duration:', round(float(times[-1] - times[0]), 3), 'seconds')
    print('Source controller:', summary.get('controller', 'unknown'))
    print('Rendering to:', output)

    # Match a constant, REAL simulated timestep in the saved data to video frames.
    frame_times = np.arange(times[0], times[-1] + 1e-9, 1.0 / args.fps)
    if len(frame_times) == 0:
        frame_times = times[:1]
    indices = np.clip(np.searchsorted(times, frame_times, side='left'), 0, len(times) - 1)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 135
    cam.elevation = -20
    cam.distance = 1.7

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    try:
        with imageio.get_writer(str(output), fps=args.fps, codec='libx264', quality=7, macro_block_size=16) as writer:
            for frame_n, idx in enumerate(indices):
                data.qpos[:] = qpos[idx]
                data.qvel[:] = qvel[idx]
                data.time = float(times[idx])
                mujoco.mj_forward(model, data)
                cam.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
                if (frame_n + 1) % 50 == 0:
                    print(f'Rendered {frame_n + 1}/{len(indices)} frames', flush=True)
    finally:
        renderer.close()
    print('VIDEO SAVED:', output)
    print('This video shows the actual recorded states; it is not a newly trained episode.')


if __name__ == '__main__':
    main()
