import argparse
import math
from pathlib import Path
from xml.parsers.expat import model
from pyexpat import model

import imageio.v2 as imageio
import mujoco
import mujoco_menagerie as mm
import numpy as np


def quat_to_yaw(q):
    w, x, y, z = map(float, q)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def load_replay(path: Path):
    data = np.load(path, allow_pickle=True)
    qpos = data["qpos"]
    qvel = data["qvel"]
    sim_time = data["sim_time"]
    return qpos, qvel, sim_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str, required=True, help="Path to saved replay npz")
    parser.add_argument("--out", type=str, required=True, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--elevation", type=float, default=-8.0)
    parser.add_argument("--azimuth-offset", type=float, default=180.0,
                        help="180 usually gives front-facing camera")
    parser.add_argument("--lookat-z", type=float, default=0.10)
    args = parser.parse_args()

    replay_path = Path(args.replay).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qpos, qvel, sim_time = load_replay(replay_path)

    model = mm.load("unitree_go2")
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.distance = args.distance
    cam.elevation = args.elevation

    duration = float(sim_time[-1])
    target_times = np.arange(0.0, duration + 1e-9, 1.0 / args.fps)
    frame_ids = np.searchsorted(sim_time, target_times, side="left")
    frame_ids = np.clip(frame_ids, 0, len(sim_time) - 1)

    writer = imageio.get_writer(str(out_path), fps=args.fps)

    for idx in frame_ids:
        data.qpos[:] = qpos[idx]
        data.qvel[:] = qvel[idx]
        mujoco.mj_forward(model, data)

        base_xyz = np.array(data.qpos[:3], dtype=float)
        base_quat = np.array(data.qpos[3:7], dtype=float)
        yaw = quat_to_yaw(base_quat)

        cam.lookat[:] = [
            base_xyz[0],
            base_xyz[1],
            base_xyz[2] + args.lookat_z,
        ]
        cam.azimuth = math.degrees(yaw) + args.azimuth_offset

        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        writer.append_data(frame)

    writer.close()
    renderer.close()

    print(f"Saved front video to: {out_path}")


if __name__ == "__main__":
    main()