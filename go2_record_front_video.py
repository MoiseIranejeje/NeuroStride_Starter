
import argparse
import math
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--view", choices=["front", "follow", "wide", "top"],
                        default="front")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--distance", type=float, default=1.8)
    args = parser.parse_args()

    replay_path = Path(args.replay).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(replay_path, allow_pickle=False) as replay:
        qpos = replay["qpos"]
        qvel = replay["qvel"]
        sim_time = replay["sim_time"].reshape(-1)

    model = mm.load("unitree_go2")
    model.vis.global_.offwidth = max(1920, args.width)
    model.vis.global_.offheight = max(1080, args.height)

    if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
        raise ValueError(
            f"Replay shape does not match model: "
            f"qpos={qpos.shape[1]} vs nq={model.nq}, "
            f"qvel={qvel.shape[1]} vs nv={model.nv}. "
            "Use the MuJoCo model used to create this replay."
        )

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model, width=args.width, height=args.height
    )

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1

    start_xyz = np.array(qpos[0, :3], dtype=float)
    start_yaw = quat_to_yaw(qpos[0, 3:7])

    # Fixed views show the full route rather than following the robot.
    if args.view == "wide":
        cam.lookat[:] = [
            start_xyz[0] + 1.35,
            start_xyz[1],
            start_xyz[2],
        ]
        cam.distance = 4.5
        cam.elevation = -30
        cam.azimuth = math.degrees(start_yaw) + 135

    elif args.view == "top":
        cam.lookat[:] = [
            start_xyz[0] + 1.35,
            start_xyz[1],
            start_xyz[2],
        ]
        cam.distance = 4.5
        cam.elevation = -89
        cam.azimuth = math.degrees(start_yaw)

    duration = float(sim_time[-1])
    target_times = np.arange(
        float(sim_time[0]),
        duration + 1e-9,
        1.0 / args.fps,
    )

    frame_ids = np.searchsorted(
        sim_time, target_times, side="left"
    )
    frame_ids = np.clip(frame_ids, 0, len(sim_time) - 1)

    try:
        with imageio.get_writer(
            str(out_path),
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for idx in frame_ids:
                data.qpos[:] = qpos[idx]
                data.qvel[:] = qvel[idx]
                mujoco.mj_forward(model, data)

                if args.view in ("front", "follow"):
                    base_xyz = np.array(
                        data.qpos[:3], dtype=float
                    )
                    yaw = quat_to_yaw(data.qpos[3:7])

                    cam.lookat[:] = [
                        base_xyz[0],
                        base_xyz[1],
                        base_xyz[2] + 0.10,
                    ]
                    cam.distance = args.distance
                    if args.view == "follow":
                        cam.elevation = -24
                        cam.azimuth = math.degrees(start_yaw) + 135
                    else:
                        cam.elevation = -8
                        cam.azimuth = math.degrees(yaw) + 180

                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                writer.append_data(frame)
    finally:
        renderer.close()

    print(f"Saved video: {out_path}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"View: {args.view}")


if __name__ == "__main__":
    main()
