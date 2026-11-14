r"""NeuroStride V5-vs-V6 independent controller audit (NOT training).

Run from NeuroStride_Starter:
    .\.venv\Scripts\python.exe neurostride_v6_audit.py

Requires gymnasium[mujoco], stable-baselines3, matplotlib, numpy.
The ten test seeds 2000-2009 are reserved from model selection; after
examining these results, do not tune on them and call them 'held-out' again.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

ENV_ID = "Ant-v5"
MAX_STEPS = 500
MODEL_DIRS = {
    "V5_best": Path("training/ann_v5_seed42"),
    "V6_best": Path("training/ann_v6_seed42"),
}


def open_policy(folder):
    model_file = folder / "best_model.zip"
    stats_file = folder / "best_vecnormalize.pkl"
    for path in (model_file, stats_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; do not substitute latest or older normalizer")
    raw = VecMonitor(DummyVecEnv([lambda: gym.make(ENV_ID, max_episode_steps=MAX_STEPS)]))
    env = VecNormalize.load(str(stats_file), raw)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_file), device="cpu")
    return model, env, raw


def audit_episode(model, env, raw, seed):
    env.seed(seed)
    observation = env.reset()
    robot = raw.venv.envs[0].unwrapped
    start_x, start_y = (float(robot.data.qpos[i]) for i in (0, 1))
    old_x, old_y = start_x, start_y
    path_length = cumulative_lateral = 0.0
    trace = []
    steps = 0
    finished_by_time_limit = False

    for step in range(1, MAX_STEPS + 1):
        action, _ = model.predict(observation, deterministic=True)
        action_vector = np.asarray(action).reshape(-1)
        observation, rewards, dones, infos = env.step(action)
        info = infos[0]
        x, y = float(info["x_position"]), float(info["y_position"])
        delta_x, delta_y = x - old_x, y - old_y
        path_length += float(np.hypot(delta_x, delta_y))
        cumulative_lateral += abs(delta_y)
        old_x, old_y = x, y
        steps = step
        done = bool(dones[0])
        finished_by_time_limit = done and bool(info.get("TimeLimit.truncated", False))
        # Important: VecEnv auto-resets on done, so post-step qpos then points
        # to the *new* episode, not the completed episode. Mark it NaN.
        torso_z = float(robot.data.qpos[2]) if not done else float("nan")
        trace.append({
            "step": step, "time_s": step * 0.05,
            "x_m": x - start_x, "y_m": y - start_y,
            "torso_z_m": torso_z,
            "action_rms": float(np.sqrt(np.mean(action_vector ** 2))),
        })
        if done:
            break

    forward = old_x - start_x
    sideways = old_y - start_y
    completed = steps == MAX_STEPS and (finished_by_time_limit or not done)
    success = int(completed and forward >= 3.0 and abs(sideways) <= 2.0)
    z_values = [t["torso_z_m"] for t in trace if np.isfinite(t["torso_z_m"])]
    return {
        "seed": seed, "steps": steps, "finished_500_steps": int(completed),
        "forward_m": forward, "sideways_m": sideways,
        "cumulative_abs_lateral_m": cumulative_lateral,
        "xy_path_length_m": path_length,
        "forward_path_efficiency": forward / path_length if path_length > 0 else 0.0,
        "torso_z_std_m": pstdev(z_values) if len(z_values) > 1 else float("nan"),
        "success": success,
    }, trace


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectory_plot(traces, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, trace in traces.items():
        t = [r["time_s"] for r in trace]
        x = [r["x_m"] for r in trace]
        y = [r["y_m"] for r in trace]
        z = [r["torso_z_m"] for r in trace]
        a = [r["action_rms"] for r in trace]
        axes[0, 0].plot(x, y, label=name)
        axes[0, 1].plot(t, y, label=name)
        axes[1, 0].plot(t, z, label=name)
        axes[1, 1].plot(t, a, label=name)
    axes[0, 0].set(title="Torso XY trajectory", xlabel="Forward X displacement (m)", ylabel="Sideways Y displacement (m)")
    axes[0, 1].set(title="Lateral displacement over time", xlabel="Time (s)", ylabel="Y displacement (m)")
    axes[1, 0].set(title="Torso height variability", xlabel="Time (s)", ylabel="Torso Z (m)")
    axes[1, 1].set(title="Motor command magnitude", xlabel="Time (s)", ylabel="Action RMS")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("NeuroStride motion audit (kinematics, NOT foot-contact gait timing)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, default=2000)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--plot-seed", type=int, default=2000)
    args = parser.parse_args()
    seeds = list(range(args.start_seed, args.start_seed + args.count))
    if args.count < 1 or args.plot_seed not in seeds:
        parser.error("--plot-seed must be in the selected seed range")
    output = Path("results/neurostride_v6_audit")
    output.mkdir(parents=True, exist_ok=True)
    all_rows, example_traces = [], {}

    for name, folder in MODEL_DIRS.items():
        model, env, raw = open_policy(folder)
        try:
            for seed in seeds:
                episode, trace = audit_episode(model, env, raw, seed)
                episode["controller"] = name
                all_rows.append(episode)
                if seed == args.plot_seed:
                    example_traces[name] = trace
                print(f"{name} seed {seed}: steps={episode['steps']} "
                      f"forward={episode['forward_m']:.2f}m "
                      f"sideways={episode['sideways_m']:.2f}m "
                      f"success={episode['success']}")
        finally:
            env.close()

    fields = ["controller", "seed", "steps", "finished_500_steps", "forward_m", "sideways_m", "cumulative_abs_lateral_m", "xy_path_length_m", "forward_path_efficiency", "torso_z_std_m", "success"]
    all_rows = [{key: row[key] for key in fields} for row in all_rows]
    write_rows(output / "heldout_episodes.csv", all_rows)
    comparison = []
    for name in MODEL_DIRS:
        sub = [r for r in all_rows if r["controller"] == name]
        comparison.append({
            "controller": name, "episodes": len(sub),
            "success_rate": mean(r["success"] for r in sub),
            "mean_survival": mean(r["steps"] for r in sub),
            "mean_forward_m": mean(r["forward_m"] for r in sub),
            "mean_abs_sideways_m": mean(abs(r["sideways_m"]) for r in sub),
            "mean_cumulative_abs_lateral_m": mean(r["cumulative_abs_lateral_m"] for r in sub),
            "mean_forward_path_efficiency": mean(r["forward_path_efficiency"] for r in sub),
            "mean_torso_z_std_m": mean(r["torso_z_std_m"] for r in sub),
        })
    write_rows(output / "heldout_comparison.csv", comparison)
    trajectory_plot(example_traces, output / f"trajectory_seed{args.plot_seed}.png")
    for summary in comparison:
        print("\n", summary)
    print(f"\nSaved per-episode CSV, comparison CSV and trajectory PNG in {output.resolve()}")
    print("CAUTION: these seeds are no longer held-out after viewing results; reserve NEW seeds for subsequent model selection.")


if __name__ == "__main__":
    main()
