
"""NeuroStride: Compare controllers on standard Ant-v5."""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)

MAX_STEPS = 500

MODELS = {
    "ann_v1": (
        Path("training/ann_seed42/ann_ppo_ant.zip"),
        Path("training/ann_seed42/ann_vecnormalize.pkl"),
    ),
    "ann_v2": (
        Path("training/ann_v2_seed42/ann_ppo_ant_v2.zip"),
        Path("training/ann_v2_seed42/ann_vecnormalize_v2.pkl"),
    ),
}


def create_raw_env(render):
    return gym.make(
        "Ant-v5",
        max_episode_steps=MAX_STEPS,
        render_mode="human" if render else None,
    )


def evaluate(controller, n_episodes, seed_start, render):
    rows = []
    is_ann = controller in MODELS

    if is_ann:
        model_file, stats_file = MODELS[controller]

        if not model_file.exists() or not stats_file.exists():
            raise FileNotFoundError(
                f"Missing files for {controller}: "
                f"{model_file}, {stats_file}"
            )

        vec = DummyVecEnv(
            [lambda: create_raw_env(render)]
        )

        env = VecNormalize.load(str(stats_file), vec)

        # Keep evaluation statistics frozen.
        env.training = False
        env.norm_reward = False

        model = PPO.load(
            str(model_file),
            device="cpu",
        )

    else:
        env = create_raw_env(render)
        model = None

    try:
        if is_ann:
            dt = float(env.venv.envs[0].unwrapped.dt)
        else:
            dt = float(env.unwrapped.dt)

        for episode_index in range(n_episodes):
            seed = seed_start + episode_index

            if is_ann:
                env.seed(seed)
                obs = env.reset()
                info = env.reset_infos[0]
            else:
                obs, info = env.reset(seed=seed)
                env.action_space.seed(seed)

            
            # Get the starting position directly from MuJoCo.
            # qpos[0] is the robot's X position.
            # qpos[1] is the robot's Y position.
            if is_ann:
                robot = env.venv.envs[0].unwrapped
            else:
                robot = env.unwrapped

            start_x = float(robot.data.qpos[0])
            start_y = float(robot.data.qpos[1])

            last_x = start_x
            last_y = start_y


            reward_total = 0.0
            mean_action_sq = 0.0
            reason = "manual_limit"

            for step in range(1, MAX_STEPS + 1):

                if is_ann:
                    action, _ = model.predict(
                        obs,
                        deterministic=True,
                    )

                    obs, reward, done, infos = env.step(action)

                    reward = float(reward[0])
                    finished = bool(done[0])
                    info = infos[0]

                    truncated = bool(
                        info.get("TimeLimit.truncated", False)
                    )

                else:
                    if controller == "random":
                        action = env.action_space.sample()
                    else:
                        action = np.zeros(
                            env.action_space.shape,
                            dtype=np.float32,
                        )

                    obs, reward, terminated, truncated, info = (
                        env.step(action)
                    )

                    finished = bool(terminated or truncated)

                reward_total += reward

                mean_action_sq += float(
                    np.mean(np.square(action))
                )

                last_x = float(info["x_position"])
                last_y = float(
                    info.get("y_position", last_y)
                )

                if finished:
                    reason = (
                        "time_limit"
                        if truncated
                        else "fell_or_unhealthy"
                    )
                    break

            dx = last_x - start_x
            dy = last_y - start_y

            row = {
                "controller": controller,
                "seed": seed,
                "steps": step,
                "reward_standard": round(reward_total, 3),
                "forward_m": round(dx, 3),
                "sideways_m": round(dy, 3),
                "forward_speed_m_s": round(
                    dx / (step * dt), 4
                ),
                "mean_action_squared": round(
                    mean_action_sq / step, 4
                ),
                "end_reason": reason,
            }

            rows.append(row)

            print(
                f"{controller} seed={seed} "
                f"steps={step} "
                f"forward={dx:.2f}m "
                f"speed={row['forward_speed_m_s']:.3f}m/s "
                f"{reason}"
            )

    finally:
        env.close()

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller",
        choices=[
            "all",
            "ann_v1",
            "ann_v2",
            "random",
            "zero",
        ],
        default="all",
    )

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    if args.controller == "all":
        controllers = list(MODELS) + ["random", "zero"]
    else:
        controllers = [args.controller]

    rows = []

    for name in controllers:
        result = evaluate(
            name,
            args.episodes,
            args.seed,
            args.render,
        )

        rows.extend(result)

        mean_distance = np.mean(
            [x["forward_m"] for x in result]
        )
        mean_speed = np.mean(
            [x["forward_speed_m_s"] for x in result]
        )
        falls = sum(
            x["end_reason"] == "fell_or_unhealthy"
            for x in result
        )

        print(
            f"\n{name}: "
            f"mean distance={mean_distance:.2f}m, "
            f"mean speed={mean_speed:.3f}m/s, "
            f"falls={falls}/{len(result)}\n"
        )

    output = Path("results") / (
        f"comparison_seed{args.seed}_{args.controller}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
