
"""NeuroStride: Evaluate and replay saved ANN V2 checkpoints."""

import argparse
import csv
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


CHECKPOINT_DIR = Path("training/ann_v2_seed42/checkpoints")
CHECKPOINTS = [50000, 101200, 151200, 201200, 251200, 301200]


def load_checkpoint(step, render=False, max_steps=500):
    model_path = CHECKPOINT_DIR / f"ann_v2_{step}_steps.zip"
    stats_path = (
        CHECKPOINT_DIR / f"ann_v2_vecnormalize_{step}_steps.pkl"
    )

    if not model_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint or normalization file for {step}"
        )

    # Use the original Ant-v5 reward for fair comparisons.
    base_env = DummyVecEnv([
        lambda: gym.make(
            "Ant-v5",
            max_episode_steps=max_steps,
            render_mode="human" if render else None,
        )
    ])

    env = VecNormalize.load(str(stats_path), base_env)

    # Freeze observation statistics during evaluation.
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), device="cpu")

    return model, env


def run_episode(model, env, seed, render=False, step_delay=0.0):
    env.seed(seed)
    obs = env.reset()

    # Read the initial position from the MuJoCo physics state.
    robot = env.venv.envs[0].unwrapped
    initial_x = float(robot.data.qpos[0])
    initial_y = float(robot.data.qpos[1])

    last_x = initial_x
    last_y = initial_y

    reward_total = 0.0
    action_effort = 0.0
    episode_steps = 0
    end_reason = "manual_limit"

    limit = env.venv.envs[0].spec.max_episode_steps
    dt = float(robot.dt)

    for i in range(limit):
        # Deterministic policy: use what the ANN has learned.
        action, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(action)
        info = infos[0]

        reward_total += float(rewards[0])
        action_effort += float(np.mean(np.square(action)))

        # Ant-v5 reports position after each physics step.
        last_x = float(info["x_position"])
        last_y = float(info["y_position"])

        episode_steps = i + 1

        if render and step_delay > 0:
            time.sleep(step_delay)

        if bool(dones[0]):
            if info.get("TimeLimit.truncated", False):
                end_reason = "time_limit"
            else:
                end_reason = "fell_or_unhealthy"
            break

    distance_x = last_x - initial_x
    distance_y = last_y - initial_y

    return {
        "steps": episode_steps,
        "reward": round(reward_total, 3),
        "forward_m": round(distance_x, 3),
        "sideways_m": round(distance_y, 3),
        "speed_m_s": round(
            distance_x / (episode_steps * dt), 4
        ),
        "mean_action_squared": round(
            action_effort / episode_steps, 4
        ),
        "end_reason": end_reason,
    }


def compare_checkpoints(args):
    all_results = []

    for checkpoint in CHECKPOINTS:
        print(f"\n=== CHECKPOINT {checkpoint:,} ===")

        model, env = load_checkpoint(
            checkpoint, max_steps=args.max_steps
        )

        results = []

        try:
            for episode in range(args.episodes):
                seed = args.seed + episode

                result = run_episode(model, env, seed)
                result["checkpoint"] = checkpoint
                result["seed"] = seed

                results.append(result)
                all_results.append(result)

        finally:
            env.close()

        mean_steps = np.mean([r["steps"] for r in results])
        mean_forward = np.mean(
            [r["forward_m"] for r in results]
        )
        mean_speed = np.mean(
            [r["speed_m_s"] for r in results]
        )

        falls = sum(
            r["end_reason"] == "fell_or_unhealthy"
            for r in results
        )

        print(f"Average survival: {mean_steps:.1f} steps")
        print(f"Average forward distance: {mean_forward:.3f} m")
        print(f"Average forward speed: {mean_speed:.3f} m/s")
        print(f"Falls: {falls}/{len(results)}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output = output_dir / "checkpoint_comparison.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(all_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\nResults saved to:", output.resolve())


def replay_checkpoint(args):
    if args.checkpoint not in CHECKPOINTS:
        raise ValueError(f"Choose from: {CHECKPOINTS}")

    model, env = load_checkpoint(
        args.checkpoint,
        render=True,
        max_steps=args.max_steps,
    )

    print(f"Replaying checkpoint {args.checkpoint:,}")
    print("The robot restarts after each episode.")
    print("Press CTRL+C in the terminal to stop.")

    episode = 0

    try:
        while args.episodes == 0 or episode < args.episodes:
            episode += 1

            result = run_episode(
                model,
                env,
                seed=args.seed + episode - 1,
                render=True,
                step_delay=args.step_delay,
            )

            print(
                f"Episode {episode}: "
                f"{result['steps']} steps | "
                f"{result['forward_m']} m forward | "
                f"{result['end_reason']}"
            )

            time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\nReplay stopped.")

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", choices=["compare", "replay"],
        default="compare",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=50000
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--step-delay", type=float, default=0.02)

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.episodes < 0 or (
        args.mode == "compare" and args.episodes == 0
    ):
        parser.error("Comparison requires at least one episode")

    if args.mode == "compare":
        compare_checkpoints(args)
    else:
        replay_checkpoint(args)


if __name__ == "__main__":
    main()
