"""Evaluate a trained ANN with real Ant-v5 episodes and save a CSV.

Examples:
  .\\.venv\\Scripts\\python.exe evaluate_ann.py --episodes 5
  .\\.venv\\Scripts\\python.exe evaluate_ann.py --episodes 1 --render
"""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("training/ann_seed42"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000,
                        help="First evaluation seed; deliberately different from training seed 42.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")

    model_file = args.run_dir / "ann_ppo_ant.zip"
    norm_file = args.run_dir / "ann_vecnormalize.pkl"
    if not model_file.is_file() or not norm_file.is_file():
        raise SystemExit(f"Missing trained model/statistics in {args.run_dir}. Run train_ann.py first.")

    # Evaluation is deterministic in policy selection; simulator initial states
    # are seeded independently. Reuse TRAINING normalization in read-only mode.
    env = DummyVecEnv([lambda: gym.make(
        "Ant-v5", max_episode_steps=args.max_steps,
        render_mode="human" if args.render else None,
    )])
    normalized_env = VecNormalize.load(str(norm_file), env)
    normalized_env.training = False
    normalized_env.norm_reward = False
    model = PPO.load(str(model_file), env=normalized_env, device="cpu")

    rows = []
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            normalized_env.seed(seed)
            obs = normalized_env.reset()
            initial_info = normalized_env.reset_infos[0]
            x_initial = float(initial_info.get("x_position", 0.0))
            last_x = x_initial
            total_reward = 0.0
            end_reason = "manual_limit"

            for step in range(1, args.max_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = normalized_env.step(action)
                total_reward += float(rewards[0])
                last_x = float(infos[0].get("x_position", last_x))
                if bool(dones[0]):
                    end_reason = "time_limit" if infos[0].get("TimeLimit.truncated", False) else "fell_or_unhealthy"
                    break

            rows.append({
                "controller": "ann_ppo",
                "seed": seed,
                "steps": step,
                "episode_reward": round(total_reward, 4),
                "forward_displacement_m": round(last_x - x_initial, 4),
                "end_reason": end_reason,
            })
            print(f"Episode {i+1}: steps={step}, reward={total_reward:.3f}, "
                  f"forward={last_x - x_initial:.3f} m, end={end_reason}")
    finally:
        normalized_env.close()

    outfile = args.output or Path("results") / f"ann_eval_seed{args.seed}.csv"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMean reward: {np.mean([r['episode_reward'] for r in rows]):.3f}")
    print(f"Mean forward displacement: {np.mean([r['forward_displacement_m'] for r in rows]):.3f} m")
    print(f"Fell/unhealthy: {sum(r['end_reason'] == 'fell_or_unhealthy' for r in rows)}/{len(rows)}")
    print(f"Saved real evaluation results to: {outfile.resolve()}")
    print("These results are from ANN PPO only; no spiking controller is trained yet.")


if __name__ == "__main__":
    main()
