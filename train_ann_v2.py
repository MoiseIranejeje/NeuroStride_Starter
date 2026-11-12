
"""NeuroStride ANN V2: PPO training for forward locomotion."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
)

MAX_STEPS = 500

# Our new experimental reward settings.
TRAIN_REWARD = {
    "healthy_reward": 0.25,
    "forward_reward_weight": 2.0,
    "ctrl_cost_weight": 0.5,
}


def make_env(seed):
    def create():
        env = gym.make(
            "Ant-v5",
            max_episode_steps=MAX_STEPS,
            **TRAIN_REWARD,
        )
        env.action_space.seed(seed)
        return env

    return create


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    # V2 has its own directory. V1 remains untouched.
    run_dir = Path("training") / f"ann_v2_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    model_file = run_dir / "ann_ppo_ant_v2.zip"
    stats_file = run_dir / "ann_vecnormalize_v2.pkl"

    # Create and monitor the robot's physics environment.
    base = DummyVecEnv([make_env(args.seed)])

    monitored = VecMonitor(
        base,
        filename=str(run_dir / "episode_monitor"),
    )

    if args.resume:
        if not model_file.is_file() or not stats_file.is_file():
            raise SystemExit(
                "Cannot resume: V2 model or normalization file missing."
            )

        env = VecNormalize.load(str(stats_file), monitored)
        env.training = True
        env.norm_reward = False

        model = PPO.load(
            str(model_file),
            env=env,
            device="cpu",
            tensorboard_log=str(run_dir / "tensorboard"),
        )

        print(f"Resuming from {model.num_timesteps:,} steps")

    else:
        if model_file.exists() or stats_file.exists():
            raise SystemExit(
                "V2 results already exist. "
                "Use --resume or choose a new --seed."
            )

        # Normalize observations but keep rewards unnormalized.
        env = VecNormalize(
            monitored,
            norm_obs=True,
            norm_reward=False,
        )

        # ANN actor-critic controller trained with PPO.
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=1,
            seed=args.seed,

            # PPO training configuration.
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,

            # Two hidden layers with 256 neurons each.
            policy_kwargs={"net_arch": [256, 256]},

            tensorboard_log=str(run_dir / "tensorboard"),
        )

    print("=== NeuroStride ANN V2 ===")
    print("Training reward settings:", TRAIN_REWARD)
    print("Observations:", env.observation_space.shape)
    print("Actions:", env.action_space.shape)
    print("Additional training steps:", args.steps)

    # Periodically save the policy and normalization statistics.
    callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ann_v2",
        save_vecnormalize=True,
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            tb_log_name="PPO_ANN_V2",
            reset_num_timesteps=not args.resume,
        )

        model.save(str(model_file))
        env.save(str(stats_file))

        manifest = {
            "env_id": "Ant-v5",
            "max_episode_steps": MAX_STEPS,
            "seed": args.seed,
            "train_reward": TRAIN_REWARD,
            "total_timesteps": int(model.num_timesteps),
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        }

        (run_dir / "manifest_v2.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print("Saved model:", model_file.resolve())
        print("Saved normalization:", stats_file.resolve())
        print("Total timesteps:", model.num_timesteps)

    finally:
        env.close()


if __name__ == "__main__":
    main()
