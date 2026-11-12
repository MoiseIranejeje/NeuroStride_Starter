"""Milestone 2: real PPO training on Gymnasium Ant-v5 (ANN actor/critic).

Windows example:
  .\\.venv\\Scripts\\python.exe train_ann.py --timesteps 20000

Run from the NeuroStride_Starter project folder. This is an ANN baseline,
NOT an SNN and NOT a reproduction of the original FullySNN robotics paper.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

ENV_ID = "Ant-v5"
MAX_EPISODE_STEPS = 500


def create_env(seed: int):
    """Factory for one *actual* Ant-v5 physics simulation."""
    def factory():
        env = gym.make(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
        env.action_space.seed(seed)
        return env
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000,
                        help="ADDITIONAL environment timesteps to train; use 20000 for first check.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Experiment directory; default training/ann_seed<seed>.")
    parser.add_argument("--resume", action="store_true",
                        help="Load the saved model and observation-normalization statistics, then continue training.")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")

    run_dir = args.run_dir or Path("training") / f"ann_seed{args.seed}"
    checkpoints = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    model_path = run_dir / "ann_ppo_ant.zip"
    stats_path = run_dir / "ann_vecnormalize.pkl"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    # VecMonitor writes true, unnormalized environment episode rewards.
    # VecNormalize normalizes *observations* only, and must be saved with the model.
    vector_env = DummyVecEnv([create_env(args.seed)])
    monitor_name = f"monitor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    monitored_env = VecMonitor(vector_env, filename=str(run_dir / monitor_name))
    if args.resume:
        if not model_path.is_file() or not stats_path.is_file():
            raise SystemExit("Cannot resume: missing saved ANN model or normalization .pkl in " + str(run_dir))
        env = VecNormalize.load(str(stats_path), monitored_env)
        env.training = True
        env.norm_reward = False
        model = PPO.load(str(model_path), env=env, device="cpu",
                         tensorboard_log=str(tensorboard_dir))
        print(f"Resuming PPO from {model.num_timesteps:,} previously recorded steps.")
    else:
        if model_path.exists():
            raise SystemExit(
                f"A trained model already exists at {model_path}. "
                "Use --resume or choose a different --run-dir / --seed to avoid overwriting it."
            )
        env = VecNormalize(monitored_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", env, verbose=1, device="cpu", seed=args.seed,
            learning_rate=3e-4, n_steps=1024, batch_size=256,
            gamma=0.99, gae_lambda=0.95,
            policy_kwargs={"net_arch": [64, 64]},
            tensorboard_log=str(tensorboard_dir),
        )

    print("=== NeuroStride M2: ANN PPO training (REAL MuJoCo) ===")
    print(f"Environment: {ENV_ID} | limit: {MAX_EPISODE_STEPS} steps/episode")
    print(f"Observation space: {env.observation_space.shape}; actions: {env.action_space.shape}")
    print(f"Seed: {args.seed}; requested additional timesteps: {args.timesteps:,}")
    print(f"TensorBoard directory: {tensorboard_dir.resolve()}")
    print("NOTE: Training may take substantial CPU time. 20k steps is a pipeline check, not proof of walking.")

    callback = CheckpointCallback(
        save_freq=10_000, save_path=str(checkpoints),
        name_prefix="ann_ppo_ant", save_vecnormalize=True,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps, callback=callback,
            tb_log_name="PPO_ANN", reset_num_timesteps=not args.resume,
            progress_bar=False,
        )
        model.save(str(model_path))
        env.save(str(stats_path))
        manifest = {
            "environment": ENV_ID,
            "episode_limit": MAX_EPISODE_STEPS,
            "controller": "ANN MlpPolicy/PPO",
            "seed": args.seed,
            "total_timesteps_reported": int(model.num_timesteps),
            "observation_normalization": True,
            "reward_normalization": False,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "gymnasium_version": gym.__version__,
            "stable_baselines3_version": stable_baselines3.__version__,
            "torch_version": torch.__version__,
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("\nSAVED training artifacts:")
        print(f" Model: {model_path.resolve()}")
        print(f" Observation stats: {stats_path.resolve()}")
        print(f" Checkpoints: {checkpoints.resolve()}")
        print(f" Actual total environment steps: {model.num_timesteps:,}")
        print("NEXT: run evaluate_ann.py for independent evaluation; do not infer success from training reward alone.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
