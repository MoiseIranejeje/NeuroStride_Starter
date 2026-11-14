
"""
NeuroStride ANN V4 — Stable Forward Locomotion

Research goal:
Train a MuJoCo Ant-v5 quadruped to move forward
while remaining upright and limiting sideways drift.

Algorithm: PPO
Controller: Artificial Neural Network
Training: From scratch
Evaluation: Deterministic, every 25,000 steps
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)


# ================================================
# 1. EXPERIMENT CONFIGURATION
# ================================================

ENV_ID = "Ant-v5"
TRAIN_EPISODE_LIMIT = 1000
EVAL_EPISODE_LIMIT = 500

# Standard Ant-v5 reward settings.
REWARD = {
    "healthy_reward": 1.0,
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.5,
}

# PPO configuration.
PPO_SETTINGS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.025,
    "ent_coef": 0.0,
}

NETWORK = [64, 64]

VALIDATION_SEEDS = [1000, 1001, 1002, 1003, 1004]

# Proposed success definition.
MIN_FORWARD_M = 3.0
MAX_SIDEWAYS_M = 2.0

EVAL_INTERVAL = 25_000
CHECKPOINT_INTERVAL = 50_000


# ================================================
# 2. CREATE THE ROBOT
# ================================================

def make_environment(seed, episode_limit):

    def create():
        env = gym.make(
            ENV_ID,
            max_episode_steps=episode_limit,
            **REWARD,
        )

        env.action_space.seed(seed)

        return env

    return create


def make_vector_environment(seed, limit, monitor=None):

    vec = DummyVecEnv([
        make_environment(seed, limit)
    ])

    return VecMonitor(
        vec,
        filename=str(monitor) if monitor else None,
    )


# ================================================
# 3. EVALUATE REAL LOCOMOTION PERFORMANCE
# ================================================

class LocomotionEvaluator(BaseCallback):
    """
    Periodically evaluate a fixed policy on five
    validation seeds.

    Save the policy with the best combined score
    for balance, forward travel and task completion.
    """

    def __init__(self, eval_env, run_dir):
        super().__init__()

        self.eval_env = eval_env
        self.run_dir = run_dir
        self.next_evaluation = None

        self.best_score = float("-inf")

        self.best_metadata = (
            run_dir / "best_selection.json"
        )

        # Preserve best score if resuming training.
        if self.best_metadata.exists():
            saved = json.loads(
                self.best_metadata.read_text(
                    encoding="utf-8"
                )
            )
            self.best_score = float(
                saved["selection_score"]
            )

        self.csv_file = run_dir / "validation_results.csv"

    def _on_training_start(self):
        self.next_evaluation = (
            self.model.num_timesteps + EVAL_INTERVAL
        )

    def _on_step(self):

        if self.num_timesteps < self.next_evaluation:
            return True

        self.next_evaluation += EVAL_INTERVAL

        # Use current training observation statistics.
        sync_envs_normalization(
            self.training_env,
            self.eval_env,
        )

        self.eval_env.training = False
        self.eval_env.norm_reward = False

        results = []

        for seed in VALIDATION_SEEDS:

            self.eval_env.seed(seed)
            obs = self.eval_env.reset()

            # Access actual MuJoCo torso coordinates.
            robot = (
                self.eval_env.venv.venv
                .envs[0].unwrapped
            )

            start_x = float(robot.data.qpos[0])
            start_y = float(robot.data.qpos[1])

            final_x = start_x
            final_y = start_y

            completed = False
            steps = 0

            for step in range(1, EVAL_EPISODE_LIMIT + 1):

                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                )

                obs, rewards, dones, infos = (
                    self.eval_env.step(action)
                )

                info = infos[0]

                final_x = float(info["x_position"])
                final_y = float(info["y_position"])
                steps = step

                if bool(dones[0]):
                    completed = bool(
                        info.get("TimeLimit.truncated", False)
                    )
                    break

            forward = final_x - start_x
            sideways = final_y - start_y

            success = (
                completed
                and forward >= MIN_FORWARD_M
                and abs(sideways) <= MAX_SIDEWAYS_M
            )

            results.append({
                "seed": seed,
                "steps": steps,
                "forward_m": forward,
                "sideways_m": sideways,
                "success": int(success),
            })

        success_rate = mean(
            r["success"] for r in results
        )

        mean_steps = mean(
            r["steps"] for r in results
        )

        mean_forward = mean(
            r["forward_m"] for r in results
        )

        mean_abs_sideways = mean(
            abs(r["sideways_m"]) for r in results
        )

        # Model-selection score, NOT the PPO reward.
        # Task completion is weighted most strongly.
        score = (
            1000.0 * success_rate
            + 0.5 * mean_steps
            + 20.0 * max(0.0, mean_forward)
            - 10.0 * mean_abs_sideways
        )

        print("\n========== V4 VALIDATION ==========")
        print("Training steps:", self.num_timesteps)
        print(f"Success rate: {success_rate:.0%}")
        print(f"Mean survival: {mean_steps:.1f}")
        print(f"Mean forward: {mean_forward:.3f} m")
        print(
            f"Mean sideways drift: "
            f"{mean_abs_sideways:.3f} m"
        )
        print(f"Selection score: {score:.2f}")
        print("===================================\n")

        # Store measurable validation data.
        file_exists = self.csv_file.exists()

        with self.csv_file.open(
            "a", newline="", encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "training_steps",
                    "success_rate",
                    "mean_survival",
                    "mean_forward_m",
                    "mean_abs_sideways_m",
                    "selection_score",
                ])

            writer.writerow([
                self.num_timesteps,
                success_rate,
                mean_steps,
                mean_forward,
                mean_abs_sideways,
                score,
            ])

        # Add validation metrics to TensorBoard.
        self.logger.record(
            "v4_validation/success_rate",
            success_rate,
        )

        self.logger.record(
            "v4_validation/mean_survival",
            mean_steps,
        )

        self.logger.record(
            "v4_validation/mean_forward_m",
            mean_forward,
        )

        # Save the strongest validated controller.
        if score > self.best_score:

            self.best_score = score

            self.model.save(
                str(self.run_dir / "best_model.zip")
            )

            self.training_env.save(
                str(
                    self.run_dir
                    / "best_vecnormalize.pkl"
                )
            )

            metadata = {
                "training_steps": self.num_timesteps,
                "selection_score": score,
                "success_rate": success_rate,
                "mean_survival": mean_steps,
                "mean_forward_m": mean_forward,
                "mean_abs_sideways_m": mean_abs_sideways,
                "validation_seeds": VALIDATION_SEEDS,
            }

            self.best_metadata.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            print(
                "NEW BEST V4 CONTROLLER SAVED"
            )

        return True


# ================================================
# 4. MAIN TRAINING
# ================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")

    # V4 has completely independent model storage.
    run_dir = (
        Path("training")
        / f"ann_v4_seed{args.seed}"
    )

    model_path = run_dir / "ann_ppo_ant_v4.zip"
    stats_path = run_dir / "ann_vecnormalize_v4.pkl"

    if args.resume:

        if not model_path.exists() or not stats_path.exists():
            raise SystemExit(
                "Cannot resume: V4 model/statistics missing."
            )

    elif model_path.exists() or stats_path.exists():

        raise SystemExit(
            "V4 model already exists. Use --resume."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    train_env = None
    eval_env = None

    try:

        monitored = make_vector_environment(
            args.seed,
            TRAIN_EPISODE_LIMIT,
            monitor=run_dir / "episode_monitor",
        )

        if args.resume:

            train_env = VecNormalize.load(
                str(stats_path),
                monitored,
            )

            train_env.training = True
            train_env.norm_obs = True
            train_env.norm_reward = True

        else:

            # Normalize both observations and rewards.
            train_env = VecNormalize(
                monitored,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
            )

        # Evaluation uses the same robot and reward
        # but a common 500-step benchmark.
        eval_monitored = make_vector_environment(
            1000,
            EVAL_EPISODE_LIMIT,
        )

        eval_env = VecNormalize(
            eval_monitored,
            norm_obs=True,
            norm_reward=False,
            training=False,
        )

        if args.resume:

            model = PPO.load(
                str(model_path),
                env=train_env,
                device="cpu",
                tensorboard_log=str(
                    run_dir / "tensorboard"
                ),
            )

        else:

            model = PPO(
                "MlpPolicy",
                train_env,

                device="cpu",
                verbose=1,
                seed=args.seed,

                policy_kwargs={
                    "net_arch": NETWORK,
                },

                tensorboard_log=str(
                    run_dir / "tensorboard"
                ),

                **PPO_SETTINGS,
            )

        checkpoint = CheckpointCallback(
            save_freq=CHECKPOINT_INTERVAL,
            save_path=str(checkpoint_dir),
            name_prefix="ann_v4",
            save_vecnormalize=True,
        )

        validator = LocomotionEvaluator(
            eval_env,
            run_dir,
        )

        callbacks = CallbackList([
            checkpoint,
            validator,
        ])

        print("\n================================")
        print("NEUROSTRIDE — ANN V4")
        print("================================")
        print("Robot:", ENV_ID)
        print("Network:", NETWORK)
        print("Training episode limit:", TRAIN_EPISODE_LIMIT)
        print("Validation episode limit:", EVAL_EPISODE_LIMIT)
        print("Reward:", REWARD)
        print("Seed:", args.seed)
        print("Additional steps:", args.steps)
        print("Resume:", args.resume)
        print("================================\n")

        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            tb_log_name="PPO_ANN_V4",
            reset_num_timesteps=not args.resume,
        )

        # Save latest controller separately from best.
        model.save(str(model_path))
        train_env.save(str(stats_path))

        manifest = {
            "experiment": "NeuroStride ANN V4",
            "environment": ENV_ID,
            "algorithm": "PPO",
            "seed": args.seed,
            "total_steps": int(model.num_timesteps),
            "train_episode_limit": TRAIN_EPISODE_LIMIT,
            "eval_episode_limit": EVAL_EPISODE_LIMIT,
            "network": NETWORK,
            "reward": REWARD,
            "ppo_settings": PPO_SETTINGS,
            "validation_seeds": VALIDATION_SEEDS,
            "success_definition": {
                "steps": EVAL_EPISODE_LIMIT,
                "min_forward_m": MIN_FORWARD_M,
                "max_abs_sideways_m": MAX_SIDEWAYS_M,
            },
        }

        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print("\nV4 TRAINING COMPLETED")
        print("Total steps:", model.num_timesteps)
        print("Latest model:", model_path.resolve())
        print("Normalization:", stats_path.resolve())
        print("Best model:", run_dir / "best_model.zip")
        print(
            "Validation log:",
            run_dir / "validation_results.csv",
        )

    finally:

        if eval_env is not None:
            eval_env.close()

        if train_env is not None:
            train_env.close()


if __name__ == "__main__":
    main()
