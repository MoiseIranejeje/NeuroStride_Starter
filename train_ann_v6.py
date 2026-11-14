r"""NeuroStride V6: cautious PPO fine-tuning from the BEST V5 controller.

Experiment: change ONLY learning rate from 3e-4 to 5e-5.
Training reward and environment wrapper remain the same as V5.
Training checkpoint selection is on 10 *new* validation reset seeds.
Separate final test seeds (2000-2009) remain untouched.

Run in NeuroStride_Starter:
    .\.venv\Scripts\python.exe train_ann_v6.py --steps 30000
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
    sync_envs_normalization,
)

ENV_ID = "Ant-v5"
REWARD_ARGS = {
    "healthy_reward": 1.0,
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.5,
}
LATERAL_PENALTY = 0.5              # Identical to V5
NEW_LEARNING_RATE = 5e-5           # Only training-hyperparameter change
TRAIN_EPISODE_LIMIT = 1000
EVAL_EPISODE_LIMIT = 500
EVAL_EVERY = 5000                  # Additional V6 training steps
VALIDATION_SEEDS = tuple(range(1200, 1210))  # Never used for V5 selection
TEST_SEEDS_RESERVED = tuple(range(2000, 2010))
MIN_FORWARD_M = 3.0
MAX_ABS_SIDEWAYS_M = 2.0


class LateralVelocityPenalty(gym.Wrapper):
    """V5 training reward: standard Ant reward minus 0.5 * abs(y_velocity)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        y_velocity = float(info["y_velocity"])
        info = dict(info)
        info["v6_lateral_penalty"] = LATERAL_PENALTY * abs(y_velocity)
        return (
            obs,
            float(reward) - info["v6_lateral_penalty"],
            terminated,
            truncated,
            info,
        )


def make_vec(seed, max_steps, shaped=False, monitor_path=None):
    def factory():
        env = gym.make(ENV_ID, max_episode_steps=max_steps, **REWARD_ARGS)
        if shaped:
            env = LateralVelocityPenalty(env)
        env.action_space.seed(seed)
        return env

    return VecMonitor(
        DummyVecEnv([factory]),
        filename=str(monitor_path) if monitor_path is not None else None,
    )


def evaluate(model, train_env, eval_env, step, dest):
    """Real deterministic performance on 10 fixed, *new* validation seeds."""
    sync_envs_normalization(train_env, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    rows = []

    for seed in VALIDATION_SEEDS:
        eval_env.seed(seed)
        observation = eval_env.reset()
        ant = eval_env.venv.venv.envs[0].unwrapped
        initial_x = float(ant.data.qpos[0])
        initial_y = float(ant.data.qpos[1])
        prev_y = initial_y
        cumulative_lateral_motion = 0.0
        final_info = None
        reached_limit = False
        steps = 0

        for steps in range(1, EVAL_EPISODE_LIMIT + 1):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, infos = eval_env.step(action)
            final_info = infos[0]
            current_y = float(final_info["y_position"])
            cumulative_lateral_motion += abs(current_y - prev_y)
            prev_y = current_y
            if bool(done[0]):
                reached_limit = bool(
                    final_info.get("TimeLimit.truncated", False)
                )
                break

        if steps == EVAL_EPISODE_LIMIT and not bool(done[0]):
            reached_limit = True

        forward = float(final_info["x_position"]) - initial_x
        sideways = float(final_info["y_position"]) - initial_y
        success = (
            reached_limit
            and forward >= MIN_FORWARD_M
            and abs(sideways) <= MAX_ABS_SIDEWAYS_M
        )
        rows.append({
            "additional_v6_steps": step,
            "seed": seed,
            "steps_survived": steps,
            "forward_m": forward,
            "sideways_m": sideways,
            "cumulative_abs_lateral_motion_m": cumulative_lateral_motion,
            "success": int(success),
        })

    episode_file = dest / "validation_episodes.csv"
    with episode_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)

    mean_steps = mean(r["steps_survived"] for r in rows)
    mean_forward = mean(r["forward_m"] for r in rows)
    mean_abs_y = mean(abs(r["sideways_m"]) for r in rows)
    mean_lateral_path = mean(
        r["cumulative_abs_lateral_motion_m"] for r in rows
    )
    rate = mean(r["success"] for r in rows)

    # Same selection scoring formula as V5, now on new validation seeds.
    score = (
        1000 * rate
        + 0.5 * mean_steps
        + 20 * max(0.0, mean_forward)
        - 10 * mean_abs_y
    )
    result = {
        "additional_v6_steps": step,
        "success_rate": rate,
        "mean_survival_steps": mean_steps,
        "mean_forward_m": mean_forward,
        "mean_abs_sideways_m": mean_abs_y,
        "mean_cumulative_abs_lateral_motion_m": mean_lateral_path,
        "selection_score": score,
    }
    print(
        f"\nV6 VALIDATION | +{step:,} steps | "
        f"success {rate:.0%} ({int(rate*len(rows))}/{len(rows)}) | "
        f"survival {mean_steps:.1f}/500 | forward {mean_forward:.3f}m | "
        f"sideways {mean_abs_y:.3f}m | score {score:.2f}",
        flush=True,
    )
    return result


class ValidationAndBestCheckpoint(BaseCallback):
    def __init__(self, train_env, eval_env, run_dir, initial_step):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.initial_step = initial_step
        self.next_check = initial_step + EVAL_EVERY
        self.best_score = -float("inf")

    def _evaluate_save(self, additional_steps):
        result = evaluate(
            self.model, self.train_env, self.eval_env,
            additional_steps, self.run_dir,
        )
        with (self.run_dir / "validation_summary.csv").open(
            "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(result))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(result)
        for name in (
            "success_rate", "mean_survival_steps", "mean_forward_m",
            "mean_abs_sideways_m", "mean_cumulative_abs_lateral_motion_m",
        ):
            self.logger.record("v6_validation/" + name, result[name])

        if result["selection_score"] > self.best_score:
            self.best_score = result["selection_score"]
            self.model.save(str(self.run_dir / "best_model.zip"))
            self.train_env.save(str(self.run_dir / "best_vecnormalize.pkl"))
            (self.run_dir / "best_selection.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print("SAVED NEW BEST V6 POLICY + MATCHING STATS", flush=True)

    def _on_training_start(self):
        # First compare the unchanged V5 best policy on the NEW V6 seeds.
        self._evaluate_save(0)

    def _on_step(self):
        if self.num_timesteps >= self.next_check:
            self._evaluate_save(self.num_timesteps - self.initial_step)
            self.next_check += EVAL_EVERY
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    src = Path("training") / f"ann_v5_seed{args.seed}"
    src_model = src / "best_model.zip"
    src_norm = src / "best_vecnormalize.pkl"
    if not src_model.is_file() or not src_norm.is_file():
        raise SystemExit(
            "Missing V5 BEST model and matching normalization files at "
            + str(src)
        )

    dest = Path("training") / f"ann_v6_seed{args.seed}"
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"V6 already exists: {dest}. Not overwriting.")
    dest.mkdir(parents=True, exist_ok=True)
    train_env = None
    eval_env = None

    try:
        train_raw = make_vec(
            args.seed, TRAIN_EPISODE_LIMIT,
            shaped=True, monitor_path=dest / "training_monitor",
        )
        train_env = VecNormalize.load(str(src_norm), train_raw)
        train_env.training = True
        train_env.norm_obs = True
        train_env.norm_reward = True

        eval_raw = make_vec(VALIDATION_SEEDS[0], EVAL_EPISODE_LIMIT)
        eval_env = VecNormalize(
            eval_raw, norm_obs=True, norm_reward=False, training=False
        )

        model = PPO.load(str(src_model), env=train_env, device="cpu")
        initial_steps = int(model.num_timesteps)
        # Only change the learning rate. SB3 uses lr_schedule for updates.
        model.learning_rate = NEW_LEARNING_RATE
        model.lr_schedule = lambda progress_remaining: NEW_LEARNING_RATE
        # Ensure reloaded model writes to the V6 log, not V5 or V4 folders.
        model.tensorboard_log = str(dest / "tensorboard")
        model.verbose = 1

        (dest / "experiment_manifest.json").write_text(
            json.dumps({
                "experiment": "NeuroStride ANN V6",
                "initial_model": str(src_model),
                "initial_vecnormalize": str(src_norm),
                "starting_model_total_steps": initial_steps,
                "additional_training_steps_requested": args.steps,
                "one_training_change": "PPO learning_rate",
                "old_learning_rate": 3e-4,
                "new_learning_rate": NEW_LEARNING_RATE,
                "training_reward": "Ant-v5 reward - 0.5 * abs(y_velocity)",
                "validation_seeds": list(VALIDATION_SEEDS),
                "held_out_test_seeds_not_used": list(TEST_SEEDS_RESERVED),
                "eval_max_steps": EVAL_EPISODE_LIMIT,
                "success": "500 steps and forward>=3m and abs(sideways)<=2m",
            }, indent=2), encoding="utf-8",
        )
        callback = ValidationAndBestCheckpoint(
            train_env, eval_env, dest, initial_steps
        )
        print("\nNEUROSTRIDE V6 — GENTLE PPO FINE-TUNING")
        print(f"Starting checkpoint: V5 best ({initial_steps:,} total steps)")
        print(f"Learning rate: 0.0003 -> {NEW_LEARNING_RATE}")
        print(f"Additional steps requested: {args.steps:,}")
        print("Validation: 10 new seeds, first run BEFORE training")
        print("Saved models are in:", dest, "\n", flush=True)
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=False,
            callback=callback,
            tb_log_name="PPO_ANN_V6",
            progress_bar=False,
        )
        model.save(str(dest / "latest_model.zip"))
        train_env.save(str(dest / "latest_vecnormalize.pkl"))
        print("\nV6 COMPLETE", flush=True)
        print("Actual total model steps:", model.num_timesteps)
        print("Best model:", dest / "best_model.zip")
        print("Best stats:", dest / "best_vecnormalize.pkl")
        print("Validation:", dest / "validation_summary.csv")
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()


if __name__ == "__main__":
    main()
