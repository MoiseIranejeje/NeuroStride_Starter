"""NeuroStride ANN V5: controlled forward motion by fine-tuning V4's best policy.

The *single experimental reward change* is a penalty for sideways velocity.
V4's best 25k policy and observation statistics are retained as a starting point.
Evaluation is deterministic, on distinct validation seeds, with original Ant reward.
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

ANT = "Ant-v5"
REWARD_ARGS = {
    "healthy_reward": 1.0,
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.5,
}
LATERAL_WEIGHT = 0.5
TRAIN_LIMIT = 1000
EVAL_LIMIT = 500
EVAL_EVERY = 10_000
VALIDATION_SEEDS = (1100, 1101, 1102, 1103, 1104)


class LateralVelocityPenalty(gym.Wrapper):
    """Keep the original observation/action spaces for V4 policy compatibility."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sideways_velocity = float(info["y_velocity"])
        penalty = LATERAL_WEIGHT * abs(sideways_velocity)
        info = dict(info)
        info["reward_without_lateral_penalty"] = float(reward)
        info["lateral_penalty"] = penalty
        return obs, float(reward) - penalty, terminated, truncated, info


def make_vec(seed, limit, shaped=False, monitor_path=None):
    def factory():
        env = gym.make(ANT, max_episode_steps=limit, **REWARD_ARGS)
        if shaped:
            env = LateralVelocityPenalty(env)
        env.action_space.seed(seed)
        return env

    return VecMonitor(
        DummyVecEnv([factory]),
        filename=str(monitor_path) if monitor_path else None,
    )


def evaluate(model, train_env, eval_env, source_step, run_dir):
    """Evaluate the same fixed policy on the same five reset seeds."""
    sync_envs_normalization(train_env, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    rows = []
    for seed in VALIDATION_SEEDS:
        eval_env.seed(seed)
        obs = eval_env.reset()
        raw_ant = eval_env.venv.venv.envs[0].unwrapped
        initial_x = float(raw_ant.data.qpos[0])
        initial_y = float(raw_ant.data.qpos[1])
        last_info = None
        steps = 0
        reached_limit = False

        for step in range(1, EVAL_LIMIT + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            last_info = infos[0]
            steps = step
            if bool(dones[0]):
                reached_limit = bool(last_info.get("TimeLimit.truncated", False))
                break

        # Gym's TimeLimit normally ends precisely at 500 steps.
        # Also support an environment that does not truncate at 500.
        if steps == EVAL_LIMIT and last_info is not None and not bool(dones[0]):
            reached_limit = True

        forward = float(last_info["x_position"]) - initial_x
        lateral = float(last_info["y_position"]) - initial_y
        success = (
            reached_limit and forward >= 3.0 and abs(lateral) <= 2.0
        )
        rows.append({
            "training_step": source_step,
            "seed": seed,
            "steps": steps,
            "forward_m": forward,
            "sideways_m": lateral,
            "success": int(success),
        })

    with (run_dir / "validation_episodes.csv").open(
        "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)

    rate = mean(r["success"] for r in rows)
    survival = mean(r["steps"] for r in rows)
    forward = mean(r["forward_m"] for r in rows)
    sideways = mean(abs(r["sideways_m"]) for r in rows)
    # Fixed selection formula (same as V4). Not PPO's reward.
    score = 1000 * rate + 0.5 * survival + 20 * max(0, forward) - 10 * sideways
    print(
        f"\nV5 VALIDATION | additional steps={source_step} "
        f"| success={rate:.0%} | survival={survival:.1f}/500 "
        f"| forward={forward:.3f}m "
        f"| |sideways|={sideways:.3f}m | score={score:.2f}",
        flush=True,
    )
    return {
        "additional_steps": source_step,
        "mean_survival": survival,
        "mean_forward_m": forward,
        "mean_abs_sideways_m": sideways,
        "success_rate": rate,
        "selection_score": score,
    }


class V5ValidationCallback(BaseCallback):
    def __init__(self, train_env, eval_env, run_dir, initial_step):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.initial_step = initial_step
        self.best_score = -float("inf")
        self.next_check = initial_step + EVAL_EVERY
        self.history = []

    def _evaluate_and_save(self, additional_steps):
        metrics = evaluate(
            self.model, self.train_env, self.eval_env,
            additional_steps, self.run_dir,
        )
        self.history.append(metrics)
        with (self.run_dir / "validation_summary.csv").open(
            "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)

        for name in ("success_rate", "mean_survival", "mean_forward_m", "mean_abs_sideways_m"):
            self.logger.record("v5_validation/" + name, metrics[name])

        if metrics["selection_score"] > self.best_score:
            self.best_score = metrics["selection_score"]
            self.model.save(str(self.run_dir / "best_model.zip"))
            self.train_env.save(str(self.run_dir / "best_vecnormalize.pkl"))
            (self.run_dir / "best_selection.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            print("Saved NEW BEST V5 checkpoint", flush=True)

    def _on_training_start(self):
        # Baseline checkpoint at 0 added steps: exactly the inherited V4 policy.
        self._evaluate_and_save(0)

    def _on_step(self):
        if self.num_timesteps >= self.next_check:
            self._evaluate_and_save(self.num_timesteps - self.initial_step)
            self.next_check += EVAL_EVERY
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be > 0")

    source = Path("training") / f"ann_v4_seed{args.seed}"
    src_model = source / "best_model.zip"
    src_stats = source / "best_vecnormalize.pkl"
    if not src_model.is_file() or not src_stats.is_file():
        raise SystemExit(f"Missing V4 best model or stats in {source}")

    dest = Path("training") / f"ann_v5_seed{args.seed}"
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"V5 results already exist: {dest} (not overwriting)")
    dest.mkdir(parents=True, exist_ok=True)

    train_env = None
    eval_env = None
    try:
        raw_train = make_vec(
            args.seed, TRAIN_LIMIT, shaped=True,
            monitor_path=dest / "training_monitor",
        )
        train_env = VecNormalize.load(str(src_stats), raw_train)
        train_env.training = True
        train_env.norm_obs = True
        train_env.norm_reward = True
        # Preserve V4 observation and reward normalization statistics.
        # Both continue adapting during V5 training.

        raw_eval = make_vec(1100, EVAL_LIMIT, shaped=False)
        eval_env = VecNormalize(
            raw_eval, norm_obs=True, norm_reward=False, training=False
        )
        model = PPO.load(str(src_model), env=train_env, device="cpu")
        initial_steps = model.num_timesteps
        callback = V5ValidationCallback(train_env, eval_env, dest, initial_steps)

        (dest / "experiment_manifest.json").write_text(
            json.dumps({
                "version": "ANN V5",
                "initial_controller": str(src_model),
                "initial_norm": str(src_stats),
                "initial_v4_steps": initial_steps,
                "new_steps_requested": args.steps,
                "seed": args.seed,
                "env": ANT,
                "reward_change": f"-{LATERAL_WEIGHT} * abs(y_velocity)",
                "ret_rms_reset": False,
                "evaluation_seeds": list(VALIDATION_SEEDS),
                "test_seeds_reserved": list(range(2000, 2010)),
            }, indent=2), encoding="utf-8"
        )

        print("\nNeuroStride V5 | Fine-tuning BEST V4 (25k)")
        print(f"Experiment: penalize |sideways velocity| by {LATERAL_WEIGHT}")
        print("First validation is BEFORE any new training.\n")
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=False,
            callback=callback,
            tb_log_name="PPO_ANN_V5",
            progress_bar=False,
        )
        model.save(str(dest / "latest_model.zip"))
        train_env.save(str(dest / "latest_vecnormalize.pkl"))
        print("\nV5 training finished. Look at validation_summary.csv")
        print("Best model and paired normalization saved in:", dest)
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()


if __name__ == "__main__":
    main()
