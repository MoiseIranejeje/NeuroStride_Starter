
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
)

folder = "training/ann_v5_seed42/"

raw_env = DummyVecEnv([
    lambda: gym.make(
        "Ant-v5",
        render_mode="human",
        max_episode_steps=500,
    )
])

env = VecNormalize.load(
    folder + "best_vecnormalize.pkl",
    raw_env,
)
env.training = False
env.norm_reward = False

model = PPO.load(
    folder + "best_model.zip",
    device="cpu",
)

env.seed(1102)
obs = env.reset()

robot = raw_env.envs[0].unwrapped
start_x = float(robot.data.qpos[0])
start_y = float(robot.data.qpos[1])

steps = 0

try:
    for i in range(500):
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, rewards, dones, infos = env.step(action)
        steps = i + 1

        if dones[0]:
            break

    info = infos[0]
    forward = float(info["x_position"]) - start_x
    sideways = float(info["y_position"]) - start_y

    print("\n=== BEST V4 REPLAY ===")
    print("Steps:", steps)
    print("Forward:", round(forward, 3), "m")
    print("Sideways:", round(sideways, 3), "m")

finally:
    env.close()

