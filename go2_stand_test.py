
import time

import mujoco
import mujoco.viewer
import mujoco_menagerie as mm
import numpy as np

model = mm.load("unitree_go2")
data = mujoco.MjData(model)

# Start in Go2's predefined standing pose.
home_id = model.key("home").id
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

assert model.nu == 12, "Expected 12 Go2 actuators"

# Use the actual joint mapping from MuJoCo.
joint_ids = model.actuator_trnid[:, 0].astype(int)
q_indices = model.jnt_qposadr[joint_ids]
v_indices = model.jnt_dofadr[joint_ids]

target = data.qpos[q_indices].copy()

# Simple position-error PD controller.
kp = np.array([20, 20, 40] * 4, dtype=float)
kd = np.array([1, 1, 2] * 4, dtype=float)

torque_min = model.actuator_ctrlrange[:, 0]
torque_max = model.actuator_ctrlrange[:, 1]

start_xy = data.qpos[:2].copy()
start_height = float(data.qpos[2])

steps = 0
fell = False

print("Starting Go2 standing test...")
print("Initial height:", round(start_height, 3), "m")

with mujoco.viewer.launch_passive(model, data) as viewer:

    total_steps = int(5.0 / model.opt.timestep)

    for step in range(total_steps):

        if not viewer.is_running():
            break

        q = data.qpos[q_indices]
        qd = data.qvel[v_indices]

        torque = kp * (target - q) - kd * qd

        data.ctrl[:] = np.clip(
            torque,
            torque_min,
            torque_max,
        )

        mujoco.mj_step(model, data)
        steps = step + 1

        if step % 10 == 0:
            viewer.sync()
            time.sleep(10 * model.opt.timestep)

        if data.qpos[2] < 0.18:
            fell = True
            break

    viewer.sync()

distance = np.linalg.norm(data.qpos[:2] - start_xy)

print("\n=== GO2 STANDING TEST ===")
print("Duration:", round(data.time, 2), "seconds")
print("Final torso height:", round(float(data.qpos[2]), 3), "m")
print("Horizontal drift:", round(float(distance), 3), "m")
print("Fell below height threshold:", fell)
print("========================")
