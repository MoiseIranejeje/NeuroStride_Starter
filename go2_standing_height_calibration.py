r"""Calibrate upright standing geometry of the exact NeuroStride MuJoCo Go2.

The manufacturer's 0.40 m dimension is overall standing height, not base qpos[2].
This is STATIC PD stand testing; no neural locomotion checkpoint is edited.

Windows examples:
  .\.venv\Scripts\python.exe go2_standing_height_calibration.py
  .\.venv\Scripts\python.exe go2_standing_height_calibration.py --extensions 0.00 0.08 0.16 0.24
  .\.venv\Scripts\python.exe go2_standing_height_calibration.py --extensions 0.16 --viewer
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
import time

import mujoco
from mujoco import viewer as mujoco_viewer
import mujoco_menagerie as mm
import numpy as np

from go2_collect_teacher import DEFAULT_JOINT_POS, EXPECTED_ACTUATORS, KP, KD

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'go2_standing_height_calibration'


def torso_angles_deg(quat):
    w, x, y, z = map(float, quat)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(float(np.clip(2 * (w*y - z*x), -1, 1)))
    return math.degrees(roll), math.degrees(pitch)


def visual_bounds(model, data):
    """World XYZ bounds of all rendered mesh vertices, excluding the ground.

    Uses MuJoCo-compiled mesh vertex positions and current geom transforms.
    It is a model visual-geometry measurement, not a manufacturer's survey.
    """
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    used = 0
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[g])
        if mesh_id < 0:
            continue
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        if count == 0:
            continue
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
        rotation = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        world = vertices @ rotation.T + data.geom_xpos[g]
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
        used += 1
    if not used:
        raise RuntimeError('No visual mesh geoms in model; cannot measure overall dimensions')
    return lo, hi


def foot_clearances(model, data):
    clearance = []
    for name in ('FL', 'FR', 'RL', 'RR'):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise RuntimeError(f'Missing collision foot geom {name}')
        radius = float(model.geom_size[gid, 0])
        clearance.append(float(data.geom_xpos[gid, 2] - radius))
    return clearance


def extension_targets(extension):
    """Propose slightly straighter knees while preserving approximate leg symmetry.

    Per leg, shift thigh by -extension/2 and knee by +extension from
    the proven ANN-reference neutral position (knee is negative).
    Any promotion to actual SNN walking REQUIRES a separate closed-loop test.
    """
    target = np.array(DEFAULT_JOINT_POS, dtype=np.float64).copy()
    for offset in (0, 3, 6, 9):
        target[offset + 1] -= extension / 2.0
        target[offset + 2] += extension
    return target


def simulate(extension, seconds=3.0, viewer=False):
    model = mm.load('unitree_go2')
    if model.nu != 12:
        raise RuntimeError(f'Expected 12 torque actuators, got {model.nu}')
    names = [model.actuator(i).name for i in range(model.nu)]
    if names != list(EXPECTED_ACTUATORS):
        raise RuntimeError(f'Actuator order mismatch: {names}')
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key < 0:
        raise RuntimeError('Unitree Go2 home keyframe missing')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[7:19] = DEFAULT_JOINT_POS.copy()
    data.qpos[2] = 0.30  # Same torso-base start height as our Go2 gait tests.
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    target = extension_targets(extension)
    for a in range(12):
        jointid = int(model.actuator_trnid[a, 0])
        qid = int(model.jnt_qposadr[jointid])
        low, high = model.jnt_range[jointid]
        if bool(model.jnt_limited[jointid]) and not (low + 0.02 <= target[a] <= high - 0.02):
            raise ValueError(f'Extension {extension:.3f}: joint {model.joint(jointid).name} target outside safe range')

    limits_lo, limits_hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    height_values, base_values, pitch_values, roll_values = [], [], [], []
    feet_values, torque_clip_samples = [], 0
    stop = 'completed'
    steps = int(round(seconds / model.opt.timestep))
    manager = mujoco_viewer.launch_passive(model, data) if viewer else contextlib.nullcontext(None)
    with manager as view:
        for k in range(steps):
            if viewer and not view.is_running():
                stop = 'viewer closed'
                break
            torque_raw = KP * (target - data.qpos[7:19]) - KD * data.qvel[6:18]
            torque_clip_samples += int(np.any((torque_raw < limits_lo) | (torque_raw > limits_hi)))
            data.ctrl[:] = np.clip(torque_raw, limits_lo, limits_hi)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.18:
                stop = 'fall or invalid state'
                break
            if viewer and k % 10 == 0:
                view.cam.lookat[:] = data.qpos[:3]
                view.sync()
                time.sleep(model.opt.timestep * 10)
            if data.time < 1.0 or k % 10 != 0:
                continue
            mujoco.mj_forward(model, data)
            low, high = visual_bounds(model, data)
            height_values.append(float(high[2]))  # Floor is z=0.
            base_values.append(float(data.qpos[2]))
            roll, pitch = torso_angles_deg(data.qpos[3:7])
            pitch_values.append(pitch)
            roll_values.append(roll)
            feet_values.append(foot_clearances(model, data))
        if viewer and view.is_running():
            view.sync()

    row = {
        'knee_extension_rad': round(extension, 3),
        'stop': stop,
        'duration_s': round(float(data.time), 3),
        'measured_visual_top_height_m': round(float(np.mean(height_values)), 4) if height_values else None,
        'measured_base_height_m': round(float(np.mean(base_values)), 4) if base_values else None,
        'height_difference_vs_40cm_m': round(float(np.mean(height_values)) - 0.4, 4) if height_values else None,
        'pitch_rms_deg': round(float(np.sqrt(np.mean(np.square(pitch_values)))), 3) if pitch_values else None,
        'roll_rms_deg': round(float(np.sqrt(np.mean(np.square(roll_values)))), 3) if roll_values else None,
        'max_abs_foot_clearance_m': round(float(np.max(np.abs(feet_values))), 4) if feet_values else None,
        'torque_clip_fraction': round(torque_clip_samples / max(steps, 1), 4),
        'note': 'Static PD stand, not an SNN locomotion result; top uses rendered mesh; Menagerie may omit hardware extras',
    }
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extensions', type=float, nargs='+', default=[0.0, 0.08, 0.16, 0.24])
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--viewer', action='store_true')
    args = ap.parse_args()
    if not 2.0 <= args.seconds <= 8.0:
        ap.error('--seconds must be 2–8')
    if any(not(0.0 <= e <= 0.24) for e in args.extensions) or len(args.extensions) > 5:
        ap.error('Use 1–5 nonnegative extensions <=0.24 radians')
    if args.viewer and len(args.extensions) != 1:
        ap.error('--viewer can display one extension at a time')
    if not args.extensions:
        ap.error('Specify at least one extension')

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for extension in args.extensions:
        row = simulate(extension, seconds=args.seconds, viewer=args.viewer)
        rows.append(row)
        print('STAND', json.dumps(row), flush=True)
    with (OUT / 'standing_geometry.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'standing_geometry.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('Saved:', OUT / 'standing_geometry.csv', flush=True)


if __name__ == '__main__':
    main()
