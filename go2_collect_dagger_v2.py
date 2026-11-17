"""NeuroStride Go2: collect on-policy SNN states with expert ANN corrective labels.

This is a DAgger *data collector*, NOT SNN v2 training and NOT proof of walking.
Existing v1 SNN controls the simulator during most frames; the prerecorded
third-party ANN teacher is queried at every policy step for target actions.
For the second trajectory of each command only, 15% of steps are randomly
teacher-controlled to visit more recoverable states (logged explicitly).

Dependencies: existing go2_collect_teacher.py, go2_snn_closed_loop.py,
trained v1 SNN checkpoint and original published ANN teacher .jit files.

Run from NeuroStride_Starter:
    .\\.venv\\Scripts\\python.exe go2_collect_dagger_v2.py

Outputs under results/go2_dagger_v2:
    dagger_rollouts.npz, episode_report.csv, metadata.json
The dataset never uses command=0.50 m/s for weight updates, preserving that
command for a diagnostic check (though NOT statistically independent anymore).
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import deque
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np
import torch

from go2_collect_teacher import (
    ACTION_SCALE, BASE_COMMAND, DECIMATION, DEFAULT_JOINT_POS,
    EXPECTED_ACTUATORS, HISTORY_LEN, KP, KD, LOW_HEIGHT_THRESHOLD,
    OBS_DIM, POLICY_DIR, EXPECTED_HASHES, build_obs,
)
from go2_snn_closed_loop import load_snn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'go2_dagger_v2'
SEED = 20261114
COMMANDS = (0.30, 0.40, 0.60)  # 0.50 is reserved for a separate diagnostic
EPISODES_PER_COMMAND = 2
SIM_SECONDS = 7.0
INTERVENTION_PROBABILITY = 0.15  # only the second episode per command
TORSO_STOP = LOW_HEIGHT_THRESHOLD


def verify_teacher_weights():
    for filename, expected in EXPECTED_HASHES.items():
        p = POLICY_DIR / filename
        if not p.is_file():
            raise FileNotFoundError(f'Missing teacher policy: {p}')
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Teacher policy SHA256 does not match publisher: {p}')
        print(f'Verified ANN teacher: {filename}')


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(SEED)
    verify_teacher_weights()
    snn, mean_x, std_x, mean_y, std_y, ckpt = load_snn()
    body = torch.jit.load(str(POLICY_DIR / 'body_latest.jit'), map_location='cpu')
    adaptation = torch.jit.load(str(POLICY_DIR / 'adaptation_module_latest.jit'), map_location='cpu')
    body.eval()
    adaptation.eval()

    model = mm.load('unitree_go2')
    actual_actuators = [model.actuator(i).name for i in range(model.nu)]
    if actual_actuators != EXPECTED_ACTUATORS or abs(model.opt.timestep - .002) > 1e-8:
        raise ValueError(f'Unexpected Go2 model: motors={actual_actuators}; dt={model.opt.timestep}')
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_id < 0:
        raise ValueError('Missing Go2 home keyframe')
    data = mujoco.MjData(model)
    lower, upper = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    observations, histories, labels, snn_actions = [], [], [], []
    episodes, times, commands, actual_actions = [], [], [], []
    torso_heights, spike_rates, expert_controls = [], [], []
    reports = []
    episode_id = 0

    for command_speed in COMMANDS:
        for replicate in range(EPISODES_PER_COMMAND):
            mujoco.mj_resetDataKeyframe(model, data, home_id)
            data.qpos[7:19] = DEFAULT_JOINT_POS
            data.qpos[2] = 0.30
            # One clean rollout; one slightly disturbed rollout per command.
            if replicate == 1:
                data.qpos[7:19] += rng.normal(0, 0.018, 12)
                data.qpos[2] += float(rng.uniform(-0.005, 0.005))
            mujoco.mj_forward(model, data)
            start_xyz = data.qpos[:3].copy()
            command = BASE_COMMAND.copy()
            command[0] = command_speed
            memory = deque((np.zeros(OBS_DIM, dtype=np.float32)
                            for _ in range(HISTORY_LEN)), maxlen=HISTORY_LEN)
            prev = np.zeros(12, dtype=np.float64)
            last = np.zeros(12, dtype=np.float64)
            joint_targets = DEFAULT_JOINT_POS.copy()
            phase = 0.0
            lowest_z = float(data.qpos[2])
            n_pre = len(labels)
            interventions = 0
            squared_action_errors = []
            reason = 'completed requested simulation duration'

            for step in range(int(SIM_SECONDS / model.opt.timestep)):
                if step % DECIMATION == 0:
                    phase = (phase + command[4]*model.opt.timestep*DECIMATION) % 1.0
                    obs = build_obs(data, command, prev, last, phase)
                    memory.append(obs)
                    frames = np.stack(memory, axis=0)
                    flat_hist = frames.reshape(-1).astype(np.float32)
                    feat = np.concatenate([obs, frames.mean(axis=0)]).astype(np.float32)
                    x_normalized = np.clip((feat - mean_x) / std_x, -5.0, 5.0)

                    with torch.inference_mode():
                        snn_std, activity = snn(torch.from_numpy(x_normalized[None]))
                        predicted = snn_std.squeeze(0).numpy() * std_y + mean_y
                        his = torch.from_numpy(flat_hist[None])
                        env_latent = adaptation(his)
                        expert = body(torch.cat([his, env_latent], dim=1)).squeeze(0).numpy()

                    if (predicted.shape != (12,) or expert.shape != (12,)
                        or not np.isfinite(predicted).all() or not np.isfinite(expert).all()):
                        reason = 'invalid neural-network action'
                        break
                    if float(np.max(np.abs(predicted))) > 8.0:
                        reason = 'SNN raw action exceeded 8 safety guard'
                        break

                    # This is the DAgger label: expert decision for the state/history
                    # the SNN and any logged interventions actually visited.
                    expert_used = (replicate == 1 and rng.random() < INTERVENTION_PROBABILITY)
                    action = expert if expert_used else predicted
                    if expert_used:
                        interventions += 1
                    action = np.clip(action, -6.0, 6.0)

                    observations.append(obs.copy())
                    histories.append(flat_hist.copy())
                    labels.append(expert.astype(np.float32).copy())
                    snn_actions.append(predicted.astype(np.float32).copy())
                    actual_actions.append(action.astype(np.float32).copy())
                    episodes.append(episode_id)
                    times.append(float(data.time))
                    commands.append(command_speed)
                    torso_heights.append(float(data.qpos[2]))
                    spike_rates.append(float(activity))
                    expert_controls.append(expert_used)
                    squared_action_errors.append(float(np.mean((expert-predicted)**2)))

                    prev = last.copy()
                    last = action.astype(np.float64)
                    joint_targets = DEFAULT_JOINT_POS + last * ACTION_SCALE

                q = data.qpos[7:19]
                qd = data.qvel[6:18]
                data.ctrl[:] = np.clip(KP * (joint_targets - q) - KD * qd, lower, upper)
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    reason = 'nonfinite MuJoCo state'
                    break
                z = float(data.qpos[2])
                lowest_z = min(lowest_z, z)
                if z < TORSO_STOP:
                    reason = 'fell below torso height threshold'
                    break

            delta = data.qpos[:3] - start_xyz
            sample_count = len(labels)-n_pre
            report = {
                'episode': episode_id,
                'command_mps': command_speed,
                'perturbed_initial_pose': bool(replicate == 1),
                'teacher_control_probability': INTERVENTION_PROBABILITY if replicate == 1 else 0.0,
                'teacher_control_steps': interventions,
                'samples': sample_count,
                'sim_seconds': round(float(data.time), 3),
                'forward_m': round(float(delta[0]), 4),
                'sideways_m': round(float(delta[1]), 4),
                'minimum_torso_height_m': round(lowest_z, 4),
                'snn_vs_teacher_rmse': round(float(np.sqrt(np.mean(squared_action_errors))), 4) if squared_action_errors else None,
                'termination': reason,
            }
            reports.append(report)
            print(f'Episode {episode_id+1}/6: command={command_speed:.2f}, '
                  f'frames={sample_count}, t={data.time:.2f}s, forward={delta[0]:+.2f}m, '
                  f'action_RMSE={report["snn_vs_teacher_rmse"]}, '
                  f'teacher interventions={interventions}, result={reason}', flush=True)
            episode_id += 1

    if not labels:
        raise RuntimeError('Collected zero teacher labels; inspect preceding logs')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / 'dagger_rollouts.npz',
        obs=np.asarray(observations, dtype=np.float32),
        history=np.asarray(histories, dtype=np.float32),
        teacher_action=np.asarray(labels, dtype=np.float32),
        snn_predicted_action=np.asarray(snn_actions, dtype=np.float32),
        executed_action=np.asarray(actual_actions, dtype=np.float32),
        episode=np.asarray(episodes, dtype=np.int16),
        command_x=np.asarray(commands, dtype=np.float32),
        sim_time=np.asarray(times, dtype=np.float32),
        torso_height=np.asarray(torso_heights, dtype=np.float32),
        mean_hidden_spike_rate=np.asarray(spike_rates, dtype=np.float32),
        expert_controlled=np.asarray(expert_controls, dtype=np.bool_),
    )
    with (OUT / 'episode_report.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    metadata = {
        'purpose': 'DAgger on-policy correction dataset for SNN v2; not v2 training',
        'source_SNN': 'NeuroStride Go2 SNN v1, offline BC pilot',
        'source_SNN_epoch': int(ckpt['epoch']),
        'source_teacher': 'Ashfak-Kausik walk-these-ways v1.0-policy ANN',
        'commands_mps': list(COMMANDS),
        'random_seed': SEED,
        'teacher_control_probability_only_perturbed_episodes': INTERVENTION_PROBABILITY,
        'samples': len(labels),
        'samples_executed_by_teacher': int(np.count_nonzero(expert_controls)),
        'note': 'No command=0.5 DAgger data; source teacher labels SNN-visited states. '
                'External ANN actions label; only SNN rollout without interventions is pure SNN.',
        'controller_limits': 'raw offsets clipped +/-6, motor torques clipped to MJCF ctrlrange',
        'limitations': 'single scene, fixed gait, one seed family, SNN-visited states '
                       'include teacher-controlled steps in perturbed rollouts',
    }
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\n=== GO2 SNN DAGGER DATA COLLECTION COMPLETE ===')
    print(f'Collected {len(labels)} ANN correction labels on SNN-visited states')
    print(f'Observed teacher-control interventions: {metadata["samples_executed_by_teacher"]}')
    print(f'Mean raw-action RMSE vs teacher: '
          f'{np.sqrt(np.mean((np.asarray(labels)-np.asarray(snn_actions))**2)):.4f}')
    print(f'Files in: {OUT}')
    print('NOT YET: SNN v2 training, validated improvement, or rough-terrain success.')


if __name__ == '__main__':
    main()
