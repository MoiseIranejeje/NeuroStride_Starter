r"""NeuroStride V7: repeated DAgger with real rollout checkpoint selection.

Place beside go2_v5_steering.py in NeuroStride_Starter. Simulation only.
  .\.venv\Scripts\python.exe go2_upright_terrain_v7.py check
  .\.venv\Scripts\python.exe go2_upright_terrain_v7.py train
  .\.venv\Scripts\python.exe go2_upright_terrain_v7.py test --checkpoint PATH

V7 is supervised SNN imitation, not reinforcement learning. The ANN labels
training states; SNN-only evaluation never calls the ANN. Classical yaw feedback
and joint torque PD remain explicit parts of both controllers. No hidden rescue.
Posture limits below are proposed engineering criteria, not published standards.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN_SEEDS = (6101, 6102, 6103)
SELECT_SEEDS = (6201, 6202)
TEST_SEEDS = (6301, 6302, 6303, 6304, 6305)
LIMITS = dict(min_height_m=0.18, max_yaw_deg=15.0, max_lateral_m=0.25,
              max_roll_rms_deg=12.0, max_pitch_rms_deg=12.0,
              max_tilt_deg=25.0, stop_tilt_deg=45.0)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime():
    global torch, mujoco, mm, make_model, features_v3, features_v3_online, base
    import torch
    import mujoco
    import mujoco_menagerie as mm
    import go2_collect_teacher as base
    from go2_v5_steering import make_model
    from train_go2_snn_v3 import features_v3, features_v3_online
    torch.set_num_threads(min(4, torch.get_num_threads()))


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError('Invalid wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def posture(q, normal, point):
    """Ground-relative roll/pitch plus gravity-relative tilt, in degrees.

    Terrain frame X follows the robot heading projected onto the actual plane.
    Clearance is VERTICAL distance above that plane, preserving the old .18m
    convention; it is torso-origin height, not foot clearance.
    """
    r = rotation(q[3:7])
    fwd = r[:, 0] - normal * np.dot(r[:, 0], normal)
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        return (90.0, 90.0, 90.0, -1.0)
    fwd /= norm
    frame = np.column_stack((fwd, np.cross(normal, fwd), normal))
    relative = frame.T @ r
    roll = math.degrees(math.atan2(relative[2, 1], relative[2, 2]))
    pitch = math.degrees(math.asin(float(np.clip(-relative[2, 0], -1, 1))))
    tilt = math.degrees(math.acos(float(np.clip(r[2, 2], -1, 1))))
    clearance = float(np.dot(normal, q[:3] - point) / normal[2])
    return roll, pitch, tilt, clearance


def passes(row, seconds, speed):
    # At the historic .45m/s and 8s this retains the original >=1.5m threshold.
    distance = max(0.15, 1.5 * (speed / .45) * (seconds / 8.0))
    return bool(row['stop'] == 'completed' and row['duration_s'] >= seconds-.001
                and row['forward_m'] >= distance
                and row['max_abs_lateral_m'] <= LIMITS['max_lateral_m']
                and row['peak_yaw_deg'] <= LIMITS['max_yaw_deg']
                and row['min_height_m'] >= LIMITS['min_height_m']
                and row['roll_rms_deg'] <= LIMITS['max_roll_rms_deg']
                and row['pitch_rms_deg'] <= LIMITS['max_pitch_rms_deg']
                and row['peak_world_tilt_deg'] <= LIMITS['max_tilt_deg'])


def load_student(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    net, stats = make_model(ckpt)
    for value, shape in zip(stats, [(350,), (350,), (12,), (12,)]):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid checkpoint normalization')
    if np.any(stats[1] <= 0) or np.any(stats[3] <= 0):
        raise ValueError('Normalization standard deviations must be positive')
    net.eval()
    return ckpt, (net, stats)


def load_teacher():
    base.verify_checkpoints()
    return tuple(torch.jit.load(str(base.POLICY_DIR / name), map_location='cpu').eval()
                 for name in ('body_latest.jit', 'adaptation_module_latest.jit'))


def choose_initial(path):
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for version in ('v6', 'v5'):
        p = ROOT / f'results/go2_snn_{version}/go2_snn_{version}_best.pt'
        if p.is_file():
            return p
    raise FileNotFoundError('Need your real V6 or V5 checkpoint; use --checkpoint PATH')


def terrain(slope):
    model = mm.load('unitree_go2')
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise RuntimeError('Expected the named Go2 floor plane')
    if model.geom_bodyid[gid] != 0:
        raise RuntimeError('Floor must belong to the world body')
    theta = -math.radians(slope)
    model.geom_quat[gid] = (math.cos(theta/2), 0, math.sin(theta/2), 0)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise RuntimeError('Expected Go2 free base plus 12 scalar joints')
    if [model.actuator(i).name for i in range(12)] != base.EXPECTED_ACTUATORS:
        raise RuntimeError('Go2 actuator order changed')
    if abs(model.opt.timestep - .002) > 1e-9:
        raise RuntimeError('Expected 500Hz physics')
    for i in range(12):
        joint = int(model.actuator_trnid[i, 0])
        if model.jnt_qposadr[joint] != 7+i or model.jnt_dofadr[joint] != 6+i:
            raise RuntimeError('Motor-to-joint index mismatch')
    probe = mujoco.MjData(model)
    mujoco.mj_forward(model, probe)
    normal = probe.geom_xmat[gid].reshape(3, 3)[:, 2].copy()
    point = probe.geom_xpos[gid].copy()
    expected = np.array([-math.sin(math.radians(slope)), 0, math.cos(math.radians(slope))])
    if not np.allclose(normal, expected, atol=1e-7):
        raise RuntimeError('Actual floor normal disagrees with requested incline')
    return model, normal, point


def rollout(slope, seed, seconds, speed, student, teacher=None, drive='student',
            collect=False, viewer=False):
    """All conditions use this one loop; eval student never queries a teacher."""
    if (collect or drive == 'expert') and teacher is None:
        raise ValueError('Teacher required for labeling or expert control')
    model, normal, point = terrain(slope)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home < 0:
        raise RuntimeError('Missing home keyframe')
    mujoco.mj_resetDataKeyframe(model, data, home)
    rng = np.random.default_rng(seed)
    data.qpos[7:19] = base.DEFAULT_JOINT_POS + rng.normal(0, .012, 12)
    floor_z = point[2] - np.dot(normal[:2], data.qpos[:2]-point[:2]) / normal[2]
    data.qpos[2] = floor_z + .30 + float(rng.uniform(-.005, .005))
    mujoco.mj_forward(model, data)
    origin = data.qpos[:2].copy()
    r = rotation(data.qpos[3:7])
    yaw0 = math.atan2(r[1, 0], r[0, 0])
    command = base.BASE_COMMAND.copy()
    command[0] = speed
    history = deque((np.zeros(70, np.float32) for _ in range(30)), maxlen=30)
    prev = np.zeros(12)
    last = np.zeros(12)
    target = base.DEFAULT_JOINT_POS.copy()
    phase = 0.0
    xs, ys, ws = [], [], []
    states, velocities, stamps, actions = [], [], [], []
    rr, pp, tt, hh, yy, lateral = [], [], [], [], [], []
    stop = 'completed'
    saturation = 0
    steps_done = 0
    if student is not None:
        net, (mx, sx, my, sy) = student
        net.eval()
        floor_id = model.geom("floor").id
    foot_ids = {model.geom(name).id for name in ("FL", "FR", "RL", "RR")}

    def grounded_feet():
        """Return one world-space contact point per paw touching the floor."""
        contacts = {}

        for contact in data.contact:
            if contact.dist > 0:
                continue

            a, b = map(int, contact.geom)
            if a == floor_id and b in foot_ids:
                contacts[b] = contact.pos.copy()
            elif b == floor_id and a in foot_ids:
                contacts[a] = contact.pos.copy()

        return contacts
    def snapshot():
        if stamps and abs(stamps[-1] - data.time) < 1e-10:
            return
        states.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        stamps.append(float(data.time))
        actions.append(last.copy())

    def measure():
        roll, pitch, tilt, height = posture(data.qpos, normal, point)
        r = rotation(data.qpos[3:7])
        yaw = math.atan2(r[1, 0], r[0, 0]) - yaw0
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        rr.append(roll); pp.append(pitch); tt.append(tilt); hh.append(height)
        yy.append(math.degrees(yaw))
        lateral.append(float(data.qpos[1] - origin[1]))
        return yaw

    yaw_error = measure()
    snapshot()
    if viewer:
        from mujoco import viewer as mj_viewer
        cm = mj_viewer.launch_passive(model, data)
    else:
        cm = contextlib.nullcontext(None)
    with cm as view:
        for step in range(round(seconds / model.opt.timestep)):
            started = time.monotonic()
            if view is not None and not view.is_running():
                stop = 'viewer closed'; break
            if step % base.DECIMATION == 0:
                command[2] = float(np.clip(-.9*yaw_error, -.2, .2))
                phase = (phase + command[4]*model.opt.timestep*base.DECIMATION) % 1
                history.append(base.build_obs(data, command, prev, last, phase))
                hist = np.stack(history).astype(np.float32)
                feat = features_v3_online(hist)
                with torch.inference_mode():
                    label = None
                    if collect or drive == 'expert':
                        h = torch.from_numpy(hist.reshape(1, -1).copy())
                        label = teacher[0](torch.cat((h, teacher[1](h)), dim=1)).squeeze(0).numpy()
                        if label.shape != (12,) or not np.isfinite(label).all() or np.max(np.abs(label)) > 12:
                            stop = 'invalid teacher action'; break
                        label = np.clip(label, -6, 6).astype(np.float32)
                    if drive == 'expert':
                        raw = label
                    else:
                        x = np.clip((feat-mx)/sx, -5, 5).astype(np.float32)
                        pred, _ = net(torch.from_numpy(x[None]))
                        raw = pred.squeeze(0).numpy()*sy+my
                if raw.shape != (12,) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 12:
                    stop = 'invalid student action'; break
                action = np.clip(raw, -6, 6).astype(np.float64)
                if collect:
                    xs.append(feat); ys.append(label)
                    # Correct student-visited states and emphasize leaning states.
                    risk = min(2.0, (abs(rr[-1])+abs(pp[-1]))/12.0)
                    ws.append((2.0 if drive == 'student' else 1.0)*(1+risk))
                prev, last = last.copy(), action.copy()
                target = base.DEFAULT_JOINT_POS + action*base.ACTION_SCALE
            torque = base.KP*(target-data.qpos[7:19]) - base.KD*data.qvel[6:18]
            lo, hi = model.actuator_ctrlrange.T
            saturation += int(np.any((torque < lo) | (torque > hi)))
            data.ctrl[:] = np.clip(torque, lo, hi)
            previous_time = float(data.time)
            mujoco.mj_step(model, data)
            steps_done += 1
            if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                    or data.time <= previous_time):
                stop = 'invalid or reset simulator state'; break
            yaw_error = measure()
            if step % base.DECIMATION == 0:
                snapshot()
                if view is not None:
                    view.cam.lookat[:] = data.qpos[:3]
                    view.sync()
            if hh[-1] < LIMITS['min_height_m']:
                stop = 'low torso above terrain'; break
            if tt[-1] > LIMITS['stop_tilt_deg']:
                stop = 'excessive tilt'; break
            if view is not None:
                time.sleep(max(0, model.opt.timestep-(time.monotonic()-started)))
    snapshot()
    row = dict(controller=drive, seed=int(seed), slope_deg=float(slope),
               duration_s=float(data.time), forward_m=float(data.qpos[0]-origin[0]),
               final_lateral_m=float(data.qpos[1]-origin[1]),
               max_abs_lateral_m=float(np.max(np.abs(lateral))),
               peak_yaw_deg=float(np.max(np.abs(yy))), min_height_m=float(min(hh)),
               roll_rms_deg=float(np.sqrt(np.mean(np.square(rr)))),
               pitch_rms_deg=float(np.sqrt(np.mean(np.square(pp)))),
               peak_world_tilt_deg=float(max(tt)),
               motor_clipped_fraction=float(saturation/max(1, steps_done)), stop=stop)
    row['passed'] = passes(row, seconds, speed)
    replay = dict(qpos=np.asarray(states), qvel=np.asarray(velocities),
                  sim_time=np.asarray(stamps), executed_action=np.asarray(actions),
                  slope_deg=np.array(slope), plane_normal=normal, plane_point=point,
                  summary_json=np.array(json.dumps(row, allow_nan=False)))
    examples = (np.asarray(xs, np.float32).reshape(-1, 350),
                np.asarray(ys, np.float32).reshape(-1, 12), np.asarray(ws, np.float32))
    return row, replay, examples


def evaluate(student, slopes, seeds, args, folder, teacher=None, drive='student'):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for slope in slopes:
        for seed in seeds:
            row, replay, _ = rollout(slope, seed, args.seconds, args.command,
                                      student, teacher, drive, viewer=args.viewer)
            rows.append(row)
            np.savez_compressed(folder/f'{drive}_slope{slope:+g}_seed{seed}.npz', **replay)
            # Save progress after every trial so an interrupted batch stays usable.
            dump(folder/'episodes.json', rows)
            print(f"{drive} slope={slope:+g} seed={seed}: pass={row['passed']} "
                  f"x={row['forward_m']:.2f}m roll/pitch="
                  f"{row['roll_rms_deg']:.1f}/{row['pitch_rms_deg']:.1f}deg "
                  f"yaw={row['peak_yaw_deg']:.1f} stop={row['stop']}", flush=True)
    with (folder/'episodes.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def ranking(rows):
    return (sum(r['passed'] for r in rows),
            -sum(r['stop'] != 'completed' for r in rows),
            -float(np.mean([r['roll_rms_deg']+r['pitch_rms_deg'] for r in rows])),
            -float(np.mean([r['peak_yaw_deg'] for r in rows])))


def acceptable(candidate, incumbent):
    """No pass-count loss on any tested grade; no >1deg RMS posture regression."""
    for slope in {r['slope_deg'] for r in incumbent}:
        a = [r for r in candidate if r['slope_deg'] == slope]
        b = [r for r in incumbent if r['slope_deg'] == slope]
        if len(a) != len(b) or sum(r['passed'] for r in a) < sum(r['passed'] for r in b):
            return False
        for metric in ('roll_rms_deg', 'pitch_rms_deg'):
            if np.mean([r[metric] for r in a]) > np.mean([r[metric] for r in b])+1.0:
                return False
        # A smaller lean is not progress if the robot simply stops walking.
        for metric, tolerance in (('forward_m', .10), ('duration_s', .02)):
            if np.mean([r[metric] for r in a]) < np.mean([r[metric] for r in b])-tolerance:
                return False
        if np.mean([r['peak_yaw_deg'] for r in a]) > np.mean([r['peak_yaw_deg'] for r in b])+1.0:
            return False
    return ranking(candidate) > ranking(incumbent)


def checkpoint(template, student, info):
    ckpt = copy.deepcopy(template)
    ckpt['model_state'] = {k: v.detach().cpu().clone() for k, v in student[0].state_dict().items()}
    ckpt['source'] = 'V7 supervised repeated DAgger; rollout-evaluated candidate'
    ckpt['v7'] = info
    return ckpt


def replay_examples():
    """Preserve prior training subsets; do not reuse earlier selection episodes."""
    chunks = []
    for name, holdout in [('go2_teacher/go2_teacher_pilot.npz', 2),
                           ('go2_dagger_v2/dagger_rollouts.npz', 5),
                           ('go2_steering_v4/steering_teacher.npz', None)]:
        p = ROOT/'results'/name
        if not p.exists():
            if name.startswith('go2_teacher'):
                raise FileNotFoundError(f'Required old walking data: {p}')
            print('Optional replay dataset absent:', p, flush=True)
            continue
        with np.load(p, allow_pickle=False) as data:
            x, y, ep = features_v3(data)
            mask = ep != holdout if holdout is not None else data['seed'] == 4101
        if mask.any():
            chunks.append((x[mask], np.clip(y[mask], -6, 6).astype(np.float32),
                           np.ones(int(mask.sum()), np.float32)))
    return chunks


def fit(student, chunks, epochs, lr, seed):
    net, (mx, sx, my, sy) = student
    x, y, w = (np.concatenate([part[i] for part in chunks]) for i in range(3))
    if not len(x) or not all(np.isfinite(a).all() for a in (x, y, w)) or np.any(w <= 0):
        raise ValueError('Training examples/weights invalid')
    X = torch.from_numpy(np.clip((x-mx)/sx, -5, 5).astype(np.float32))
    Y = torch.from_numpy(((y-my)/sy).astype(np.float32))
    W = torch.from_numpy(w)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    losses = []
    net.train()
    for epoch in range(epochs):
        total = 0.0
        order = rng.permutation(len(X))
        for start in range(0, len(X), 96):
            ix = order[start:start+96]
            pred, spikes = net(X[ix])
            per = torch.nn.functional.smooth_l1_loss(pred, Y[ix], reduction='none', beta=.5).mean(1)
            loss = (per*W[ix]).sum()/W[ix].sum() + 1e-4*spikes
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(ix)
        losses.append(total/len(X))
        print(f'  epoch {epoch+1}/{epochs} loss={losses[-1]:.6f}', flush=True)
    net.eval()
    return losses


def train(args, out, initial):
    template, student = load_student(initial)
    teacher = load_teacher()
    chunks = replay_examples()
    selected_path = out/'selected.pt'
    # Preserve truthful provenance when no candidate improves.
    torch.save(template, selected_path)
    status = dict(initial_checkpoint=str(initial), initial_sha256=sha(initial),
                  selected_checkpoint=str(selected_path), learned_update_accepted=False,
                  training='Supervised repeated DAgger, not PPO', stages=[])
    dump(out/'training_summary.json', status)
    for stage in range(1, len(args.grades)+1):
        slopes = [0.0] + [sign*g for g in args.grades[1:stage] for sign in (1, -1)]
        print('\nCurriculum slopes:', slopes, flush=True)
        folder = out/f'stage_{stage}'
        expert = evaluate(None, slopes, TRAIN_SEEDS, args, folder/'expert', teacher, 'expert')
        if not all(r['passed'] for r in expert):
            status['blocked_reason'] = 'Teacher failed upright criteria: do not imitate an unqualified gait.'
            dump(out/'training_summary.json', status)
            print(status['blocked_reason'], flush=True)
            return
        # Fresh expert histories for each actual grade. Saved with provenance.
        for slope in slopes:
            for seed in TRAIN_SEEDS:
                _, _, examples = rollout(slope, seed, args.seconds, args.command,
                                          None, teacher, 'expert', collect=True)
                chunks.append(examples)
                np.savez_compressed(folder/f'expert_labels_s{slope:+g}_seed{seed}.npz',
                                    x=examples[0], y=examples[1], weight=examples[2],
                                    seed=seed, slope_deg=slope, source='expert')
        incumbent = evaluate(student, slopes, SELECT_SEEDS, args, folder/'baseline')
        stage_report = dict(slopes=slopes, rounds=[], completed=False)
        status['stages'].append(stage_report)
        for round_id in range(1, args.rounds+1):
            rdir = folder/f'round_{round_id}'
            rdir.mkdir(parents=True)
            before = copy.deepcopy(student[0].state_dict())
            for slope in slopes:
                for seed in TRAIN_SEEDS:
                    row, replay, examples = rollout(slope, seed, args.seconds, args.command,
                                                     student, teacher, collect=True)
                    if not len(examples[0]):
                        raise RuntimeError('No corrective labels: '+row['stop'])
                    chunks.append(examples)
                    slug = f's{slope:+g}_seed{seed}'
                    np.savez_compressed(rdir/f'visited_{slug}.npz', **replay)
                    np.savez_compressed(rdir/f'labels_{slug}.npz', x=examples[0], y=examples[1],
                                        weight=examples[2], seed=seed, slope_deg=slope,
                                        source='student', policy_sha256=sha(selected_path))
            losses = fit(student, chunks, args.epochs, args.lr, 7100+stage*100+round_id)
            rows = evaluate(student, slopes, SELECT_SEEDS, args, rdir/'selection')
            accepted = acceptable(rows, incumbent)
            info = dict(train_seeds=list(TRAIN_SEEDS), selection_seeds=list(SELECT_SEEDS),
                        reserved_test_seeds=list(TEST_SEEDS), grades=slopes, stage=stage,
                        round=round_id, accepted=accepted, limits=LIMITS,
                        source_checkpoint_sha256=sha(initial))
            candidate = checkpoint(template, student, info)
            torch.save(candidate, rdir/'candidate.pt')
            if accepted:
                torch.save(candidate, selected_path)
                incumbent = rows
                status['learned_update_accepted'] = True
            else:
                student[0].load_state_dict(before)
            stage_report['rounds'].append(dict(round=round_id, accepted=accepted,
                                               losses=losses, selected_passes=sum(r['passed'] for r in incumbent)))
            dump(out/'training_summary.json', status)
            print('Candidate ACCEPTED' if accepted else 'Candidate rejected; restored incumbent', flush=True)
        stage_report['completed'] = all(r['passed'] for r in incumbent)
        dump(out/'training_summary.json', status)
        if not stage_report['completed']:
            status['blocked_reason'] = 'Selected student has not passed this stage; no advance to steeper slopes.'
            dump(out/'training_summary.json', status)
            break
    print('\nSelected checkpoint:', selected_path, flush=True)
    print('Selection is not final evidence. Run test on reserved seeds.', flush=True)


def self_test():
    for slope in (-3, 0, 3):
        t = math.radians(slope)
        n = np.array([-math.sin(t), 0., math.cos(t)])
        q = np.array([2., 0., math.tan(t)*2+.30, math.cos(t/2), 0., -math.sin(t/2), 0.])
        roll, pitch, tilt, height = posture(q, n, np.zeros(3))
        assert abs(roll) < 1e-8 and abs(pitch) < 1e-8
        assert abs(height-.30) < 1e-8 and abs(tilt-abs(slope)) < 1e-6
    assert set(TRAIN_SEEDS).isdisjoint(SELECT_SEEDS)
    assert set(TEST_SEEDS).isdisjoint(TRAIN_SEEDS+SELECT_SEEDS)
    row = dict(stop='completed', duration_s=8., forward_m=1.6, max_abs_lateral_m=.1,
               peak_yaw_deg=10., min_height_m=.25, roll_rms_deg=3., pitch_rms_deg=5.,
               peak_world_tilt_deg=10., slope_deg=0.)
    assert passes(row, 8., .45)
    for key, value in [('roll_rms_deg', 13), ('pitch_rms_deg', 13),
                        ('max_abs_lateral_m', .26), ('min_height_m', .17),
                        ('peak_world_tilt_deg', 26), ('stop', 'viewer closed')]:
        assert not passes({**row, key:value}, 8., .45), key
    a = {**row, 'passed': True}
    assert not acceptable([{**a, 'passed':False}], [a])
    assert not acceptable([a], [a])
    assert acceptable([{**a, 'pitch_rms_deg':4.}], [a])
    print('PASS: slope geometry, upright gates, split disjointness, selection safeguards')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', choices=('self-test', 'check', 'train', 'test'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--baseline', type=Path, help='Optional paired SNN comparison during test')
    p.add_argument('--grades', nargs='+', type=float, default=[0., 1., 2.],
                   help='Training curriculum, ascending nonnegative grades; trains both signs')
    p.add_argument('--slopes', nargs='+', type=float, default=[0., 1., -1., 2., -2.])
    p.add_argument('--seeds', nargs='+', type=int, default=list(TEST_SEEDS))
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--command', type=float, default=.45)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.00005)
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--ann', action='store_true', help='Separate teacher baseline during test')
    args = p.parse_args()
    if args.task == 'self-test':
        self_test(); return
    if not (1 <= args.seconds <= 20 and .1 <= args.command <= .6):
        p.error('Use duration 1..20s and speed .1...6m/s')
    if (not args.grades or args.grades[0] != 0 or args.grades != sorted(set(args.grades))
            or any(not math.isfinite(x) or not 0 <= x <= 3 for x in args.grades)):
        p.error('Grades must start at 0 and increase uniquely, at most 3 degrees')
    if len(set(args.slopes)) != len(args.slopes) or any(not math.isfinite(x) or abs(x)>3 for x in args.slopes):
        p.error('Use unique finite slopes within +/-3 degrees')
    if not (1 <= args.rounds <= 5 and 1 <= args.epochs <= 30 and 0 < args.lr <= .001):
        p.error('Invalid training settings')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        p.error('Use unique nonnegative seeds')
    if args.task == 'test' and set(args.seeds).intersection(TRAIN_SEEDS+SELECT_SEEDS):
        p.error('Test seeds must not overlap training or selection seeds')
    if args.viewer and (args.task != 'test' or len(args.seeds)*len(args.slopes) != 1 or args.ann or args.baseline):
        p.error('Viewer requires one test seed/slope and no paired baseline')
    if args.task == 'test' and args.checkpoint is None:
        p.error('Test requires explicit --checkpoint PATH printed by training')
    runtime()
    initial = choose_initial(args.checkpoint)
    _, student = load_student(initial)
    if args.task == 'check':
        load_teacher()
        replay_examples()
        model, n, point = terrain(2.)
        print('PASS: model, checkpoint, teacher hashes, training data, +2deg floor normal', n)
        print('Initialization:', initial)
        print('This preflight does not establish walking ability.')
        return
    random.seed(20261119); np.random.seed(20261119); torch.manual_seed(20261119)
    out = ROOT/'results/go2_upright_v7'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'mujoco', 'mujoco-menagerie')}
    dump(out/'manifest.json', dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
         versions=versions, limits=LIMITS, source_sha256=sha(Path(__file__)),
         initial_checkpoint=str(initial), initial_sha256=sha(initial),
         train_seeds=TRAIN_SEEDS, selection_seeds=SELECT_SEEDS, reserved_test_seeds=TEST_SEEDS))
    print('Results:', out, flush=True)
    if args.task == 'train':
        train(args, out, initial)
    else:
        rows = evaluate(student, args.slopes, args.seeds, args, out/'student')
        report = {'student_passes':sum(r['passed'] for r in rows), 'episodes':len(rows)}
        if args.baseline:
            _, old = load_student(args.baseline)
            old_rows = evaluate(old, args.slopes, args.seeds, args, out/'baseline')
            report['baseline_passes'] = sum(r['passed'] for r in old_rows)
            report['baseline_sha256'] = sha(args.baseline)
        if args.ann:
            teacher_rows = evaluate(None, args.slopes, args.seeds, args, out/'ann', load_teacher(), 'expert')
            report['teacher_passes'] = sum(r['passed'] for r in teacher_rows)
        report['note'] = 'Initial-pose seeds in one simulator, not broad terrain or hardware robustness.'
        dump(out/'test_summary.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
