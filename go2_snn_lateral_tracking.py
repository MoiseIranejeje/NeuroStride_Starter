"""NeuroStride | sideways path tracking with neural-only yaw at inference.

Preserves the validated frozen V6 walking SNN and the existing yaw5 LIF SNN.
Adds a SEPARATELY TRAINED LIF lateral-position correction head. At inference:
  yaw_rate = clamp(yaw_SNN(heading_error) + lateral_SNN(cross_track), +/- cap)
No ANN and no classical lateral/yaw feedback calculation runs in ``neural``.
Shared low-level PD motors are unchanged. The residual head learns by supervised
imitation of a hand-designed lateral control law; THIS IS NOT PPO/RL.

Windows VS Code PowerShell from NeuroStride_Starter (save THIS .py beside
existing go2_yaw5_fasttrack.py, go2_snn_yaw_distillation.py):

  python go2_snn_lateral_tracking.py sweep --seeds 7001 7003 7005 --speeds 0.35 0.55
  python go2_snn_lateral_tracking.py train --epochs 35
  python go2_snn_lateral_tracking.py test --seeds 7001 7002 7003 7004 7005 --speeds 0.35 0.55

Reserve new seeds 7101..7110 for one-time validation after a successful dev test.
Viewer reruns conditions (not an exact NPZ replay); use one mode/speed/seed.
All outputs use results/go2_snn_lateral_tracking, leaving old CSVs untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import go2_yaw5_fasttrack as fast
import go2_snn_yaw_distillation as yaw
import go2_snn_self_steering as ss

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "go2_snn_lateral_tracking"
SELECT = OUT / "selected_lateral_gain.json"
CHECKPOINT = OUT / "lateral_spiking_controller.pt"
YAW_WEIGHTS = ROOT / "results" / "go2_yaw5_fasttrack" / "yaw5_spiking_controller.pt"
LATERAL_COMMAND_LIMIT = 0.15    # rad/s, bounded residual steering authority
LATERAL_NORMALIZER = 0.20      # metres; fixed network input scale
GAIN_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)  # (rad/s) per metre
TRAIN_SEED = 20261121


def make_lateral_net():
    """A true 2-hidden-layer LIF spiking network; analog command readout."""
    return yaw.make_yaw_net()  # feature_dim=1, two 64-LIF layers, 12 spike steps


def lateral_reference(cross_track_m, gain):
    """CLASSICAL DIAGNOSTIC and OFFLINE TRAINING TARGET ONLY."""
    y = np.asarray(cross_track_m, dtype=np.float32)
    return np.clip(-float(gain) * y, -LATERAL_COMMAND_LIMIT,
                   LATERAL_COMMAND_LIMIT).astype(np.float32)


def lateral_neural_command(net, cross_track_m):
    """Pure LIF correction, with neutral subtraction for zero at zero offset."""
    if isinstance(cross_track_m, torch.Tensor):
        lateral = cross_track_m.float().reshape(-1, 1)
    else:
        lateral = torch.as_tensor(cross_track_m, dtype=torch.float32).reshape(-1, 1)
    x = (lateral / LATERAL_NORMALIZER).clamp(-3, 3)
    raw, spikes = net(x)
    neutral, _ = net(torch.zeros_like(x))
    residual = LATERAL_COMMAND_LIMIT * (raw[:, 0] - neutral[:, 0])
    return residual.clamp(-LATERAL_COMMAND_LIMIT, LATERAL_COMMAND_LIMIT), spikes


def load_yaw_model():
    if not YAW_WEIGHTS.is_file():
        raise FileNotFoundError(f"Validated yaw SNN checkpoint missing: {YAW_WEIGHTS}")
    if not fast.SELECTION.is_file():
        raise FileNotFoundError(f"Original yaw-law selection missing: {fast.SELECTION}")
    selected = fast.read_selection()
    net = yaw.make_yaw_net()
    cp = torch.load(YAW_WEIGHTS, map_location="cpu", weights_only=True)
    net.load_state_dict(cp["state_dict"])
    net.eval()
    return net, selected


def model_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cross_track_m(qpos, origin_xy, yaw_initial):
    """Signed lateral displacement perpendicular to the starting heading."""
    dx = float(qpos[0] - origin_xy[0])
    dy = float(qpos[1] - origin_xy[1])
    return -math.sin(yaw_initial) * dx + math.cos(yaw_initial) * dy


def control_command(yaw_net, head_err_deg, lateral_pos_m, selected,
                    mode, lateral_gain=0.0, lateral_net=None):
    """Return actual yaw command; no classical control in baseline/neural."""
    with torch.inference_mode():
        yaw_cmd, yaw_spikes = fast.neural_command(yaw_net, [head_err_deg])
    base_cmd = float(yaw_cmd.item())
    if mode == "baseline":
        correction = 0.0
        lateral_spikes = 0.0
    elif mode == "diagnostic":
        correction = float(lateral_reference([lateral_pos_m], lateral_gain)[0])
        lateral_spikes = 0.0
    elif mode == "neural":
        if lateral_net is None:
            raise RuntimeError("Neural lateral mode needs its trained LIF checkpoint")
        with torch.inference_mode():
            residual, lateral_spikes_tensor = lateral_neural_command(lateral_net, [lateral_pos_m])
        correction = float(residual.item())
        lateral_spikes = float(lateral_spikes_tensor)
    else:
        raise ValueError(mode)
    cap = float(selected["cap_rad_s"])
    return float(np.clip(base_cmd + correction, -cap, cap)), float(yaw_spikes), lateral_spikes


def rollout(model, base, yaw_net, selected, mode, seed, speed, seconds,
            lateral_gain=0.0, lateral_net=None, viewer=False):
    """Same MuJoCo/PD/50Hz V6 loop as go2_snn_yaw_distillation.run.

    Expanded snapshot records full trajectory and cross-track per physics step
    for a true peak path-error measurement, beyond final sideways position.
    """
    import mujoco
    from go2_heading_benchmark import heading_rad, wrap_pi
    from go2_first_terrain_pilot import replay_metrics

    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home < 0:
        raise RuntimeError("MuJoCo Go2 'home' keyframe missing")
    mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[7:19] = ss.DEFAULT_JOINT_POS
    data.qpos[2] = 0.30
    rng = np.random.default_rng(seed)
    data.qpos[7:19] += rng.normal(0, 0.012, 12)
    data.qpos[2] += float(rng.uniform(-0.005, 0.005))
    mujoco.mj_forward(model, data)
    if model.nu != 12 or abs(model.opt.timestep - 0.002) > 1e-9:
        raise RuntimeError("Unexpected Go2 actuator count/timestep")
    if [model.actuator(i).name for i in range(12)] != ss.EXPECTED_ACTUATORS:
        raise RuntimeError("Go2 actuator order unexpectedly changed")

    origin = data.qpos[:2].copy()
    yaw_initial = heading_rad(data.qpos[3:7])
    cmd = ss.BASE_COMMAND.copy()
    cmd[0] = float(speed)
    h = deque((np.zeros(ss.OBS_DIM, np.float32) for _ in range(ss.HISTORY_LEN)),
              maxlen=ss.HISTORY_LEN)
    prev = np.zeros(12, dtype=np.float64)
    last = prev.copy()
    target = ss.DEFAULT_JOINT_POS.copy()
    phase = 0.0
    lows, highs = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    stop = "completed"
    peak_yaw, peak_cross, min_height = 0.0, 0.0, 10.0
    decisions, spike_sum, inference_seconds = 0, 0.0, 0.0
    snapshots = dict(qpos=[], qvel=[], sim_time=[], joint_torque=[], yaw_cmd=[],
                     cross_track_m=[], heading_error_deg=[])

    def snapshot():
        stamp = float(data.time)
        if snapshots["sim_time"] and stamp == snapshots["sim_time"][-1]:
            return
        xpos = cross_track_m(data.qpos, origin, yaw_initial)
        heading = math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - yaw_initial))
        snapshots["qpos"].append(data.qpos.copy())
        snapshots["qvel"].append(data.qvel.copy())
        snapshots["sim_time"].append(stamp)
        snapshots["joint_torque"].append(data.ctrl.copy())
        snapshots["yaw_cmd"].append(float(cmd[2]))
        snapshots["cross_track_m"].append(xpos)
        snapshots["heading_error_deg"].append(heading)

    snapshot()
    cm = contextlib.nullcontext(None)
    if viewer:
        import mujoco.viewer
        cm = mujoco.viewer.launch_passive(model, data)
    with cm as view:
        for step in range(int(round(seconds / model.opt.timestep))):
            if viewer and not view.is_running():
                stop = "viewer closed"
                break
            if step % ss.DECIMATION == 0:
                heading = math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - yaw_initial))
                sideways = cross_track_m(data.qpos, origin, yaw_initial)
                t0 = time.perf_counter()
                cmd[2], s_yaw, s_lateral = control_command(
                    yaw_net, heading, sideways, selected, mode,
                    lateral_gain=lateral_gain, lateral_net=lateral_net)
                spike_sum += s_yaw + s_lateral
                phase = (phase + cmd[4] * model.opt.timestep * ss.DECIMATION) % 1.0
                obs = ss.build_obs(data, cmd, prev, last, phase)
                h.append(obs)
                history = np.stack(h).astype(np.float32)
                with torch.inference_mode():
                    actor, stats = base
                    mx, sx, my, sy = stats
                    feat = ss.local_features(history[None])[0]
                    x = np.clip((feat - mx) / sx, -5, 5).astype(np.float32)
                    raw, spikes = actor(torch.from_numpy(x[None]))
                    action = raw.squeeze(0).numpy() * sy + my
                spike_sum += float(spikes)
                inference_seconds += time.perf_counter() - t0
                if (action.shape != (12,) or not np.all(np.isfinite(action))
                        or np.max(np.abs(action)) > 12):
                    stop = "invalid motor action"
                    break
                action = np.clip(action, -6, 6).astype(np.float64)
                prev, last = last.copy(), action.copy()
                target = ss.DEFAULT_JOINT_POS + last * ss.ACTION_SCALE
                decisions += 1
            data.ctrl[:] = np.clip(
                ss.KP * (target - data.qpos[7:19]) - ss.KD * data.qvel[6:18],
                lows, highs)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                stop = "nonfinite state"
                break
            z_local = float(data.qpos[2])  # flat floor; no slope in this experiment
            min_height = min(min_height, z_local)
            heading_now = math.degrees(wrap_pi(heading_rad(data.qpos[3:7]) - yaw_initial))
            peak_yaw = max(peak_yaw, abs(heading_now))
            peak_cross = max(peak_cross, abs(cross_track_m(data.qpos, origin, yaw_initial)))
            if step % ss.DECIMATION == 0:
                snapshot()
                if viewer and view.is_running():
                    view.cam.lookat[:] = data.qpos[:3]
                    view.sync()
                    time.sleep(model.opt.timestep * ss.DECIMATION)
            if z_local < 0.18:
                stop = "torso below local ground height threshold"
                break
        snapshot()

    OUT.joinpath("replays").mkdir(parents=True, exist_ok=True)
    name = f"{mode}_k{lateral_gain:.2f}_speed{speed:.2f}_seed{seed}"
    replay = OUT / "replays" / (name + ".npz")
    if replay.exists():
        archive = OUT / "replays" / (name + "__previous_" +
                                     f"{time.time_ns()}" + ".npz")
        replay.rename(archive)
    np.savez_compressed(replay, **{k: np.asarray(v) for k, v in snapshots.items()})
    dx, dy = (float(n) for n in data.qpos[:2] - origin)
    final_cross = cross_track_m(data.qpos, origin, yaw_initial)
    passed15 = bool(stop == "completed" and data.time >= seconds - 0.002
                    and dx >= 1.5 and abs(dy) <= 0.25
                    and peak_yaw <= 15.0 and min_height >= 0.18)
    posture = replay_metrics(replay, 0.0)
    return dict(mode=mode, seed=int(seed), command_mps=float(speed),
                time_s=round(float(data.time), 4), forward_m=round(dx, 5),
                sideways_m=round(dy, 5), final_cross_track_m=round(final_cross, 5),
                peak_cross_track_m=round(peak_cross, 5),
                max_abs_yaw_deg=round(peak_yaw, 4),
                pitch_rms_deg=round(float(posture["pitch_rms_deg"]), 4),
                roll_rms_deg=round(float(posture["roll_rms_deg"]), 4),
                minimum_local_height_m=round(min_height, 5),
                pass_15deg=passed15,
                pass_10deg=bool(passed15 and peak_yaw <= 10.0),
                pass_path10cm=bool(passed15 and peak_cross <= 0.10),
                pass_combined=bool(passed15 and peak_yaw <= 10.0 and peak_cross <= 0.10),
                stop=stop, mean_spike_rate=round(spike_sum/max(decisions, 1), 5),
                wall_inference_ms=round(1000.0*inference_seconds/max(decisions, 1), 3),
                replay=str(replay))


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test(seeds, speeds, seconds, modes, viewer=False):
    from go2_first_terrain_pilot import build_terrain
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if viewer and len(seeds)*len(speeds)*len(modes) != 1:
        raise ValueError("--viewer requires one seed, one speed, one mode")
    base = fast.prepare_base()
    yaw_net, selected = load_yaw_model()
    yaw.YAW_LIMIT = float(selected["cap_rad_s"])
    if not SELECT.is_file():
        raise FileNotFoundError(f"Run sweep first: {SELECT}")
    selection = json.loads(SELECT.read_text(encoding="utf-8"))
    gain = float(selection["lateral_gain"])
    lateral_net = None
    if "neural" in modes:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Train lateral SNN first: {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        if abs(float(ckpt["lateral_gain"]) - gain) > 1e-8:
            raise RuntimeError("Trained lateral head doesn't match selected diagnostic gain")
        lateral_net = make_lateral_net()
        lateral_net.load_state_dict(ckpt["state_dict"])
        lateral_net.eval()
    world = build_terrain(0.0)
    rows = []
    for speed in speeds:
        for seed in seeds:
            for mode in modes:
                r = rollout(world, base, yaw_net, selected, mode, seed, speed,
                            seconds, lateral_gain=gain, lateral_net=lateral_net,
                            viewer=viewer)
                rows.append(r)
                print(f"{mode:>10} speed={speed:.2f} seed={seed}: "
                      f"X={r['forward_m']:.3f}m Y={r['sideways_m']:+.3f}m "
                      f"peak_path={r['peak_cross_track_m']:.3f}m "
                      f"yaw={r['max_abs_yaw_deg']:.2f}° "
                      f"pass10={r['pass_10deg']} combined={r['pass_combined']}", flush=True)
    write_csv(rows, OUT / "evaluation" / "episodes.csv")
    summary = {}
    for speed in speeds:
        for mode in modes:
            g = [r for r in rows if r["mode"] == mode and r["command_mps"] == speed]
            summary[f"{mode}_speed{speed:.2f}"] = dict(
                episodes=len(g), passes_15=sum(x["pass_15deg"] for x in g),
                passes_10=sum(x["pass_10deg"] for x in g),
                passes_combined=sum(x["pass_combined"] for x in g),
                mean_abs_final_Y_m=float(np.mean([abs(x["sideways_m"]) for x in g])),
                mean_peak_cross_track_m=float(np.mean([x["peak_cross_track_m"] for x in g])),
                mean_peak_yaw_deg=float(np.mean([x["max_abs_yaw_deg"] for x in g])))
    (OUT/"evaluation"/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def sweep(seeds, speeds, seconds):
    from go2_first_terrain_pilot import build_terrain
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base = fast.prepare_base()
    yaw_net, selected = load_yaw_model()
    yaw.YAW_LIMIT = float(selected["cap_rad_s"])
    world = build_terrain(0.0)
    rows, ranked = [], []
    for gain in GAIN_CANDIDATES:
        group = []
        for speed in speeds:
            for seed in seeds:
                mode = "baseline" if gain == 0.0 else "diagnostic"
                r = rollout(world, base, yaw_net, selected, mode, seed, speed,
                            seconds, lateral_gain=gain)
                group.append(r)
                rows.append(dict(lateral_gain=gain, **r))
                print(f"gain={gain:.2f} speed={speed:.2f} seed={seed}: "
                      f"finalY={r['sideways_m']:+.3f}m "
                      f"peakPath={r['peak_cross_track_m']:.3f}m "
                      f"yaw={r['max_abs_yaw_deg']:.2f}° "
                      f"8sec={r['stop']=='completed'}", flush=True)
        valid = all(r["pass_15deg"] and
                    (r["max_abs_yaw_deg"] <= (10 if r["command_mps"] >= .50 else 12))
                    for r in group)
        ranked.append(dict(lateral_gain=gain, valid=bool(valid),
                           mean_abs_final_Y_m=float(np.mean([abs(r["sideways_m"]) for r in group])),
                           mean_peak_cross_track_m=float(np.mean([r["peak_cross_track_m"] for r in group])),
                           worst_peak_cross_track_m=float(max(r["peak_cross_track_m"] for r in group)),
                           worst_yaw_deg=float(max(r["max_abs_yaw_deg"] for r in group)),
                           pass_combined=sum(r["pass_combined"] for r in group),
                           episodes=len(group)))
    write_csv(rows, OUT / "sweep_episodes.csv")
    ranked.sort(key=lambda r: (not r["valid"], r["mean_peak_cross_track_m"],
                               r["mean_abs_final_Y_m"]))
    (OUT / "sweep_ranking.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    safe = [r for r in ranked if r["valid"]]
    if not safe:
        raise RuntimeError("No lateral gain preserved walking/heading; keep original yaw SNN")
    top = safe[0]
    baseline = next((r for r in ranked if r["lateral_gain"] == 0.0), None)
    improvement = (baseline is not None and
                   top["mean_peak_cross_track_m"] <= 0.85*baseline["mean_peak_cross_track_m"])
    choice = dict(top, baseline_mean_peak_cross_track_m=baseline["mean_peak_cross_track_m"],
                  improved_path_by_15pct=bool(improvement), development_seeds=seeds,
                  development_speeds=speeds,
                  teacher="lateral residual -gain * cross_track (OFFLINE/diagnostic only)",
                  next="train neural residual only if gain >0 and improvement demonstrated")
    SELECT.parent.mkdir(parents=True, exist_ok=True)
    SELECT.write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print("\nSELECTED:", json.dumps(choice, indent=2), flush=True)
    if not improvement or top["lateral_gain"] == 0.0:
        print("NO strong path benefit demonstrated; DO NOT assume neural distillation will help.", flush=True)


def train(epochs, lr, batch_size):
    if not SELECT.is_file():
        raise FileNotFoundError(f"Run 'sweep' first: {SELECT}")
    selection = json.loads(SELECT.read_text(encoding="utf-8"))
    gain = float(selection["lateral_gain"])
    if gain <= 0 or not selection.get("improved_path_by_15pct"):
        raise RuntimeError("Diagnostic lateral law lacks >=15% path improvement; retain baseline")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(TRAIN_SEED)
    rng = np.random.default_rng(TRAIN_SEED)
    train_y = rng.uniform(-.35, .35, size=3600).astype(np.float32)
    valid_y = np.linspace(-.35, .35, 401, dtype=np.float32)
    net = make_lateral_net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    target = torch.from_numpy(lateral_reference(train_y, gain))
    xv = torch.from_numpy(valid_y)
    yv = torch.from_numpy(lateral_reference(valid_y, gain))
    x = torch.from_numpy(train_y)
    best = float("inf")
    history = []
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for idx in np.array_split(rng.permutation(len(x)), math.ceil(len(x)/batch_size)):
            pred, spikes = lateral_neural_command(net, x[idx])
            loss = ((pred - target[idx])**2).mean() + 1e-6*spikes
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.inference_mode():
            prediction, spikes = lateral_neural_command(net, xv)
            err = prediction-yv
            rmse = float(torch.sqrt((err*err).mean()))
            worst = float(err.abs().max())
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)),
                            held_grid_rmse_rad_s=rmse, held_grid_max_error_rad_s=worst,
                            mean_lif_spike_rate=float(spikes)))
        if rmse < best:
            best = rmse
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                            lateral_gain=gain, epoch=epoch, held_grid_rmse_rad_s=rmse,
                            learning="supervised classical lateral residual distillation; NOT RL",
                            input="measured cross track metres normalized by 0.2",
                            residual_cap_rad_s=LATERAL_COMMAND_LIMIT), CHECKPOINT)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d}/{epochs}: held RMSE={rmse:.5f} "
                  f"max={worst:.5f} rad/s", flush=True)
    write_csv(history, OUT / "training_history.csv")
    (OUT / "training_summary.json").write_text(json.dumps(dict(
        best_held_grid_rmse_rad_s=best,
        selected_gain=gain,
        method="offline classical lateral residual -> LIF SNN; NOT RL",
        warning="offline RMSE does not guarantee robot path-tracking improvement",
        checkpoint=str(CHECKPOINT))), encoding="utf-8")
    print("Saved", CHECKPOINT, "sha256", model_hash(CHECKPOINT), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="job", required=True)
    p1 = sub.add_parser("sweep")
    p2 = sub.add_parser("train")
    p3 = sub.add_parser("test")
    for p in (p1, p3):
        p.add_argument("--seeds", nargs="+", type=int, default=[7001, 7003, 7005])
        p.add_argument("--speeds", nargs="+", type=float, default=[0.35, 0.55])
        p.add_argument("--seconds", type=float, default=8.0)
    p2.add_argument("--epochs", type=int, default=35)
    p2.add_argument("--lr", type=float, default=.003)
    p2.add_argument("--batch", type=int, default=256)
    p3.add_argument("--modes", nargs="+", choices=("baseline", "neural", "diagnostic"),
                    default=["baseline", "neural", "diagnostic"])
    p3.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.job == "train":
        if not 1 <= args.epochs <= 120 or not 1e-5 <= args.lr <= .01 or not 32 <= args.batch <= 1024:
            parser.error("Train: epochs 1..120, lr 1e-5..0.01, batch 32..1024")
        train(args.epochs,args.lr,args.batch)
    else:
        if not 1 <= len(args.seeds) <= 20 or len(set(args.seeds)) != len(args.seeds):
            parser.error("Choose 1..20 unique seeds")
        if not 1 <= len(args.speeds) <= 5 or any(s < .1 or s > .7 for s in args.speeds):
            parser.error("Choose 1..5 speeds between 0.1 and 0.7 m/s")
        if not 1 <= args.seconds <= 12:
            parser.error("Duration must be 1..12 seconds")
        if args.job == "test" and args.viewer and len(args.seeds)*len(args.speeds)*len(args.modes)!=1:
            parser.error("Viewer requires one mode, seed, speed")
        if args.job == "sweep":
            sweep(args.seeds,args.speeds,args.seconds)
        else:
            test(args.seeds,args.speeds,args.seconds,args.modes,args.viewer)


if __name__ == "__main__":
    main()
