# NeuroStride Go2 — SNN V2 + live training previews + exact episode replay

Date: 14 November 2026. Platform: Windows PowerShell, Python `.venv`, CPU / Intel UHD, MuJoCo Menagerie Unitree Go2.

## What actually happened

Six DAgger V1-driven MuJoCo episodes provided 914 labeled policy states; all stopped early at the 18 cm torso-height threshold. 81 of 914 recorded motor choices were from the external ANN teacher during mixed-control rollouts. There was **no seven-second SNN success**. This is not a final gait-learning result.

On the uploaded genuine data, **V2 fine-tuning completed** (30 epochs, same 2-layer surrogate-gradient LIF spiking actor):

- Training: 1,800 original expert observations plus 799 DAgger-correction observations. Teacher DAgger offsets clipped to executable ±6 before loss computation. DAgger samples weighted 2× relative to ordinary expert demonstrations. Original V1 feature/target normalization retained.
- Validation for choosing a V2 checkpoint: **all 115 samples from DAgger episode 3**, kept out of V2 weight updates. V1 error = 1.2725, best V2 epoch 30 = **0.9141 RMSE** (clipped raw joint-offset units, ~28% lower).
- Old ordinary teacher trajectory (0.50 m/s) diagnostic: V1 error 0.07334, V2 error **0.09605** (worse). It was used for V1 checkpoint selection before, so is not untouched test data.
- **No V2 independent closed-loop evaluation has been done here.** V2 may still fall. Spikes per neuron per time step are not joules/energy efficiency.

## Install without disrupting your existing project

Extract `NeuroStride_Go2_V2_Visual_Lab.zip` directly into `C:\Users\STUDENT_SMART_0\Desktop\NeuroStride_Starter` keeping subfolders. Check these files exist:

```
NeuroStride_Starter/
    go2_visual_replay.py
    train_go2_snn_v2_live.py
    train_go2_snn_v1.py                 (already there)
    go2_collect_teacher.py             (already there)
    go2_snn_closed_loop.py             (already there)
    results/
        go2_snn_v1/go2_snn_v1_best.pt (already there)
        go2_snn_v2/go2_snn_v2_best.pt <-- new trained model
        go2_teacher/go2_teacher_pilot.npz (already there)
        go2_dagger_v2/dagger_rollouts.npz  (already there)
```

## A — Watch the NEW V2 spiking actor walking/failing live

```
.\.venv\Scripts\python.exe go2_visual_replay.py record --controller snn --checkpoint results/go2_snn_v2/go2_snn_v2_best.pt --command 0.50 --seconds 8 --viewer --label v2_first
```

The MuJoCo window shows the SNN-controlled Go2 **without querying the ANN teacher**. The script prints forward motion, sideways drift, torso height and termination reason; it also saves a `.npz` recording in `results/go2_replays/`. If it falls, that is the actual result. DO NOT report it as walking successfully before seeing the output.

To compare with the SAME base conditions and commanded speed, execute V1:

```
.\.venv\Scripts\python.exe go2_visual_replay.py record --controller snn --checkpoint results/go2_snn_v1/go2_snn_v1_best.pt --command 0.50 --seconds 8 --viewer --label v1_control
```

Or replay the external ANN teacher (this is NOT SNN):

```
.\.venv\Scripts\python.exe go2_visual_replay.py record --controller teacher --command 0.50 --seconds 8 --viewer --label pretrained_ann
```

## B — Replay the EXACT saved episode

Use the filename printed under `RECORDED:` from Step A. Example ONLY — use the exact path printed by your own run:

```
.\.venv\Scripts\python.exe go2_visual_replay.py play --file "results/go2_replays/snn_v2_first_cmd0.50_3632ms.npz" --loop
```

You can also add `--speed 0.5` for half-speed replay, `--speed 2` for faster. Playback re-displays saved joint/body state frames exactly, without policy inference or physics integration. The old DAgger dataset lacks `qpos` and `qvel`, so the six historical DAgger episodes cannot be replayed precisely unless recollected with recording enabled.

## C — See SNN weight training *and* live Go2 previews

```
.\.venv\Scripts\python.exe train_go2_snn_v2_live.py --live --epochs 30 --preview-every 10
```

Every ten epochs, supervised weight updates **pause** and the current checkpoint is tested independently inside a MuJoCo viewer, then training resumes. This is the honest visualization for **offline behavior cloning plus DAgger**: robot movement is an *evaluation preview*, not a continuous on-policy reinforcement learning episode producing gradients. Training progress and the new checkpoints are written to `results/go2_snn_v2_live` so the supplied `results/go2_snn_v2/go2_snn_v2_best.pt` is not overwritten.

This uses existing `results/go2_teacher/go2_teacher_pilot.npz` and `results/go2_dagger_v2/dagger_rollouts.npz` and requires that both were collected as described. For performance on a CPU PC, close other resource-intensive apps during rendered previews; `record` without `--viewer` is faster.

## Research integrity / limitations

- Source ANN teacher: Ashfak-Kausik/rl-locomotion-learning v1.0-policy; weights verified separately by SHA-256. Source training data and provenance are third-party.
- NeuroStride V1 and V2 are **spiking student actors** trained via supervised imitation with analog motor-action readout; not fully spiking end-to-end. No claim about electrical energy.
- DAgger correction rollout includes samples from both SNN-controlled and teacher-controlled frames, specifically labeled. A mixed-control run is not SNN-only success.
- No rough-terrain SNN locomotion yet. Use fresh robot starting conditions and terrain variations for future robustness evaluation, and report sample sizes.
- `go2_visual_replay.py` requires MuJoCo/Menagerie on the user's Windows machine. Python syntax and PyTorch V2 training have been checked in the assistant environment; the viewer/integration still requires testing on your machine.
