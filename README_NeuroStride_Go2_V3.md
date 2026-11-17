# NeuroStride Go2 — SNN V3 (16 November 2026)

## What has actually been done

V3 is a trained surrogate-gradient LIF SNN model, initialized **exactly from the
user's V2 weights** and fine-tuned using the user-provided ANN-teacher and
DAgger datasets. This is **supervised learning** of 12 joint-position-offset
commands. MuJoCo converts them to torques through the unchanged low-level PD
controller. It is not PPO training or a demonstration of neuromorphic energy
savings.

**V3's new input:** 350 values: current Go2 observation (70), mean of past
30 observations (70), and observations from 1, 5 and 15 policy decisions ago
(70 each). This retains temporal variation absent from V2's 140-input model.
Still two hidden spiking LIF layers with 128 neurons each, 8 internal spike
steps per 50 Hz motor-control decision and 12 continuous action outputs.

**Offline learning evidence (not walking):**

- Warm-start V2-to-V3 verification: maximum output discrepancy **0**, when
  extra temporal weights are zero.
- 40 training epochs; 1,800 normal ANN teacher samples and 731 DAgger
  corrective samples; full DAgger episode 5 (183 frames) reserved for V3
  checkpoint selection (other 5 DAgger episodes used in training).
- DAgger ep5 clipped-action RMSE: **0.3682527 (initial V2) → 0.3503002
  (selected V3)**, roughly 4.9% improvement in this *nonindependent* metric.
- Old .50 m/s ANN teacher-state diagnostic RMSE: **0.0960476 (V2) →
  0.0947675 (V3)**. The old .50 command was previously involved in V1
  selection. Neither metric is an untouched test, and neither establishes
  stable closed-loop walking.
- Original data are four deterministic ANN expert runs and six short SNN V1
  DAgger runs on flat ground. This remains a limited dataset.

## Install in your EXISTING working NeuroStride_Starter project

Extract the entire ZIP to **the project root** (the directory with
`go2_collect_teacher.py`). Keep nested `results/...` folders. You do NOT
need to retrain V3 to evaluate the provided model.

The package contains a copy of the original v2 checkpoint/data as exact
reproduction inputs and scripts. Preserve your existing research files.

You need `torch`, `numpy`, `mujoco`, and `mujoco-menagerie` installed in the
same Python interpreter. No NVIDIA CUDA GPU is required for inference.

### ONE decisive comparison (both V2 and V3 run with identical starting states)

Works in GitHub Codespaces **Bash** or local Windows PowerShell:

```bash
python go2_compare_v3.py --commands 0.45 --seeds 3101 3102 3103 --seconds 8
```

This executes 6 total **headless MuJoCo rollouts**: V2 and V3 for each of
three deterministic initial-pose perturbations. Neither ANN teacher nor its
`.jit` files are loaded. Each run outputs:

- time until torso-height threshold or 8 simulated seconds
- net forward and lateral displacement, cumulative lateral travel
- min torso height and explicit termination reason
- neural-inference count and mean hidden spiking rate
- pilot pass: 8s survival, >=1.5m net forward, <=0.5m |final Y|, min Z >=0.18m

Results written under `results/go2_snn_v3/evaluation/`:

- `paired_results.csv`
- `paired_summary.json`
- exact recorded `v2_cmd0.45_seed3101.npz`, `v3_cmd0.45_seed3101.npz`, etc.

**Interpretation rules:** The three seeds are exploratory and constitute a
small sample, not a robustness conclusion. They become exposed after use and
cannot later serve as untouched final-test seeds. Because V2 closed-loop
behavior has NOT been reported yet, do not claim V3 is better before checking
these paired results. A movement failure on V3 remains a valid negative result.

### Live viewer on your Windows desktop (not ordinary Codespaces)

```powershell
python go2_compare_v3.py --commands 0.45 --seeds 3101 --seconds 8 --viewer
```

Only V3 is shown in the live viewer; V2 is run headlessly under the same
initial pose. No policy gradients happen here: it is model evaluation.

### Exact episode replay (Windows GUI)

```powershell
python go2_visual_replay.py play --file results/go2_snn_v3/evaluation/v3_cmd0.45_seed3101.npz --speed 0.7
```

Replay is of actual stored MuJoCo qpos/qvel states, not a freshly generated
walk. You can also replay V2 by changing `v3` to `v2` in the filename.

### MP4 playback in headless Codespaces

Install additional components if needed (`imageio`, `imageio-ffmpeg`, working
software OpenGL libraries) and render using the supplied renderer:

```bash
MUJOCO_GL=osmesa python go2_render_mp4.py \
  --file results/go2_snn_v3/evaluation/v3_cmd0.45_seed3101.npz
```

It should save under `results/go2_videos/`. If Codespaces lacks OSMesa,
install its system library or use a local Windows viewer instead.

## Reproduce V3 supervised learning (OPTIONAL; may take time on CPU)

```bash
python train_go2_snn_v3.py --epochs 40
```

This uses the authentic stored `results/go2_teacher/go2_teacher_pilot.npz`,
`results/go2_dagger_v2/dagger_rollouts.npz`, and
`results/go2_snn_v2/go2_snn_v2_best.pt`. It overwrites only the V3 output
folder. The supplied V3 checkpoint was already trained and verified; no need
to do this before testing.

## Next objective and week-of-Nov16 success gates

1. **Nov 16–17:** Independently evaluate V2 and V3 on same starts and commands.
   Save simulator traces and collect failure metrics; keep the better real
   closed-loop controller, even if that is V2.
2. **Nov 18–19:** If both fail, gather corrective observations from the currently
   best SNN under additional starting states; fine-tune and measure another
   paired evaluation. Avoid claims about improvement based only on action RMSE.
3. **Nov 20–22:** Aim for >=8s without torso threshold violation and >=1.5m
   forward with <=0.5m final drift on multiple new starts. Only if stable on
   flat ground, attempt a *small* uneven-terrain variation and record evidence.

These are checkpoints, not guaranteed results. As of V3 release, **SNN V3
closed-loop locomotion remains UNTESTED** in this runtime because MuJoCo and
Menagerie are unavailable here. Full uneven-terrain locomotion and energy
advantage remain unproven.
