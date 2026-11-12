# NeuroStride — Milestone 2: ANN learning baseline

## What this is

This is **real** PPO reinforcement learning on Gymnasium Ant-v5, using a regular artificial neural network (ANN). The Ant has 105 observations and 8 motor actions. It deliberately uses the same 500-step experiment limit as Milestone 1. This is NOT an SNN, proof of stable walking, or an exact reproduction of Jiang et al.'s robotics paper.

The ZIP is an **add-on** to your existing `NeuroStride_Starter` folder; it contains only new files, so your Milestone 1 files and CSVs should remain intact.

## Install on Windows (PowerShell in NeuroStride_Starter)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_m2.txt
.\.venv\Scripts\python.exe -c "import stable_baselines3, tensorboard, mujoco; print('M2 dependencies OK')"
```

PyTorch is a large dependency. Stable-Baselines3 and TensorBoard require additional storage. Your working MuJoCo installation should remain unchanged.

## 1. Small training run to check the pipeline

```powershell
.\.venv\Scripts\python.exe train_ann.py --timesteps 20000
```

Expected **file paths**, not expected success metrics:

- `training/ann_seed42/ann_ppo_ant.zip` — trained ANN policy and value networks
- `training/ann_seed42/ann_vecnormalize.pkl` — observation normalization statistics required for evaluation
- `training/ann_seed42/checkpoints/` — snapshots with normalization statistics, at 10k-step intervals
- `training/ann_seed42/tensorboard/` — real training metrics
- `training/ann_seed42/monitor_*.monitor.csv` — real training episode results

**Do not judge walking performance from 20,000 steps.** The trainer may run slightly beyond the requested step count to complete a PPO rollout. There is no preset reward or guaranteed success.

## 2. Show the learning dashboard

Open a SECOND PowerShell window in the same project directory and run:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir training\ann_seed42\tensorboard
```

Open the localhost URL that TensorBoard prints, typically `http://localhost:6006/`. Examine `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `train/approx_kl`, `train/value_loss`, and related training metrics when available. Early in training, some metrics are blank until episodes finish. Charts are generated from actual log files, not artificial demo data.

## 3. Evaluate without exploration noise

```powershell
.\.venv\Scripts\python.exe evaluate_ann.py --episodes 5
```

This uses five evaluation seeds, starting at 1000, which are distinct from the default training seed. It writes `results/ann_eval_seed1000.csv`. The evaluation policy is deterministic. An evaluation mean from only five episodes is preliminary rather than publishable evidence.

## 4. Watch learned behavior in the 3D viewer

```powershell
.\.venv\Scripts\python.exe evaluate_ann.py --episodes 1 --render
```

No need to run `--render` during training because graphics slow it down significantly.

## 5. Continue the same ANN training run

```powershell
.\.venv\Scripts\python.exe train_ann.py --timesteps 180000 --resume
```

This adds about 180k more environment steps to your original 20k. Both model weights and observation-normalization statistics are loaded. PPO may overshoot by a partial rollout. You may need substantially more training to see reliable walking.

## Scientifically valid follow-up

After training longer, evaluate the trained ANN and the random baseline on matching seeds (`--seed 1000`) and 500-step limits. Run multiple seeds and retain raw results before making claims about performance. Future SNN comparisons must use the same tasks, observation space, actions, and evaluation procedures.

## Troubleshooting

- No `tensorboard.exe`: verify the `pip install -r requirements_m2.txt` command completed in the **same** `.venv` you used for M1.
- `ModuleNotFoundError: torch`: re-run the dependencies install. Torch is required for PPO.
- `FileNotFoundError` for model or normalization stats: finish training and verify the `training/ann_seed42/` folder exists.
- Slow training: this uses CPU by default, no video recording and one simulator; first try the 20k run before committing to larger experiments.
- If you rerun training without `--resume`, the script stops instead of overwriting your saved ANN model.

**Scientific status:** M1 successfully demonstrated the physics simulator on your computer. M2 code is prepared and syntax-checked, but training/evaluation results depend on running these files on your computer. Never represent anticipated results as observed results.
