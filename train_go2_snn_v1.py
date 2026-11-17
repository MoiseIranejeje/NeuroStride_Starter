"""NeuroStride Go2 SNN v1: reproducible behavior cloning from ANN-teacher dataset.

Prerequisite: results/go2_teacher/go2_teacher_pilot.npz made by go2_collect_teacher.py.
Run: .\\.venv\\Scripts\\python.exe train_go2_snn_v1.py

Spiking model: fixed Gaussian population encoder -> 2 LIF spiking layers
with surrogate-gradient backpropagation -> analog readout of output spike rates.
Output is 12 raw joint-position offsets, NOT joint torques; downstream low-level
controller must scale these offsets and use torque-limited PD, like teacher.

Important: A four-episode pilot with the same initial pose is not a robust data
set. The command 0.50 m/s is held out as a *validation* command for checkpoint
selection, not a final independent test. More demonstration trajectories and
closed-loop SNN evaluation are necessary before claims of successful locomotion.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "go2_teacher" / "go2_teacher_pilot.npz"
OUT = ROOT / "results" / "go2_snn_v1"
SEED = 42
STEPS = 8
HIDDEN = 128
POPULATIONS = 3
EPOCHS = 75
BATCH_SIZE = 128
LR = 0.002
VALIDATE_EPISODE = 2  # command_x=0.5 m/s, not seen in SNN weight updates


class SurrogateSpike(torch.autograd.Function):
    """Binary spikes, smooth surrogate gradient w.r.t. membrane voltage."""
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (voltage,) = ctx.saved_tensors
        return grad_output / (1 + 4 * voltage.abs()).square()


class Go2SpikingActor(nn.Module):
    """Input 70 current observations + 70 mean-history summaries = 140 features.

    3 fixed Gaussian populations per feature generate binary spike trains.
    Two learned discrete LIF layers process them. A linear rate decoder maps
    spikes to 12 continuous joint-position-offset commands. Continuous readout
    is an interface to actuators, not itself a spiking neuron.
    """
    def __init__(self, feature_dim=140, population=POPULATIONS, hidden=HIDDEN, steps=STEPS):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.population = int(population)
        self.hidden = int(hidden)
        self.steps = int(steps)
        self.encoder = nn.Linear(feature_dim * population, hidden)
        self.recurrent = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, 12)
        self.register_buffer("centers", torch.linspace(-2.5, 2.5, population))
        self.register_buffer("phases", torch.remainder(torch.arange(feature_dim * population).float() * 0.61803398875, 1))
        self.sigma = 1.6
        self.decay = 0.7
        self.threshold = 0.8
        nn.init.constant_(self.encoder.bias, 0.35)
        nn.init.constant_(self.recurrent.bias, 0.35)

    def forward(self, normalized_features):
        if normalized_features.ndim != 2 or normalized_features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected [batch,{self.feature_dim}] features")
        z = normalized_features.float().clamp(-4, 4).unsqueeze(-1)
        rates = torch.exp(-0.5 * ((z - self.centers) / self.sigma).square()).flatten(1)
        rates = rates.clamp(0, 1)
        v1 = torch.zeros((z.shape[0], self.hidden), dtype=z.dtype, device=z.device)
        v2 = torch.zeros_like(v1)
        count1 = torch.zeros_like(v1)
        count2 = torch.zeros_like(v2)
        for step in range(self.steps):
            # Deterministic pulse-density encoding: binary events for each input.
            inp = (torch.floor(self.phases + (step+1)*rates) -
                   torch.floor(self.phases + step*rates)).clamp(0, 1)
            mem1 = self.decay * v1 + self.encoder(inp)
            s1 = SurrogateSpike.apply(mem1 - self.threshold)
            v1 = mem1 - self.threshold * s1.detach()
            mem2 = self.decay * v2 + self.recurrent(s1)
            s2 = SurrogateSpike.apply(mem2 - self.threshold)
            v2 = mem2 - self.threshold * s2.detach()
            count1 += s1
            count2 += s2
        rate1 = count1 / self.steps
        rate2 = count2 / self.steps
        output_standardized = self.readout(rate2)
        # Activity is not an estimate of energy consumption.
        spike_rate = (rate1.mean() + rate2.mean()) / 2
        return output_standardized, spike_rate


def features_from_npz(npz):
    obs = np.asarray(npz["obs"], dtype=np.float32)
    hist = np.asarray(npz["history"], dtype=np.float32)
    target = np.asarray(npz["teacher_action"], dtype=np.float32)
    episode = np.asarray(npz["episode"])
    if obs.ndim != 2 or obs.shape[1] != 70 or hist.shape != (len(obs), 2100) or target.shape != (len(obs), 12):
        raise ValueError("Incorrect teacher data shapes")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(hist)) or not np.all(np.isfinite(target)):
        raise ValueError("Non-finite samples detected")
    # MuJoCo teacher histories include the current observation as final frame.
    if not np.allclose(hist.reshape(-1, 30, 70)[:, -1, :], obs, atol=1e-6):
        raise ValueError("Teacher history is not aligned with current observations")
    history_mean = hist.reshape(-1, 30, 70).mean(axis=1)
    x = np.concatenate([obs, history_mean], axis=1).astype(np.float32)
    return x, target, episode


def run_training():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not DATA.is_file():
        raise FileNotFoundError(f"Run go2_collect_teacher.py first: {DATA}")
    with np.load(DATA, allow_pickle=False) as raw:
        x, y, ep = features_from_npz(raw)
    if set(np.unique(ep).tolist()) != {0, 1, 2, 3}:
        raise ValueError("Expected four unique teacher episodes 0,1,2,3")
    train_idx = np.where(ep != VALIDATE_EPISODE)[0]
    val_idx = np.where(ep == VALIDATE_EPISODE)[0]
    mean_x = x[train_idx].mean(axis=0)
    std_x = np.maximum(x[train_idx].std(axis=0), 0.05)
    mean_y = y[train_idx].mean(axis=0)
    std_y = np.maximum(y[train_idx].std(axis=0), 0.1)
    xx = np.clip((x - mean_x) / std_x, -5, 5).astype(np.float32)
    yy = ((y - mean_y) / std_y).astype(np.float32)
    train_data = TensorDataset(torch.from_numpy(xx[train_idx]), torch.from_numpy(yy[train_idx]))
    val_x = torch.from_numpy(xx[val_idx])
    val_target = torch.from_numpy(y[val_idx])
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = Go2SpikingActor(feature_dim=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    OUT.mkdir(exist_ok=True, parents=True)
    best_rmse = float("inf")
    history = []
    print(f"Pilot: {len(train_idx)} train / {len(val_idx)} held-out-command validation samples")
    print("Train commands: 0.3, 0.4, 0.6 m/s; validation command: 0.5 m/s")
    print("Training a 2-layer surrogate-gradient LIF SNN, not an ANN MLP")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            pred, spike_rate = model(bx)
            loss = loss_fn(pred, by) + 0.0001 * spike_rate
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            yp, spikes = model(val_x)
            yp = yp * torch.from_numpy(std_y) + torch.from_numpy(mean_y)
            err = yp - val_target
            rmse = float(err.square().mean().sqrt())
            mae = float(err.abs().mean())
            spike = float(spikes)
        history.append({"epoch": epoch, "train_standardized_mse": float(np.mean(losses)),
                        "val_rmse_raw_actions": rmse, "val_mae_raw_actions": mae,
                        "val_hidden_spike_rate": spike})
        if rmse < best_rmse:
            best_rmse = rmse
            best_ep = epoch
            torch.save({
                "model_state": model.state_dict(),
                "feature_mean": torch.from_numpy(mean_x.copy()),
                "feature_std": torch.from_numpy(std_x.copy()),
                "target_mean": torch.from_numpy(mean_y.copy()),
                "target_std": torch.from_numpy(std_y.copy()),
                "config": {"feature_dim": 140, "population": POPULATIONS, "hidden": HIDDEN,
                           "steps": STEPS, "seed": SEED, "train_episodes": [0,1,3],
                           "validation_episode": 2},
                "epoch": epoch,
                "val_rmse_raw_actions": rmse,
            }, OUT / "go2_snn_v1_best.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:3d} | train std MSE {np.mean(losses):.4f} | "
                  f"val raw RMSE {rmse:.4f} | val MAE {mae:.4f} | spike rate {spike:.3f}", flush=True)
    baseline = float(np.sqrt(np.square(y[val_idx]-mean_y).mean()))
    with (OUT / "training_history.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    summary = {"train_samples": int(len(train_idx)), "validation_samples": int(len(val_idx)),
               "validation_command": 0.5, "best_epoch": int(best_ep),
               "best_validation_rmse_raw_action": best_rmse,
               "constant_action_baseline_rmse": baseline,
               "limitations": "Pilot offline action imitation. Validation used for checkpoint selection; no closed-loop SNN locomotion yet. No independent terrain tests or energy measurement."}
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print("=== SNN PILOT TRAINING COMPLETE ===")
    print(json.dumps(summary, indent=2))
    print("Weights:", OUT / "go2_snn_v1_best.pt")


if __name__ == "__main__":
    run_training()
