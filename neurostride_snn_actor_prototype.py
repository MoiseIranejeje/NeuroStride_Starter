"""NeuroStride SNN actor *prototype*, NOT a trained policy or a paper replication.

Install dependency (inside the user's venv):
    python -m pip install norse
Smoke test:
    python neurostride_snn_actor_prototype.py

Conceptually follows Jiang et al. (2024), arxiv:2310.05022v2:
population encoding -> LIF spike layers -> population rate decoding.
Important simplification: input population centers are fixed, and Poisson spike
sampling does not propagate gradients through the encoder. Trainable LIF weights
still support surrogate gradients via Norse, but this is NOT PPO-integrated.
"""
import torch
from torch import nn
from norse.torch import LIFCell


class PopulationLIFActor(nn.Module):
    """105 normalized Ant observations -> 8 continuous motor commands."""

    def __init__(self, obs_dim=105, actions=8, populations=4, hidden=128, time_steps=8):
        super().__init__()
        self.obs_dim = obs_dim
        self.actions = actions
        self.populations = populations
        self.time_steps = time_steps

        # Non-learned encoder centers: baseline only; compare against learnable
        # encoding in a later *controlled* experiment.
        self.register_buffer("centers", torch.linspace(-3.0, 3.0, populations))
        self.sigma = 1.5
        self.input_layer = nn.Linear(obs_dim * populations, hidden)
        self.hidden_lif1 = LIFCell()
        self.hidden_layer = nn.Linear(hidden, hidden)
        self.hidden_lif2 = LIFCell()
        self.output_layer = nn.Linear(hidden, actions * populations)
        self.output_lif = LIFCell()
        self.readout = nn.Parameter(torch.randn(actions, populations) * 0.1)
        self.bias = nn.Parameter(torch.zeros(actions))

    def forward(self, normalized_obs):
        if normalized_obs.ndim != 2 or normalized_obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected batch of [N, {self.obs_dim}] observations")
        x = normalized_obs.float().clamp(-5, 5)
        rates = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.sigma) ** 2)
        rates = rates.clamp(0, 1)
        batch_size = x.shape[0]
        first_state = second_state = output_state = None
        spike_counts = torch.zeros(
            batch_size, self.actions, self.populations,
            dtype=x.dtype, device=x.device,
        )
        total_spikes = torch.zeros((), dtype=x.dtype, device=x.device)

        for _ in range(self.time_steps):
            # Stochastic spike encoding. For evaluation repeat with fixed torch
            # RNG seed or use multiple trials to quantify noise variability.
            spike_in = torch.bernoulli(rates).reshape(batch_size, -1)
            spikes1, first_state = self.hidden_lif1(
                self.input_layer(spike_in), first_state
            )
            spikes2, second_state = self.hidden_lif2(
                self.hidden_layer(spikes1), second_state
            )
            spikes_out, output_state = self.output_lif(
                self.output_layer(spikes2), output_state
            )
            spike_counts = spike_counts + spikes_out.reshape(
                batch_size, self.actions, self.populations
            )
            total_spikes = total_spikes + spikes1.sum() + spikes2.sum() + spikes_out.sum()

        firing_rates = spike_counts / self.time_steps
        motor_commands = torch.tanh(
            (firing_rates * self.readout.unsqueeze(0)).sum(dim=-1) + self.bias
        )
        # Spike counts are activity metrics, NOT measured electrical power.
        return motor_commands, total_spikes


if __name__ == "__main__":
    torch.manual_seed(42)
    actor = PopulationLIFActor()
    actions, spike_count = actor(torch.randn(2, 105))
    print("Output shape (expected 2x8):", tuple(actions.shape))
    print("Motor command range:", float(actions.min()), "to", float(actions.max()))
    print("Total sampled hidden/output spikes:", float(spike_count))
    print("UNTRAINED PROTOTYPE ONLY. No walking or energy advantage is claimed.")
