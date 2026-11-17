
"""Plot actual Go2 posture and movement recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from go2_v3_upright_sweep import euler_deg_from_wxyz

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "go2_upright_sweep"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=3101
    )

    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[1.0, 0.8, 0.6]
    )

    parser.add_argument(
        "--command", type=float, default=0.45
    )

    parser.add_argument(
        "--source-dir", type=Path, default=OUT
    )

    args = parser.parse_args()

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9),
        layout="constrained"
    )

    plotted = 0

    for alpha in args.alphas:

        path = args.source_dir / (
            f"v3_alpha{alpha:.2f}"
            f"_cmd{args.command:.2f}"
            f"_seed{args.seed}.npz"
        )

        if not path.is_file():
            print("Missing recording:", path)
            continue

        with np.load(
            path, allow_pickle=False
        ) as file:
            q = file["qpos"].copy()
            t = file["sim_time"].copy()

            summary = json.loads(
                str(file["summary_json"].item())
            )

        rpy = np.array([
            euler_deg_from_wxyz(quat)
            for quat in q[:, 3:7]
        ])

        yaw = (
            np.unwrap(np.deg2rad(rpy[:, 2]))
            - np.deg2rad(rpy[0, 2])
        ) * 180 / np.pi

        roll = rpy[:, 0]
        pitch = rpy[:, 1]

        label = (
            f"alpha={alpha:.2f} "
            f"({summary['duration_s']:.2f}s)"
        )

        axes[0, 0].plot(
            q[:, 0] - q[0, 0],
            q[:, 1] - q[0, 1],
            label=label
        )

        axes[0, 1].plot(
            t, roll,
            label=f"roll {alpha:.2f}"
        )

        axes[0, 1].plot(
            t, pitch,
            linestyle="--",
            label=f"pitch {alpha:.2f}"
        )

        axes[1, 0].plot(
            t, yaw, label=label
        )

        axes[1, 1].plot(
            t, q[:, 2], label=label
        )

        plotted += 1

    if not plotted:
        raise SystemExit(
            "No recordings found. Run the upright sweep first."
        )

    axes[0, 0].set(
        title="Horizontal XY trajectory",
        xlabel="Forward X (m)",
        ylabel="Sideways Y (m)"
    )

    axes[0, 0].axis("equal")

    axes[0, 1].set(
        title="Body roll and pitch",
        xlabel="Simulation time (s)",
        ylabel="Angle (degrees)"
    )

    axes[1, 0].set(
        title="Heading / yaw drift",
        xlabel="Simulation time (s)",
        ylabel="Angle (degrees)"
    )

    axes[1, 1].set(
        title="Torso height",
        xlabel="Simulation time (s)",
        ylabel="Height (m)"
    )

    axes[1, 1].axhline(
        0.18,
        linestyle=":",
        color="red",
        label="Height threshold"
    )

    for ax in axes.flat:
        ax.grid(alpha=0.28)
        ax.legend(fontsize="small")

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    destination = OUT / (
        f"posture_seed{args.seed}.png"
    )

    fig.suptitle(
        "NeuroStride Go2 — Upright Walking Experiment\n"
        f"Seed {args.seed}, command {args.command:.2f} m/s"
    )

    fig.savefig(
        destination, dpi=150
    )

    plt.close(fig)

    print("Saved posture plot:", destination)


if __name__ == "__main__":
    main()
