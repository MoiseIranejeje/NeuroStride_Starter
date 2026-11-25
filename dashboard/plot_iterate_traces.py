"""Plot the matched baseline and height-feedback replays for Iterate seed 7601."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_trace(path):
    with np.load(path) as replay:
        time = replay["sim_time"]
        pose = replay["qpos"]
        position = pose[:, :2] - pose[0, :2]
        w, x, y, z = pose[:, 3:7].T
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return time, position, replay["pitch_deg"], replay["roll_deg"], np.degrees(yaw - yaw[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    args = parser.parse_args()
    dashboard = Path(__file__).resolve().parent
    media = dashboard / "public" / "media"
    provenance = json.loads((media / "go2-hero-provenance.json").read_text())
    neural_path = args.replay_dir / provenance["source_filename"]
    if hashlib.sha256(neural_path.read_bytes()).hexdigest() != provenance["source_sha256"]:
        raise ValueError("Neural replay does not match the published video source checksum")

    baseline = read_trace(args.replay_dir / "baseline_v0.45_seed7601.npz")
    neural = read_trace(neural_path)
    for trace in (baseline, neural):
        assert np.isclose(trace[0][-1], 8)
        assert all(np.isfinite(values).all() for values in trace)

    plots = [
        ("trajectory", "Forward displacement (m)", "Sideways displacement (m)", 1, (-.25, .25)),
        ("pitch", "Time (s)", "Torso pitch (°)", 2, ()),
        ("roll", "Time (s)", "Torso roll (°)", 3, ()),
        ("heading", "Time (s)", "Heading error (°)", 4, (-10, 10)),
    ]
    for name, x_label, y_label, column, limits in plots:
        fig, ax = plt.subplots(figsize=(6.3, 3.6), layout="constrained")
        for trace, label, color, style in (
            (baseline, "Original controller", "#175b87", "--"),
            (neural, "Height-feedback SNN", "#287a50", "-"),
        ):
            x_data = trace[1][:, 0] if column == 1 else trace[0]
            y_data = trace[1][:, 1] if column == 1 else trace[column]
            ax.plot(x_data, y_data, color=color, linestyle=style, linewidth=1.7, label=label)
        for limit in limits:
            ax.axhline(limit, color="#99511b", linestyle=":", linewidth=1.2)
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.grid(color="#dfe5e2", alpha=.85)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9)
        fig.savefig(media / f"go2-seed7601-{name}.png", dpi=160)
        plt.close(fig)

    print("Final XY (m):", baseline[1][-1], neural[1][-1])
    print("Saved-frame peak heading error (degrees):", *(np.abs(t[4]).max() for t in (baseline, neural)))


if __name__ == "__main__":
    main()
