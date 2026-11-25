"""Plot Iterate episode comparisons and matched replays."""
import argparse
import csv
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


def plot_seed_comparisons(dashboard, media, study_id, series, output_prefix):
    with (dashboard / "src/evidence" / f"{study_id}-episodes.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    seeds = [7601, 7602, 7603]
    plots = [
        ("height", "mean_visual_top_height_after_075s_m", "Top-of-robot height (cm)", 100, 40, (34, 43)),
        ("yaw", "max_abs_yaw_deg", "Peak heading error (°)", 1, 10, (0, 12)),
        ("drift", "sideways_m", "Final sideways offset (cm)", 100, 25, (0, 110)),
    ]
    for name, field, label, scale, reference, bounds in plots:
        fig, ax = plt.subplots(figsize=(4.5, 3.3), layout="constrained")
        ax.set_facecolor("#f4f8f6")
        for index, (mode, title, color) in enumerate(series):
            values = {int(row["seed"]): abs(float(row[field])) * scale
                      for row in rows if row["mode"] == mode}
            if set(values) != set(seeds):
                raise ValueError(f"Unexpected seeds for {study_id}/{mode}: {values.keys()}")
            low, high = min(values.values()), max(values.values())
            average = np.mean(list(values.values()))
            if not bounds[0] <= low <= high <= bounds[1]:
                raise ValueError(f"Plot bounds exclude values for {study_id}/{mode}/{name}")
            ax.hlines(index, low, high, color=color, linewidth=3)
            ax.vlines([low, high], index - .08, index + .08, color=color, linewidth=1.5)
            ax.scatter(average, index, color=color, edgecolor="white", marker="D",
                       s=50, linewidth=.8, zorder=3)
            ax.annotate(f"{average:.2f}", (average, index), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold", color="#142825")
        ax.axvline(reference, color="#99511b", linestyle=":", linewidth=1.6)
        ax.set(xlabel=label, xlim=bounds, ylim=(len(series) - .55, -.65),
               yticks=range(len(series)), yticklabels=[title for _, title, _ in series])
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.grid(axis="x", color="#d6e2de")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.savefig(media / f"{output_prefix}-{name}.png", dpi=160)
        plt.close(fig)


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

    original = ("baseline", "Original\ncontroller", "#175b87")
    neural_series = ("neural", "Height-feedback\nSNN", "#287a50")
    plot_seed_comparisons(dashboard, media, "development", (original, neural_series),
                          "go2-seed-comparison")
    for study_id, modes in (
        ("extension_scan", (
            ("fixed04", "+0.04 rad", "#99511b"),
            ("fixed08", "+0.08 rad", "#6750a4"),
            ("fixed12", "+0.12 rad", "#287a50"),
        )),
        ("development", (
            ("fixed16", "+0.16 rad", "#99511b"),
            ("fixed24", "+0.24 rad", "#6750a4"),
            neural_series,
        )),
    ):
        plot_seed_comparisons(dashboard, media, study_id, (original, *modes),
                              f"go2-study-{study_id}")

    print("Final XY (m):", baseline[1][-1], neural[1][-1])
    print("Saved-frame peak heading error (degrees):", *(np.abs(t[4]).max() for t in (baseline, neural)))


if __name__ == "__main__":
    main()
