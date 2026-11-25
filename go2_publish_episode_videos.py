"""Publish the dashboard episode MP4s from the original saved MuJoCo states."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from go2_publish_hero_video import ROOT, find_ffmpeg, file_sha256, make_video, read_replay_provenance


DASHBOARD = ROOT / "dashboard"
WIDTH, HEIGHT, FPS = 960, 540, 20
STUDIES = {
    "extension_scan": "go2_knee_extension_probe/extension_scan",
    "development": "go2_snn_upright_walk/development",
}


def publish(study, source, row, ffmpeg):
    provenance = read_replay_provenance(source)
    summary = provenance["episode_summary"]
    for key in ("mode", "seed", "command_mps", "duration_s", "forward_m",
                "sideways_m", "max_abs_yaw_deg", "pass_10deg"):
        if str(summary[key]) != row[key]:
            raise ValueError(f"{study}/{source.name}: replay metadata differs from CSV at {key}")

    filename = f"{source.stem}.mp4"
    relative_path = f"episodes/{study}/{filename}"
    output = DASHBOARD / "public" / "media" / relative_path
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="go2_episode_") as temp:
        rendered = Path(temp) / "rendered.mp4"
        encoded = Path(temp) / "encoded.mp4"
        subprocess.run([
            sys.executable, str(ROOT / "go2_record_front_video.py"), "--replay", str(source),
            "--out", str(rendered), "--view", "follow", "--distance", "1.45",
            "--fps", str(FPS), "--width", str(WIDTH), "--height", str(HEIGHT),
        ], cwd=ROOT, check=True)
        make_video(ffmpeg, rendered, encoded, "mp4", FPS, WIDTH, HEIGHT,
                   float(row["duration_s"]), mp4_crf=19)
        subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(encoded),
                        "-f", "null", "-"], check=True)
        output.write_bytes(encoded.read_bytes())
    print(f"Verified {relative_path}", flush=True)
    return f"{study}/{row['replay']}", {
        "path": relative_path,
        "source_sha256": provenance["source_sha256"],
        "sha256": file_sha256(output),
        "seconds": provenance["simulation_seconds"],
        "dimensions_px": [WIDTH, HEIGHT],
        "camera": "three-quarter tracking",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    ffmpeg = find_ffmpeg()
    jobs = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for study, directory in STUDIES.items():
            with (DASHBOARD / "src/evidence" / f"{study}-episodes.csv").open(newline="") as export:
                for row in csv.DictReader(export):
                    source = args.results_dir / directory / row["replay"]
                    jobs.append(pool.submit(publish, study, source, row, ffmpeg))
        manifest = dict(job.result() for job in jobs)
    (DASHBOARD / "src/evidence/episode-videos.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
