r"""Build a small, genuine MuJoCo hero clip for the NeuroStride Vercel site.

From the NeuroStride_Starter project root:

    .\.venv\Scripts\python.exe go2_publish_hero_video.py --replay "results\go2_snn_upright_walk\development\replays\neural_v0.45_seed7601.npz"

Or optimize an existing real recording:

    .\.venv\Scripts\python.exe go2_publish_hero_video.py --input-mp4 "results\go2_videos\your_real_recording.mp4"

The first command re-renders actual saved qpos/qvel, not a fabricated animation.
The outputs are Vercel static assets at dashboard/public/media/go2-hero.webm
and dashboard/public/media/go2-hero.mp4. The website detects them automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "dashboard" / "public" / "media"


def existing_file(text: str) -> Path:
    candidate = Path(text)
    path = candidate if candidate.is_absolute() else ROOT / candidate
    if not path.is_file():
        raise FileNotFoundError(f"Recording does not exist: {path}")
    return path.resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "FFmpeg is required. Install into your active venv with "
            r".\.venv\Scripts\python.exe -m pip install imageio-ffmpeg"
        ) from exc


def run(cmd: list[str]) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def read_replay_provenance(path: Path) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=False) as recording:
        if not all(k in recording.files for k in ("qpos", "qvel", "sim_time", "summary_json")):
            raise ValueError("The replay lacks exact MuJoCo state arrays and episode summary")
        frames = len(recording["sim_time"])
        if frames < 2 or recording["qpos"].shape[0] != frames:
            raise ValueError("Replay has incomplete simulator samples")
        summary = json.loads(str(recording["summary_json"].item()))
        seconds = float(recording["sim_time"][-1] - recording["sim_time"][0])

    match = re.search(r"seed(\d+)", path.stem)
    return {
        "source_type": "MuJoCo qpos/qvel state replay",
        "source_filename": path.name,
        "source_sha256": file_sha256(path),
        "seed_from_filename": int(match.group(1)) if match else None,
        "simulation_seconds": round(seconds, 3),
        "episode_summary": summary,
        "note": "Rendered from saved simulator states. Not a new model evaluation.",
    }


def make_video(ffmpeg: str, source: Path, target: Path, kind: str, fps: int, width: int, height: int, seconds: float) -> None:
    common = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-t", str(seconds), "-an",
        "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos",
    ]
    if kind == "webm":
        # Constant quality / no target bitrate: good for short, looped web footage.
        args = ["-c:v", "libvpx-vp9", "-deadline", "good", "-cpu-used", "4",
                "-row-mt", "1", "-b:v", "0", "-crf", "40", str(target)]
    else:
        args = ["-c:v", "libx264", "-preset", "medium", "-crf", "29",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    run(common + args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--replay", help="Real .npz MuJoCo replay with qpos/qvel")
    inputs.add_argument("--input-mp4", help="Existing MP4 recording of the real simulation")
    parser.add_argument("--view", choices=["wide", "front", "top"], default="wide")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    if not (10 <= args.fps <= 30 and 320 <= args.width <= 1280 and 240 <= args.height <= 720):
        parser.error("Use 10–30 fps, width 320–1280, height 240–720")
    if args.width % 2 or args.height % 2 or not (2 <= args.seconds <= 12):
        parser.error("Width and height must be even; clip length must be 2–12 seconds")

    source = existing_file(args.replay or args.input_mp4)
    ffmpeg = find_ffmpeg()
    MEDIA.mkdir(parents=True, exist_ok=True)

    if args.replay:
        if source.suffix.lower() != ".npz":
            parser.error("--replay must be a MuJoCo .npz state recording")
        provenance = read_replay_provenance(source)
    else:
        if source.suffix.lower() != ".mp4":
            parser.error("--input-mp4 must be an MP4 recording")
        provenance = {
            "source_type": "supplied MP4",
            "source_filename": source.name,
            "source_sha256": file_sha256(source),
            "note": "Footage origin must be verified manually against the recorded MuJoCo episode.",
        }

    with TemporaryDirectory(prefix="neurostride_hero_") as temp:
        if args.replay:
            source_mp4 = Path(temp) / "source.mp4"
            render = ROOT / "go2_record_front_video.py"
            if not render.is_file():
                raise FileNotFoundError(f"Expected source renderer: {render}")
            run([
                sys.executable, str(render), "--replay", str(source),
                "--out", str(source_mp4), "--view", args.view,
                "--fps", str(args.fps), "--width", str(args.width),
                "--height", str(args.height),
            ])
        else:
            source_mp4 = source

        webm_path = MEDIA / "go2-hero.webm"
        mp4_path = MEDIA / "go2-hero.mp4"
        webm_ok = True
        try:
            make_video(ffmpeg, source_mp4, webm_path, "webm", args.fps, args.width, args.height, args.seconds)
        except subprocess.CalledProcessError:
            webm_ok = False
            webm_path.unlink(missing_ok=True)
            print("VP9/WebM codec unavailable in this FFmpeg build; MP4 fallback will still work.")

        make_video(ffmpeg, source_mp4, mp4_path, "mp4", args.fps, args.width, args.height, args.seconds)

    outputs = [p for p in (webm_path, mp4_path) if p.is_file()]
    metadata = {
        **provenance,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "display": "muted looping hero background, decorative playback",
        "frame_rate_fps": args.fps,
        "dimensions_px": [args.width, args.height],
        "clip_max_duration_seconds": args.seconds,
        "webm_available": webm_ok,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in outputs],
    }
    (MEDIA / "go2-hero-provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nReal MuJoCo footage prepared:")
    for p in outputs:
        print(f"  {p.relative_to(ROOT)} ({p.stat().st_size / (1024 * 1024):.2f} MiB)")
    print("Push the real video files and provenance JSON to the GitHub PR branch.")
    print("If WebM exceeds ~3 MiB, consider a lower resolution or a shorter clip.")


if __name__ == "__main__":
    main()
