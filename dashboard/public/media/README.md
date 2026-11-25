# NeuroStride real-video media folder

This directory intentionally contains **no video asset yet**. The homepage uses
its existing static dark background until a verified recorded clip is committed.

After you have an actual Go2 MuJoCo episode replay, run from the repository root:

    .\.venv\Scripts\python.exe go2_publish_hero_video.py --replay "results\go2_snn_upright_walk\development\replays\neural_v0.45_seed7601.npz"

Alternatively, use a real recorded MP4:

    .\.venv\Scripts\python.exe go2_publish_hero_video.py --input-mp4 "results\go2_videos\real_go2_walk.mp4"

The script writes the following when encoding support is available:

- go2-hero.webm — VP9, low-bitrate, small autoplay background, preferred in supporting browsers
- go2-hero.mp4 — H.264 fallback, muted and looped
- go2-hero-provenance.json — input SHA-256, source metadata and actual output sizes

Both video files are **real replay renders** or independently verified user-supplied
MuJoCo captures; do not copy in a stock or AI-generated video presented as a trial.

Commit these outputs to the same GitHub PR branch after checking them. Vercel
will serve them under /media/. The React homepage checks for valid files before
showing the background; when absent it remains a static, readable dark hero.

For page-weight control, aim for a 6–8 s silent loop, 20 fps, 896×504 or smaller,
and around 1–3 MiB WebM. The actual filesize is printed by the encoder.
A decorative autoplay video is turned off for visitors requesting reduced motion
or reduced data and pauses when not visible.

The full experiment's downloadable CSV/JSON results remain independent from this
visual background. Do not describe the background as a new policy evaluation.
