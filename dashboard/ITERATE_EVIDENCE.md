# Iterate evidence manifest

This inventory covers the Go2 MuJoCo evidence in the research dashboard. Both studies use development seeds 7601–7603 at a commanded 0.45 m/s for up to 8 seconds on flat ground.

| Study ID | Configurations | Episode records | Grouped results | Evaluation script |
| --- | --- | --- | --- | --- |
| `extension_scan` | baseline, fixed04, fixed08, fixed12 | [12 CSV rows](src/evidence/extension_scan-episodes.csv) | [4 JSON groups](src/evidence/extension_scan-summary.json) | [Fixed knee-extension sweep](../go2_knee_extension_probe.py) |
| `development` | baseline, fixed16, fixed24, neural | [12 CSV rows](src/evidence/development-episodes.csv) | [4 JSON groups](src/evidence/development-summary.json) | [Neural upright-controller evaluation](../go2_snn_upright_walk.py) |

The baseline rows appear in both exports. Compare them within each study and avoid counting them as independent runs when combining studies.

## Episode schema

Use `(study_id, mode, seed, command_mps)` as the episode key. The study ID comes from the export prefix. Grouped JSON uses `speed_mps` for the commanded speed and groups over seeds by mode and speed.

| CSV field | Meaning |
| --- | --- |
| `mode`, `seed`, `command_mps` | Controller configuration, integer simulation seed, and commanded speed in m/s |
| `duration_s`, `stop` | Elapsed simulation time and termination reason |
| `forward_m`, `sideways_m`, `peak_cross_track_m` | Forward displacement, signed final sideways displacement, and maximum sideways distance from the start line |
| `max_abs_yaw_deg` | Largest absolute heading error in degrees |
| `minimum_root_height_m`, `mean_visual_top_height_after_075s_m` | Minimum body-root height and average top-of-mesh height after the first 0.75 seconds |
| `pass_5deg`, `pass_10deg`, `pass_15deg` | Boolean outcome for the full task at each heading limit |
| `replay` | Path to the saved simulator states, relative to the original study output directory |

The full-task criteria also cover run duration, forward distance, final sideways displacement and minimum root height. The evaluation scripts define the thresholds. Additional CSV diagnostics include torso pitch and roll, knee extension and torque clipping.

## Published recordings

The [WebM](public/media/go2-hero.webm) and [MP4](public/media/go2-hero.mp4) are two encodings of the same 8-second replay. They map to `(development, neural, 7601, 0.45)`. The episode completed 8 seconds and ended 0.65973 m sideways, exceeding the 0.25 m limit. The [provenance record](public/media/go2-hero-provenance.json) lists the source filename and checksums for both encodings.

The episode explorer also includes a 960 × 540 MP4 for every exported row in both studies. A three-quarter tracking camera keeps the robot in view while the saved simulator states determine its motion. The [episode replay manifest](src/evidence/episode-videos.json) maps study-relative replay paths to the corresponding MP4 and records the source and video checksums. The publisher verifies the saved replay metadata against the CSV and decodes each MP4 before listing it. Regenerate the clips with `python go2_publish_episode_videos.py <original-results-directory>` from the repository root.

## Seed 7601 trace plots

The trajectory, pitch, roll and heading plots use saved frames from the baseline and neural replays in the development study. The neural replay checksum was verified directly against the local source and matches the published video provenance. Regenerate the PNGs with `python dashboard/plot_iterate_traces.py <development-replay-directory>` using NumPy and Matplotlib.

## Evidence still to collect

- Raw simulation replays referenced by the exports are outside this checkout. The neural seed 7601 source replay was verified in the local simulator checkout.
- Research references and any evidence outside these two studies still need to be inventoried.
