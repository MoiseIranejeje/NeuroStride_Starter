# NeuroStride Go2 research dashboard

This Vite, React, TypeScript and Tailwind site compares the MuJoCo knee-extension sweep with the neural upright-controller experiment. The dashboard summarizes the original exports in `src/evidence/` and includes the corresponding per-episode CSVs. CSV replay paths are relative to the original run directories. Replay files are not bundled. Both studies used development seeds 7601–7603, with eight-second runs at a commanded 0.45 m/s.

| Export prefix | Simulator script | Original run directory |
| --- | --- | --- |
| `extension_scan` | `go2_knee_extension_probe.py` | `results/go2_knee_extension_probe/extension_scan` |
| `development` | `go2_snn_upright_walk.py` | `results/go2_snn_upright_walk/development` |

## Run locally

From `dashboard/`, use Node 24 and pnpm 11.25.0.

```sh
pnpm install --frozen-lockfile
pnpm dev
pnpm build
pnpm preview
```

## Vercel deployment and GitHub pull requests

The site is hosted through Vercel, connected to the repository's \`main\` branch.
The root directory for the Vite project is \`dashboard\`, build command
\`pnpm run build\`, and published output is \`dist\`.

Keep development on feature branches and inspect Vercel PR previews before
merging to \`main\`. Do not merge unverified dashboard changes into production.

## Adding a genuine Go2 hero recording

The hero background is **optional**. Without actual media, it remains a static
dark background. Never insert illustrative motion and present it as an experiment.

1. Choose an **actual saved replay** \`.npz\` from a Go2 MuJoCo test
   (the \`go2_snn_upright_walk.py\` experiments save replays under
   \`results/go2_snn_upright_walk/development/replays/\` on the simulator PC).
2. At the repository root, run the video encoder, for example:

   \`\`\`powershell
   .\\.venv\\Scripts\\python.exe go2_publish_hero_video.py --replay "results\\go2_snn_upright_walk\\development\\replays\\neural_v0.45_seed7601.npz"
   \`\`\`

3. Inspect the generated muted 20 fps video for genuine Go2 motion. The script
   prepares \`dashboard/public/media/go2-hero.webm\` (preferred) and
   \`go2-hero.mp4\` (H.264 fallback), plus a provenance JSON file.
   You can also use \`--input-mp4\` to compress a genuine pre-existing recording.
4. Commit the generated files to your **PR branch**. Vercel will expose the
   assets at \`/media/go2-hero.webm\` and \`/media/go2-hero.mp4\`.
   The website discovers real assets automatically, pauses the background when
   offscreen, and avoids autoplay under reduced-motion/reduced-data settings.

Target a 6–8 second, muted clip around 1–3 MiB to limit initial page weight.
Original per-episode measurements remain in \`src/evidence/\`, with the real
recording's identity documented in \`go2-hero-provenance.json\`.
