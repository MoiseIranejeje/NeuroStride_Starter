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

## Deploy and review on Render

Create a **Static Site** and connect the GitHub repository. Grant the Render GitHub app access to this private repository. The deployed website and its downloadable evidence will be reachable at the site URL.

| Render setting | Value |
| --- | --- |
| Branch for the first review deploy | `codex/go2-research-dashboard` |
| Root Directory | `dashboard` |
| Build Command | `npx --yes pnpm@11.25.0 install --frozen-lockfile && npx --yes pnpm@11.25.0 build` |
| Publish Directory | `dist` |
| Environment variables | `NODE_VERSION=24.22.3` and `SKIP_INSTALL_DEPS=true` |

The dashboard folder first appears in this PR. Deploy the feature branch to review the first version before merging. That URL is the branch site's URL. After review and merge, switch the site's linked branch to `main`.

For subsequent PRs, go to the site's **Previews** page in Render and set **Pull Request Previews** to **Automatic** or **Manual**. Automatic applies to PRs targeting the linked branch. Manual uses the GitHub label `render-preview` or the marker `[render preview]` in the PR title. Open the preview from **View deployment** on the GitHub PR. Render refreshes the preview after pushes and removes it on merge or close.

[Render static sites](https://render.com/docs/static-sites) · [Render service previews](https://render.com/docs/service-previews)
