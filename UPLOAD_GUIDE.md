# GitHub Upload Guide

This repository is meant to be uploaded as a lightweight research/code release.
Do not upload every local artifact.

## What to keep

- Source code under `controlled_system/`, `maze/`, and `le-wm/`.
- Root `README.md`, this upload guide, and per-folder README/notes.
- Paper-facing figures and compact provenance files.
- Small CSV/JSON summaries that make the results auditable.

## What to exclude

- The two external demo folders/files. They are intentionally outside this release.
- Demo/video artifacts such as `.mp4`, `.mov`, `.avi`, `.mkv`, or `.webm`.
- Embedded git histories: `le-wm/.git/`, `controlled_system/Manifold/.git/`, and any other nested `.git/`.
- Model/checkpoint/data binaries: `.pt`, `.pth`, `.ckpt`, `.npz`, `.npy`, `.h5`, `.hdf5`.
- Bulky generated run directories:
  - `le-wm/outputs/`
  - `maze/maze1/outputs/runs/`
  - `controlled_system/Toy/experiments/`
- Local runtime junk: `__pycache__/`, logs, W&B runs, virtual environments.
- Stray shell/editor artifacts such as files ending in `:`.

## Recommended upload workflow

Because this working directory contains nested git repositories, the safest
approach is to create a clean staging copy and initialize Git there.

From the parent directory:

```bash
rsync -av \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.npz' \
  --exclude='*.npy' \
  --exclude='*.h5' \
  --exclude='*.hdf5' \
  --exclude='*.mp4' \
  --exclude='*.mov' \
  --exclude='*.avi' \
  --exclude='*.mkv' \
  --exclude='*.webm' \
  --exclude='*:' \
  --exclude='le-wm/outputs/' \
  --exclude='maze/maze1/outputs/runs/' \
  --exclude='controlled_system/Toy/experiments/' \
  --exclude='demo/' \
  --exclude='demos/' \
  github/ github-public/
```

Then initialize and inspect:

```bash
cd github-public
git init
git add .
git status --short
git ls-files | sed -n '1,200p'
```

Before committing, check for accidental large files:

```bash
find . -type f -size +20M -print
```

If this prints anything, inspect it before committing.

## LeWM attribution

`le-wm/` is based on LeWorldModel. Keep the upstream README and license:

- `le-wm/README.md`
- `le-wm/LICENSE`

The public description should say that this project uses LeWM as a base system
and adds predictive-realization measurements, latent-geometry diagnostics,
ablations, and modification experiments on top of it.
