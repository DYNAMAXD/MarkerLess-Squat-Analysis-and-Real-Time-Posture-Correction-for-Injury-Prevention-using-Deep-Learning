---
title: AlphaPose + MotionBERT 3D Pose & Mesh
emoji: 🕺
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# AlphaPose + MotionBERT — Video to 3D Joints & Mesh (no-Docker build)

Same pipeline as the Docker version, built as a plain Gradio SDK Space
instead: `packages.txt` installs apt build tools, `pre-requirements.txt`
installs torch before anything tries to compile against it, and
`requirements.txt` installs AlphaPose in-place (`-e ./AlphaPose`, which
compiles its C++ extensions) plus MotionBERT's own requirements.

## ⚠️ Hardware: do NOT select ZeroGPU for this Space

This pipeline runs AlphaPose and MotionBERT as **subprocesses**
(`subprocess.run(...)`, calling their own `demo_inference.py` /
`infer_wild.py` / `infer_wild_mesh.py` scripts). ZeroGPU only attaches a
GPU to the exact process running inside an `@spaces.GPU`-decorated
function for the duration of that call — it explicitly does not extend to
forked/subprocess children. Under ZeroGPU, these subprocesses would just
silently run on CPU (slow) and likely hit ZeroGPU's per-call timeout on
top of that. It also pins torch to a specific short list of versions tied
to its own worker fleet, which is a constant source of build errors for
code like AlphaPose that predates torch 2.x.

**Pick one of these hardware tiers instead, in Space settings:**
- A dedicated GPU (T4-small / L4 / A10G) — billed while running, but a
  normal always-attached GPU that subprocesses see like any other program.
- CPU-basic (free) — works, just slow (think minutes per video, not seconds).

See `SETUP.md` for exactly what to put where, and `check_setup.py` to
verify everything's in place before you push.
