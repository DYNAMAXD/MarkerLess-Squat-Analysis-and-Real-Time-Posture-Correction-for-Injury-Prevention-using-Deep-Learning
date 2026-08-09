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
instead: `packages.txt` installs apt build tools and `pre-requirements.txt`
installs torch. AlphaPose (compiling its C++ extensions) and MotionBERT's
own requirements are deliberately installed at **app startup** rather than
in requirements.txt — see the comment in `requirements.txt` and
`ensure_alphapose_and_motionbert_installed()` in `app.py` for why (HF's
builder installs requirements.txt before your repo's own files exist, so
any local-path reference there always fails).

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

## A note on ZeroGPU hardware

If you're deploying this on **ZeroGPU** (the free/pay-per-call shared-GPU
tier), be aware this pipeline's architecture doesn't fit it well: ZeroGPU
grants GPU access only inside a function decorated with `@spaces.GPU`,
scoped to that call *in the same Python process*. This app instead calls
AlphaPose's and MotionBERT's own scripts as **separate subprocesses**
(`subprocess.run(...)`) — that's what lets it reuse their unmodified,
correct code instead of re-implementing it. Those child processes are not
guaranteed to see the GPU ZeroGPU granted to the parent.

In practice: this may fall back to CPU inside every subprocess even on a
ZeroGPU Space, making it very slow (and possibly timing out ZeroGPU's
per-call duration limit) without ever throwing a clear error.

**Recommended:** use standard dedicated GPU hardware (T4 small/medium,
L4, or A10G) instead of ZeroGPU for this Space. It costs by the hour
rather than per-call, but it actually gives the subprocesses real,
persistent GPU access. If you want to stay on ZeroGPU, the pipeline would
need restructuring to run the AlphaPose/MotionBERT model code in-process
(importing their Python modules directly and calling them as functions)
rather than via subprocess — a substantially bigger rewrite than this repo
currently does.
