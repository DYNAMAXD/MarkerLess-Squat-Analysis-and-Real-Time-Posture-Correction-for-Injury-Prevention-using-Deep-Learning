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

See `SETUP.md` for exactly what to put where, and `check_setup.py` to
verify everything's in place before you push.
