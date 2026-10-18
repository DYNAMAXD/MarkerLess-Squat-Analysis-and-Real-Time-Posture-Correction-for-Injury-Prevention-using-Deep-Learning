---
title: Squat Analysis
emoji: 🏋️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12.12"
app_file: app.py
pinned: false
short_description: Two-camera squat videos to errors per rep on a 3D body
---

# Squat Analysis

Upload the front and side video of a squat set and get every rep checked: the app finds the clap, calibrates the
two cameras from the board held before it, rebuilds the squat in 3D (Pose2Sim + RTMPose), builds the 3D body
(MotionBERT + SMPL), detects the error in each rep with the trained CNN-LSTM and colours the body with the painted
template of that error, weighted by the model's attention over the frames.

## Use

1. **Camera calibration**: keep **Dataset cameras** (the DJI pair the training data was recorded with, 1920x1080),
   or upload a Pose2Sim `calibration.toml`, or the two intrinsic videos of your own cameras.
2. **Squat videos**: the front and the side video (board held still where both cameras see it, one clap, then
   the squats).
3. **Start analysis**. Progress shows on the left; the results (3D body, keypoint videos, error and fixes per rep,
   timeline, downloads) open below when it is done.

## ZeroGPU

Only the body-keypoint step (RTMPose) runs on the GPU - about a minute of GPU time per analysis, taken from the
visitor's daily ZeroGPU quota (signed-in free accounts have 5 minutes a day, PRO 40). Everything else runs on the
CPU. When no GPU is available (quota used up, queue) the keypoints are computed on the CPU instead: slower, same
result. One analysis runs at a time. Sessions are kept until the Space restarts, and everyone who opens the Space
can open them: make the Space **private** if the videos must not be seen by others.

Space variables: `PRODUCT_POSE_MODE` (`performance` = as the training data, default; `balanced`, `lightweight`),
`GPU_MAX_SECONDS` (upper limit of one GPU call, default 240).

## Contents

`app.py` (this page), `final_product/` (pipeline helpers, results page, model export), `two_view/` (clap sync,
calibration, Pose2Sim run, joint angles, mesh), `research/squat_research/` (model code), `keypoint_pipeline.py` +
`MotionBERT/` (mesh model), `saliency_maps/` (painted templates), `presets/` (the dataset cameras' calibration).
SMPL (`MotionBERT/data/mesh/SMPL_NEUTRAL.pkl`) is licensed for non-commercial research use by the Max Planck
Institute (https://smpl.is.tue.mpg.de/); MotionBERT under its own licence (`MotionBERT/LICENSE`).
