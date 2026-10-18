# Hugging Face Space (Gradio + ZeroGPU)

Runs the final product's pipeline in the cloud on a free **ZeroGPU** Space: open the Space, drop the front and side
squat videos, get the error in every rep on the 3D body (the product's own results page, inside the Gradio page).
No local PC needed.

| Step | Where | |
|---|---|---|
| calibration | CPU | the dataset cameras (ready), a calibration `.toml`, or two intrinsic videos |
| prepare | CPU | `run_pose2sim.py --phase prepare`: clap sync, board &rarr; extrinsics, videos cut after the clap |
| pose | **GPU** | RTMPose keypoints: `run_pose2sim.pose_phase()` inside `@spaces.GPU` (CPU if no GPU is usable) |
| finish | CPU | `run_pose2sim.py --phase finish`: triangulation, joint angles, reps |
| mesh | CPU | `p2s_mesh.py`: MotionBERT 3D body |
| model | CPU | `squat_research.predict`: CNN-LSTM error per rep + attention |

Only the pose step uses the GPU (about a minute per analysis), because every visitor has a small daily ZeroGPU
quota (signed-in free accounts 5 minutes, PRO 40).

## Build the upload folder

```
python hf_space/build_space.py
```

writes `hf_space/upload/` (git-ignored, about 320 MB) with only the files the Space needs. It needs the trained model
in `final_product/squat_model_export/` (copy of `research/results/04_final/export/`) and the MotionBERT checkpoint +
SMPL files (Git LFS in this repo: `git lfs pull`).

## Create the Space

1. https://huggingface.co/new-space -> a name, SDK **Gradio** (Blank), hardware **ZeroGPU** (a free account can host
   up to 2 ZeroGPU Spaces once it is older than 30 days with a verified e-mail), **Private** if the videos must not be
   seen by others (everyone who can open the Space can open every session).
2. Upload the **contents** of `hf_space/upload/` (not the folder itself) - `README.md`, `app.py`, `requirements.txt`,
   `packages.txt` must end up at the top level of the Space:
   - in the browser: Files -> Add file -> Upload files, drag everything in (large files go to Git LFS by themselves), or
   - `huggingface-cli upload <user>/<space> hf_space/upload . --repo-type space`, or
   - git: clone the Space, copy the contents in, `git lfs install`, commit, push.
3. The Space installs `packages.txt` (apt) and `requirements.txt` (pip) and starts `app.py`. The first build takes
   about 10-15 minutes. Errors show under **Logs** (Build / Container).

The Space's `README.md` (from `SPACE_README.md`) sets `sdk: gradio`, `sdk_version: 6.30.0`, `python_version: 3.12.12`
(ZeroGPU supports 3.10.13 and 3.12.12; Pose2Sim needs 3.11+). Space variables: `PRODUCT_POSE_MODE`
(`performance` default, as the training data; `balanced`, `lightweight`), `GPU_MAX_SECONDS` (default 240).

## Change the model on the Space

The Space uses `models/squat_model/` (`model.pt`, `config.json`, `norm_stats.json` = the export of
`research/notebooks/04_final_model.ipynb`) or the same files as `models/squat_model.zip`. To change it: rename the old
one (e.g. `squat_model_2026-10-18`), upload the new one as `models/squat_model` (folder or `.zip`); the Space restarts
and the line **Error model** at the top of the page shows the new one (name, training date, CV score, file). Without a
`squat_model` the newest model in `models/` is used. A model whose `feature_version` differs from the Space's
`research/squat_research` is refused (reps shown without errors) - upload `research/` too when the features changed.
Details: `models/README.md` in the Space (from `models_README.md`).

## Files here

| File | |
|---|---|
| `app.py` | the Gradio page and the pipeline steps |
| `requirements.txt`, `packages.txt` | pip and apt packages of the Space (one environment) |
| `SPACE_README.md` | becomes the Space's `README.md` (Space settings in its header) |
| `space.gitattributes` | becomes the Space's `.gitattributes` (Git LFS for `*.bin`, `*.pkl`, `*.pt`, ...) |
| `presets/cameras/Dataset_cameras/` | intrinsics of the dataset's DJI pair (1920x1080) |
| `models_README.md` | becomes the Space's `models/README.md` (how to change the model) |
| `build_space.py` | assembles `upload/` |

## Run it locally (optional, CPU)

From the built folder, with a Python that has the packages of `requirements.txt` plus `gradio==6.30.0` (Pose2Sim may
stay in its own environment):

```
cd hf_space/upload
set POSE2SIM_PY=C:\Pose2SimVenv\Scripts\python.exe
python app.py            -> http://127.0.0.1:7860
```
