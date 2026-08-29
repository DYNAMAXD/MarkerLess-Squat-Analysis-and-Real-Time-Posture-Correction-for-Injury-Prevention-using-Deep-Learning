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

# AlphaPose + MotionBERT — Video → 2D Keypoints → 3D Pose → SMPL Mesh

A single Gradio Space that takes an uploaded MP4, runs a four‑stage CV pipeline
(**YOLOv11 → AlphaPose → MotionBERT → optional SMPL mesh**) entirely **in‑process**
on CPU, and returns a per‑frame keypoint CSV, a 2D‑skeleton overlay video, and
(optionally) a rendered 3D mesh video — plus a live "debug console" in the UI
since Spaces give you no terminal.

This document is a from‑the‑source technical write‑up: everything below was
read directly out of `app.py`, `keypoint_pipeline.py`, `check_setup.py`, the
requirements files, and the vendored `AlphaPose/` and `MotionBERT/` trees, not
summarized from the Space's marketing copy. Where the code disagrees with the
repo's own `README.md`/`SETUP.md`, that's called out explicitly in
[§12](#12-known-issues-inconsistencies--gotchas) — several of the shipped docs
describe an **earlier, subprocess‑based version** of this app that no longer
matches what's actually running.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Layout](#3-repository-layout)
4. [The Pipeline, Stage by Stage](#4-the-pipeline-stage-by-stage)
5. [The Application Layer (`app.py`)](#5-the-application-layer-apppy)
6. [Models & Weights Inventory](#6-models--weights-inventory)
7. [Dependency Stack, Explained Line by Line](#7-dependency-stack-explained-line-by-line)
8. [Setup & Deployment Guide (corrected)](#8-setup--deployment-guide-corrected)
9. [Running Locally / CLI Usage](#9-running-locally--cli-usage)
10. [Function Reference (`keypoint_pipeline.py`)](#10-function-reference-keypoint_pipelinepy)
11. [Output Formats](#11-output-formats)
12. [Known Issues, Inconsistencies & Gotchas](#12-known-issues-inconsistencies--gotchas)
13. [Unfinished / Scaffolded Feature: Mesh Saliency](#13-unfinished--scaffolded-feature-mesh-saliency)
14. [Licensing & Attribution](#14-licensing--attribution)

---

## 1. What This Project Does

You upload a video of a person moving. The app:

1. Finds the person in every frame (**YOLOv11**, object detection).
2. Estimates their 2D body keypoints in every frame (**AlphaPose**, Halpe‑26
   joint set — 26 keypoints including face and foot points, not just COCO's 17).
3. Lifts that 2D sequence into 3D joint coordinates (**MotionBERT**, a
   transformer trained on Human3.6M).
4. Optionally regresses a full 3D human mesh (**MotionBERT's mesh head +
   SMPL**, 6890 vertices per frame) instead of just sparse joints.
5. Hands back:
   - a CSV with one row per `(frame, joint)` containing 2D (and optionally 3D)
     coordinates and detection confidence,
   - an MP4 with the 2D skeleton drawn over the original footage,
   - if mesh mode was on, a second MP4 rendering the reconstructed mesh.

The code comments reveal the intended downstream use: this is scaffolding for
a **squat‑form / exercise‑correctness analysis project** — see
[§13](#13-unfinished--scaffolded-feature-mesh-saliency) for the explainability
hooks already stubbed in for that.

## 2. High-Level Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │                app.py (Gradio UI)             │
                         │  video upload, checkboxes, det_conf slider,   │
                         │  live "Debug Console" textbox                  │
                         └───────────────────────┬────────────────────────┘
                                                  │ calls process_video()
                                                  ▼
┌───────────────────────────── keypoint_pipeline.py ─────────────────────────────┐
│                                                                                  │
│  Video file (cv2.VideoCapture, frame-by-frame)                                  │
│         │                                                                       │
│         ▼                                                                       │
│  STAGE 1 — YOLOv11 (yolo11s.pt, class "person" only)                            │
│         │  → boxes [x1,y1,x2,y2,conf], top box kept as "primary person"         │
│         ▼                                                                       │
│  STAGE 2 — AlphaPose (halpe26_fast_res50_256x192.pth)                           │
│         │  → 26 (x, y, score) 2D keypoints per frame, in original frame coords  │
│         ├──────────────► draw_overlay() → overlay_<name>.mp4                    │
│         ▼                                                                       │
│  Remap Halpe‑26 → Human3.6M‑17  (halpe26_to_h36m17, interpolates Spine/Neck)    │
│         │                                                                       │
│         ▼                                                                       │
│  Normalize to ~[-1, 1] image-plane coords (normalize_2d_for_motionbert)         │
│         │                                                                       │
│         ├─────────────────────────────┬─────────────────────────────────────┐  │
│         ▼ (if use_3d)                 ▼ (if use_mesh)                       │  │
│  STAGE 3 — MotionBERT pose3d    STAGE 4 — MotionBERT mesh (DSTformer +      │  │
│  (DSTformer, FT_MB_lite_...)    MeshRegressor, FT_MB_release_..._pw3d)      │  │
│         │  → (T,17,3) 3D joints        │  → (T,6890,3) SMPL verts +        │  │
│         │                              │     (T,17,3) mesh-derived joints  │  │
│         │                              ▼                                    │  │
│         │                     render_mesh_video() → mesh_<name>.mp4         │  │
│         ▼                                                                    │  │
│  Write CSV: frame, joint, x_2d, y_2d, score_2d [, x_3d, y_3d, z_3d]          │  │
└──────────────────────────────────────────────────────────────────────────────┘
```

Everything in the box above runs **in the same Python process**, on **CPU**
(`DEVICE = "cpu"` is hardcoded — see [§12](#12-known-issues-inconsistencies--gotchas)),
sequentially, frame‑by‑frame for stages 1–2, then in `clip_len=243`‑frame
chunks for stages 3–4 (MotionBERT's fixed receptive field).

## 3. Repository Layout

```
<space root>/
├── README.md                 ← HF Space card (this file's YAML header origin; historically
│                                described a different, subprocess-based architecture)
├── SETUP.md                   ← deployment walkthrough (partially stale, see §12)
├── app.py                     ← Gradio UI + live debug console
├── keypoint_pipeline.py       ← the entire CV pipeline, single file
├── check_setup.py             ← pre-push file-existence checker (partially stale, see §12)
├── packages.txt                ← apt packages installed at Space build time
├── pre-requirements.txt       ← pip packages installed BEFORE requirements.txt
├── requirements.txt           ← main pip dependency list
├── yolo11s.pt                  ← YOLOv11-small detector weights (19.3 MB)
│
├── AlphaPose/                  ← full vendored clone of MVIG-SJTU/AlphaPose (418 MB)
│   ├── alphapose/               (the installable Python package: models, utils, config)
│   ├── configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml   (cfg this app loads)
│   ├── pretrained_models/halpe26_fast_res50_256x192.pth        (2D pose weights)
│   ├── detector/, trackers/, scripts/, docs/, examples/, model_files/
│   └── setup.py, setup.cfg, LICENSE, README.md
│
└── MotionBERT/                 ← full vendored clone of Walter0807/MotionBERT (328 MB)
    ├── lib/model/DSTformer.py         (the transformer backbone both heads share)
    ├── lib/model/model_mesh.py        (MeshRegressor: DSTformer + SMPL head)
    ├── lib/utils/utils_smpl.py        (SMPL face-topology helper, get_smpl_faces())
    ├── configs/pose3d/MB_ft_h36m_global_lite.yaml
    ├── configs/mesh/MB_ft_pw3d.yaml
    ├── checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
    ├── checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin
    ├── data/mesh/                     (EMPTY in this repo — you must add the 4 SMPL
    │                                    asset files yourself, see §6; a SMPL_NEUTRAL.pkl
    │                                    was committed at one point and later deleted,
    │                                    presumably for licensing reasons)
    ├── infer_wild.py, infer_wild_mesh.py   (MotionBERT's own CLI scripts — NOT called
    │                                         by this app; kept only because they came
    │                                         with the clone)
    └── train.py, train_mesh.py, train_action*.py, LICENSE, README.md
```

`AlphaPose/` and `MotionBERT/` are **unmodified upstream source trees** — this
project doesn't patch them, it imports their Python packages directly
(`sys.path.insert(...)`) and calls their model classes with hand‑written
pre/post‑processing. Everything project‑specific lives in the four files at
the repo root.

## 4. The Pipeline, Stage by Stage

All of this lives in `keypoint_pipeline.py`. `ROOT` is resolved from
`Path(__file__).resolve().parent`, and every other path (`ALPHAPOSE_DIR`,
`MOTIONBERT_DIR`, checkpoint paths, `SMPL_DATA_DIR`) is derived from it, so
the whole thing assumes the exact directory layout in §3.

### 4.1 Video ingestion

`process_video()` opens the file with `cv2.VideoCapture`, reads `fps`, `w`,
`h`, and the (approximate) frame count, and iterates frame‑by‑frame with
`cap.read()` until it runs out. There is no frame skipping/sampling — every
frame is run through the full 2D pipeline.

### 4.2 Stage 1 — YOLOv11 person detection

- `load_yolo()` loads `yolo11s.pt` via `ultralytics.YOLO(...)`. This is the
  **small** variant of Ultralytics' YOLOv11 (the pickle's opcode list — C2PSA,
  C3k2, SPPF, DFL heads etc. — confirms it's a YOLOv11 detection checkpoint,
  not v8).
- `detect_persons()` calls `yolo_model.predict(..., conf=det_conf,
  classes=[0], device="cpu")`. `classes=[0]` restricts detections to COCO
  class 0 (`person`) — nothing else is ever detected or passed downstream.
- Returns a list of `[x1, y1, x2, y2, conf]` boxes.
- In `process_video()`, boxes are sorted by confidence and **only the single
  highest‑confidence box per frame** is kept ("primary person"). There is no
  multi‑person output and no tracking/ID association across frames — if the
  highest‑confidence detection flips between two people in a multi‑person
  video, the resulting keypoint sequence will silently jump between bodies.
- If no person is detected in a frame, the pipeline writes an all‑zero
  `(26,3)` keypoint row and passes the raw (undrawn) frame through to the
  overlay video, rather than skipping the frame or interpolating.

### 4.3 Stage 2 — AlphaPose 2D keypoints (Halpe‑26)

- `load_alphapose()` loads the config at
  `AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml` via
  AlphaPose's own `update_config`, builds the pose model with
  `alphapose.models.builder.build_sppe`, and loads
  `halpe26_fast_res50_256x192.pth` — a **ResNet‑50‑backbone "FastPose"** model
  trained on the **Halpe** dataset's 26‑keypoint annotation format (COCO's 17
  body points plus head/neck/hip and both feet's big toe/small toe/heel).
- `_crop_and_normalize()` does the pre‑processing manually rather than reusing
  AlphaPose's own dataloader/transform pipeline:
  - pads the YOLO box by 10% on each side,
  - resizes the crop to **256×192** (height×width — matches the config name),
  - converts BGR→RGB, scales to `[0,1]`,
  - subtracts a fixed per‑channel mean `[0.406, 0.457, 0.480]` (no std
    division — this is AlphaPose's own normalization constants, not
    ImageNet's `[0.485, 0.456, 0.406]`/`[0.229,0.224,0.225]`),
  - converts to a `(1,3,256,192)` tensor.
- `run_alphapose_on_frame()` runs the model to get `(1, 26, H, W)` heatmaps,
  then for each of the 26 joints takes a **plain `argmax`** over the heatmap
  (`np.unravel_index(np.argmax(...))`) and rescales the `(px, py)` heatmap
  cell back into the padded box's coordinate frame. This is a from‑scratch,
  simplified decoder — it does **not** use AlphaPose's own post‑processing
  (no DARK/soft‑argmax sub‑pixel refinement, no flip‑test averaging), so
  expect slightly coarser localization than AlphaPose's official demo script.
- Output: one `(26, 3)` array of `[x, y, heatmap_peak_score]` per frame, in
  original video pixel coordinates.

### 4.4 Joint remapping — Halpe‑26 → Human3.6M‑17

MotionBERT's public checkpoints are trained on **Human3.6M's 17‑joint
skeleton**, not Halpe‑26, so a hand‑authored correspondence table
(`HALPE26_TO_H36M17`) remaps what overlaps:

| H36M‑17 idx | H36M‑17 name | ← Halpe‑26 idx | Halpe‑26 name |
|---|---|---|---|
| 0 | Hip | 19 | Hip |
| 1 | RHip | 12 | RHip |
| 2 | RKnee | 14 | RKnee |
| 3 | RAnkle | 16 | RAnkle |
| 4 | LHip | 11 | LHip |
| 5 | LKnee | 13 | LKnee |
| 6 | LAnkle | 15 | LAnkle |
| 7 | Spine | — | **interpolated**: `mid(Hip, Thorax)` |
| 8 | Thorax | 18 | Neck |
| 9 | Neck/Nose | — | **interpolated**: `mid(Thorax, Head)` |
| 10 | Head | 17 | Head |
| 11 | LShoulder | 5 | LShoulder |
| 12 | LElbow | 7 | LElbow |
| 13 | LWrist | 9 | LWrist |
| 14 | RShoulder | 6 | RShoulder |
| 15 | RElbow | 8 | RElbow |
| 16 | RWrist | 10 | RWrist |

Halpe‑26's face points (LEye/REye/LEar/REar) and foot points (toes/heels) have
**no H36M equivalent and are simply dropped** for the 3D/mesh stages — they
still appear in the CSV's 2D‑only columns, but never reach MotionBERT.

### 4.5 Normalization for MotionBERT

`normalize_2d_for_motionbert()` centers each frame's 17 H36M points on the
frame center and scales by half of `max(frame_w, frame_h)`, producing
roughly `[-1, 1]`‑ranged coordinates — this matches the input convention
MotionBERT's pretrained checkpoints were trained on (pixel‑space 2D
detections normalized by image size, not real‑world units).

### 4.6 Stage 3 — MotionBERT 3D lift

- `load_motionbert()` reads `configs/pose3d/MB_ft_h36m_global_lite.yaml` via
  MotionBERT's `lib.utils.tools.get_config`, then builds a **`DSTformer`**
  (MotionBERT's dual‑stream spatio‑temporal transformer backbone) sized from
  the config's `dim_feat`/`dim_rep`/`depth`/`num_heads`/`mlp_ratio`/`maxlen`/
  `num_joints`, and loads the `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`
  checkpoint (state dict under the `"model_pos"` key, with a `module.` prefix
  strip for `DataParallel`‑saved checkpoints).
- `lift_sequence_to_3d()` feeds the normalized `(T, 17, 3)` sequence through
  in **243‑frame chunks** (`clip_len=243`, MotionBERT's fixed temporal
  receptive field / positional‑embedding length). The final, shorter chunk is
  padded by **repeating its last frame** rather than zero‑padding, to avoid
  feeding the model an unnatural "freeze to zero" transition — chunks are
  processed independently with **no cross‑chunk smoothing**, so a visible
  discontinuity is possible every 243 frames on long videos.
- Output: `(T, 17, 3)` 3D coordinates per frame, in the model's own scale
  (root‑relative, millimeter‑like units per MotionBERT's training
  convention — not calibrated to any real‑world unit from this app alone).

### 4.7 Stage 4 (optional) — MotionBERT mesh / SMPL regression

Enabled by the "Also reconstruct a 3D mesh" checkbox.

- `check_smpl_assets()` runs **first**, before any model is loaded, and
  raises a `FileNotFoundError` listing exactly which of the four required
  files are missing from `MotionBERT/data/mesh/` — a deliberate fail‑fast
  design so a long video doesn't get processed for minutes only to crash at
  the very last stage:
  - `SMPL_NEUTRAL.pkl` — the actual SMPL body model (blend shapes + joint
    regressor + skinning weights). **License‑gated**: must be registered for
    and downloaded from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) as
    `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`, then renamed.
  - `smpl_mean_params.npz`, `J_regressor_extra.npy`,
    `J_regressor_h36m_correct.npy` — standard SPIN‑repo data files
    ([nkolot/SPIN](https://github.com/nkolot/SPIN)) reused by nearly every
    SMPL‑based mesh‑recovery project (SPIN, VIBE, MotionBERT itself).
- `load_motionbert_mesh()` builds the **same `DSTformer` backbone**, but
  wraps it in MotionBERT's `MeshRegressor` head (`lib.model.model_mesh`),
  configured from `configs/mesh/MB_ft_pw3d.yaml`, and loads the
  `FT_MB_release_MB_ft_pw3d/best_epoch.bin` checkpoint (under the `"model"`
  key). `args.data_root` is overridden to the absolute `SMPL_DATA_DIR` so the
  mesh head's own SMPL‑loading code resolves the assets regardless of the
  process's working directory.
- `lift_sequence_to_mesh()` runs the **same normalized H36M‑17 sequence**
  used for the pose3d stage (not a separate mesh‑specific 2D format) through
  the mesh model in the same 243‑frame chunks (**no flip‑test averaging**,
  which the docstring explicitly notes is an accuracy refinement being
  skipped for simplicity). Returns:
  - `verts`: `(T, 6890, 3)` SMPL mesh vertex positions (6890 is SMPL's fixed
    vertex count), root‑relative, millimeter‑scale,
  - `kp_3d`: `(T, 17, 3)` H36M joints regressed *from the mesh* (via SMPL's
    joint regressor) — a second, independent 3D‑joint estimate distinct from
    Stage 3's `kpts_3d`, used only for mesh‑adjacent purposes, not merged
    back into the CSV.
- `smplx==0.1.28` in `requirements.txt` supplies the Python SMPL/SMPL‑X body
  model classes that MotionBERT's mesh code depends on; `chumpy` is pulled in
  transitively because the classic `SMPL_NEUTRAL.pkl` stores its blend‑shape
  arrays as `chumpy.Ch` objects, and unpickling it requires `chumpy` to be
  importable even though nothing in this app calls it directly (see the
  numpy‑compatibility note in [§12](#12-known-issues-inconsistencies--gotchas)).

### 4.8 Rendering

**Overlay video** (`draw_overlay`, always produced): draws the 26 Halpe
keypoints as filled circles and a fixed list of skeleton edges
(`HALPE26_EDGES`) as lines directly onto each original frame with OpenCV,
using a fixed 0.05 confidence threshold per joint/edge.

**Mesh video** (`render_mesh_video`, only if mesh mode succeeded): renders
each frame's `(6890, 3)` vertex cloud with **Matplotlib's `Agg` backend**
(`mpl_toolkits.mplot3d`, `ax.plot_trisurf`) rather than a real 3D renderer
(pyrender/OSMesa/EGL) — the code comment explains this tradeoff explicitly:
offscreen GPU renderers are fragile in headless containers, so a slower but
dependency‑free matplotlib render was chosen instead. Face triangles come
from `lib.utils.utils_smpl.get_smpl_faces()`; if that import fails for any
reason, it falls back to a plain point‑cloud scatter of the vertices. The
axis limits are computed once from the *entire* clip's vertex range (so
scale stays consistent across frames) and the camera uses a **fixed**
`elev=-90, azim=-90` view — despite the docstring calling this a
"turntable‑style" render, the camera does **not** rotate frame to frame; it's
a static top‑down orthographic view (see [§12](#12-known-issues-inconsistencies--gotchas)).

### 4.9 CSV output

See [§11](#11-output-formats).

## 5. The Application Layer (`app.py`)

`app.py` is intentionally thin — one screen, one button — but has one
non‑obvious piece of infrastructure: **a live debug console**, because HF
Spaces give you no SSH/terminal access to watch stdout while a job runs.

- **`QueueLogHandler`**: a `logging.Handler` subclass whose `emit()` pushes
  formatted log lines into a `queue.Queue` instead of printing them.
- **`run_pipeline()`** (the Gradio click handler, a **generator function**):
  1. Attaches a fresh `QueueLogHandler` to `keypoint_pipeline.LOGGER`.
  2. Starts `process_video(...)` on a **background `threading.Thread`**
     (`worker()`), so the main coroutine stays free to keep yielding UI
     updates.
  3. Loops `while t.is_alive()`, draining the queue and `yield`‑ing the
     accumulated console text (plus `None` for the not‑yet‑ready outputs)
     roughly every 0.3s — this is what makes Gradio's textbox appear to
     stream live.
  4. On completion, does a final queue drain, then yields the CSV/overlay/
     mesh file paths (or the captured traceback text, on failure) as the
     last message.
  5. Removes its log handler in a `finally`‑equivalent path so repeated runs
     don't stack duplicate handlers on the shared module logger.
- **Gradio layout**: a `gr.Blocks` app with a video input, two checkboxes
  (`use_3d`, `use_mesh`), a `det_conf` slider (0.1–0.9, default 0.5, passed
  straight through to YOLO's `conf=`), a run button, and four outputs — the
  debug textbox, the CSV file, the overlay video, and the mesh video.
- **`_run()` / `@spaces.GPU(duration=1)`**: a one‑line wrapper function
  decorated with HF's ZeroGPU helper, present purely so ZeroGPU's Space
  builder sees at least one `@spaces.GPU`‑decorated function (a static
  requirement for ZeroGPU hardware to allocate anything at all).
  **This function is never actually called anywhere in the file** —
  `run_pipeline()` calls `process_video()` directly inside `worker()`,
  bypassing `_run()` entirely. It is dead code that exists only to satisfy
  ZeroGPU's build‑time check. See [§12](#12-known-issues-inconsistencies--gotchas)
  for why this matters.

## 6. Models & Weights Inventory

| File | Origin | Purpose | Expected path | Approx. size* |
|---|---|---|---|---|
| `yolo11s.pt` | Ultralytics YOLOv11‑small, COCO‑pretrained | Person detection (class 0 only) | repo root | ~19–20 MB |
| `halpe26_fast_res50_256x192.pth` | AlphaPose's own model zoo (ResNet‑50 FastPose, Halpe‑26) | 2D keypoint heatmaps | `AlphaPose/pretrained_models/` | ~130 MB |
| `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` | MotionBERT model zoo | 2D→3D joint lifting (DSTformer) | `MotionBERT/checkpoint/pose3d/.../` | ~61 MB |
| `FT_MB_release_MB_ft_pw3d/best_epoch.bin` | MotionBERT model zoo | SMPL mesh regression (DSTformer + MeshRegressor) | `MotionBERT/checkpoint/mesh/.../` | ~162 MB |
| `SMPL_NEUTRAL.pkl` | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) (**license‑gated, not redistributable**) | SMPL body model (blend shapes, skinning, joint regressor) | `MotionBERT/data/mesh/` | ~40 MB |
| `smpl_mean_params.npz` | [SPIN repo](https://github.com/nkolot/SPIN) data bundle | Mean pose/shape init for the mesh head | `MotionBERT/data/mesh/` | small |
| `J_regressor_extra.npy` | SPIN repo data bundle | Extra joint regressor (non‑SMPL joints, e.g. for other eval sets) | `MotionBERT/data/mesh/` | small |
| `J_regressor_h36m_correct.npy` | SPIN repo data bundle | H36M‑specific joint regressor from mesh vertices | `MotionBERT/data/mesh/` | small |

\* Sizes are the repo's own `SETUP.md` accounting, not independently
re‑measured file‑by‑file here. The four SMPL files are **not included** in
this Space (`MotionBERT/data/mesh/` ships empty; a `SMPL_NEUTRAL.pkl` was
committed once and later deleted from the Space, consistent with the
license restriction), so mesh mode will not work until you add them
yourself.

## 7. Dependency Stack, Explained Line by Line

**`packages.txt`** (apt packages, installed at Space build time, before any
pip step):
```
ffmpeg          # video encode/decode; cv2.VideoWriter's mp4v backend and
                # general MP4 muxing rely on this being present
libsm6          # X11 Session Management lib — a transitive runtime dependency
libxext6        # of OpenCV's build even in headless/server use
libgl1          # OpenGL runtime — needed by opencv-python-headless's own
                # linked libs on some manylinux wheels, even without a display
build-essential # gcc/g++/make — needed to compile cython_bbox and any native
                # extensions pulled in transitively
```

**`pre-requirements.txt`** (installed *before* `requirements.txt`, per the HF
Spaces convention for anything a later `pip install` needs to already be
importable):
```
numpy<2.0            # AlphaPose/MotionBERT-era code predates numpy 2.x's API changes
setuptools<81 wheel  # older setuptools needed for some legacy build_ext paths
torch==2.8.0          # pinned specifically for ZeroGPU compatibility (ZeroGPU
torchvision==0.23.0   # only supports a fixed short list of torch builds) —
                       # see §12 for why this pin is largely moot today
cython cython_bbox    # native extensions some pose/detection code paths use
```
The file's own top comment is explicit that **this app does not actually do
any editable/native install of AlphaPose** (`pip install -e ./AlphaPose`) —
it imports AlphaPose's Python package in‑process via `sys.path.insert`
instead — so the torch pin here is a defensive leftover rather than a hard
requirement of the current code path.

**`requirements.txt`** (main pip stack):
```
gradio==4.44.1                                  # UI framework
gradio_client==1.3.0, fastapi==0.115.2,
starlette==0.40.0, pydantic<2.10                # pinned together for gradio 4.44.1 compatibility
huggingface_hub==0.25.2                          # HF API client gradio depends on
torch, torchvision                               # (unpinned here; version comes from pre-requirements.txt)
opencv-python-headless                           # video I/O, drawing, no GUI deps
numpy<2.0                                        # consistent with pre-requirements.txt
ultralytics                                      # YOLOv11 model loading/inference
easydict, pyyaml                                 # AlphaPose/MotionBERT config-object conventions
scipy                                             # numerical utilities used by both vendored repos
cython, munkres, tqdm, tensorboardX               # AlphaPose/MotionBERT training-code dependencies
                                                   # (munkres = Hungarian algorithm, used by AlphaPose's
                                                   # multi-person tracker, which this app doesn't invoke)
matplotlib                                        # mesh-video rendering (Agg backend)
smplx==0.1.28                                     # SMPL/SMPL-X body model classes for the mesh stage
```
A trailing comment flags a **known fragility**: `chumpy` (needed to unpickle
the classic `SMPL_NEUTRAL.pkl`, which stores arrays as `chumpy.Ch` objects) is
unmaintained and breaks on `numpy>=1.24` because it references removed NumPy
aliases (`np.bool`, `np.object`, `np.int`). If the mesh stage fails with an
`AttributeError` originating in `chumpy`, the fix noted in the file itself is
to tighten the pin to `numpy<1.24,>=1.19` and redeploy.

## 8. Setup & Deployment Guide (corrected)

The repo's own `SETUP.md` is a good starting point but drifted from the
current code in two places (noted inline below). Corrected steps:

1. **Clone the two upstream repos locally** (not inside the Space repo yet):
   ```bash
   git clone https://github.com/MVIG-SJTU/AlphaPose.git
   git clone https://github.com/Walter0807/MotionBERT.git
   ```
2. **Place the weight files** at the exact paths in [§6](#6-models--weights-inventory).
   In particular:
   - `AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth`
   - `MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`
   - `MotionBERT/checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin`
   - `MotionBERT/data/mesh/SMPL_NEUTRAL.pkl` — **note**: directly under
     `data/mesh/`, *not* `data/mesh/smpl/`. `SETUP.md` and `check_setup.py`
     both say `data/mesh/smpl/SMPL_NEUTRAL.pkl`; the actual runtime path
     resolved by `keypoint_pipeline.py`'s `SMPL_DATA_DIR = MOTIONBERT_DIR /
     "data" / "mesh"` has **no `smpl/` subfolder**. Follow the code, not
     those two docs, or the fail‑fast check in `check_smpl_assets()` will
     report the file missing even when it's present at the documented‑but‑wrong path.
   - `MotionBERT/data/mesh/smpl_mean_params.npz`,
     `J_regressor_extra.npy`, `J_regressor_h36m_correct.npy` — same
     directory, from the SPIN repo's data bundle.
   - `yolo11s.pt` in the repo root.
3. **Clone your empty HF Space repo** locally and move `AlphaPose/`,
   `MotionBERT/`, `yolo11s.pt`, and the six root files
   (`README.md`, `packages.txt`, `pre-requirements.txt`, `requirements.txt`,
   `app.py`, `check_setup.py`) into it.
4. **Git‑LFS the binaries** before committing:
   ```bash
   cd <space-name>
   git lfs install
   git lfs track "*.pth" "*.pt" "*.pkl" "*.bin" "*.npz" "*.npy"
   git add .gitattributes
   ```
5. **Run `check_setup.py`** — but be aware it checks the older
   `data/mesh/smpl/SMPL_NEUTRAL.pkl` path (see step 2), so a "MISS" there
   doesn't necessarily mean the actual pipeline will fail; verify against
   `SMPL_DATA_DIR`/`SMPL_REQUIRED_FILES` in `keypoint_pipeline.py` directly
   if in doubt.
6. `git add . && git commit -m "..." && git push`.
7. **Set the Space's hardware tier deliberately.** As of this writing the
   live Space is configured on **ZeroGPU** ("Running on Zero" in the Space
   header) — see [§12](#12-known-issues-inconsistencies--gotchas) for why
   that choice currently has no effect on speed either way, and why the
   Space's own `README.md` explicitly argues against ZeroGPU for a different
   (now‑outdated) reason.
8. Watch the **Build logs** tab for the `packages.txt`/`pre-requirements.txt`/
   `requirements.txt` install, then the **app/container logs** (or the app's
   own Debug Console once it loads) for model‑loading errors.

## 9. Running Locally / CLI Usage

`keypoint_pipeline.py` has its own `argparse` entry point, independent of
Gradio:

```bash
python keypoint_pipeline.py path/to/video.mp4              # 2D + 3D lift, no mesh
python keypoint_pipeline.py path/to/video.mp4 --no-3d       # 2D only
python keypoint_pipeline.py path/to/video.mp4 --mesh        # 2D + 3D + mesh
```

Outputs land in `./outputs/` (created next to `keypoint_pipeline.py`) as
`<video_stem>_keypoints.csv`, `<video_stem>_overlay.mp4`, and (with `--mesh`)
`<video_stem>_mesh.mp4`. This bypasses `app.py`/Gradio entirely and is the
fastest way to debug the pipeline itself without the UI in the loop.

## 10. Function Reference (`keypoint_pipeline.py`)

| Function | Role |
|---|---|
| `halpe26_to_h36m17(kpts_2d)` | Remaps a `(26,3)` Halpe frame to `(17,3)` H36M order; interpolates Spine & Neck/Nose. |
| `load_yolo()` | Loads `yolo11s.pt` via `ultralytics.YOLO`. |
| `detect_persons(yolo_model, frame, conf)` | Runs YOLO, returns person‑class boxes above `conf`. |
| `load_alphapose()` | Builds AlphaPose's SPPE model from the Halpe‑26 config and loads its checkpoint. |
| `_crop_and_normalize(frame, box)` | Pads/crops/resizes a detection box to 256×192 and normalizes it for AlphaPose. |
| `run_alphapose_on_frame(model, frame, boxes)` | Runs AlphaPose on each box; argmax‑decodes heatmaps into `(26,3)` keypoints. |
| `load_motionbert()` | Builds `DSTformer` from the pose3d config and loads the lift checkpoint. |
| `lift_sequence_to_3d(model, seq_2d, clip_len)` | Chunks a normalized 2D sequence through MotionBERT to get 3D joints. |
| `normalize_2d_for_motionbert(kpts, w, h)` | Centers/scales 2D coords into MotionBERT's expected range. |
| `check_smpl_assets()` | Verifies the 4 SMPL files exist; raises a detailed error listing what's missing. |
| `load_motionbert_mesh()` | Builds `DSTformer` + `MeshRegressor` from the mesh config and loads the mesh checkpoint. |
| `lift_sequence_to_mesh(model, seq_2d, clip_len)` | Chunks the same normalized sequence through the mesh model; returns verts + mesh‑derived joints. |
| `render_mesh_video(verts, out_path, fps, smpl_faces)` | Matplotlib `Agg`‑based fixed‑camera render of the SMPL mesh sequence to MP4. |
| `draw_overlay(frame, kpts_2d_26)` | Draws Halpe‑26 skeleton lines/points onto a frame with OpenCV. |
| `process_video(video_path, use_3d, use_mesh, det_conf, progress_cb)` | The orchestrator — runs all stages in order and writes the CSV/videos. |
| `compute_mesh_saliency(...)` / `_broadcast_joint_saliency_to_vertices(...)` / `render_mesh_saliency_video(...)` | Unused Grad‑CAM‑style scaffolding — see [§13](#13-unfinished--scaffolded-feature-mesh-saliency). |

## 11. Output Formats

**CSV** — one row per `(frame, joint)`, 26 joints × N frames rows total.

- If 3D lift was **enabled**: `frame, joint, x_2d, y_2d, score_2d, x_3d, y_3d, z_3d`.
  For the 9 Halpe joints with no H36M correspondence (face points, foot
  points), `x_3d`/`y_3d`/`z_3d` are left as **empty strings**, not zeros or
  `NaN` — worth handling explicitly if you parse this with `pandas` (empty
  string, not `0.0`, in a numeric column will upcast it to `object`/`NaN`
  depending on parser settings).
- If 3D lift was **disabled**: `frame, joint, x_2d, y_2d, score_2d` only.

**Overlay video** — same resolution/fps as the input, `mp4v`‑encoded, 2D
skeleton drawn in green (edges) and red (joints) over the original footage.

**Mesh video** (mesh mode only) — a separate, fixed 600×600px‑figure render
(`figsize=(6,6), dpi=100` → 600×600 output frames) at the source video's fps,
independent resolution from the input/overlay videos.

## 12. Known Issues, Inconsistencies & Gotchas

These are things a technical reader should know before trusting or extending
this repo — gathered directly from reading the code, not speculation:

1. **`DEVICE = "cpu"` is hardcoded** at the top of `keypoint_pipeline.py`.
   Every model (`YOLO`, AlphaPose, MotionBERT pose3d, MotionBERT mesh) is
   moved to `"cpu"` and every YOLO call is passed `device="cpu"` explicitly.
   **This means the Space's hardware tier has zero effect on inference
   speed today** — a dedicated A10G and CPU‑basic will run this pipeline at
   the same (CPU) speed. This directly contradicts the urgency of the
   hardware‑tier advice in the Space's own `README.md`.
2. **The Space's own `README.md` and `SETUP.md` describe an earlier,
   subprocess‑based architecture** (`ensure_alphapose_and_motionbert_installed()`,
   calling AlphaPose's `demo_inference.py` and MotionBERT's `infer_wild.py`/
   `infer_wild_mesh.py` as child processes via `subprocess.run`). The current
   `keypoint_pipeline.py` does **not** do this — it imports both packages'
   Python modules in‑process via `sys.path.insert(...)` and re‑implements the
   pre/post‑processing itself. The elaborate ZeroGPU‑subprocess warning in
   `README.md` ("child processes don't inherit the GPU grant") no longer
   describes what the code does, though the *practical* advice (ZeroGPU
   won't help this app) still happens to hold — now because of point 1, not
   because of subprocess GPU inheritance.
3. **As observed on the live Space page, hardware is currently set to
   ZeroGPU** ("Running on Zero"), which the Space's own README explicitly
   recommends against. Combined with point 1, the attached GPU (during the
   brief `@spaces.GPU(duration=1)` window) does nothing useful anyway.
4. **`app.py`'s `_run()` function, decorated with `@spaces.GPU(duration=1)`,
   is never called.** `run_pipeline()` invokes `process_video()` directly
   inside a background thread, bypassing `_run()` entirely. The decorator
   exists solely to satisfy ZeroGPU's build‑time requirement that at least
   one function in the app be `@spaces.GPU`‑decorated — it is otherwise dead
   code, and its `duration=1` (one second) wouldn't be enough for real
   inference even if it were wired up.
5. **`check_setup.py` and `SETUP.md` both check for
   `MotionBERT/data/mesh/smpl/SMPL_NEUTRAL.pkl`** (with an extra `smpl/`
   subdirectory), but `keypoint_pipeline.py`'s `SMPL_DATA_DIR` resolves to
   `MotionBERT/data/mesh/` directly — no `smpl/` subfolder. A file placed at
   the path the docs/checker describe will be invisible to the actual
   pipeline, and vice versa. (Consistent with this: the Space's own commit
   history shows a `SMPL_NEUTRAL.pkl` once committed and later deleted
   directly under `MotionBERT/data/mesh/`, matching the code's path, not the
   docs' path.)
6. **Only the single highest‑confidence YOLO detection per frame is used.**
   No multi‑person support, no identity tracking across frames — in a
   multi‑person video, which "person" the keypoints belong to can change
   frame to frame without warning.
7. **2D keypoint decoding is a simplified, hand‑written heatmap argmax**, not
   AlphaPose's own (more accurate) post‑processing pipeline — no sub‑pixel
   refinement, no flip‑test/multi‑scale averaging.
8. **The Halpe‑26 → H36M‑17 joint remap is an approximate, hand‑authored
   table**; Spine and Neck/Nose have no Halpe equivalent and are linearly
   interpolated from neighboring joints rather than derived from the model.
9. **`render_mesh_video`'s camera does not rotate** (`elev=-90, azim=-90`
   fixed every frame) despite the function's own docstring calling it a
   "turntable‑style" render.
10. **SMPL assets are intentionally not bundled** (license restriction) —
    mesh mode will always fail on a fresh clone until you obtain and place
    the four files yourself; this is by design (`check_smpl_assets()`
    fails fast with an actionable message rather than crashing deep inside
    `smplx`).
11. **`chumpy`/NumPy version landmine**: unpickling a classic `SMPL_NEUTRAL.pkl`
    requires `chumpy`, which breaks on `numpy>=1.24`. If you hit a `chumpy`
    `AttributeError`, the fix is tightening `numpy` in `requirements.txt`
    (see [§7](#7-dependency-stack-explained-line-by-line)), not a code change.
12. **AlphaPose's license is non‑commercial/academic‑research‑only**
    (Shanghai Jiao Tong University's SJTU research license, not MIT/Apache);
    MotionBERT is Apache 2.0. Anything you build on top of this Space
    inherits AlphaPose's more restrictive terms — see [§14](#14-licensing--attribution).
13. **No frame sampling** — every single frame of the input video goes
    through YOLO + AlphaPose. On CPU (see point 1), this is the dominant
    cost; a 30‑second, 30fps clip is ~900 full detection+pose passes.

## 13. Unfinished / Scaffolded Feature: Mesh Saliency

`keypoint_pipeline.py` contains three functions —
`compute_mesh_saliency()`, `_broadcast_joint_saliency_to_vertices()`, and
`render_mesh_saliency_video()` — explicitly marked in a block comment as
**"FUTURE"** scaffolding, not wired into `process_video()` anywhere:

- `compute_mesh_saliency()` implements a generic **Grad‑CAM**: it registers
  forward/backward hooks on a classifier's modules, backprops from a chosen
  output class, and combines activations with gradients
  (`ReLU(mean(grad) * activation)`) into a per‑joint saliency map, optionally
  broadcasting it to per‑vertex saliency via nearest‑joint lookup.
- The intent, per the comment, is a **planned squat‑form‑analysis feature**:
  once a downstream classifier exists that judges "correct" vs. "incorrect"
  exercise reps from a pose/mesh sequence, this code would let the app point
  to *which joint or mesh region* drove that classification — i.e.,
  visual explainability for a not‑yet‑built model.
- `render_mesh_saliency_video()` is the matching renderer: same fixed‑camera
  matplotlib approach as `render_mesh_video()`, but color‑maps each vertex by
  its saliency score (`jet` colormap) instead of a flat color.
- **None of this does anything today** — there is no classifier to explain,
  and no call site invokes these functions. They're included here only
  because the prompt asked for "everything," and because a future
  contributor extending this repo should know this groundwork already
  exists rather than re‑implementing it.

## 14. Licensing & Attribution

This Space vendors two independent upstream projects with **different
licenses** — know which one governs which part before reusing this code:

- **AlphaPose** (`AlphaPose/`, `MVIG-SJTU/AlphaPose`) — a custom
  **Shanghai Jiao Tong University academic/non‑commercial research license**.
  It grants use for noncommercial internal research only, prohibits
  redistribution, sublicensing, or commercial use, and requires derivatives
  to remain under the same restriction. It also bundles third‑party notices
  (BSD‑style Torch distro code, Apache‑2.0 TensorFlow/PyraNet code, MIT
  tf‑faster‑rcnn code, BSD pose‑hg‑demo code) for components AlphaPose
  itself incorporates.
- **MotionBERT** (`MotionBERT/`, `Walter0807/MotionBERT`) — **Apache License
  2.0**, a permissive license allowing commercial use, modification, and
  redistribution with attribution.
- **SMPL** (not bundled — obtained separately per [§6](#6-models--weights-inventory)) —
  governed by its own registration‑gated license from the Max Planck
  Institute; redistribution of the model file itself is not permitted, which
  is exactly why it's absent from this repo.
- **YOLOv11** (`yolo11s.pt`) — Ultralytics' AGPL‑3.0 / enterprise dual
  license (as of the Ultralytics versions this pipeline targets); check
  Ultralytics' current licensing terms if deploying this commercially.

Net effect: **because AlphaPose's own license is non‑commercial‑research‑only,
this pipeline as a whole cannot be used commercially without a separate
agreement from AlphaPose's authors**, regardless of MotionBERT's more
permissive Apache terms. This applies to the app code in this repo as a
derivative that depends on and imports AlphaPose directly.
