# two_view — front + side camera preprocessing

Calibrated two-camera 3D squat analysis with Pose2Sim: a **side** and a **front** camera, synced by a clap, calibrated
from a chessboard, RTMPose 2D keypoints triangulated into 3D joints, joint angles and reps per recording. The batch
page runs it over a whole dataset; the final product (`final_product/`) and the research notebooks (`research/`) use
its output.

(The earlier method - the single-view AlphaPose pipeline run on each video and fused per rep, `run_two_view.py` /
the website's "Classic" mode - was removed on 2026-10-18 together with AlphaPose.)

## Website (easiest)

Run `two_view\start_website.ps1` (right-click > Run with PowerShell). First run installs the environment at
`C:\Proj1Venv` (~10 min); later runs just open http://127.0.0.1:5000. Upload the videos: the page runs the Pose2Sim
3D analysis, then shows both videos together, angle graphs (click a curve to jump to that moment), a per-rep table
and downloads. Past analyses are listed on the left.
Files: `web_app.py` (server), `web/index.html` (page), `web_results.py` (packages results for the page).

## Pose2Sim 3D mode (calibrated, side + front camera)

The website's default tab, **3D analysis**, triangulates real 3D joints from a **side** and a **front** camera
(roughly at right angles). The **Extrinsic calibration** tab does only the clap sync + calibration, so you can
calibrate once and reuse it.

Two recording modes:

| Mode | Videos per camera (side + front) | Website |
|---|---|---|
| 1 | intrinsic video; full-length video: board held still where both cameras see it &rarr; clap &rarr; squats | 3D analysis, "Calibrate now" |
| 2 | intrinsic video; extrinsic video: board seen by both + one clap; squat videos: clap &rarr; squats &rarr; clap | Extrinsic calibration once, then 3D analysis with "Reuse calibration" + "clap at the start AND the end" |

A whole dataset (mode 1 or 2): see **Batch preprocessing** below.

| Input | What it shows |
|---|---|
| side / front intrinsic video | a checkerboard moved around in front of that camera (40-60 s, over the whole frame incl. corners, at different tilts), same resolution/fps/lens mode as the other videos |
| side / front full-length video (mode 1) | the checkerboard held **still** where both cameras see it, a **clap**, then the squats |
| side / front extrinsic video (mode 2) | the checkerboard still where both cameras see it + one clap (any order) |
| side / front squat video (mode 2) | clap, squats, clap |

```
side_full  ─┐ sync (p2s_sync.py): all sharp sounds of both tracks are lined up -> offset; the clap is the
front_full ─┘   strongest sound heard in BOTH videos -> clap frame per camera
                  (two-clap squat videos: the 2 strongest such sounds >= 3 s apart; their offsets give the drift)
                  ├─ calibration: intrinsics (Pose2Sim, from the intrinsic videos or an earlier calibration)
                  │               extrinsics (p2s_extrinsics.py): scan both videos for the board (every 0.25 s,
                  │               4 CPU processes) - mode 1 only up to the clap, extrinsic videos whole - then
                  │               check frame by frame where both cameras see it, back from the clap then forward
                  │               (side 549/front 449, 548/448, ...), stopping after 45 still pairs; use the
                  │               pair where the board is still and largest -> solvePnP
                  └─ cut both videos from clap+0.5 s (to end clap-0.5 s with two claps; the front is
                     time-stretched if the clocks drifted > half a frame) -> synced videos
   ─► Pose2Sim.poseEstimation() (RTMPose, CPU or --device cuda) ─► personAssociation + triangulation + filtering ─► .trc
   ─► p2s_biomech.py ─► angles.csv, advancedangles.csv
```

Command line (uses the separate Pose2Sim environment; quote paths that contain spaces):

```
:: full analysis, calibrating from the intrinsic videos + the board before the clap
C:\Pose2SimVenv\Scripts\python.exe two_view/run_pose2sim.py --name subj01 ^
    --side side_full.mp4 --front front_full.mp4 --intr-side side_intr.mp4 --intr-front front_intr.mp4 ^
    --intr-corners 6,7 --intr-square 60 --ext-corners 6,7 --ext-square 60

:: intrinsics only (reuse later with --intr-from <out>/calibration.toml)
C:\Pose2SimVenv\Scripts\python.exe two_view/run_pose2sim.py --mode intrinsic --name intr01 ^
    --intr-side side_intr.mp4 --intr-front front_intr.mp4 --intr-corners 6,7 --intr-square 41

:: extrinsic calibration only (intrinsics from videos, or --intr-from an earlier calibration.toml)
C:\Pose2SimVenv\Scripts\python.exe two_view/run_pose2sim.py --mode extrinsic --name calib01 ^
    --side side_full.mp4 --front front_full.mp4 --intr-side side_intr.mp4 --intr-front front_intr.mp4

:: full analysis reusing a complete calibration (no board needed in these videos); mode 2 = --squat-claps 2
C:\Pose2SimVenv\Scripts\python.exe two_view/run_pose2sim.py --name subj02 --squat-claps 2 ^
    --side side_squat.mp4 --front front_squat.mp4 --calib-from outputs/two_view/calib01/calibration.toml
```

Other options: `--device auto|cuda|cpu` (default cpu), `--pose-mode lightweight|balanced|performance`,
`--out-dir` / `--work-dir` (output and Pose2Sim working folders instead of `outputs/...`).

Outputs in `outputs/two_view/<name>/`:

- `angles.csv`: `frame, joint, x_3d, y_3d, z_3d` for LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip,
  LKnee, RKnee, LAnkle, RAnkle, Head, Neck, Hip. Metres, Y up, origin at the board corner marked "origin".
- `advancedangles.csv`: `frame`, the 14 angles (same formulas as `keypoint_pipeline.py`), `com_keypoints`,
  `com_mesh`, `com_final`. `com_mesh` is empty because Pose2Sim has no SMPL mesh, so `com_final = com_keypoints`.
- `reps.csv`: `rep_no, start_frame, end_frame` for every detected squat rep (same frame numbering as above).
- `calibration.toml`, `board_side.jpg` / `board_front.jpg` (detected corners + origin: both must mark the same
  physical corner), `rep_features.csv`, `points3d.trc`, `log.txt` (the full run log, also shown live on the website).

`frame` is the frame number in the synced videos (frame 0 = 0.5 s after the clap, the same instant in both cameras).
The Pose2Sim project folder (config, intrinsic frames, 2D poses, overlay videos) is `outputs/pose2sim/<name>/trial/`.

Notes
- Pose2Sim runs in `C:\Pose2SimVenv` (created by `start_website.ps1` or `start_batch.ps1`; exact versions in
  `requirements_pose2sim.txt`). It needs numpy 2 and another OpenCV build, which would break the MotionBERT
  environment if installed there. The website runs on the CPU; the batch uses the GPU when it can.
- Board corner counts are **inner** corners (default 6×7). Use one even and one odd count so both cameras agree on
  the board's orientation. Hold the extrinsic board so both cameras see its face (about 45° to each), and fill a
  good part of each view: the log warns when the board is under ~200 px across in a camera.
- World "up": with a held or standing board, up comes from the two cameras (assumes level tripods); with
  "flat on the floor", up is the board's normal.
- Nothing pops up during a run (no detection windows, no manual clicking). If no frame has the board in both
  cameras, the run stops and the log says how often each camera saw it.

## Batch preprocessing (whole dataset, GPU)

Every person's front + side squat videos &rarr; CSV files, unattended for hours on a Windows PC with an NVIDIA GPU
(also runs on a CPU). Default = recording **mode 1**: each squat video starts with the board held still where both
cameras see it, then a clap, then the squats. The intrinsics are computed once; every person gets their own
extrinsics from the board in their own videos, so a camera that moved between people does not spoil the others.

1. Copy the repo to that PC and put the dataset in one folder:
   ```
   D:\squat_dataset\
       intrinsic_front.mp4   intrinsic_side.mp4      (required; or folders of these names holding one video each)
       extrinsic_front.mp4   extrinsic_side.mp4      (optional: fallback for a person whose board is not found)
       mapping.csv           person_id,front_id,side_id,error_performed
                             P01,P01_front,P01_side,"knee_valgus, heels_up"
       ...squat videos...    named <front_id> / <side_id> (extension optional), anywhere under the folder
   ```
   Give the videos unique names: both cameras number their files the same way (DJI_0001 ...), and a name found
   twice makes that row "ambiguous" (skipped).
2. Run `two_view\start_batch.ps1` (right-click > Run with PowerShell). The first time it installs Python 3.11 (if
   missing) and `C:\Pose2SimVenv` with `onnxruntime-gpu` + CUDA/cuDNN, then opens http://127.0.0.1:5050.
   **The GPU build needs NVIDIA driver 580 or newer** (CUDA 13); the script warns if the driver is older, and the
   page header must say "onnxruntime CUDA ready" - otherwise everything runs on the much slower CPU.
3. On the page: dataset folder &rarr; **Scan folder** (lists the calibration videos and each CSV row with its videos
   found or missing) &rarr; check the settings (board inner corners + measured square size, board held = vertical,
   claps per squat video, device "GPU (CUDA) only" to fail fast instead of crawling on the CPU) &rarr;
   **Start preprocessing**. Keep the PowerShell window open. Try 2-3 persons first and check their overlays.

The page shows progress, time left, GPU load, the intrinsics, a table of persons (frames, reps, which extrinsics were
used, extrinsic reprojection error, camera distance/angle, triangulation error, links to the overlay videos and board
images) and the live logs. A person whose camera distance differs from the batch median by more than 10 cm is
flagged ("cameras moved or bad board frame?"). **Stop after this person** / **Stop now**; pressing Start again
resumes (finished persons are skipped).

Output (`<dataset>\preprocessed_data\` by default):
- `<person_id>\` &mdash; `angles.csv`, `advancedangles.csv`, `reps.csv`, `rep_features.csv`, `points3d.trc`,
  `calibration.toml`, `board_side.jpg` / `board_front.jpg`, `meta.json` (person_id, front/side ids,
  `error_performed`, the split `errors` list, which extrinsics were used), `log.txt`, `results.json`,
  `front.mp4` / `side.mp4` (540p pose overlays for checking).
- `_calibration\intrinsic\` (+ `_calibration\extrinsic\` from the optional extrinsic videos; computed once, reused
  while the videos and board settings stay the same), `summary.csv` (one row per person), `batch_log.txt`,
  `batch_status.json`, `_work\<person_id>\` (Pose2Sim working folder with the 2D keypoints; its large videos are
  deleted unless "keep intermediate videos").

Mode 2 (one extrinsic recording for everybody, squat videos without a board): choose "One calibration from
extrinsic_front/side" (`--extrinsics shared`) and usually "Two claps" (`--squat-claps 2`).

Without the page: `.\two_view\start_batch.ps1 -Dataset D:\squat_dataset -ExtraArgs "--intr-square 41 --ext-square 41"`,
or `C:\Pose2SimVenv\Scripts\python.exe two_view\batch_preprocess.py --input D:\squat_dataset --check` to only list
what was found. All options: `batch_preprocess.py --help`.

## 3D mesh + saliency viewer

MotionBERT's SMPL body mesh - the same model and steps as the single-view pipeline's mesh stage - made from the 2D
keypoints Pose2Sim detected during preprocessing, played back in 3D next to the front and side videos, with each
labelled error's **painted saliency template** overlaid on the moving body.

1. **Run** `two_view\start_mesh_viewer.ps1` &rarr; http://127.0.0.1:5070. It uses `C:\Proj1Venv` (torch + smplx), the
   MotionBERT mesh checkpoint and the SMPL files in `MotionBERT\data\mesh\` (licence-gated, not in git &mdash; copy
   them to any other PC).
2. **Load folder**: `<dataset>\preprocessed_data` (batch output) or `outputs\two_view` (website runs). Click a
   recording; if it has no mesh yet, **Build 3D mesh** (~5-15 s on a CPU). Whole dataset at once:
   `C:\Proj1Venv\Scripts\python.exe two_view\p2s_mesh.py <dataset>\preprocessed_data` (skips persons already done).
   The 2D keypoints come from the Pose2Sim working folder (`_work\<person>\trial\pose-associated\`), which the
   batch keeps.
3. **Which camera**: the **side** camera by default. Seen from the front, squat depth is along the camera's line of
   sight and a single-camera model underestimates it badly (a deep squat came out as a slight knee bend); from the
   side it matches the video. **Rebuild…** on the page switches to the front camera or to an SMPL fit to the
   triangulated 3D points (`--method fit`).
4. **Templates**: open the **fault painter** link on the page (`mesh_saliency_painter.html`), load
   `smpl_neutral_tpose.obj` (download link on the page), paint where the fault shows on the body, give it the
   same name as the error in the CSV (`knee_valgus`, `heels_up`, ... - case, spaces and punctuation don't matter),
   **Save**, then **Upload templates** with the downloaded `*_saliency.json`. Templates live in
   `two_view\saliency_templates\` and apply to every recording labelled with that error.
5. **Watch**: the chips under the viewer are the recording's errors (dashed = no template yet); pick one or
   "All errors". **When to show it**: follow squat depth (strongest at the bottom of each rep, default), the
   whole of every rep, or all the time. The timeline shows the reps (shaded), squat depth (grey) and the overlay
   strength (orange); click it or a rep chip to jump. Drag to orbit, wheel to zoom, Front/Side/Back/Top buttons,
   **Record** saves the 3D view as a .webm video.

`<recording>\mesh\`: `verts.bin` (int16 mm vertices, (T, 6890, 3)), `mesh.json` (frames, fps, method, camera,
facing), `log.txt`. MotionBERT's mesh is root-relative; it is turned Y-up and stood on the floor frame by frame
(feet planted). The fit method also writes `smpl_params.npz` and fills `com_mesh` / `com_final` in
`advancedangles.csv`.

The saliency is a **painted prior** (where on the body each fault shows), timed by the squat - not a model's
explanation. A trained classifier's per-frame saliency (Grad-CAM scaffolding in `keypoint_pipeline.py`) can be
fed into the same overlay later.

## Squat error model

The per-rep error classifier (CNN-LSTM with attention pooling over the frames, chosen out of six neural models and
three classical baselines by 5-fold cross-validation grouped by person) is built in `research/` from this folder's
batch output (`preprocessed_data/`); see `research/README.md`. The final product runs it with
`python -m squat_research.predict <recording folder> --model <export folder>` (`PYTHONPATH=research`) and paints
its attention weights &alpha; on the 3D body.

## Final product

The end-user web app (front + side video &rarr; errors per rep, 3D body with saliency, synced videos) lives in
`final_product/` (see `final_product/README.md`); it runs this folder's `run_pose2sim.py` and `p2s_mesh.py`.
The model hand-off (architecture, build instructions, sample data) is in `ai_lead_handoff/`.

## Run from the command line

`run_pose2sim.py` (in `C:\Pose2SimVenv`); the examples are in its docstring. One recording, own board before the clap:

```
C:\Pose2SimVenv\Scripts\python.exe two_view/run_pose2sim.py --name subj01 --side side.mp4 --front front.mp4 \
    --intr-from <intrinsics calibration.toml> --intr-square 41 --ext-square 41
```

`--phase prepare | pose | finish` runs it in parts (the ZeroGPU Space runs only the pose step on the GPU).

## Files

| File | Purpose |
|---|---|
| `sync_clap.py` | Finds the clap in a video's audio (ffmpeg via imageio-ffmpeg) |
| `rep_fusion.py` | Detects reps on knee flexion and builds per-rep features |
| `run_pose2sim.py` | Pose2Sim 3D mode end to end (full analysis, extrinsic or intrinsic calibration only; `--phase`) |
| `p2s_sync.py` | Side/front sync by lining up all sharp sounds (robust when each mic hears a different sound loudest) |
| `p2s_extrinsics.py` | Board search (coarse + frame by frame), corner ordering, extrinsic solve |
| `p2s_biomech.py` | Pose2Sim .trc &rarr; `angles.csv` + `advancedangles.csv` |
| `batch_preprocess.py` | Whole dataset: intrinsics once, then `run_pose2sim.py` per CSV row (own-board or shared extrinsics); resumable |
| `batch_app.py`, `web/batch.html` | Batch page (http://127.0.0.1:5050) |
| `start_batch.ps1` | Sets up the Pose2Sim environment (GPU if present) and opens the batch page |
| `requirements_pose2sim.txt` | Exact package versions of the Pose2Sim environment |
| `p2s_mesh.py` | MotionBERT SMPL mesh from a camera's 2D keypoints (or an SMPL fit to `points3d.trc`); one recording or a folder |
| `mesh_app.py`, `web/mesh.html`, `start_mesh_viewer.ps1` | 3D mesh + saliency viewer (http://127.0.0.1:5070) |
| `saliency_templates/` | Painted per-error saliency maps (`<error>_saliency.json`) |

## Limitations

- Needs a clear clap in both videos (a confidence below 8 triggers a warning).
- Reps are found from knee flexion; slow or partial reps below 20° prominence are ignored.
- Fusion is at feature level; there is no shared 3D coordinate system between views.
