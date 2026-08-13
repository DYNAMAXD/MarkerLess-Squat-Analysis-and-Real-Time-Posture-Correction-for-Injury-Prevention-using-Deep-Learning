"""
app.py
======
Single-file pipeline:  video  ->  YOLOv11 (person box)  ->  AlphaPose (2D, Halpe-26)
                                 ->  MotionBERT (3D lift, Human3.6M-17)
                                 ->  keypoints_2d.csv, keypoints_3d.csv, overlay.mp4

Assumes the following are ALREADY present next to this file (as you described):

    ./yolo11s.pt
    ./AlphaPose/                                    (full repo, already built/installed)
    ./AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth
    ./MotionBERT/                                   (full repo)
    ./MotionBERT/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin

Nothing is downloaded and nothing is re-cloned. Mesh recovery (MotionBERT
infer_wild_mesh.py + the FT_MB_release_MB_ft_pw3d checkpoint you already
uploaded) is intentionally NOT wired in yet -- you said you'll ask for that
next, so this version only takes the pipeline as far as 3D joint keypoints.

Design notes (why it's built this way)
---------------------------------------
* AlphaPose is driven through its OWN, official `scripts/demo_inference.py`
  as a subprocess, using its built-in `--detfile` mode. That mode makes
  AlphaPose skip its internal detector entirely and read bounding boxes
  from a JSON file we hand it -- this is how YOLOv11 is used as the
  detector instead of AlphaPose's default YOLOv3-SPP, without touching a
  single line of AlphaPose's source. Running it as a subprocess (rather
  than importing alphapose.* in-process) also avoids fighting AlphaPose's
  own torch.multiprocessing setup and keeps its heavy import graph out of
  the Gradio process.
* MotionBERT IS imported in-process (`lib.model.DSTformer`, `lib.utils.*`)
  because in-the-wild 3D lifting is just a small forward pass over the
  2D keypoints -- there's no benefit to shelling out, and infer_wild.py's
  exact math (halpe26->h36m mapping, normalization, flip test-time
  augmentation, root-relative fix-up) is reproduced verbatim below so the
  numbers match what infer_wild.py would have produced.
* CSVs are written in pixel coordinates (2D: original image pixels,
  3D: MotionBERT's output rescaled back to the pixel frame, same
  convention as infer_wild.py's `--pixel` mode) so they're directly
  usable without decoding a separate normalization step.

Usage
-----
As a Gradio app (Hugging Face Spaces entry point):
    python app.py

As a one-off CLI run (useful to sanity check before wiring into Spaces):
    python app.py --video /path/to/clip.mp4 --outdir /path/to/out
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 0. PATHS -- edit ONLY if your folder layout differs from what's listed above
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

ALPHAPOSE_DIR   = os.path.join(REPO_ROOT, "AlphaPose")
MOTIONBERT_DIR  = os.path.join(REPO_ROOT, "MotionBERT")
YOLO_WEIGHTS    = os.path.join(REPO_ROOT, "yolo11s.pt")

ALPHAPOSE_CFG   = os.path.join(ALPHAPOSE_DIR, "configs", "halpe_26", "resnet",
                                "256x192_res50_lr1e-3_1x.yaml")
ALPHAPOSE_CKPT  = os.path.join(ALPHAPOSE_DIR, "pretrained_models",
                                "halpe26_fast_res50_256x192.pth")

MOTIONBERT_CFG  = os.path.join(MOTIONBERT_DIR, "configs", "pose3d",
                                "MB_ft_h36m_global_lite.yaml")
MOTIONBERT_CKPT = os.path.join(MOTIONBERT_DIR, "checkpoint", "pose3d",
                                "FT_MB_lite_MB_ft_h36m_global_lite", "best_epoch.bin")

REQUIRED_PATHS = {
    "AlphaPose repo": ALPHAPOSE_DIR,
    "MotionBERT repo": MOTIONBERT_DIR,
    "YOLOv11 weights (yolo11s.pt)": YOLO_WEIGHTS,
    "AlphaPose halpe26 config": ALPHAPOSE_CFG,
    "AlphaPose halpe26 checkpoint": ALPHAPOSE_CKPT,
    "MotionBERT pose3d config": MOTIONBERT_CFG,
    "MotionBERT pose3d checkpoint": MOTIONBERT_CKPT,
}


def check_paths():
    missing = [f"  - {name}: {path}" for name, path in REQUIRED_PATHS.items()
               if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing expected file(s)/folder(s):\n" + "\n".join(missing) +
            "\n\nEdit the PATHS section at the top of app.py if your layout differs."
        )


# MotionBERT is imported in-process, so its `lib` package needs to be on sys.path.
# (AlphaPose is only ever launched as a subprocess, so it does NOT need this --
#  it already exposes `alphapose`/`detector`/`trackers` globally from its own
#  `setup.py develop` build.)
sys.path.insert(0, MOTIONBERT_DIR)


# ---------------------------------------------------------------------------
# 1. Joint tables
# ---------------------------------------------------------------------------

# Halpe-26 body layout, exactly as produced by the halpe26_fast_res50_256x192
# AlphaPose model (see MotionBERT/lib/data/dataset_wild.py::halpe2h36m docstring).
HALPE26_JOINTS = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "Hip",
    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
]

# Edges used only for drawing the overlay video (not biomechanically load-bearing).
HALPE26_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),                # face
    (17, 18), (18, 19),                            # head - neck - hip (spine line)
    (18, 5), (18, 6),                               # neck -> shoulders
    (5, 7), (7, 9),                                 # left arm
    (6, 8), (8, 10),                                # right arm
    (19, 11), (19, 12),                             # hip -> l/r hip
    (11, 13), (13, 15),                             # left leg
    (12, 14), (14, 16),                             # right leg
    (15, 20), (15, 22), (15, 24),                   # left ankle -> left foot
    (16, 21), (16, 23), (16, 25),                   # right ankle -> right foot
]

# Human3.6M 17-joint order that halpe2h36m() below produces (and that
# MotionBERT was trained on).
H36M_JOINTS = [
    "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Nose", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]


def halpe2h36m(x):
    """
    Verbatim port of MotionBERT/lib/data/dataset_wild.py::halpe2h36m
    Input:  x, shape (T, 26, C)  -- Halpe-26 keypoints
    Output: shape (T, 17, C)     -- Human3.6M-17 keypoints
    """
    T, V, C = x.shape
    y = np.zeros([T, 17, C], dtype=x.dtype)
    y[:, 0, :] = x[:, 19, :]
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 7, :] = (x[:, 18, :] + x[:, 19, :]) * 0.5
    y[:, 8, :] = x[:, 18, :]
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = x[:, 17, :]
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]
    return y


def flip_data(data):
    """Verbatim port of MotionBERT/lib/utils/utils_data.py::flip_data"""
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped = data.clone()
    flipped[..., 0] *= -1
    flipped[..., left_joints + right_joints, :] = flipped[..., right_joints + left_joints, :]
    return flipped


# ---------------------------------------------------------------------------
# 2. Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fp = os.path.join(out_dir, f"{idx:06d}.jpg")
        cv2.imwrite(fp, frame)
        frame_paths.append(fp)
        idx += 1
    cap.release()
    return frame_paths, fps, width, height


# ---------------------------------------------------------------------------
# 3. YOLOv11 person detection -> AlphaPose detfile JSON
# ---------------------------------------------------------------------------
_yolo_model = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_WEIGHTS)
    return _yolo_model


def run_yolo_detection(frame_paths, conf_thres=0.25):
    """
    Runs YOLOv11 person detection (COCO class 0) on every extracted frame and
    keeps the single highest-confidence person box per frame (single-subject
    pipeline, matches the squat/exercise-form use case). Returns a list of
    detections in the exact schema AlphaPose's FileDetectionLoader expects:
        {"image_id": <abs frame path>, "category_id": 1,
         "bbox": [x, y, w, h], "score": float, "idx": 0}
    Frames with no detected person are simply omitted.
    """
    model = get_yolo()
    detections = []
    results = model.predict(source=frame_paths, classes=[0], conf=conf_thres,
                             verbose=False, stream=True)
    for fp, r in zip(frame_paths, results):
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().tolist()
        score = float(confs[best])
        detections.append({
            "image_id": fp,
            "category_id": 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": score,
            "idx": 0,
        })
    return detections


# ---------------------------------------------------------------------------
# 4. AlphaPose (subprocess, official demo_inference.py, --detfile mode)
# ---------------------------------------------------------------------------
def run_alphapose(detections, out_dir):
    if not detections:
        raise RuntimeError("No person detections to hand to AlphaPose.")

    detfile_path = os.path.join(out_dir, "detections.json")
    with open(detfile_path, "w") as f:
        json.dump(detections, f)

    cmd = [
        sys.executable,
        os.path.join(ALPHAPOSE_DIR, "scripts", "demo_inference.py"),
        "--cfg", ALPHAPOSE_CFG,
        "--checkpoint", ALPHAPOSE_CKPT,
        "--detfile", detfile_path,
        "--outdir", out_dir,
        "--sp",
        "--gpus", "0" if torch.cuda.is_available() else "-1",
    ]
    proc = subprocess.run(cmd, cwd=ALPHAPOSE_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "AlphaPose subprocess failed.\n\n--- stdout (tail) ---\n"
            f"{proc.stdout[-3000:]}\n\n--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )

    results_json = os.path.join(out_dir, "alphapose-results.json")
    if not os.path.exists(results_json):
        raise RuntimeError(
            "AlphaPose finished but alphapose-results.json was not produced.\n"
            f"stdout tail:\n{proc.stdout[-2000:]}"
        )
    with open(results_json) as f:
        return json.load(f)


def build_2d_table(ap_results, fps):
    """
    ap_results: parsed alphapose-results.json (list of per-detection dicts,
    'image_id' is the basename we gave each frame, e.g. '000123.jpg').
    Returns: (dataframe, sorted list of frame indices, {frame_idx: (26,3) array})
    """
    by_frame = {}
    for item in ap_results:
        frame_idx = int(os.path.splitext(item["image_id"])[0])
        kpts = np.array(item["keypoints"], dtype=np.float32).reshape(-1, 3)  # (26,3)
        by_frame[frame_idx] = kpts

    frame_indices = sorted(by_frame.keys())
    rows = []
    for fi in frame_indices:
        kpts = by_frame[fi]
        row = {"frame": fi, "time_sec": fi / fps}
        for j, name in enumerate(HALPE26_JOINTS):
            row[f"{name}_x"] = float(kpts[j, 0])
            row[f"{name}_y"] = float(kpts[j, 1])
            row[f"{name}_conf"] = float(kpts[j, 2])
        rows.append(row)
    return pd.DataFrame(rows), frame_indices, by_frame


# ---------------------------------------------------------------------------
# 5. MotionBERT: ordered 2D (halpe26) -> 3D (h36m17), in pixel coordinates
# ---------------------------------------------------------------------------
_motionbert_model = None
_motionbert_args = None


def _load_state_dict_flexible(model, state_dict):
    """
    The official checkpoint was saved from a DataParallel-wrapped model
    (keys prefixed 'module.'). infer_wild.py only wraps in DataParallel when
    CUDA is available, which means straight `strict=True` loading breaks on
    CPU-only machines (a known MotionBERT CPU-inference issue). This adds/
    strips the 'module.' prefix as needed so it loads correctly either way.
    """
    model_keys = list(model.state_dict().keys())
    model_has_module = any(k.startswith("module.") for k in model_keys)
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
    if ckpt_has_module and not model_has_module:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    elif model_has_module and not ckpt_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def load_motionbert_model():
    global _motionbert_model, _motionbert_args
    if _motionbert_model is not None:
        return _motionbert_model, _motionbert_args

    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone

    args = get_config(MOTIONBERT_CFG)
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone).cuda()

    ckpt = torch.load(MOTIONBERT_CKPT, map_location=lambda storage, loc: storage)
    _load_state_dict_flexible(model_backbone, ckpt["model_pos"])
    model_backbone.eval()

    _motionbert_model, _motionbert_args = model_backbone, args
    return model_backbone, args


def lift_to_3d(model_backbone, args, ordered_kpts_26, vid_size):
    """
    ordered_kpts_26: (T, 26, 3) float array, strictly in ascending-frame order
                      (only frames where a person was detected).
    vid_size: (width, height) of the ORIGINAL video.
    Returns: (T, 17, 3) float array, x/y in original pixel coordinates.

    This mirrors infer_wild.py's `--pixel` code path exactly:
      1. halpe26 -> h36m17
      2. center + scale by vid_size (the SAME normalization the model was
         run with -- no additional crop_scale, matching --pixel mode)
      3. per-clip (<=243 frames) forward pass with flip test-time augmentation
      4. root-relative z fix-up
      5. inverse of step 2, to land back in pixel coordinates
    """
    motion_h36m = halpe2h36m(ordered_kpts_26.astype(np.float32))  # (T,17,3)
    w, h = vid_size
    scale = min(w, h) / 2.0
    motion_h36m[:, :, :2] = motion_h36m[:, :, :2] - np.array([w, h], dtype=np.float32) / 2.0
    motion_h36m[:, :, :2] = motion_h36m[:, :, :2] / scale

    clip_len = args.maxlen
    T = motion_h36m.shape[0]
    device = next(model_backbone.parameters()).device

    all_preds = []
    with torch.no_grad():
        for st in range(0, T, clip_len):
            end = min(st + clip_len, T)
            clip = torch.from_numpy(motion_h36m[st:end]).float().unsqueeze(0).to(device)  # (1,t,17,3)
            if args.no_conf:
                clip = clip[..., :2]
            if args.flip:
                pred1 = model_backbone(clip)
                pred2 = flip_data(model_backbone(flip_data(clip)))
                pred = (pred1 + pred2) / 2.0
            else:
                pred = model_backbone(clip)
            if args.rootrel:
                pred[:, :, 0, :] = 0
            else:
                pred[:, 0, 0, 2] = 0
            all_preds.append(pred.squeeze(0).cpu().numpy())
    motion_3d = np.concatenate(all_preds, axis=0)  # (T,17,3), normalized

    motion_3d = motion_3d * (min(w, h) / 2.0)
    motion_3d[:, :, :2] = motion_3d[:, :, :2] + np.array([w, h], dtype=np.float32) / 2.0
    return motion_3d


def build_3d_table(motion_3d, frame_indices, fps):
    rows = []
    for i, fi in enumerate(frame_indices):
        row = {"frame": fi, "time_sec": fi / fps}
        for j, name in enumerate(H36M_JOINTS):
            row[f"{name}_x"] = float(motion_3d[i, j, 0])
            row[f"{name}_y"] = float(motion_3d[i, j, 1])
            row[f"{name}_z"] = float(motion_3d[i, j, 2])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Overlay video (2D Halpe-26 skeleton drawn on the original frames)
# ---------------------------------------------------------------------------
def render_overlay_video(video_path, by_frame_kpts, fps, width, height, out_path,
                          conf_thres=0.3):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kpts = by_frame_kpts.get(idx)
        if kpts is not None:
            for a, b in HALPE26_SKELETON:
                xa, ya, ca = kpts[a]
                xb, yb, cb = kpts[b]
                if ca > conf_thres and cb > conf_thres:
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)
            for x, y, c in kpts:
                if c > conf_thres:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(video_path, det_conf=0.25, work_dir=None, progress_cb=None):
    """
    progress_cb(fraction: float, desc: str) -> None   (optional, e.g. gr.Progress)
    Returns: (path_to_2d_csv, path_to_3d_csv, path_to_overlay_mp4)
    """
    def _progress(frac, desc):
        if progress_cb is not None:
            progress_cb(frac, desc=desc)
        print(f"[{frac*100:5.1f}%] {desc}")

    check_paths()

    own_tmp = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="pose3d_")
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    _progress(0.05, "Extracting frames")
    frame_paths, fps, width, height = extract_frames(video_path, frames_dir)
    if not frame_paths:
        raise RuntimeError("Could not read any frames from the video.")

    _progress(0.20, "Detecting person with YOLOv11")
    detections = run_yolo_detection(frame_paths, conf_thres=det_conf)
    if not detections:
        raise RuntimeError(
            "YOLOv11 did not detect a person in any frame. "
            "Try lowering the detection-confidence threshold."
        )

    _progress(0.40, "Running AlphaPose (2D keypoints, Halpe-26)")
    ap_results = run_alphapose(detections, work_dir)

    _progress(0.60, "Building 2D CSV")
    df2d, frame_indices, by_frame = build_2d_table(ap_results, fps)

    _progress(0.70, "Lifting to 3D with MotionBERT")
    ordered = np.stack([by_frame[fi] for fi in frame_indices], axis=0)  # (T,26,3)
    model_backbone, mb_args = load_motionbert_model()
    motion_3d = lift_to_3d(model_backbone, mb_args, ordered, (width, height))
    df3d = build_3d_table(motion_3d, frame_indices, fps)

    _progress(0.85, "Rendering overlay video")
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_2d_csv = os.path.join(work_dir, f"{base}_keypoints_2d.csv")
    out_3d_csv = os.path.join(work_dir, f"{base}_keypoints_3d.csv")
    out_overlay = os.path.join(work_dir, f"{base}_overlay.mp4")

    df2d.to_csv(out_2d_csv, index=False)
    df3d.to_csv(out_3d_csv, index=False)
    render_overlay_video(video_path, by_frame, fps, width, height, out_overlay)

    _progress(1.0, "Done")

    if own_tmp:
        # frames/ dir and detections.json/alphapose-results.json stay in work_dir
        # for debugging; only the frames themselves are large, so clean those up.
        try:
            for fp in frame_paths:
                os.remove(fp)
        except OSError:
            pass

    return out_2d_csv, out_3d_csv, out_overlay


# ---------------------------------------------------------------------------
# 8. Gradio app
# ---------------------------------------------------------------------------
def build_demo():
    import gradio as gr

    def _run(video_path, det_conf, progress=gr.Progress()):
        if video_path is None:
            raise gr.Error("Please upload a video first.")
        try:
            out_dir = tempfile.mkdtemp(prefix="pose3d_")
            return run_pipeline(video_path, det_conf=det_conf, work_dir=out_dir,
                                 progress_cb=progress)
        except Exception as e:
            raise gr.Error(str(e))

    with gr.Blocks(title="AlphaPose + MotionBERT: 2D/3D keypoints") as demo:
        gr.Markdown(
            "## Video &rarr; 2D keypoints (AlphaPose, Halpe-26) "
            "&rarr; 3D keypoints (MotionBERT, Human3.6M-17)\n"
            "Detector: YOLOv11 (`yolo11s.pt`). Outputs two CSVs (2D pixel "
            "keypoints, 3D lifted keypoints) and a video with the 2D "
            "skeleton overlaid."
        )
        video_in = gr.Video(label="Input video")
        det_conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05,
                              label="YOLOv11 person-detection confidence")
        run_btn = gr.Button("Run", variant="primary")
        with gr.Row():
            out_2d = gr.File(label="2D keypoints CSV (Halpe-26, pixel coords)")
            out_3d = gr.File(label="3D keypoints CSV (H36M-17)")
        out_video = gr.Video(label="Overlay video")

        run_btn.click(fn=_run, inputs=[video_in, det_conf],
                       outputs=[out_2d, out_3d, out_video])

    return demo


# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                         help="If given, run once from the CLI instead of launching Gradio.")
    parser.add_argument("--outdir", type=str, default="./pose3d_out")
    parser.add_argument("--conf", type=float, default=0.25)
    cli_args = parser.parse_args()

    if cli_args.video:
        check_paths()
        os.makedirs(cli_args.outdir, exist_ok=True)
        csv2d, csv3d, overlay = run_pipeline(cli_args.video, det_conf=cli_args.conf,
                                              work_dir=cli_args.outdir)
        print("\n2D CSV   :", csv2d)
        print("3D CSV   :", csv3d)
        print("Overlay  :", overlay)
    else:
        build_demo().queue().launch(server_name="0.0.0.0")
