"""
keypoint_pipeline.py

Single-file pipeline:
  Video -> YOLOv11 (person detection) -> AlphaPose (halpe26_fast_res50_256x192, 2D kpts)
        -> MotionBERT (FT_MB_lite_MB_ft_h36m_global_lite, 3D lift)
        -> CSV (frame, joint, x, y, z, score) + overlay video (2D keypoints drawn on frames)

Assumes this repo layout (already uploaded, NOT re-downloaded here):

  <space_root>/
    AlphaPose/                     (MVIG-SJTU/AlphaPose source)
      pretrained_models/halpe26_fast_res50_256x192.pth
    MotionBERT/                    (Walter0807/MotionBERT source)
      checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
      checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin   (not used yet, mesh stage later)
    yolo11s.pt

Every stage logs through the module-level `LOGGER`, which app.py taps into
to stream a live debug console in the UI (since there is no terminal access
in the HF Space).
"""

import os
import sys
import csv
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import cv2
import torch

# --------------------------------------------------------------------------
# Paths — adjust ROOT if your Space's working dir differs. Everything else
# is derived from ROOT so you don't have to touch multiple files.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ALPHAPOSE_DIR = ROOT / "AlphaPose"
MOTIONBERT_DIR = ROOT / "MotionBERT"
YOLO_WEIGHTS = ROOT / "yolo11s.pt"
ALPHAPOSE_CKPT = ALPHAPOSE_DIR / "pretrained_models" / "halpe26_fast_res50_256x192.pth"
MOTIONBERT_POSE3D_CKPT = (
    MOTIONBERT_DIR / "checkpoint" / "pose3d" / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Logging — captured by app.py's DebugLogHandler and shown live in the UI.
# --------------------------------------------------------------------------
LOGGER = logging.getLogger("kpt_pipeline")
LOGGER.setLevel(logging.DEBUG)
if not LOGGER.handlers:
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(_sh)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Halpe26 joint names, in the order AlphaPose's halpe26 config outputs them.
HALPE26_JOINTS = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe",
    "LHeel", "RHeel",
]

# Halpe26 skeleton edges (index pairs) for drawing the overlay.
HALPE26_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 18), (18, 5), (18, 6), (18, 19),
    (19, 11), (19, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 20), (15, 24), (20, 22), (16, 21), (16, 25), (21, 23),
]

# H36M 17-joint order that MotionBERT's pretrained pose3d model expects.
H36M17_JOINTS = [
    "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck/Nose", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]

# Mapping from Halpe26 index -> H36M17 index (approximate correspondences;
# H36M has no separate nose so we reuse Head there).
HALPE26_TO_H36M17 = {
    19: 0,   # Hip -> Hip
    12: 1,   # RHip
    14: 2,   # RKnee
    16: 3,   # RAnkle
    11: 4,   # LHip
    13: 5,   # LKnee
    15: 6,   # LAnkle
    18: 8,   # Neck -> Thorax
    17: 10,  # Head -> Head
    5: 11,   # LShoulder
    7: 12,   # LElbow
    9: 13,   # LWrist
    6: 14,   # RShoulder
    8: 15,   # RElbow
    10: 16,  # RWrist
}


def halpe26_to_h36m17(kpts_2d: np.ndarray) -> np.ndarray:
    """kpts_2d: (26, 3) [x, y, score] -> (17, 3) H36M order.
    Spine (7) and Neck/Nose (9) are interpolated since Halpe26 has no direct match."""
    out = np.zeros((17, 3), dtype=np.float32)
    for h_idx, m_idx in HALPE26_TO_H36M17.items():
        out[m_idx] = kpts_2d[h_idx]
    out[7] = (out[0] + out[8]) / 2.0        # Spine ~= mid(Hip, Thorax)
    out[9] = (out[8] + out[10]) / 2.0       # Neck/Nose ~= mid(Thorax, Head)
    return out


# ==========================================================================
# Stage 1: YOLOv11 person detection
# ==========================================================================
def load_yolo():
    LOGGER.info(f"Loading YOLOv11 weights from {YOLO_WEIGHTS}")
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS}")
    from ultralytics import YOLO
    model = YOLO(str(YOLO_WEIGHTS))
    LOGGER.info("YOLOv11 loaded.")
    return model


def detect_persons(yolo_model, frame_bgr, conf=0.5):
    """Returns list of [x1, y1, x2, y2, conf] boxes for class 'person' (COCO id 0)."""
    results = yolo_model.predict(source=frame_bgr, conf=conf, classes=[0], verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            c = float(b.conf[0].cpu().numpy())
            boxes.append(xyxy + [c])
    return boxes


# ==========================================================================
# Stage 2: AlphaPose 2D keypoints (halpe26, fast_res50_256x192)
# ==========================================================================
def load_alphapose():
    LOGGER.info(f"Loading AlphaPose (halpe26_fast_res50_256x192) from {ALPHAPOSE_CKPT}")
    if not ALPHAPOSE_CKPT.exists():
        raise FileNotFoundError(f"AlphaPose checkpoint not found at {ALPHAPOSE_CKPT}")
    if str(ALPHAPOSE_DIR) not in sys.path:
        sys.path.insert(0, str(ALPHAPOSE_DIR))

    from alphapose.models import builder
    from alphapose.utils.config import update_config

    # This cfg matches AlphaPose's configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml
    cfg_path = ALPHAPOSE_DIR / "configs" / "halpe_26" / "resnet" / "256x192_res50_lr1e-3_1x.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Expected AlphaPose config at {cfg_path}. If your AlphaPose folder layout "
            "differs, update cfg_path in load_alphapose()."
        )
    cfg = update_config(str(cfg_path))

    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    state_dict = torch.load(str(ALPHAPOSE_CKPT), map_location=DEVICE)
    pose_model.load_state_dict(state_dict)
    pose_model.to(DEVICE)
    pose_model.eval()
    LOGGER.info("AlphaPose model loaded.")
    return pose_model, cfg


def _crop_and_normalize(frame_bgr, box, input_size=(256, 192)):
    x1, y1, x2, y2, _ = box
    h, w = frame_bgr.shape[:2]
    # pad box a bit
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - 0.1 * bw)
    y1 = max(0, y1 - 0.1 * bh)
    x2 = min(w, x2 + 0.1 * bw)
    y2 = min(h, y2 + 0.1 * bh)
    crop = frame_bgr[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return None, None
    resized = cv2.resize(crop, (input_size[1], input_size[0]))
    inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = (inp - np.array([0.406, 0.457, 0.480])) / np.array([1.0, 1.0, 1.0])
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    box_meta = (x1, y1, x2 - x1, y2 - y1)
    return inp, box_meta


def run_alphapose_on_frame(pose_model, frame_bgr, boxes):
    """Returns list of (26,3) arrays [x,y,score] in original-frame coordinates,
    one per detected person box."""
    outputs = []
    for box in boxes:
        inp, box_meta = _crop_and_normalize(frame_bgr, box)
        if inp is None:
            continue
        with torch.no_grad():
            heatmaps = pose_model(inp)  # (1, 26, H, W)
        hm = heatmaps[0].cpu().numpy()
        n_joints, hm_h, hm_w = hm.shape
        x1, y1, bw, bh = box_meta
        kpts = np.zeros((n_joints, 3), dtype=np.float32)
        for j in range(n_joints):
            idx = np.unravel_index(np.argmax(hm[j]), hm[j].shape)
            score = float(hm[j][idx])
            py, px = idx
            kpts[j, 0] = x1 + (px / hm_w) * bw
            kpts[j, 1] = y1 + (py / hm_h) * bh
            kpts[j, 2] = score
        outputs.append(kpts)
    return outputs


# ==========================================================================
# Stage 3: MotionBERT 3D lift (H36M lite checkpoint)
# ==========================================================================
def load_motionbert():
    LOGGER.info(f"Loading MotionBERT pose3d checkpoint from {MOTIONBERT_POSE3D_CKPT}")
    if not MOTIONBERT_POSE3D_CKPT.exists():
        raise FileNotFoundError(f"MotionBERT checkpoint not found at {MOTIONBERT_POSE3D_CKPT}")
    if str(MOTIONBERT_DIR) not in sys.path:
        sys.path.insert(0, str(MOTIONBERT_DIR))

    from lib.utils.tools import get_config
    from lib.model.DSTformer import DSTformer

    cfg_path = MOTIONBERT_DIR / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Expected MotionBERT config at {cfg_path}. Update cfg_path in load_motionbert() "
            "if your config file name/location differs."
        )
    args = get_config(str(cfg_path))

    model = DSTformer(
        dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep,
        depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        norm_layer=torch.nn.LayerNorm, maxlen=args.maxlen,
        num_joints=args.num_joints,
    )
    ckpt = torch.load(str(MOTIONBERT_POSE3D_CKPT), map_location=DEVICE)
    state_dict = ckpt.get("model_pos", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    LOGGER.info("MotionBERT loaded.")
    return model, args


def lift_sequence_to_3d(mb_model, seq_2d, clip_len=243):
    """seq_2d: (T, 17, 3) normalized [x,y,score] in [-1,1]-ish range.
    Returns (T, 17, 3) predicted 3D coords. Processed in clip_len chunks
    (MotionBERT's default receptive field) with simple padding for the tail."""
    T = seq_2d.shape[0]
    preds = np.zeros((T, 17, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, T, clip_len):
            end = min(start + clip_len, T)
            clip = seq_2d[start:end]
            pad = clip_len - clip.shape[0]
            if pad > 0:
                clip = np.concatenate([clip, np.repeat(clip[-1:], pad, axis=0)], axis=0)
            x = torch.from_numpy(clip).float().unsqueeze(0).to(DEVICE)  # (1, clip_len, 17, 3)
            out = mb_model(x)  # (1, clip_len, 17, 3)
            out = out[0].cpu().numpy()[: end - start]
            preds[start:end] = out
    return preds


def normalize_2d_for_motionbert(kpts_2d_h36m, frame_w, frame_h):
    """MotionBERT expects roughly [-1, 1] normalized image-plane coords
    (x,y centered, scaled by half the max frame dimension)."""
    out = kpts_2d_h36m.copy()
    scale = max(frame_w, frame_h) / 2.0
    out[:, 0] = (out[:, 0] - frame_w / 2.0) / scale
    out[:, 1] = (out[:, 1] - frame_h / 2.0) / scale
    return out


# ==========================================================================
# Overlay video drawing (2D keypoints only, for the "keypoints task" pass)
# ==========================================================================
def draw_overlay(frame_bgr, kpts_2d_26):
    frame = frame_bgr.copy()
    for (a, b) in HALPE26_EDGES:
        if kpts_2d_26[a, 2] > 0.05 and kpts_2d_26[b, 2] > 0.05:
            pa = tuple(kpts_2d_26[a, :2].astype(int))
            pb = tuple(kpts_2d_26[b, :2].astype(int))
            cv2.line(frame, pa, pb, (0, 255, 0), 2)
    for j in range(kpts_2d_26.shape[0]):
        if kpts_2d_26[j, 2] > 0.05:
            p = tuple(kpts_2d_26[j, :2].astype(int))
            cv2.circle(frame, p, 3, (0, 0, 255), -1)
    return frame


# ==========================================================================
# Main entry point
# ==========================================================================
def process_video(video_path: str, use_3d: bool = True, det_conf: float = 0.5, progress_cb=None):
    """
    Runs the full pipeline on a video file.

    Returns: (csv_path, overlay_video_path)
    Raises on any hard failure (with full traceback logged via LOGGER).
    """
    video_path = str(video_path)
    stem = Path(video_path).stem
    csv_path = str(OUTPUT_DIR / f"{stem}_keypoints.csv")
    overlay_path = str(OUTPUT_DIR / f"{stem}_overlay.mp4")

    LOGGER.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LOGGER.info(f"Video: {w}x{h} @ {fps:.2f}fps, ~{n_frames} frames")

    yolo_model = load_yolo()
    pose_model, _ = load_alphapose()
    mb_model = None
    if use_3d:
        mb_model, _ = load_motionbert()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    all_frames_2d = []   # per-frame primary-person (26,3)
    frame_idx = 0

    LOGGER.info("Starting frame-by-frame inference (Stage 1+2: YOLOv11 + AlphaPose)...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        try:
            boxes = detect_persons(yolo_model, frame, conf=det_conf)
            if not boxes:
                LOGGER.debug(f"Frame {frame_idx}: no person detected")
                kpts = np.zeros((26, 3), dtype=np.float32)
                all_frames_2d.append(kpts)
                writer.write(frame)
                frame_idx += 1
                if progress_cb:
                    progress_cb(frame_idx, n_frames)
                continue

            # take highest-confidence box as primary person
            boxes.sort(key=lambda b: b[4], reverse=True)
            per_person_kpts = run_alphapose_on_frame(pose_model, frame, boxes[:1])
            kpts = per_person_kpts[0] if per_person_kpts else np.zeros((26, 3), dtype=np.float32)
            all_frames_2d.append(kpts)

            overlay_frame = draw_overlay(frame, kpts)
            writer.write(overlay_frame)

        except Exception:
            LOGGER.error(f"Frame {frame_idx} failed:\n{traceback.format_exc()}")
            all_frames_2d.append(np.zeros((26, 3), dtype=np.float32))
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 15 == 0:
            LOGGER.info(f"Processed {frame_idx}/{n_frames} frames")
        if progress_cb:
            progress_cb(frame_idx, n_frames)

    cap.release()
    writer.release()
    LOGGER.info(f"Overlay video written: {overlay_path}")

    all_frames_2d = np.stack(all_frames_2d, axis=0)  # (T, 26, 3)

    kpts_3d = None
    if use_3d and mb_model is not None:
        LOGGER.info("Stage 3: lifting to 3D with MotionBERT...")
        try:
            h36m_seq = np.stack([halpe26_to_h36m17(f) for f in all_frames_2d], axis=0)  # (T,17,3)
            h36m_seq_norm = np.stack(
                [normalize_2d_for_motionbert(f, w, h) for f in h36m_seq], axis=0
            )
            kpts_3d = lift_sequence_to_3d(mb_model, h36m_seq_norm)  # (T,17,3)
            LOGGER.info("3D lift complete.")
        except Exception:
            LOGGER.error(f"MotionBERT 3D lift failed, continuing with 2D-only CSV:\n{traceback.format_exc()}")
            kpts_3d = None

    LOGGER.info(f"Writing CSV: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        if kpts_3d is not None:
            writer_csv.writerow(["frame", "joint", "x_2d", "y_2d", "score_2d", "x_3d", "y_3d", "z_3d"])
            for t in range(all_frames_2d.shape[0]):
                for j, name in enumerate(HALPE26_JOINTS):
                    x2, y2, s2 = all_frames_2d[t, j]
                    if j in HALPE26_TO_H36M17:
                        m_idx = HALPE26_TO_H36M17[j]
                        x3, y3, z3 = kpts_3d[t, m_idx]
                    else:
                        x3 = y3 = z3 = ""
                    writer_csv.writerow([t, name, x2, y2, s2, x3, y3, z3])
        else:
            writer_csv.writerow(["frame", "joint", "x_2d", "y_2d", "score_2d"])
            for t in range(all_frames_2d.shape[0]):
                for j, name in enumerate(HALPE26_JOINTS):
                    x2, y2, s2 = all_frames_2d[t, j]
                    writer_csv.writerow([t, name, x2, y2, s2])

    LOGGER.info("Pipeline finished successfully.")
    return csv_path, overlay_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--no-3d", action="store_true")
    a = p.parse_args()
    process_video(a.video, use_3d=not a.no_3d)
