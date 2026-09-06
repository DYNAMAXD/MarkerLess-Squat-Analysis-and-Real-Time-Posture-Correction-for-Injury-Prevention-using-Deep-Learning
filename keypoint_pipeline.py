"""
keypoint_pipeline.py

Single-file pipeline:
  Video -> YOLOv11 (person detection) -> AlphaPose (halpe26_fast_res50_256x192, 2D kpts)
        -> MotionBERT (FT_MB_lite_MB_ft_h36m_global_lite, 3D lift)
        -> MotionBERT mesh (FT_MB_release_MB_ft_pw3d, SMPL mesh regression) [optional]
        -> CSV (frame, joint, x, y, z, score) + overlay video (2D keypoints drawn on frames)
           + mesh video (turntable-style SMPL mesh render) [optional]

Assumes this repo layout (already uploaded, NOT re-downloaded here):

  <space_root>/
    AlphaPose/                     (MVIG-SJTU/AlphaPose source)
      pretrained_models/halpe26_fast_res50_256x192.pth
    MotionBERT/                    (Walter0807/MotionBERT source)
      checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin
      checkpoint/mesh/FT_MB_release_MB_ft_pw3d/best_epoch.bin
      data/mesh/                   (SMPL assets -- see check_smpl_assets(), NOT bundled with
                                     MotionBERT; you must obtain these separately, see below)
    yolo11s.pt

SMPL assets required for the mesh stage (put in MotionBERT/data/mesh/):
  - SMPL_NEUTRAL.pkl          (from SMPLify-X / SMPL website -- license-gated, cannot be
                                redistributed here; register at https://smpl.is.tue.mpg.de/
                                download the "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
                                and rename it to SMPL_NEUTRAL.pkl)
  - smpl_mean_params.npz      (from the SPIN repo: https://github.com/nkolot/SPIN)
  - J_regressor_extra.npy     (from the SPIN repo)
  - J_regressor_h36m_correct.npy  (from the SPIN repo / MotionBERT's own data prep docs)
  These are standard files used by almost every SMPL-based mesh-recovery project (SPIN,
  VIBE, MotionBERT, ...) -- if you already have a data/mesh/ or data/smpl/ folder from any
  of those, you can usually reuse it directly.

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
import os

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
MOTIONBERT_MESH_CKPT = (
    MOTIONBERT_DIR / "checkpoint" / "mesh" / "FT_MB_release_MB_ft_pw3d" / "best_epoch.bin"
)
SMPL_DATA_DIR = MOTIONBERT_DIR / "data" / "mesh"
SMPL_REQUIRED_FILES = [
    "SMPL_NEUTRAL.pkl",
    "smpl_mean_params.npz",
    "J_regressor_extra.npy",
    "J_regressor_h36m_correct.npy",
]

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

DEVICE = "cpu"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_device(requested_device="auto"):
    """
    Set the computation device for the pipeline.

    requested_device:
        "auto" -> GPU if CUDA is available, otherwise CPU
        "gpu"  -> GPU if CUDA is available, otherwise CPU
        "cpu"  -> CPU
    """
    global DEVICE

    requested_device = (requested_device or "auto").lower()

    if requested_device == "cpu":
        DEVICE = "cpu"

    elif requested_device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            DEVICE = "cuda"
        else:
            DEVICE = "cpu"
            LOGGER.warning(
                "GPU was requested, but CUDA is not available. Falling back to CPU."
            )

    else:  # auto
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LOGGER.info(
        f"Computation device selected: {DEVICE.upper()}"
    )

    if DEVICE == "cuda":
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
        LOGGER.info(
            f"CUDA version available to PyTorch: {torch.version.cuda}"
        )

    return DEVICE


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

# H36M 17-joint order that MotionBERT's pretrained pose3d/mesh models expect.
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

# SMPL H36M-17 skeleton edges, used only for the fallback skeleton-only mesh
# render if face triangles aren't available for some reason.
H36M17_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9),
    (8, 11), (8, 14), (9, 10), (11, 12), (12, 13), (14, 15), (15, 16),
]


def halpe26_to_h36m17(kpts_2d: np.ndarray) -> np.ndarray:
    """kpts_2d: (26, 3) [x, y, score] -> (17, 3) H36M order.
    Spine (7) and Neck/Nose (9) are interpolated since Halpe26 has no direct match."""
    out = np.zeros((17, 3), dtype=np.float32)
    for h_idx, m_idx in HALPE26_TO_H36M17.items():
        out[m_idx] = kpts_2d[h_idx]
    out[7] = (out[0] + out[8]) / 2.0        # Spine ~= mid(Hip, Thorax)
    out[9] = (out[8] + out[10]) / 2.0       # Neck/Nose ~= mid(Thorax, Head)
    return out

# H36M-17 joint indices, for readability when computing angles below.
(H36M_HIP, H36M_RHIP, H36M_RKNEE, H36M_RANKLE, H36M_LHIP, H36M_LKNEE, H36M_LANKLE,
 H36M_SPINE, H36M_THORAX, H36M_NECK, H36M_HEAD,
 H36M_LSHOULDER, H36M_LELBOW, H36M_LWRIST, H36M_RSHOULDER, H36M_RELBOW, H36M_RWRIST) = range(17)

EXTRACTED_ANGLE_COLUMNS = [
    "knee_valgus_L", "knee_valgus_R",
    "head_forward_angle",
    "squat_depth_knee_deg",
    "sagittal_flexion_trunk",
    "hip_angle_L", "hip_angle_R",
    "hip_flexion_L", "hip_flexion_R",
    "knee_flexion_L", "knee_flexion_R",
    "ankle_dorsiflexion_proxy_L", "ankle_dorsiflexion_proxy_R",
    "lumbar_curvature_proxy",
]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _angle_between(v1, v2):
    """Unsigned angle in degrees between two 3D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _signed_angle_in_plane(v_from, v_to, x_axis, y_axis):
    """Signed angle in degrees from v_from to v_to, both measured within the
    plane spanned by orthonormal (x_axis, y_axis), as atan2(y_comp, x_comp).
    Positive = rotation from x_axis toward y_axis."""
    def _ang(v):
        return np.degrees(np.arctan2(np.dot(v, y_axis), np.dot(v, x_axis)))
    diff = _ang(v_to) - _ang(v_from)
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return float(diff)


def _build_anatomical_frame(kpts_3d: np.ndarray):
    """Derives a per-clip, gravity-aligned anatomical frame (up/right/forward)
    from the body's own geometry, instead of assuming MotionBERT's raw output
    axes correspond to real-world up/forward/sideways -- that correspondence
    isn't guaranteed by the model. Assumes a static camera and that the frame
    with the straightest knees is a standing reference (true for a squat clip
    with a visible top position).

    Returns (calib_frame_idx, up_axis, right_axis, fwd_axis), unit vectors,
    fixed for the whole clip.
    """
    T = kpts_3d.shape[0]
    knee_ext = np.array([
        _angle_between(kpts_3d[t, H36M_LHIP] - kpts_3d[t, H36M_LKNEE],
                        kpts_3d[t, H36M_LANKLE] - kpts_3d[t, H36M_LKNEE]) +
        _angle_between(kpts_3d[t, H36M_RHIP] - kpts_3d[t, H36M_RKNEE],
                        kpts_3d[t, H36M_RANKLE] - kpts_3d[t, H36M_RKNEE])
        for t in range(T)
    ])
    calib = int(np.nanargmax(knee_ext))  # straightest-knee frame = standing reference

    j = kpts_3d[calib]
    up = _unit(j[H36M_THORAX] - j[H36M_HIP])
    right_raw = j[H36M_RHIP] - j[H36M_LHIP]
    right = _unit(right_raw - np.dot(right_raw, up) * up)  # orthogonalize vs up
    fwd = _unit(np.cross(up, right))
    return calib, up, right, fwd


def compute_biomechanics_angles(kpts_3d: np.ndarray) -> list:
    """kpts_3d: (T, 17, 3) H36M-17-order 3D joints. Returns a list of T dicts
    keyed by EXTRACTED_ANGLE_COLUMNS.

    An anatomical frame (up/right/forward) is derived once per clip from the
    standing-most frame (see _build_anatomical_frame). Angles that are
    naturally "zero at standing" (hip_flexion, ankle_dorsiflexion_proxy) are
    calibrated against that same frame. hip_angle and knee_flexion are pure
    3-point joint angles and don't need calibration -- they're camera/axis
    independent by construction.

    LIMITATIONS (please read):
      - knee_valgus_L/R: Frontal Plane Projection Angle (FPPA), the standard
        clinical metric for dynamic knee valgus during a squat. Positive =
        knee deviating toward the midline (valgus), negative = varus.
      - ankle_dorsiflexion_proxy_L/R: H36M-17 has NO foot/toe keypoint (Halpe's
        toe/heel points are dropped in halpe26_to_h36m17), so true ankle
        dorsiflexion (foot-to-shank angle) cannot be computed. This is a
        proxy: shank forward lean relative to vertical, zeroed at standing.
        It tracks dorsiflexion during a squat but is not the same measurement
        a goniometer on the foot would give.
      - lumbar_curvature_proxy: the H36M "Spine" joint is defined as the exact
        midpoint of Hip and Thorax (see halpe26_to_h36m17), so it's always
        perfectly colinear with them -- any angle computed there is
        degenerate (always 180 deg, zero information). There is no real
        curvature signal available without an actual mid-back/chest keypoint,
        which nothing upstream provides. This column reports the trunk's
        overall sagittal lean (same as sagittal_flexion_trunk) so it isn't
        left blank -- it is NOT a measurement of spinal curvature.
    """
    T = kpts_3d.shape[0]
    calib, up, right, fwd = _build_anatomical_frame(kpts_3d)

    def _raw(j):
        thigh_sag_L = _signed_angle_in_plane(up, j[H36M_LKNEE] - j[H36M_LHIP], up, fwd)
        thigh_sag_R = _signed_angle_in_plane(up, j[H36M_RKNEE] - j[H36M_RHIP], up, fwd)
        shank_sag_L = _signed_angle_in_plane(up, j[H36M_LANKLE] - j[H36M_LKNEE], up, fwd)
        shank_sag_R = _signed_angle_in_plane(up, j[H36M_RANKLE] - j[H36M_RKNEE], up, fwd)
        trunk_sag = _signed_angle_in_plane(up, j[H36M_THORAX] - j[H36M_HIP], up, fwd)
        return thigh_sag_L, thigh_sag_R, shank_sag_L, shank_sag_R, trunk_sag

    tsL_c, tsR_c, ssL_c, ssR_c, trunk_c = _raw(kpts_3d[calib])
    hip_flex_ref_L = tsL_c - trunk_c
    hip_flex_ref_R = tsR_c - trunk_c
    ankle_ref_L = ssL_c
    ankle_ref_R = ssR_c

    rows = []
    for t in range(T):
        j = kpts_3d[t]
        thigh_sag_L, thigh_sag_R, shank_sag_L, shank_sag_R, trunk_sag = _raw(j)

        # 1. Knee medial/lateral deviation (Frontal Plane Projection Angle)
        knee_valgus_L = _signed_angle_in_plane(j[H36M_LKNEE] - j[H36M_LHIP], j[H36M_LANKLE] - j[H36M_LKNEE], up, right)
        knee_valgus_R = -_signed_angle_in_plane(j[H36M_RKNEE] - j[H36M_RHIP], j[H36M_RANKLE] - j[H36M_RKNEE], up, right)

        # 2. Head positioning (forward head tilt, sagittal plane, 0 = head over thorax)
        head_forward_angle = _signed_angle_in_plane(up, j[H36M_HEAD] - j[H36M_THORAX], up, fwd)

        # 5. Hip angle (3D included angle, trunk vs thigh)
        hip_angle_L = _angle_between(j[H36M_THORAX] - j[H36M_LHIP], j[H36M_LKNEE] - j[H36M_LHIP])
        hip_angle_R = _angle_between(j[H36M_THORAX] - j[H36M_RHIP], j[H36M_RKNEE] - j[H36M_RHIP])

        # 6. Hip flexion (sagittal-plane, zeroed at standing calibration frame)
        hip_flexion_L = (thigh_sag_L - trunk_sag) - hip_flex_ref_L
        hip_flexion_R = (thigh_sag_R - trunk_sag) - hip_flex_ref_R

        # 7. Knee flexion angle (clinical convention: 0 deg = fully extended)
        knee_flexion_L = 180.0 - _angle_between(j[H36M_LHIP] - j[H36M_LKNEE], j[H36M_LANKLE] - j[H36M_LKNEE])
        knee_flexion_R = 180.0 - _angle_between(j[H36M_RHIP] - j[H36M_RKNEE], j[H36M_RANKLE] - j[H36M_RKNEE])

        # 3. Squat depth via knee angle (avg knee flexion of both legs)
        squat_depth_knee_deg = float(np.nanmean([knee_flexion_L, knee_flexion_R]))

        # 4. Sagittal flexion of the trunk (forward lean, 0 = trunk vertical)
        sagittal_flexion_trunk = trunk_sag

        # 8. Ankle dorsiflexion proxy (shank forward lean vs vertical, zeroed at standing)
        ankle_dorsiflexion_proxy_L = shank_sag_L - ankle_ref_L
        ankle_dorsiflexion_proxy_R = shank_sag_R - ankle_ref_R

        # 9. Lumbar curvature -- NOT actually measurable, see docstring. Duplicate of trunk lean.
        lumbar_curvature_proxy = sagittal_flexion_trunk

        rows.append({
            "knee_valgus_L": knee_valgus_L, "knee_valgus_R": knee_valgus_R,
            "head_forward_angle": head_forward_angle,
            "squat_depth_knee_deg": squat_depth_knee_deg,
            "sagittal_flexion_trunk": sagittal_flexion_trunk,
            "hip_angle_L": hip_angle_L, "hip_angle_R": hip_angle_R,
            "hip_flexion_L": hip_flexion_L, "hip_flexion_R": hip_flexion_R,
            "knee_flexion_L": knee_flexion_L, "knee_flexion_R": knee_flexion_R,
            "ankle_dorsiflexion_proxy_L": ankle_dorsiflexion_proxy_L,
            "ankle_dorsiflexion_proxy_R": ankle_dorsiflexion_proxy_R,
            "lumbar_curvature_proxy": lumbar_curvature_proxy,
        })
    return rows


def write_extracted_angles_csv(kpts_3d: np.ndarray, out_path: str):
    """Writes one row per frame: frame index + the 8 angle columns above."""
    rows = compute_biomechanics_angles(kpts_3d)
    LOGGER.info(f"Writing extracted angles CSV: {out_path}")
    with open(out_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["frame"] + EXTRACTED_ANGLE_COLUMNS)
        for t, row in enumerate(rows):
            writer_csv.writerow([t] + [row[c] for c in EXTRACTED_ANGLE_COLUMNS])
    LOGGER.info("Extracted angles CSV written.")


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
    results = yolo_model.predict(source=frame_bgr, conf=conf, classes=[0], device=DEVICE, verbose=False)
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
# Stage 4: MotionBERT mesh (SMPL) — optional
# ==========================================================================
def check_smpl_assets():
    """Verifies the SMPL data bundle exists before we even try to import smplx
    or build the mesh model. Raises a clear, actionable FileNotFoundError
    listing exactly what's missing rather than failing deep inside smplx with
    an opaque pickle/IO error."""
    missing = [f for f in SMPL_REQUIRED_FILES if not (SMPL_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Mesh stage requires SMPL body-model assets that are NOT bundled with "
            "MotionBERT or AlphaPose (license-gated, must be obtained separately). "
            f"Missing from {SMPL_DATA_DIR}: {missing}. "
            "Get SMPL_NEUTRAL.pkl from https://smpl.is.tue.mpg.de/ (register, download "
            "'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', rename it), and "
            "smpl_mean_params.npz / J_regressor_extra.npy / J_regressor_h36m_correct.npy "
            "from the SPIN repo (https://github.com/nkolot/SPIN, see their data bundle). "
            f"Place all four files directly in {SMPL_DATA_DIR}."
        )


def load_motionbert_mesh():
    """Loads the MeshRegressor (DSTformer backbone + SMPL head) from the
    FT_MB_release_MB_ft_pw3d checkpoint. Requires the SMPL asset bundle —
    see check_smpl_assets()."""
    LOGGER.info(f"Loading MotionBERT mesh checkpoint from {MOTIONBERT_MESH_CKPT}")
    if not MOTIONBERT_MESH_CKPT.exists():
        raise FileNotFoundError(f"MotionBERT mesh checkpoint not found at {MOTIONBERT_MESH_CKPT}")
    check_smpl_assets()

    if str(MOTIONBERT_DIR) not in sys.path:
        sys.path.insert(0, str(MOTIONBERT_DIR))

    from lib.utils.tools import get_config
    from lib.model.DSTformer import DSTformer
    from lib.model.model_mesh import MeshRegressor

    cfg_path = MOTIONBERT_DIR / "configs" / "mesh" / "MB_ft_pw3d.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Expected MotionBERT mesh config at {cfg_path}. Update cfg_path in "
            "load_motionbert_mesh() if your config file name/location differs."
        )
    args = get_config(str(cfg_path))
    # args.data_root in the stock yaml is a relative path ("data/mesh"); pin it to our
    # absolute SMPL_DATA_DIR so it resolves correctly regardless of cwd.
    args.data_root = str(SMPL_DATA_DIR)

    LOGGER.info("Building MotionBERT mesh backbone (DSTformer, mesh config)...")
    backbone = DSTformer(
        dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep,
        depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        norm_layer=torch.nn.LayerNorm, maxlen=args.maxlen,
        num_joints=args.num_joints,
    )
    model = MeshRegressor(
        args, backbone=backbone, dim_rep=args.dim_rep,
        hidden_dim=args.hidden_dim, dropout_ratio=args.dropout,
    )

    ckpt = torch.load(str(MOTIONBERT_MESH_CKPT), map_location=DEVICE)
    state_dict = ckpt.get("model", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    LOGGER.info("MotionBERT mesh model loaded.")
    return model, args


def lift_sequence_to_mesh(mesh_model, seq_2d, clip_len=243):
    """seq_2d: (T, 17, 3) normalized [x,y,score] in [-1,1]-ish range, same
    input the pose3d model takes.
    Returns:
      verts: (T, 6890, 3) SMPL mesh vertices in millimeters, root-relative
      kp_3d: (T, 17, 3) H36M joints regressed from the mesh, in millimeters
    Processed in clip_len chunks (MotionBERT's receptive field), no flip-test
    averaging (that's an accuracy refinement, not required for this to work)."""
    T = seq_2d.shape[0]
    verts_all = np.zeros((T, 6890, 3), dtype=np.float32)
    kp3d_all = np.zeros((T, 17, 3), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, T, clip_len):
            end = min(start + clip_len, T)
            clip = seq_2d[start:end]
            pad = clip_len - clip.shape[0]
            if pad > 0:
                clip = np.concatenate([clip, np.repeat(clip[-1:], pad, axis=0)], axis=0)
            x = torch.from_numpy(clip).float().unsqueeze(0).to(DEVICE)  # (1, clip_len, 17, 3)
            out = mesh_model(x)[0]  # dict: 'theta', 'verts' (1,T,6890,3), 'kp_3d' (1,T,17,3)
            v = out["verts"][0].cpu().numpy()[: end - start]
            k = out["kp_3d"][0].cpu().numpy()[: end - start]
            verts_all[start:end] = v
            kp3d_all[start:end] = k
    return verts_all, kp3d_all


def render_mesh_video(verts, out_path, fps=25.0, smpl_faces=None):
    """Renders a turntable-fixed SMPL mesh video from (T, 6890, 3) vertices,
    matplotlib-based (same approach MotionBERT's own vismo.motion2video_mesh
    uses) so no extra offscreen-rendering deps (pyrender/OSMesa/EGL) are
    needed — those are notoriously fragile in headless containers.
    Falls back to a plain skeleton line-render if smpl_faces isn't available
    for any reason."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LOGGER.info(f"Rendering mesh video ({verts.shape[0]} frames) to {out_path}")
    T = verts.shape[0]
    X, Y, Z = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2]
    max_range = float(np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()) / 2.0
    mid_x, mid_y, mid_z = float((X.max() + X.min()) * 0.5), float((Y.max() + Y.min()) * 0.5), float((Z.max() + Z.min()) * 0.5)
    max_range = max(max_range, 1e-3)

    fig = plt.figure(figsize=(6, 6), dpi=100)
    frame_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for f in range(T):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=-90, azim=-90)
        ax.set_axis_off()
        v = verts[f]
        if smpl_faces is not None:
            ax.plot_trisurf(v[:, 0], v[:, 1], triangles=smpl_faces, Z=v[:, 2],
                             color=(166 / 255.0, 188 / 255.0, 218 / 255.0, 0.9))
        else:
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=1, c="steelblue")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if frame_writer is None:
            h, w = img_bgr.shape[:2]
            frame_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        frame_writer.write(img_bgr)
        if f % 15 == 0:
            LOGGER.info(f"Mesh render: {f}/{T} frames")

    if frame_writer is not None:
        frame_writer.release()
    plt.close(fig)
    LOGGER.info(f"Mesh video written: {out_path}")


# ==========================================================================
# FUTURE: Grad-CAM-style saliency overlay on the mesh.
# Not called anywhere in the pipeline yet — this is scaffolding for the
# squat-form-analysis project's planned "point to the responsible muscle /
# joint" explainability feature. Left here as a clean, reusable helper so
# that when a downstream classifier (e.g. "correct rep" vs "wrong rep") is
# plugged in, wiring up per-vertex/per-joint saliency is a few lines, not a
# redesign.
# ==========================================================================
def compute_mesh_saliency(classifier_model, seq_2d, target_class_idx, mesh_verts=None,
                           layer_name=None):
    """Grad-CAM-style saliency for a downstream classifier that consumes
    pose/mesh sequences (e.g. a squat-form correctness classifier).

    This is intentionally generic and NOT wired into process_video() yet —
    there's no classifier trained yet. When one exists, the intended usage
    is:

        heatmap = compute_mesh_saliency(
            classifier_model=my_squat_classifier,
            seq_2d=h36m_seq_norm,             # same (T,17,3) input the pose/mesh
                                               # models already consume in this file
            target_class_idx=predicted_wrong_class,
            mesh_verts=verts,                 # (T, 6890, 3) from lift_sequence_to_mesh,
                                               # optional — enables per-vertex coloring
            layer_name="backbone.some_layer", # the layer whose activations you want
                                               # Grad-CAM over; None = use classifier's
                                               # final feature layer via a registered hook
        )

    Args:
        classifier_model: any nn.Module that takes the same (N,T,J,3) input shape
            this pipeline already uses and outputs per-class logits.
        seq_2d: (T, 17, 3) normalized 2D sequence (H36M order), the same tensor
            format fed to lift_sequence_to_3d / lift_sequence_to_mesh.
        target_class_idx: int, which output class to backprop from (e.g. the
            predicted "incorrect form" class).
        mesh_verts: optional (T, 6890, 3) vertex array. If provided, the
            per-joint saliency is broadcast to per-vertex via nearest-joint
            assignment so it can be rendered as a heatmap on the actual mesh
            (via render_mesh_saliency_video below) instead of just per-joint.
        layer_name: dotted module path to hook for Grad-CAM. If None, hooks
            are registered on every leaf module and the first one with a
            gradient is used — fine for exploration, name it explicitly once
            you know which layer is meaningful for your model.

    Returns:
        dict with:
          'joint_saliency': (T, 17) float array, per-joint/per-frame importance
              (ReLU'd, normalized to [0,1])
          'vertex_saliency': (T, 6890) float array or None if mesh_verts wasn't
              given — per-vertex importance for mesh heatmap rendering

    Implementation notes for when this gets wired up:
      - Uses standard Grad-CAM: forward hook captures activations A, backward
        hook captures gradients dY/dA, saliency = ReLU(mean_spatial(dY/dA) * A).
      - Per-vertex assignment (mesh_verts branch) uses a nearest-H36M-joint
        lookup per vertex, computed once and cached — not a learned skinning
        weight, so treat it as a coarse visualization, not an anatomically
        precise attribution.
    """
    activations = {}
    gradients = {}

    def _fwd_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    def _bwd_hook(name):
        def hook(module, grad_in, grad_out):
            gradients[name] = grad_out[0].detach()
        return hook

    handles = []
    modules_to_hook = (
        [(layer_name, dict(classifier_model.named_modules())[layer_name])]
        if layer_name else list(classifier_model.named_modules())
    )
    for name, module in modules_to_hook:
        handles.append(module.register_forward_hook(_fwd_hook(name)))
        handles.append(module.register_full_backward_hook(_bwd_hook(name)))

    try:
        classifier_model.eval()
        x = torch.from_numpy(seq_2d).float().unsqueeze(0).to(DEVICE)
        x.requires_grad_(True)
        logits = classifier_model(x)
        score = logits[0, target_class_idx]
        classifier_model.zero_grad(set_to_none=True)
        score.backward()

        # Use the first hooked layer that actually produced a gradient.
        used_name = layer_name
        if used_name is None:
            for name in activations:
                if name in gradients:
                    used_name = name
                    break
        if used_name is None:
            raise RuntimeError(
                "compute_mesh_saliency: no hooked layer produced a gradient. "
                "Pass an explicit layer_name once the classifier architecture is known."
            )

        A = activations[used_name]     # activations, shape depends on layer
        dA = gradients[used_name]      # gradients w.r.t. those activations
        weights = dA.mean(dim=tuple(range(1, dA.dim() - 1)), keepdim=True) if dA.dim() > 2 else dA
        cam = torch.relu((weights * A).sum(dim=-1))  # collapse feature dim
        cam = cam.squeeze(0).cpu().numpy()
        # Normalize to [0,1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        T = seq_2d.shape[0]
        if cam.shape[0] != T:
            # Layer output isn't per-frame/per-joint shaped — caller needs a
            # more specific layer_name for a meaningful per-joint breakdown.
            joint_saliency = np.tile(cam.reshape(1, -1)[:, :17], (T, 1)) if cam.size >= 17 else np.zeros((T, 17))
        else:
            joint_saliency = cam.reshape(T, -1)[:, :17]

        vertex_saliency = None
        if mesh_verts is not None:
            vertex_saliency = _broadcast_joint_saliency_to_vertices(joint_saliency, mesh_verts)

        return {"joint_saliency": joint_saliency, "vertex_saliency": vertex_saliency}
    finally:
        for h in handles:
            h.remove()


def _broadcast_joint_saliency_to_vertices(joint_saliency, mesh_verts, joint_positions=None):
    """Nearest-joint lookup to spread (T,17) joint saliency onto (T,6890)
    vertices for heatmap rendering. Coarse by design — see docstring above."""
    T, V, _ = mesh_verts.shape
    vertex_saliency = np.zeros((T, V), dtype=np.float32)
    for t in range(T):
        if joint_positions is not None:
            jp = joint_positions[t]  # (17,3)
        else:
            # Without explicit joint 3D positions, approximate joints as the
            # mean of the whole mesh split into 17 rough chunks — placeholder
            # only; pass real joint_positions (e.g. kp_3d from lift_sequence_to_mesh)
            # for a meaningful assignment.
            chunk = V // 17
            jp = np.stack([mesh_verts[t, i * chunk:(i + 1) * chunk].mean(axis=0) for i in range(17)])
        dists = np.linalg.norm(mesh_verts[t][:, None, :] - jp[None, :, :], axis=-1)  # (V,17)
        nearest = dists.argmin(axis=1)  # (V,)
        vertex_saliency[t] = joint_saliency[t][nearest]
    return vertex_saliency


def render_mesh_saliency_video(verts, vertex_saliency, out_path, fps=25.0, smpl_faces=None):
    """Companion renderer for compute_mesh_saliency's output — draws the mesh
    colored by per-vertex saliency (red = high importance) instead of a flat
    color. Not called anywhere yet; wire up once compute_mesh_saliency has a
    real classifier behind it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    T = verts.shape[0]
    X, Y, Z = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2]
    max_range = float(np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()) / 2.0
    mid_x, mid_y, mid_z = float((X.max() + X.min()) * 0.5), float((Y.max() + Y.min()) * 0.5), float((Z.max() + Z.min()) * 0.5)
    max_range = max(max_range, 1e-3)

    fig = plt.figure(figsize=(6, 6), dpi=100)
    frame_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cmap = cm.get_cmap("jet")

    for f in range(T):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=-90, azim=-90)
        ax.set_axis_off()
        v = verts[f]
        colors = cmap(vertex_saliency[f])
        if smpl_faces is not None:
            face_vals = vertex_saliency[f][smpl_faces].mean(axis=1)
            ax.plot_trisurf(v[:, 0], v[:, 1], triangles=smpl_faces, Z=v[:, 2],
                             facecolor=cmap(face_vals), shade=False)
        else:
            ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=2, c=colors)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if frame_writer is None:
            h, w = img_bgr.shape[:2]
            frame_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        frame_writer.write(img_bgr)

    if frame_writer is not None:
        frame_writer.release()
    plt.close(fig)


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
# def process_video(video_path: str, use_3d: bool = True, use_mesh: bool = False,
#                    det_conf: float = 0.5, progress_cb=None):
def process_video(
            video_path: str,
            use_3d: bool = True,
            use_mesh: bool = False,
            det_conf: float = 0.5,
            progress_cb=None,
            output_name=None,
        ):
    """
    Runs the full pipeline on a video file.

    Returns: (csv_path, overlay_video_path, mesh_video_path)
      mesh_video_path is None if use_mesh is False.
    Raises on any hard failure (with full traceback logged via LOGGER).
    """
    # stem = Path(video_path).stem
    stem = output_name or Path(video_path).stem

    # Create a dedicated folder for this video
    video_output_dir = OUTPUT_DIR / stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = str(video_output_dir / f"{stem}_keypoints.csv")
    overlay_path = str(video_output_dir / f"{stem}_overlay.mp4")
    mesh_path = str(video_output_dir / f"{stem}_mesh.mp4")

    LOGGER.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LOGGER.info(f"Video: {w}x{h} @ {fps:.2f}fps, ~{n_frames} frames")
    if use_mesh:
        LOGGER.info("Mesh reconstruction requested — will run after 2D/3D stages.")

    yolo_model = load_yolo()
    pose_model, _ = load_alphapose()
    mb_model = None
    if use_3d or use_mesh:
        # mesh stage needs the same normalized H36M 2D sequence the pose3d
        # lift uses, so we still build it below regardless of which of
        # use_3d / use_mesh is on. mb_model itself (pose3d lift) is only
        # loaded if use_3d is requested.
        pass
    if use_3d:
        mb_model, _ = load_motionbert()

    mesh_model = None
    if use_mesh:
        # Fail fast, before spending time on frame-by-frame inference, if the
        # SMPL assets aren't there — check_smpl_assets() gives a clear error.
        try:
            check_smpl_assets()
        except FileNotFoundError as e:
            LOGGER.error(str(e))
            raise
        mesh_model, _ = load_motionbert_mesh()

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

    h36m_seq_norm = None
    if use_3d or use_mesh:
        h36m_seq = np.stack([halpe26_to_h36m17(f) for f in all_frames_2d], axis=0)  # (T,17,3)
        h36m_seq_norm = np.stack(
            [normalize_2d_for_motionbert(f, w, h) for f in h36m_seq], axis=0
        )

    kpts_3d = None
    if use_3d and mb_model is not None:
        LOGGER.info("Stage 3: lifting to 3D with MotionBERT...")
        try:
            kpts_3d = lift_sequence_to_3d(mb_model, h36m_seq_norm)  # (T,17,3)
            LOGGER.info("3D lift complete.")
        except Exception:
            LOGGER.error(f"MotionBERT 3D lift failed, continuing with 2D-only CSV:\n{traceback.format_exc()}")
            kpts_3d = None

    mesh_kp3d = None
    mesh_video_result = None
    if use_mesh and mesh_model is not None:
    
        LOGGER.info("Stage 4: reconstructing SMPL mesh with MotionBERT...")
        try:
            verts, mesh_kp3d = lift_sequence_to_mesh(mesh_model, h36m_seq_norm)
            LOGGER.info(f"Mesh regression complete: {verts.shape[0]} frames, {verts.shape[1]} vertices/frame.")

            smpl_faces = None
            try:
                from lib.utils.utils_smpl import get_smpl_faces
                smpl_faces = get_smpl_faces()
                LOGGER.info(f"Loaded SMPL face topology ({len(smpl_faces)} triangles) for mesh rendering.")
            except Exception:
                LOGGER.error(
                    "Could not load SMPL face topology for shaded rendering; "
                    f"falling back to point-cloud mesh render.\n{traceback.format_exc()}"
                )

            render_mesh_video(verts, mesh_path, fps=fps, smpl_faces=smpl_faces)
            mesh_video_result = mesh_path
        except FileNotFoundError:
            # check_smpl_assets() already raised earlier if assets were missing;
            # this catches any other missing-file surprise from smplx itself.
            raise
        except Exception:
            LOGGER.error(f"Mesh reconstruction failed, continuing without mesh video:\n{traceback.format_exc()}")
            mesh_video_result = None

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

    angles_source_3d = kpts_3d if kpts_3d is not None else mesh_kp3d
    if angles_source_3d is not None: 
        extracted_data_path = str(video_output_dir / "extracted_data.csv")
        try:
            write_extracted_angles_csv(angles_source_3d, extracted_data_path)
        except Exception:
            LOGGER.error(f"Failed to write extracted_data.csv:\n{traceback.format_exc()}")
    else:
        LOGGER.info("No 3D joints available (use_3d/use_mesh both off, or 3D lift failed) -- skipping extracted_data.csv.")

    LOGGER.info("Pipeline finished successfully.")
    return csv_path, overlay_path, mesh_video_result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--no-3d", action="store_true")
    p.add_argument("--mesh", action="store_true")
    a = p.parse_args()
    process_video(a.video, use_3d=not a.no_3d, use_mesh=a.mesh)
