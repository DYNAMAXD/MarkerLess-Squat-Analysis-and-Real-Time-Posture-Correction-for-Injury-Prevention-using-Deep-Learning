"""
p2s_biomech.py -- turns Pose2Sim's triangulated 3D keypoints (.trc) into

    angles.csv          frame, joint, x_3d, y_3d, z_3d     (15 joints per frame, long format)
    advancedangles.csv  frame, knee_valgus_L ... lumbar_curvature_proxy, com_keypoints, com_mesh, com_final

The angle and centre-of-mass formulas are the SAME as keypoint_pipeline.py (compute_biomechanics_angles,
compute_keypoint_com, compute_final_com). They are copied here because keypoint_pipeline imports torch at load time,
and the Pose2Sim environment does not have torch. If you change a formula there, change it here too.

Coordinates are Pose2Sim's world frame (metres, Y up, origin = the extrinsic checkerboard).
com_mesh is empty: Pose2Sim has no SMPL mesh, so com_final = com_keypoints (same rule as the single-view pipeline
when the mesh stage is off).
"""
import csv

import numpy as np
import pandas as pd

# The 15 joints written to angles.csv (all are HALPE_26 / "Body_with_feet" markers in Pose2Sim).
OUT_JOINTS = ["LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
              "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Head", "Neck", "Hip"]

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

# H36M-17 order used by the angle formulas (same as keypoint_pipeline.H36M17_JOINTS).
(H36M_HIP, H36M_RHIP, H36M_RKNEE, H36M_RANKLE, H36M_LHIP, H36M_LKNEE, H36M_LANKLE,
 H36M_SPINE, H36M_THORAX, H36M_NECK, H36M_HEAD,
 H36M_LSHOULDER, H36M_LELBOW, H36M_LWRIST, H36M_RSHOULDER, H36M_RELBOW, H36M_RWRIST) = range(17)

# HALPE marker name -> H36M index (same correspondences as keypoint_pipeline.HALPE26_TO_H36M17).
HALPE_TO_H36M = {
    "Hip": H36M_HIP, "RHip": H36M_RHIP, "RKnee": H36M_RKNEE, "RAnkle": H36M_RANKLE,
    "LHip": H36M_LHIP, "LKnee": H36M_LKNEE, "LAnkle": H36M_LANKLE,
    "Neck": H36M_THORAX, "Head": H36M_HEAD,
    "LShoulder": H36M_LSHOULDER, "LElbow": H36M_LELBOW, "LWrist": H36M_LWRIST,
    "RShoulder": H36M_RSHOULDER, "RElbow": H36M_RELBOW, "RWrist": H36M_RWRIST,
}


# --------------------------------------------------------------------------- TRC reading
def read_trc(path):
    """Returns (frames, times, {marker: (T,3) array}). Frame numbers are taken from the Frame# column."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    df = pd.read_csv(path, sep="\t", skiprows=5, header=None)
    data = df.iloc[:, :2 + 3 * len(names)].to_numpy(dtype=float)  # keep NaN columns: a marker may be all-NaN
    frames, times = data[:, 0].astype(int), data[:, 1]
    xyz = data[:, 2:2 + 3 * len(names)].reshape(len(data), len(names), 3)
    return frames, times, {n: xyz[:, i, :] for i, n in enumerate(names)}


def to_h36m(markers):
    """{marker: (T,3)} -> (T,17,3) H36M array. Spine and Neck/Nose are interpolated exactly like
    keypoint_pipeline.halpe26_to_h36m17."""
    missing = [n for n in HALPE_TO_H36M if n not in markers]
    if missing:
        raise ValueError(f"TRC file is missing markers {missing}. Is pose_model = 'Body_with_feet' (HALPE_26)?")
    T = len(next(iter(markers.values())))
    out = np.full((T, 17, 3), np.nan, dtype=np.float64)
    for name, idx in HALPE_TO_H36M.items():
        out[:, idx] = markers[name]
    out[:, H36M_SPINE] = (out[:, H36M_HIP] + out[:, H36M_THORAX]) / 2.0
    out[:, H36M_NECK] = (out[:, H36M_THORAX] + out[:, H36M_HEAD]) / 2.0
    return out


# --------------------------------------------------------------------------- math (copied from keypoint_pipeline.py)
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6 or not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        return float("nan")
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _signed_angle_in_plane(v_from, v_to, x_axis, y_axis):
    def _ang(v):
        return np.degrees(np.arctan2(np.dot(v, y_axis), np.dot(v, x_axis)))
    diff = _ang(v_to) - _ang(v_from)
    if not np.isfinite(diff):
        return float("nan")
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return float(diff)


def _wrap(d):
    """Difference of two signed angles -> [-180, 180]. Leg segments point DOWN (~ +/-180 deg from 'up'), so their
    signed angles flip between +179 and -179; without wrapping, hip_flexion / ankle proxy jump by 360 deg.
    (keypoint_pipeline.py does not wrap these; this is the one intentional difference.)"""
    return float((d + 180.0) % 360.0 - 180.0) if np.isfinite(d) else float("nan")


def _build_anatomical_frame(kpts_3d):
    """Standing reference = frame with the straightest knees (see keypoint_pipeline._build_anatomical_frame)."""
    T = kpts_3d.shape[0]
    knee_ext = np.array([
        _angle_between(kpts_3d[t, H36M_LHIP] - kpts_3d[t, H36M_LKNEE], kpts_3d[t, H36M_LANKLE] - kpts_3d[t, H36M_LKNEE]) +
        _angle_between(kpts_3d[t, H36M_RHIP] - kpts_3d[t, H36M_RKNEE], kpts_3d[t, H36M_RANKLE] - kpts_3d[t, H36M_RKNEE])
        for t in range(T)
    ])
    calib = int(np.nanargmax(knee_ext))
    j = kpts_3d[calib]
    up = _unit(j[H36M_THORAX] - j[H36M_HIP])
    right_raw = j[H36M_RHIP] - j[H36M_LHIP]
    right = _unit(right_raw - np.dot(right_raw, up) * up)
    fwd = _unit(np.cross(up, right))
    return calib, up, right, fwd


def compute_biomechanics_angles(kpts_3d):
    """Same definitions and limitations as keypoint_pipeline.compute_biomechanics_angles."""
    T = kpts_3d.shape[0]
    calib, up, right, fwd = _build_anatomical_frame(kpts_3d)

    def _raw(j):
        return (_signed_angle_in_plane(up, j[H36M_LKNEE] - j[H36M_LHIP], up, fwd),
                _signed_angle_in_plane(up, j[H36M_RKNEE] - j[H36M_RHIP], up, fwd),
                _signed_angle_in_plane(up, j[H36M_LANKLE] - j[H36M_LKNEE], up, fwd),
                _signed_angle_in_plane(up, j[H36M_RANKLE] - j[H36M_RKNEE], up, fwd),
                _signed_angle_in_plane(up, j[H36M_THORAX] - j[H36M_HIP], up, fwd))

    tsL_c, tsR_c, ssL_c, ssR_c, trunk_c = _raw(kpts_3d[calib])
    hip_flex_ref_L, hip_flex_ref_R = _wrap(tsL_c - trunk_c), _wrap(tsR_c - trunk_c)

    rows = []
    for t in range(T):
        j = kpts_3d[t]
        thigh_sag_L, thigh_sag_R, shank_sag_L, shank_sag_R, trunk_sag = _raw(j)
        knee_flexion_L = 180.0 - _angle_between(j[H36M_LHIP] - j[H36M_LKNEE], j[H36M_LANKLE] - j[H36M_LKNEE])
        knee_flexion_R = 180.0 - _angle_between(j[H36M_RHIP] - j[H36M_RKNEE], j[H36M_RANKLE] - j[H36M_RKNEE])
        kf = [v for v in (knee_flexion_L, knee_flexion_R) if np.isfinite(v)]
        rows.append({
            "knee_valgus_L": _signed_angle_in_plane(j[H36M_LKNEE] - j[H36M_LHIP], j[H36M_LANKLE] - j[H36M_LKNEE], up, right),
            "knee_valgus_R": -_signed_angle_in_plane(j[H36M_RKNEE] - j[H36M_RHIP], j[H36M_RANKLE] - j[H36M_RKNEE], up, right),
            "head_forward_angle": _signed_angle_in_plane(up, j[H36M_HEAD] - j[H36M_THORAX], up, fwd),
            "squat_depth_knee_deg": float(np.mean(kf)) if kf else float("nan"),
            "sagittal_flexion_trunk": trunk_sag,
            "hip_angle_L": _angle_between(j[H36M_THORAX] - j[H36M_LHIP], j[H36M_LKNEE] - j[H36M_LHIP]),
            "hip_angle_R": _angle_between(j[H36M_THORAX] - j[H36M_RHIP], j[H36M_RKNEE] - j[H36M_RHIP]),
            "hip_flexion_L": _wrap(_wrap(thigh_sag_L - trunk_sag) - hip_flex_ref_L),
            "hip_flexion_R": _wrap(_wrap(thigh_sag_R - trunk_sag) - hip_flex_ref_R),
            "knee_flexion_L": knee_flexion_L, "knee_flexion_R": knee_flexion_R,
            "ankle_dorsiflexion_proxy_L": _wrap(shank_sag_L - ssL_c),
            "ankle_dorsiflexion_proxy_R": _wrap(shank_sag_R - ssR_c),
            "lumbar_curvature_proxy": trunk_sag,  # NOT spinal curvature, see keypoint_pipeline docstring
        })
    return rows, calib


SEGMENT_MASS_FRACTIONS = {
    "head": 0.0694, "trunk": 0.4346,
    "upper_arm_L": 0.0271, "upper_arm_R": 0.0271, "forearm_L": 0.0162, "forearm_R": 0.0162,
    "hand_L": 0.0061, "hand_R": 0.0061, "thigh_L": 0.1416, "thigh_R": 0.1416,
    "shank_L": 0.0433, "shank_R": 0.0433, "foot_L": 0.0137, "foot_R": 0.0137,
}


def compute_keypoint_com(kpts_3d):
    """Segment-mass-weighted COM, same model as keypoint_pipeline.compute_keypoint_com."""
    segs = [  # (mass name, p1, p2) -- p2=None means a point mass at p1
        ("head", H36M_HEAD, None), ("trunk", H36M_HIP, H36M_THORAX),
        ("upper_arm_L", H36M_LSHOULDER, H36M_LELBOW), ("forearm_L", H36M_LELBOW, H36M_LWRIST), ("hand_L", H36M_LWRIST, None),
        ("upper_arm_R", H36M_RSHOULDER, H36M_RELBOW), ("forearm_R", H36M_RELBOW, H36M_RWRIST), ("hand_R", H36M_RWRIST, None),
        ("thigh_L", H36M_LHIP, H36M_LKNEE), ("shank_L", H36M_LKNEE, H36M_LANKLE), ("foot_L", H36M_LANKLE, None),
        ("thigh_R", H36M_RHIP, H36M_RKNEE), ("shank_R", H36M_RKNEE, H36M_RANKLE), ("foot_R", H36M_RANKLE, None),
    ]
    T = kpts_3d.shape[0]
    com = np.full((T, 3), np.nan)
    for t in range(T):
        j = kpts_3d[t]
        pts, ms = [], []
        for name, a, b in segs:
            p = j[a] if b is None else 0.5 * (j[a] + j[b])
            if np.all(np.isfinite(p)):
                pts.append(p)
                ms.append(SEGMENT_MASS_FRACTIONS[name])
        if pts:
            ms = np.asarray(ms) / np.sum(ms)
            com[t] = np.sum(np.asarray(pts) * ms[:, None], axis=0)
    return com


def _fmt_com(c):
    return "" if c is None or not np.all(np.isfinite(c)) else f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f}"


# --------------------------------------------------------------------------- writers
def write_outputs(trc_path, angles_csv, advanced_csv, log=print):
    """Reads the TRC, writes angles.csv + advancedangles.csv. Returns a DataFrame of the advanced angles
    (with a numeric 'frame' column) for rep detection / the website."""
    # Frame# = frame index in the synced (post-clap) videos. Pose2Sim drops leading/trailing frames it could not
    # triangulate, so the first row is not necessarily frame 0.
    frames, times, markers = read_trc(trc_path)
    log(f"[angles] {len(frames)} frames, {len(markers)} markers in {trc_path}")

    missing = [n for n in OUT_JOINTS if n not in markers]
    if missing:
        raise ValueError(f"TRC is missing joints {missing}")
    with open(angles_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "joint", "x_3d", "y_3d", "z_3d"])
        for t, f in enumerate(frames):
            for n in OUT_JOINTS:
                x, y, z = markers[n][t]
                w.writerow([f, n, *("" if not np.isfinite(v) else f"{v:.6f}" for v in (x, y, z))])
    log(f"[angles] wrote {angles_csv}")

    kpts = to_h36m(markers)
    rows, calib = compute_biomechanics_angles(kpts)
    log(f"[angles] standing reference frame (straightest knees) = {frames[calib]}")
    com_kp = compute_keypoint_com(kpts)
    with open(advanced_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame"] + EXTRACTED_ANGLE_COLUMNS + ["com_keypoints", "com_mesh", "com_final"])
        for t, row in enumerate(rows):
            w.writerow([frames[t], *[row[c] for c in EXTRACTED_ANGLE_COLUMNS],
                        _fmt_com(com_kp[t]), "", _fmt_com(com_kp[t])])
    log(f"[angles] wrote {advanced_csv}")

    df = pd.DataFrame(rows)
    df.insert(0, "frame", frames)
    nan_pct = 100 * df[EXTRACTED_ANGLE_COLUMNS].isna().mean().mean()
    if nan_pct > 5:
        log(f"[angles] WARNING: {nan_pct:.0f}% of angle values are empty (joints not triangulated in those frames)")
    return df, times
