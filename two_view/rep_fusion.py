"""
rep_fusion.py -- per-rep fusion of the front-view and side-view angle tables.

Input : the two extracted_data.csv files the existing pipeline already writes
        (one from the front video, one from the side video).
Output: ONE table, one row per squat repetition, with front-view features and
        side-view features side by side (rep_features.csv).

Why per rep: a repetition is the natural unit for "was this squat good?", and
matching rep N in one view to rep N in the other only needs the videos to be
roughly aligned (the clap offset), not frame-perfect.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ANGLE_COLS = [
    "knee_valgus_L", "knee_valgus_R", "head_forward_angle", "squat_depth_knee_deg",
    "sagittal_flexion_trunk", "hip_angle_L", "hip_angle_R", "hip_flexion_L", "hip_flexion_R",
    "knee_flexion_L", "knee_flexion_R", "ankle_dorsiflexion_proxy_L", "ankle_dorsiflexion_proxy_R",
    "lumbar_curvature_proxy",
]


def load_angles(path):
    df = pd.read_csv(path)
    missing = [c for c in ["frame"] + ANGLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[["frame"] + ANGLE_COLS].reset_index(drop=True)


def detect_reps(depth, fps, min_prominence_deg=20.0, min_gap_s=1.0, smooth_s=0.25):
    """Find squat reps as peaks of knee flexion (bottom of the squat).
    Returns a list of dicts {start, bottom, end} in frame indices."""
    s = pd.Series(np.asarray(depth, dtype=float)).interpolate(limit_direction="both")
    if s.isna().all():
        return []
    win = max(3, int(smooth_s * fps) | 1)
    sm = np.convolve(s.to_numpy(), np.ones(win) / win, mode="same")
    prom = max(min_prominence_deg, 0.3 * (np.percentile(sm, 95) - np.percentile(sm, 5)))
    peaks, _ = find_peaks(sm, prominence=prom, distance=max(1, int(min_gap_s * fps)))
    reps = []
    for k, p in enumerate(peaks):
        lo = peaks[k - 1] if k > 0 else 0
        hi = peaks[k + 1] if k + 1 < len(peaks) else len(sm) - 1
        start = lo + int(np.argmin(sm[lo:p + 1]))
        end = p + int(np.argmin(sm[p:hi + 1]))
        reps.append({"start": int(start), "bottom": int(p), "end": int(end)})
    return reps


def rep_features(df, rep, fps, prefix):
    """Features for one rep: value at the bottom, plus min/max over the rep, for every angle."""
    seg = df.iloc[rep["start"]:rep["end"] + 1]
    row = {}
    for m in ANGLE_COLS:
        row[f"{prefix}{m}_bottom"] = df[m].iloc[rep["bottom"]]
        row[f"{prefix}{m}_min"] = seg[m].min()
        row[f"{prefix}{m}_max"] = seg[m].max()
    row[f"{prefix}rep_dur_s"] = (rep["end"] - rep["start"]) / fps
    row[f"{prefix}descent_s"] = (rep["bottom"] - rep["start"]) / fps
    row[f"{prefix}ascent_s"] = (rep["end"] - rep["bottom"]) / fps
    return row


def match_reps(front_reps, side_reps, fps_front, fps_side, offset_s=None, tol_s=0.75):
    """Pair front reps with side reps. With a clap offset: nearest bottom time within tol_s.
    Without one: pair by order. Returns a list of (front_idx, side_idx)."""
    if offset_s is None:
        return [(i, i) for i in range(min(len(front_reps), len(side_reps)))]
    pairs, used = [], set()
    for i, fr in enumerate(front_reps):
        expected_side_t = fr["bottom"] / fps_front + offset_s
        best, best_d = None, tol_s
        for j, sr in enumerate(side_reps):
            d = abs(sr["bottom"] / fps_side - expected_side_t)
            if j not in used and d <= best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def fuse_views(front_csv, side_csv, fps_front, fps_side, offset_s=None):
    front, side = load_angles(front_csv), load_angles(side_csv)
    # Reps are detected on the knee-flexion curve of each view independently.
    f_reps = detect_reps(front["squat_depth_knee_deg"], fps_front)
    s_reps = detect_reps(side["squat_depth_knee_deg"], fps_side)
    pairs = match_reps(f_reps, s_reps, fps_front, fps_side, offset_s)
    rows = []
    for n, (i, j) in enumerate(pairs, start=1):
        row = {"rep_id": n,
               "front_bottom_s": round(f_reps[i]["bottom"] / fps_front, 3),
               "side_bottom_s": round(s_reps[j]["bottom"] / fps_side, 3)}
        row.update(rep_features(front, f_reps[i], fps_front, "front_"))
        row.update(rep_features(side, s_reps[j], fps_side, "side_"))
        rows.append(row)
    return {
        "front": front, "side": side,
        "front_reps": f_reps, "side_reps": s_reps,
        "pairs": pairs, "table": pd.DataFrame(rows),
    }
