"""
features.py - one recording folder -> per-rep arrays every model can use.

Based on squat_error_study/src/squat_study/features.py (the cleaning that compared best), with three fixes:
  1. "up" is the real standing direction ankles -> hips (median over the standing frames), not the nearest world axis
     (that snap was off by up to 19 deg on this data and looked like a constant forward lean)
  2. a joint missing for a whole recording stays missing until after hip-centring and is then put at the hip (0),
     instead of becoming the world origin (0,0,0), which gave impossible positions
  3. each rep is trimmed to the movement (start / end where knee flexion is within 10 % of its range above the
     standing minimum), so rep 1 no longer contains the standing time before the first squat

Pipeline per recording:  load -> clean (zero joints = missing, interpolate, rebuild Hip) -> body frame
(lateral / up / forward, hip-centred, / leg length, L/R swap check) -> per rep: cut, trim, velocity, 5 extra
features, resample to T frames -> tempo numbers -> summary row (rep_features.csv, tree models).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ANGLE_COLS = ["knee_valgus_L", "knee_valgus_R", "head_forward_angle", "squat_depth_knee_deg", "sagittal_flexion_trunk",
              "hip_angle_L", "hip_angle_R", "hip_flexion_L", "hip_flexion_R", "knee_flexion_L", "knee_flexion_R",
              "ankle_dorsiflexion_proxy_L", "ankle_dorsiflexion_proxy_R", "lumbar_curvature_proxy"]
EXTRA_COLS = ["knee_offset_L", "knee_offset_R", "heel_height_L", "heel_height_R", "hip_height_rel"]
FRAME_FEATURE_NAMES = ANGLE_COLS + EXTRA_COLS                      # F = 19
TEMPO_NAMES = ["rep_dur_s", "descent_s", "ascent_s", "bottom_frac"]
JOINT_CHANNELS = ["pos_lat", "pos_up", "pos_fwd", "vel_lat", "vel_up", "vel_fwd"]
CORE_JOINTS = ["Hip", "RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel", "RBigToe", "LBigToe"]
TRUNK_JOINT = "Neck"


# ------------------------------------------------------------------------------------------------- loading
def read_trc(path):
    """-> (frames [N], joints [N,J,3] metres with NaN = missing, names [J], fps)."""
    lines = Path(path).read_text().splitlines()
    fps = float(lines[2].split("\t")[0])
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    df = pd.read_csv(path, sep="\t", skiprows=5, header=None)
    arr = df.iloc[:, : 2 + 3 * len(names)].to_numpy(dtype=float)
    joints = arr[:, 2:].reshape(len(arr), len(names), 3)
    joints[np.all(joints == 0.0, axis=2)] = np.nan        # Pose2Sim writes untriangulated joints as exactly (0,0,0)
    return arr[:, 0].astype(int), joints, names, fps


def _interp_nan(a):
    """Linear interpolation of NaNs along axis 0; a column that is NaN everywhere stays NaN."""
    flat = pd.DataFrame(a.reshape(len(a), -1)).interpolate(limit_direction="both")
    return flat.to_numpy().reshape(a.shape)


def load_recording(rec_dir):
    rec_dir = Path(rec_dir)
    frames, joints, names, fps = read_trc(rec_dir / "points3d.trc")
    ix = {n: i for i, n in enumerate(names)}
    hip_mid = (joints[:, ix["RHip"]] + joints[:, ix["LHip"]]) / 2
    gap = np.isnan(joints[:, ix["Hip"]]).any(axis=1) & ~np.isnan(hip_mid).any(axis=1)
    joints[gap, ix["Hip"]] = hip_mid[gap]                 # Hip missing but both hip joints present
    missing = np.isnan(joints).any(axis=2)                # [N,J]
    joints = _interp_nan(joints)                          # fully missing joints stay NaN (fix 2)
    adv = pd.read_csv(rec_dir / "advancedangles.csv").set_index("frame").reindex(frames)
    angles = _interp_nan(adv[ANGLE_COLS].to_numpy(dtype=float))
    reps = pd.read_csv(rec_dir / "reps.csv")
    summ = pd.read_csv(rec_dir / "rep_features.csv") if (rec_dir / "rep_features.csv").exists() else None
    return dict(frames=frames, joints=joints, names=names, fps=fps, angles=angles, reps=reps, summary=summ, missing=missing)


# ------------------------------------------------------------------------------------------------- body frame
def leg_length(j, names):
    ix = {n: i for i, n in enumerate(names)}
    out = []
    for s in ("L", "R"):
        h, k, a = j[:, ix[s + "Hip"]], j[:, ix[s + "Knee"]], j[:, ix[s + "Ankle"]]
        out.append(np.linalg.norm(h - k, axis=1) + np.linalg.norm(k - a, axis=1))
    return float(np.nanmedian(np.mean(out, axis=0)))


def body_axes(j, names):
    """-> (R rows = lateral (to the person's left), up, forward; swapped; tilt_deg).
    up = median ankle->hip direction over the STANDING frames (hips in the top 20 % of their height) - fix 1;
    forward = heel->toe of both feet made perpendicular to up; lateral = up x forward.
    tilt_deg = angle between this up and the nearest world axis (what the snapped version would have used)."""
    ix = {n: i for i, n in enumerate(names)}
    mid = lambda a, b: (j[:, ix[a]] + j[:, ix[b]]) / 2   # noqa: E731
    v = mid("RHip", "LHip") - mid("RAnkle", "LAnkle")
    med = np.nanmedian(v, axis=0)
    a = int(np.argmax(np.abs(med)))
    axis = np.zeros(3)
    axis[a] = np.sign(med[a]) or 1.0
    height = mid("RHip", "LHip") @ axis
    standing = height >= np.nanpercentile(height, 80)
    up = np.nanmedian(v[standing], axis=0)
    up /= np.linalg.norm(up)
    foot = np.nanmedian((j[:, ix["RBigToe"]] - j[:, ix["RHeel"]] + j[:, ix["LBigToe"]] - j[:, ix["LHeel"]]) / 2, axis=0)
    fwd = foot - np.dot(foot, up) * up
    fwd /= np.linalg.norm(fwd) + 1e-9
    lat = np.cross(up, fwd)
    swapped = bool(np.dot(np.nanmedian(j[:, ix["LHip"]] - j[:, ix["RHip"]], axis=0), lat) < 0)
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(up, axis), -1, 1))))
    return np.stack([lat, up, fwd]), swapped, tilt


def _swap_token(name):
    toks = ["R" if t == "L" else "L" if t == "R" else t for t in name.split("_")]
    out = "_".join(toks)
    if out == name and name[:1] in "LR" and name[1:2].isupper():
        out = ("R" if name[0] == "L" else "L") + name[1:]
    return out


def swap_permutation(names):
    ix = {n: i for i, n in enumerate(names)}
    return [ix.get(_swap_token(n), i) for i, n in enumerate(names)]


def extra_features(b, names):
    """knee offset from the hip-ankle line (+ = valgus / inward) L/R and heel height above the big toe L/R."""
    ix = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(b), 4))
    for k, s in enumerate(("L", "R")):
        hip, knee, ank = b[:, ix[s + "Hip"]], b[:, ix[s + "Knee"]], b[:, ix[s + "Ankle"]]
        d = ank - hip
        t = np.sum((knee - hip) * d, axis=1) / (np.sum(d * d, axis=1) + 1e-9)
        off = knee[:, 0] - (hip + t[:, None] * d)[:, 0]
        out[:, k] = -off if s == "L" else off
        out[:, 2 + k] = b[:, ix[s + "Heel"], 1] - b[:, ix[s + "BigToe"], 1]
    return out


# ------------------------------------------------------------------------------------------------- per rep
def resample(a, T):
    n = len(a)
    if n == T:
        return a.astype(np.float32)
    x_old, x_new = np.linspace(0, 1, n), np.linspace(0, 1, T)
    flat = a.reshape(n, -1)
    out = np.stack([np.interp(x_new, x_old, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape((T,) + a.shape[1:]).astype(np.float32)


def trim_rep(flex, s, e, frac=0.10):
    """Rep frames s..e (inclusive) -> (start, end, bottom) trimmed to the movement (fix 3).
    The standing level is taken separately before and after the bottom: people often rest between reps with the
    knees slightly bent (e.g. 15 deg) but stand fully straight only at the end, and one common level would leave
    that bent-knee rest in the rep."""
    f = flex[s:e + 1]
    b = int(np.nanargmax(f))
    hi = f[b]
    lo_pre, lo_post = np.nanmin(f[:b + 1]), np.nanmin(f[b:])
    before = np.where(f[:b] < lo_pre + frac * (hi - lo_pre))[0]
    after = np.where(f[b + 1:] < lo_post + frac * (hi - lo_post))[0]
    return (s + int(before[-1]) if len(before) else s), (s + b + 1 + int(after[0]) if len(after) else e), s + b


def recording_to_reps(rec_dir, data_cfg):
    """-> (list of rep dicts, info). Each rep: joints [T,J,6], frame [T,19], tempo [4], summary [S] or None,
    rep_no, start/end/bottom frames (trimmed), duration_s, missing (core), trunk_missing, flex [T] (knee flexion)."""
    T, trim = int(data_cfg["T"]), bool(data_cfg.get("trim", True))
    rec = load_recording(rec_dir)
    j, names, fps, angles, miss = rec["joints"], rec["names"], rec["fps"], rec["angles"], rec["missing"]
    leg = leg_length(j, names)
    R, swapped, tilt = body_axes(j, names)
    summ = rec["summary"]
    summ_cols = [c for c in summ.columns if c != "rep_id"] if summ is not None else []
    ps = list(range(len(summ_cols)))
    if swapped:
        pj = swap_permutation(names)
        j, miss = j[:, pj], miss[:, pj]
        angles = angles[:, swap_permutation(ANGLE_COLS)]
        ps = swap_permutation(summ_cols)
    ix = {n: i for i, n in enumerate(names)}
    if not data_cfg.get("body_frame", True):      # ablation: keep the camera-world axes (Pose2Sim's calibration board)
        R = np.eye(3)
    scale = leg if data_cfg.get("leg_norm", True) else 1.0   # ablation: metres instead of leg lengths
    b = ((j - j[:, ix["Hip"]][:, None, :]) @ R.T) / scale     # body frame, NaN where a joint is never seen
    vel = np.gradient(b, 1.0 / fps, axis=0)
    extras = extra_features(b, names)
    hip_up = (j[:, ix["Hip"]] @ R[1]) / scale
    flex = np.nanmean(angles[:, [ANGLE_COLS.index("knee_flexion_L"), ANGLE_COLS.index("knee_flexion_R")]], axis=1)
    core = [ix[n] for n in CORE_JOINTS]
    miss_core, miss_trunk = miss[:, core].any(axis=1), miss[:, ix[TRUNK_JOINT]]
    frames = rec["frames"]
    reps, dropped = [], []
    for _, r in rec["reps"].iterrows():
        s = int(np.searchsorted(frames, r["start_frame"]))
        e = int(np.searchsorted(frames, r["end_frame"], side="right")) - 1
        rep_no = int(r["rep_no"])
        if e - s < 8 or not np.isfinite(flex[s:e + 1]).any():
            dropped.append((rep_no, "too short / no knee flexion"))
            continue
        bot = s + int(np.nanargmax(flex[s:e + 1]))
        if trim:
            s, e, bot = trim_rep(flex, s, e, data_cfg.get("trim_frac", 0.10))
        dur = (e - s) / fps
        if e - s < 8 or dur < data_cfg["min_rep_s"] or dur > data_cfg["max_rep_s"]:
            dropped.append((rep_no, f"duration {dur:.2f} s"))
            continue
        descent = (bot - s) / fps
        tempo = np.array([dur, descent, dur - descent, descent / max(dur, 1e-6)], dtype=np.float32)
        sl = slice(s, e + 1)
        fr = np.concatenate([angles[sl], extras[sl], (hip_up[sl] - hip_up[s])[:, None]], axis=1)
        jt = np.nan_to_num(np.concatenate([b[sl], vel[sl]], axis=2))   # a never-seen joint sits at the hip, still
        summary = None
        if summ is not None:
            row = summ[summ["rep_id"] == rep_no]
            summary = row[summ_cols].to_numpy(dtype=np.float32)[0][ps] if len(row) == 1 else np.full(len(summ_cols), np.nan, np.float32)
        item = dict(joints=resample(jt, T), frame=resample(np.nan_to_num(fr), T), tempo=tempo, summary=summary,
                    rep_no=rep_no, start=int(frames[s]), end=int(frames[e]), bottom=int(frames[bot]), duration_s=float(dur),
                    missing=float(miss_core[sl].mean()), trunk_missing=float(miss_trunk[sl].mean()),
                    flex=resample(np.nan_to_num(flex[sl]), T))
        reps.append(item)
    info = dict(leg_length_m=leg, lr_swapped=swapped, up_tilt_deg=tilt, fps=fps, n_reps=len(reps), dropped=dropped,
                joint_names=names, summary_names=summ_cols, n_frames=int(len(frames)),
                missing_per_joint={n: round(float(miss[:, i].mean()), 3) for i, n in enumerate(names) if miss[:, i].mean() > 0.05})
    return reps, info
