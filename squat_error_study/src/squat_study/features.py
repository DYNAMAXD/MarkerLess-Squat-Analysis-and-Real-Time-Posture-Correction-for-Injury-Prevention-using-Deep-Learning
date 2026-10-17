"""
features.py   (MODULE - imported by scripts/build_dataset.py)
==============================================================
Turns one recording folder (points3d.trc, advancedangles.csv, reps.csv, results.json,
rep_features.csv) into per-rep arrays that every model can use.

Pipeline per recording
----------------------
 1. load        read the 3D joints (metres), the frame-wise angles and the rep boundaries
 2. body frame  express the joints in the person's own axes (lateral / up / forward),
                hip-centred and divided by leg length, so people of different size and
                camera orientation become comparable
 3. per rep     cut [start, end], compute velocity, add 5 extra frame features,
                resample everything to T frames (linear interpolation)
 4. tempo       4 numbers that resampling would otherwise hide
 5. summary     the per-rep summary row from rep_features.csv (for the tree baselines)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. Constants
# ----------------------------------------------------------------------------
# The 14 frame-wise angles taken from advancedangles.csv (the com_* columns are not used).
ANGLE_COLS = [
    "knee_valgus_L", "knee_valgus_R", "head_forward_angle", "squat_depth_knee_deg",
    "sagittal_flexion_trunk", "hip_angle_L", "hip_angle_R", "hip_flexion_L", "hip_flexion_R",
    "knee_flexion_L", "knee_flexion_R", "ankle_dorsiflexion_proxy_L",
    "ankle_dorsiflexion_proxy_R", "lumbar_curvature_proxy",
]
# 5 extra frame features computed from the joints in the body frame.
EXTRA_COLS = ["knee_valgus_body_L", "knee_valgus_body_R", "heel_height_L", "heel_height_R",
              "hip_height_rel"]
FRAME_FEATURE_NAMES = ANGLE_COLS + EXTRA_COLS            # F = 19
TEMPO_NAMES = ["rep_dur_s", "descent_s", "ascent_s", "bottom_frac"]
# joints that must be present for a rep to be trustworthy (used for the missing-data fraction)
CORE_JOINTS = ["Hip", "RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel",
               "RBigToe", "LBigToe"]
# the trunk angles of the pipeline need the Neck; where it is missing they are garbage (see README)
TRUNK_JOINT = "Neck"
JOINT_CHANNELS = ["pos_lat", "pos_up", "pos_fwd", "vel_lat", "vel_up", "vel_fwd"]


# ----------------------------------------------------------------------------
# 2. Loading a recording
# ----------------------------------------------------------------------------
def read_trc(path):
    """Returns (frames[N], joints[N,J,3] in metres, names[J], fps)."""
    lines = Path(path).read_text().splitlines()
    fps = float(lines[2].split("\t")[0])
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    j = len(names)
    df = pd.read_csv(path, sep="\t", skiprows=5, header=None)
    arr = df.iloc[:, : 2 + 3 * j].to_numpy(dtype=float)
    frames = arr[:, 0].astype(int)
    joints = arr[:, 2:].reshape(len(arr), j, 3)
    # Pose2Sim writes a joint it could not triangulate as exactly (0,0,0): that is MISSING, not a position
    joints[np.all(joints == 0.0, axis=2)] = np.nan
    return frames, joints, names, fps


def _interp_nan(a):
    """Linear interpolation of NaNs along axis 0 (works on any trailing shape)."""
    flat = pd.DataFrame(a.reshape(len(a), -1)).interpolate(limit_direction="both")
    return flat.to_numpy().reshape(a.shape)


def load_recording(rec_dir):
    rec_dir = Path(rec_dir)
    frames, joints, names, fps = read_trc(rec_dir / "points3d.trc")
    ix0 = {n: i for i, n in enumerate(names)}
    hip_mid = (joints[:, ix0["RHip"]] + joints[:, ix0["LHip"]]) / 2
    hip_gap = np.isnan(joints[:, ix0["Hip"]]).any(axis=1) & ~np.isnan(hip_mid).any(axis=1)
    joints[hip_gap, ix0["Hip"]] = hip_mid[hip_gap]          # Hip missing but both hip joints present
    missing = np.isnan(joints).any(axis=2)                  # [N, J] True where the joint was missing
    joints = np.nan_to_num(_interp_nan(joints))             # a joint missing in ALL frames becomes 0

    adv = pd.read_csv(rec_dir / "advancedangles.csv").set_index("frame")
    adv = adv.reindex(frames)[ANGLE_COLS]
    angles = _interp_nan(adv.to_numpy(dtype=float))

    reps = pd.read_csv(rec_dir / "reps.csv")
    res = json.loads((rec_dir / "results.json").read_text())
    bottoms = {r["id"]: r["t"][1] for r in res.get("reps", []) if len(r.get("t", [])) == 3}

    summ = pd.read_csv(rec_dir / "rep_features.csv")
    return dict(frames=frames, joints=joints, names=names, fps=fps, angles=angles,
                reps=reps, bottoms=bottoms, summary=summ, missing=missing)


# ----------------------------------------------------------------------------
# 3. Body frame
# ----------------------------------------------------------------------------
def leg_length(j, names):
    """Median over frames of the mean L/R (hip-knee + knee-ankle) distance, in metres."""
    ix = {n: i for i, n in enumerate(names)}
    out = []
    for s in ("L", "R"):
        h, k, a = j[:, ix[s + "Hip"]], j[:, ix[s + "Knee"]], j[:, ix[s + "Ankle"]]
        out.append(np.linalg.norm(h - k, axis=1) + np.linalg.norm(k - a, axis=1))
    return float(np.median(np.mean(out, axis=0)))


def body_axes(j, names):
    """
    Per-recording body axes, returned as a (3,3) matrix R whose ROWS are
    lateral (right -> left), up, forward, in world coordinates, plus swapped = True when the
    pipeline's Left/Right labels are the wrong way round for this recording.

    * up      = mean(hips) - mean(ankles)        (does not need the Neck, which is often missing)
    * forward = mean(toe - heel) of both feet    (the feet point forward in a squat)
    * lateral = up x forward                     (right-handed: forward x lateral = up)
    The vertical world axis is picked from the data, so the Pose2Sim axis convention does not matter.
    """
    ix = {n: i for i, n in enumerate(names)}
    mid = lambda a, b: (j[:, ix[a]] + j[:, ix[b]]) / 2
    trunk = np.median(mid("RHip", "LHip") - mid("RAnkle", "LAnkle"), axis=0)
    a = int(np.argmax(np.abs(trunk)))
    up = np.zeros(3)
    up[a] = np.sign(trunk[a]) or 1.0

    foot = np.median((j[:, ix["RBigToe"]] - j[:, ix["RHeel"]] + j[:, ix["LBigToe"]] - j[:, ix["LHeel"]]) / 2, axis=0)
    fwd = foot - np.dot(foot, up) * up
    fwd /= np.linalg.norm(fwd) + 1e-9
    lat = np.cross(up, fwd)                              # points to the person's LEFT
    hips = np.median(j[:, ix["LHip"]] - j[:, ix["RHip"]], axis=0)
    swapped = bool(np.dot(hips, lat) < 0)                # the "L" hip is on the person's right
    return np.stack([lat, up, fwd]), swapped


def _swap_token(name):
    """knee_valgus_L_bottom <-> knee_valgus_R_bottom ; RKnee <-> LKnee ; hip_flexion_L <-> hip_flexion_R"""
    toks = name.split("_")
    toks = ["R" if t == "L" else "L" if t == "R" else t for t in toks]
    out = "_".join(toks)
    if out == name and name[:1] in "LR" and name[1:2].isupper():       # joint names such as LKnee / RHeel
        out = ("R" if name[0] == "L" else "L") + name[1:]
    return out


def swap_permutation(names):
    """Index permutation that exchanges every left/right pair in `names`."""
    ix = {n: i for i, n in enumerate(names)}
    return [ix.get(_swap_token(n), i) for i, n in enumerate(names)]


def to_body_frame(j, names, R, leg):
    """World joints [N,J,3] -> hip-centred, leg-length-normalised body-frame joints [N,J,3]."""
    ix = {n: i for i, n in enumerate(names)}
    rel = j - j[:, ix["Hip"]][:, None, :]
    return (rel @ R.T) / leg                     # columns: lateral, up, forward


def extra_features(b, j_abs_up, names, rep_slice):
    """The 5 extra frame features, for ALL frames. b = body-frame joints [N,J,3]."""
    ix = {n: i for i, n in enumerate(names)}
    out = np.zeros((len(b), 5))
    for k, s in enumerate(("L", "R")):
        hip, knee, ank = b[:, ix[s + "Hip"]], b[:, ix[s + "Knee"]], b[:, ix[s + "Ankle"]]
        d = ank - hip
        t = np.sum((knee - hip) * d, axis=1) / (np.sum(d * d, axis=1) + 1e-9)
        line = hip + t[:, None] * d
        off = knee[:, 0] - line[:, 0]            # lateral offset of the knee from the hip-ankle line
        # + = knee collapses inward (valgus). Left leg: inward = towards -lateral, right leg: towards +lateral.
        out[:, k] = -off if s == "L" else off
        out[:, 2 + k] = b[:, ix[s + "Heel"], 1] - b[:, ix[s + "BigToe"], 1]
    return out


# ----------------------------------------------------------------------------
# 4. Resampling helper
# ----------------------------------------------------------------------------
def resample(a, T):
    """Resample along axis 0 to T frames by linear interpolation. a: [n, ...]."""
    n = len(a)
    if n == T:
        return a.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, T)
    flat = a.reshape(n, -1)
    out = np.stack([np.interp(x_new, x_old, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    return out.reshape((T,) + a.shape[1:]).astype(np.float32)


# ----------------------------------------------------------------------------
# 5. One recording -> per-rep arrays
# ----------------------------------------------------------------------------
def recording_to_reps(rec_dir, T=64, min_rep_s=1.0, max_rep_s=12.0):
    """
    Returns (list_of_rep_dicts, info_dict). Each rep dict has:
      joints  [T, J, 6]  position (3) + velocity (3) in the body frame
      frame   [T, F]     F = 19 frame features (14 angles + 5 extras)
      tempo   [4]        rep_dur_s, descent_s, ascent_s, bottom_frac
      summary [S]        the rep_features.csv row (summary stats) - for the tree models
      rep_no  int
      missing float   fraction of the rep frames with a missing lower-body joint
      trunk_missing float   fraction of the rep frames with the Neck missing (trunk angles unreliable)
    """
    rec = load_recording(rec_dir)
    j, names, fps = rec["joints"], rec["names"], rec["fps"]
    leg = leg_length(j, names)
    R, swapped = body_axes(j, names)
    summ = rec["summary"]
    summ_cols = [c for c in summ.columns if c != "rep_id"]
    angles = rec["angles"]
    miss_all = rec["missing"]
    if swapped:                                           # put anatomical left/right back where they belong
        pj = swap_permutation(names)
        j, miss_all = j[:, pj], miss_all[:, pj]
        angles = angles[:, swap_permutation(ANGLE_COLS)]
        ps = swap_permutation(summ_cols)
    else:
        ps = list(range(len(summ_cols)))
    b = to_body_frame(j, names, R, leg)
    core_ix = [names.index(n) for n in CORE_JOINTS]
    miss_core = miss_all[:, core_ix].any(axis=1)         # [N] frame has a missing lower-body joint
    miss_trunk = miss_all[:, names.index(TRUNK_JOINT)]   # [N] Neck missing

    # velocity in leg lengths per second, from the real time step
    vel = np.gradient(b, 1.0 / fps, axis=0)

    extras = extra_features(b, None, names, None)
    ix = {n: i for i, n in enumerate(names)}
    hip_up_world = (j[:, ix["Hip"]] @ R[1]) / leg          # world hip height, in leg lengths

    frames = rec["frames"]
    reps, dropped = [], 0
    for _, r in rec["reps"].iterrows():
        s = int(np.searchsorted(frames, r["start_frame"]))
        e = int(np.searchsorted(frames, r["end_frame"], side="right"))
        dur = (e - s - 1) / fps
        if e - s < 8 or dur < min_rep_s or dur > max_rep_s:
            dropped += 1
            continue
        rep_no = int(r["rep_no"])

        # bottom time: from results.json when available, else the lowest hip
        t_bottom = rec["bottoms"].get(rep_no)
        if t_bottom is not None:
            t0 = frames[s] / fps
            descent = float(np.clip(t_bottom - t0, 0.1, dur - 0.1))
        else:
            descent = float((np.argmin(hip_up_world[s:e])) / fps)
        ascent = dur - descent
        tempo = np.array([dur, descent, ascent, descent / max(dur, 1e-6)], dtype=np.float32)

        hip_rel = (hip_up_world[s:e] - hip_up_world[s])[:, None]
        fr = np.concatenate([angles[s:e], extras[s:e, :4], hip_rel], axis=1)
        jt = np.concatenate([b[s:e], vel[s:e]], axis=2)            # [n, J, 6]

        srow = summ[summ["rep_id"] == rep_no]
        if len(srow) != 1:
            dropped += 1
            continue
        summary = srow[summ_cols].to_numpy(dtype=np.float32)[0][ps]

        item = dict(joints=resample(jt, T), frame=resample(fr, T), tempo=tempo,
                    summary=summary, rep_no=rep_no, missing=float(miss_core[s:e].mean()),
                    trunk_missing=float(miss_trunk[s:e].mean()))
        if np.isnan(item["joints"]).any() or np.isnan(item["frame"]).any():
            dropped += 1
            continue
        reps.append(item)

    miss_per_joint = {n: round(float(miss_all[:, i].mean()), 3) for i, n in enumerate(names)}
    info = dict(leg_length_m=leg, lr_swapped=swapped, fps=fps, n_reps=len(reps),
                n_dropped=dropped, joint_names=names, summary_names=summ_cols,
                missing_per_joint={k: v for k, v in miss_per_joint.items() if v > 0.05})
    return reps, info
