"""
postprocess.py -- turns one finished run into what the website's angle viewer needs.

Input  (inside a run folder):  results/points3d.trc   Pose2Sim 3D joints (metres; frame 0 = 0.5 s after the clap)
                               results/advancedangles.csv   the pipeline's own angles (optional)
Output (same run folder):      angles_all.csv    one row per frame, one column per angle (see below)
                               angles_meta.json  names, labels, left/right pairing for the viewer
                               processed_videos/ synced front/side videos + 2D-pose overlay videos (moved here, one copy)

Angles written, for the LEFT and the RIGHT side (suffix _L / _R)
  3-point joint angles, the inside angle between the two segments (180 deg = straight limb), exactly what
  Kinovea's Angle tool shows:   knee (hip-knee-ankle), hip (shoulder-hip-knee), shoulder (hip-shoulder-elbow),
                                elbow (shoulder-elbow-wrist), ankle (knee-ankle-big toe)
    each in two versions:  _3d  = full 3D angle
                           _sag = projected on the side-view (sagittal) plane, which is what a side camera sees.
                                  The plane is found from the data: its normal is the direction in which that
                                  body side's joints move the least during the recording.
  trunk_lean_3d / trunk_lean_sag: angle between the hip-to-shoulder line and straight up (0 = upright). "Up" is the
                                  mean hip-over-ankle direction in frames with straight legs.
plus every column of the pipeline's own advancedangles.csv (knee_flexion_L/R, hip_angle_L/R, knee_valgus_L/R, ...).

Left/Right are the labels of the pose model (the person's own left and right).
Run by hand:  C:\\Pose2SimVenv\\Scripts\\python.exe pose2sim_web\\postprocess.py <run folder>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = {            # name -> (proximal, vertex, distal); the side letter R/L is put in front of each marker
    "knee": ("Hip", "Knee", "Ankle"),
    "hip": ("Shoulder", "Hip", "Knee"),
    "shoulder": ("Hip", "Shoulder", "Elbow"),
    "elbow": ("Shoulder", "Elbow", "Wrist"),
    "ankle": ("Knee", "Ankle", "BigToe"),
}
LABELS = {"knee": "Knee", "hip": "Hip (shoulder-hip-knee)", "shoulder": "Shoulder", "elbow": "Elbow", "ankle": "Ankle (knee-ankle-toe)"}

# the pipeline's own angles: id -> (label, has left/right pair)
PIPELINE = {
    "knee_flexion": ("Knee flexion", True), "hip_angle": ("Hip angle (neck-hip-knee)", True),
    "hip_flexion": ("Hip flexion", True), "knee_valgus": ("Knee valgus", True),
    "ankle_dorsiflexion_proxy": ("Ankle dorsiflexion proxy", True),
    "sagittal_flexion_trunk": ("Trunk flexion", False), "head_forward_angle": ("Head forward angle", False),
    "squat_depth_knee_deg": ("Squat depth (mean knee flexion)", False), "lumbar_curvature_proxy": ("Lumbar proxy", False),
}


def read_trc(path):
    """Returns (frames, times, {marker: (N,3) array}); Pose2Sim's exact (0,0,0) means 'not triangulated' -> NaN."""
    lines = Path(path).read_text().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    df = pd.read_csv(path, sep="\t", skiprows=5, header=None)
    a = df.iloc[:, : 2 + 3 * len(names)].to_numpy(dtype=float)
    xyz = a[:, 2:].reshape(len(a), len(names), 3)
    xyz[np.all(xyz == 0.0, axis=2)] = np.nan
    return a[:, 0].astype(int), a[:, 1], {n: xyz[:, i] for i, n in enumerate(names)}


def interior(p, v, d):
    a, b = p - v, d - v
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def lateral_axis(markers, names):
    """Direction of least movement of the given markers over the recording (normal of the sagittal plane)."""
    parts = []
    for n in names:
        if n in markers:
            p = markers[n][~np.isnan(markers[n]).any(axis=1)]
            if len(p):
                parts.append(p - p.mean(axis=0))      # each marker around its OWN mean: only its motion counts
    if not parts or sum(len(p) for p in parts) < 30:
        return None
    return np.linalg.svd(np.concatenate(parts, axis=0), full_matrices=False)[2][-1]


def flatten(x, lat):
    return x if lat is None else x - np.outer(x @ lat, lat)


def compute(trc_path):
    frames, times, m = read_trc(trc_path)
    out = {"frame": frames, "t": np.round(times, 4)}
    have = lambda *ns: all(n in m for n in ns)  # noqa: E731
    for side in ("L", "R"):
        names = [side + k for k in ("Hip", "Knee", "Ankle", "Shoulder", "Elbow")]
        lat = lateral_axis(m, names) if have(*names) else None
        for j, (p, v, d) in JOINTS.items():
            pn, vn, dn = side + p, side + v, side + d
            if not have(pn, vn, dn):
                continue
            out[f"{j}_{side}_3d"] = interior(m[pn], m[vn], m[dn])
            out[f"{j}_{side}_sag"] = (interior(flatten(m[pn], lat), flatten(m[vn], lat), flatten(m[dn], lat))
                                      if lat is not None else np.full(len(frames), np.nan))
    # trunk lean against straight up
    if have("RHip", "LHip", "RAnkle", "LAnkle", "RShoulder", "LShoulder"):
        hip, sho = (m["RHip"] + m["LHip"]) / 2, (m["RShoulder"] + m["LShoulder"]) / 2
        ank = (m["RAnkle"] + m["LAnkle"]) / 2
    elif have("RHip", "RAnkle", "RShoulder"):
        hip, sho, ank = m["RHip"], m["RShoulder"], m["RAnkle"]
    else:
        hip = sho = ank = None
    if hip is not None:
        straight = np.zeros(len(frames), bool)
        if "knee_R_3d" in out and "knee_L_3d" in out:
            straight = (out["knee_R_3d"] > 165) & (out["knee_L_3d"] > 165)
        up_src = (hip - ank)[straight] if straight.sum() > 5 else (hip - ank)
        up = np.nanmean(up_src, axis=0)
        trunk = sho - hip
        lat = lateral_axis(m, [s + k for s in "LR" for k in ("Hip", "Knee", "Ankle", "Shoulder")])
        out["trunk_lean_3d"] = interior(trunk, np.zeros_like(trunk), np.tile(up, (len(frames), 1)))
        out["trunk_lean_sag"] = interior(flatten(trunk, lat), np.zeros_like(trunk), np.tile(flatten(up[None], lat)[0], (len(frames), 1)))
    return pd.DataFrame(out)


def meta_for(columns):
    items = []
    for j, lab in LABELS.items():
        for v, vlab in (("sag", "side-view plane"), ("3d", "full 3D")):
            if f"{j}_L_{v}" in columns or f"{j}_R_{v}" in columns:
                items.append({"id": f"{j}_{{S}}_{v}", "key": j, "label": lab, "group": "joint", "variant": v, "pair": True,
                              "note": vlab})
    for v in ("sag", "3d"):
        if f"trunk_lean_{v}" in columns:
            items.append({"id": f"trunk_lean_{v}", "key": "trunk_lean", "label": "Trunk lean from vertical", "group": "joint",
                          "variant": v, "pair": False})
    for k, (lab, pair) in PIPELINE.items():
        if pair and (f"{k}_L" in columns or f"{k}_R" in columns):
            items.append({"id": f"{k}_{{S}}", "key": k, "label": lab, "group": "pipeline", "pair": True})
        elif not pair and k in columns:
            items.append({"id": k, "key": k, "label": lab, "group": "pipeline", "pair": False})
    return items


def build(run_dir):
    """Writes angles_all.csv and angles_meta.json into run_dir. Returns the number of frames."""
    run_dir = Path(run_dir)
    df = compute(run_dir / "results" / "points3d.trc")
    adv_path = run_dir / "results" / "advancedangles.csv"
    if adv_path.exists():
        adv = pd.read_csv(adv_path).drop(columns=["com_keypoints", "com_mesh", "com_final"], errors="ignore")
        df = df.merge(adv, on="frame", how="left")
    df.round(3).to_csv(run_dir / "angles_all.csv", index=False)
    (run_dir / "angles_meta.json").write_text(json.dumps({"items": meta_for(set(df.columns)), "n_frames": int(len(df))}), encoding="utf-8")
    return len(df)


def collect_videos(run_dir):
    """Moves the synced videos and the 2D-pose overlay videos out of the Pose2Sim project folder (one copy, saves disk)."""
    run_dir = Path(run_dir)
    trial = run_dir / "pose2sim_project" / "trial"
    dst = run_dir / "processed_videos"
    dst.mkdir(exist_ok=True)
    cams = {"side": "cam01_side", "front": "cam02_front"}
    copied = []
    for cam, stem in cams.items():
        src = trial / "videos" / f"{stem}.mp4"
        if src.exists():
            shutil.move(str(src), str(dst / f"{cam}_synced.mp4"))
            copied.append(f"{cam}_synced.mp4")
        for ov in sorted((trial / "pose").glob(f"{stem}*.mp4")):
            shutil.move(str(ov), str(dst / f"{cam}_pose_overlay.mp4"))
            copied.append(f"{cam}_pose_overlay.mp4")
            break
    return copied


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: postprocess.py <run folder>")
    n = build(sys.argv[1])
    print(f"wrote angles_all.csv and angles_meta.json ({n} frames)")
    print("videos copied:", collect_videos(sys.argv[1]))
