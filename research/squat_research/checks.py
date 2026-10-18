"""
checks.py - preprocessing sanity checks on the built dataset (no training). Each check returns pass/fail with numbers,
so notebook 01 can show that the preprocessing is correct before any model is trained.

    from squat_research.checks import run_checks
    table = run_checks(ds)          # ds = load_dataset(cfg, filters=False) or with filters
"""
import numpy as np
import pandas as pd

from .features import FRAME_FEATURE_NAMES


def _ix(ds):
    return {n: i for i, n in enumerate(ds["meta"]["joint_names"])}


def run_checks(ds):
    J, F = _ix(ds), {n: i for i, n in enumerate(FRAME_FEATURE_NAMES)}
    P, V = ds["joints"][..., :3], ds["joints"][..., 3:]
    flex = ds["flex"]
    T = flex.shape[1]
    rows = []

    def add(group, check, ok, value, expect):
        rows.append(dict(group=group, check=check, result="PASS" if ok else "FAIL", value=value, expected=expect))

    # ---- values
    add("values", "no NaN / inf in joints, frame features, tempo",
        bool(np.isfinite(ds["joints"]).all() and np.isfinite(ds["frame"]).all() and np.isfinite(ds["tempo"]).all()), "-", "all finite")
    mx = float(np.abs(P).max())
    add("values", "joint positions within 2.5 leg lengths of the hip", mx <= 2.5, f"max {mx:.2f}", "<= 2.5")
    vmax = float(np.percentile(np.linalg.norm(V, axis=-1).max(axis=(1, 2)), 99))
    add("values", "joint speed (99th percentile of the per-rep maximum)", vmax < 15, f"{vmax:.1f} leg lengths/s", "< 15")
    # ---- body frame (first frame = standing, after trimming)
    s = P[:, 0]
    knee_below_hip = np.mean((s[:, J["LKnee"], 1] < 0) & (s[:, J["RKnee"], 1] < 0))
    ankle_below_knee = np.mean((s[:, J["LAnkle"], 1] < s[:, J["LKnee"], 1]) & (s[:, J["RAnkle"], 1] < s[:, J["RKnee"], 1]))
    toe_front = np.mean((s[:, J["LBigToe"], 2] > s[:, J["LHeel"], 2]) & (s[:, J["RBigToe"], 2] > s[:, J["RHeel"], 2]))
    left_left = np.mean((s[:, J["LHip"], 0] > 0) & (s[:, J["RHip"], 0] < 0))
    for name, v in (("knees below the hip at the rep start", knee_below_hip), ("ankles below the knees", ankle_below_knee),
                    ("toes in front of the heels (forward axis)", toe_front), ("left hip on the left side (lateral axis)", left_left)):
        add("body frame", name, v >= 0.99, f"{100 * v:.1f} % of reps", ">= 99 %")
    leg = np.array([r["leg_length_m"] for r in ds["meta"]["recordings"].values()])
    add("body frame", "leg length per recording", bool(((leg > 0.6) & (leg < 1.05)).all()), f"{leg.min():.2f}-{leg.max():.2f} m", "0.60-1.05 m")
    # ---- trimming: no flat standing stretch left at the start / end, bottom inside the rep
    rng = flex.max(1) - flex.min(1)
    k = max(2, int(0.15 * T))
    flat_start = np.mean((flex[:, k] - flex[:, 0]) < 0.05 * rng)
    flat_end = np.mean((flex[:, -k - 1] - flex[:, -1]) < 0.05 * rng)
    add("trimming", "rep starts moving (knee flexion rises within the first 15 %)", flat_start < 0.03, f"{100 * flat_start:.1f} % flat", "< 3 %")
    add("trimming", "rep ends moving (knee flexion falls within the last 15 %)", flat_end < 0.03, f"{100 * flat_end:.1f} % flat", "< 3 %")
    bot = flex.argmax(1) / (T - 1)
    add("trimming", "bottom of the squat inside the rep (10-90 %)", np.mean((bot > 0.1) & (bot < 0.9)) > 0.98,
        f"median at {100 * np.median(bot):.0f} %", "10-90 %")
    add("trimming", "squat depth (max - min knee flexion) at least 30 deg", np.mean(rng >= 30) > 0.97, f"{100 * np.mean(rng >= 30):.1f} % of reps", "> 97 %")
    # ---- tempo
    t = ds["tempo"]
    add("tempo", "descent + ascent = duration", bool(np.allclose(t[:, 1] + t[:, 2], t[:, 0], atol=1e-3)), "-", "equal")
    add("tempo", "duration 0.5-10 s after trimming", bool(((t[:, 0] >= 0.5) & (t[:, 0] <= 10)).all()), f"{t[:, 0].min():.2f}-{t[:, 0].max():.2f} s", "0.5-10 s")
    # ---- class signatures (median per class must point the expected way)
    C = ds["classes"]
    y = ds["y"]
    fr = ds["frame"]
    knee_off = fr[:, :, [F["knee_offset_L"], F["knee_offset_R"]]].mean(2)
    sig = {
        "knee_valgus: largest inward knee offset": (knee_off.max(1), "knee_valgus", "max"),
        "knee_varus: largest outward knee offset": (knee_off.min(1), "knee_varus", "min"),
        "heel_lift: highest heel above the toe": (fr[:, :, [F["heel_height_L"], F["heel_height_R"]]].max(2).max(1), "heel_lift", "max"),
        "insufficient_depth: least knee flexion": (fr[:, :, [F["knee_flexion_L"], F["knee_flexion_R"]]].mean(2).max(1), "insufficient_depth", "min"),
        "insufficient_depth: smallest hip drop": (-fr[:, :, F["hip_height_rel"]].min(1), "insufficient_depth", "min"),
    }
    for name, (v, cls, how) in sig.items():
        med = pd.Series(v).groupby(np.array(C)[y]).median()
        ok = med.idxmax() == cls if how == "max" else med.idxmin() == cls
        add("class signatures", name, ok, f"{cls} {med[cls]:.3f}; next {med.drop(cls).max() if how == 'max' else med.drop(cls).min():.3f}", how)
    return pd.DataFrame(rows)
