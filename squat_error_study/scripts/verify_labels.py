"""
verify_labels.py   (SCRIPT - step 1 of the ground-truth check)
=============================================================
Measures, for every rep, the one quantity that defines its recording's error, and flags reps
whose measurement does not look like the error. It never looks at model results.

RULE (fixed BEFORE looking at any numbers, do not tune it afterwards)
---------------------------------------------------------------------
Per error class there is ONE measure, taken from the body-frame data already in reps_T64.npz:

  insufficient_depth  depth  = -min(hip_height_rel)           (leg lengths; smaller = shallower)  -> error if LOW
  forward_lean        lean   = atan2(Neck_fwd, Neck_up) at the bottom frame, degrees               -> error if HIGH
  knee_valgus         valg   = mean(knee_valgus_body_L/R) at the bottom frame                      -> error if HIGH
  knee_varus          varus  = -valg                                                               -> error if HIGH
  heel_lift           heel   = max over the rep of mean(heel_height_L/R)                           -> error if HIGH

Reference = the pooled reps of the 'correct' recordings (same filters as training).
A rep "shows the error" when its measure is beyond the reference 75th percentile
(lower than the 25th for depth). Otherwise it is flagged "weak".
A recording is flagged when fewer than 30 % of its reps show the error.
(Persons differ in size/flexibility, so the per-person comparison to that person's own
'correct' recording is reported next to it, as extra information - it is not part of the rule.)

Run from squat_error_study/ :
    C:\\squat_venv\\Scripts\\python.exe scripts\\verify_labels.py
Writes results/label_check/rep_measures.csv and results/label_check/recording_flags.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NPZ = Path("data/processed/reps_T64.npz")
OUT = Path("results/label_check")
MAX_TRUNK_MISSING, MAX_MISSING_CORE = 0.5, 0.25      # same as configs/default.yaml
SHOW_Q, REC_MIN_FRAC = 0.75, 0.30                    # the fixed rule
ERROR_CLASSES = ["insufficient_depth", "forward_lean", "knee_valgus", "knee_varus", "heel_lift"]
MEASURE_OF = dict(insufficient_depth="depth", forward_lean="lean", knee_valgus="valg",
                  knee_varus="varus", heel_lift="heel")


def main():
    d = np.load(NPZ, allow_pickle=False)
    meta = json.load(open(NPZ.with_suffix(".json")))
    jn, fn = meta["joint_names"], meta["frame_feature_names"]
    keep = (d["missing"] <= MAX_MISSING_CORE) & (d["trunk_missing"] <= MAX_TRUNK_MISSING)
    J, F = d["joints"][keep], d["frame"][keep]
    n = len(J)
    bot = F[:, :, fn.index("hip_height_rel")].argmin(axis=1)           # bottom = lowest hip
    r = np.arange(n)
    neck = J[r, bot, jn.index("Neck"), :3]                              # lat, up, fwd (hip-centred)
    valg = F[r, bot, fn.index("knee_valgus_body_L")] / 2 + F[r, bot, fn.index("knee_valgus_body_R")] / 2
    heel = ((F[:, :, fn.index("heel_height_L")] + F[:, :, fn.index("heel_height_R")]) / 2).max(axis=1)
    df = pd.DataFrame(dict(
        recording=d["recording"][keep], person=d["person"][keep], rep_no=d["rep_no"][keep],
        label=d["label"][keep],
        depth=-F[:, :, fn.index("hip_height_rel")].min(axis=1),
        lean=np.degrees(np.arctan2(neck[:, 2], neck[:, 1])),
        valg=valg, varus=-valg, heel=heel))

    ref = df[df.label == "correct"]
    shows = np.zeros(n, bool)
    thr = {}
    for cls in ERROR_CLASSES:
        m = MEASURE_OF[cls]
        sel = (df.label == cls).to_numpy()
        if m == "depth":
            thr[cls] = ref[m].quantile(1 - SHOW_Q)
            shows[sel] = df.loc[sel, m] < thr[cls]
        else:
            thr[cls] = ref[m].quantile(SHOW_Q)
            shows[sel] = df.loc[sel, m] > thr[cls]
    df["shows_error"] = np.where(df.label == "correct", np.nan, shows.astype(float))

    # per-person comparison to that person's own correct recording (information only)
    own = ref.groupby("person")[["depth", "lean", "valg", "heel"]].median().add_suffix("_own_correct")
    df = df.join(own, on="person")

    OUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(OUT / "rep_measures.csv", index=False)

    err = df[df.label != "correct"]
    rec = err.groupby(["recording", "person", "label"]).agg(
        reps=("rep_no", "size"), frac_showing=("shows_error", "mean")).reset_index()
    rec["flag_recording"] = rec.frac_showing < REC_MIN_FRAC
    rec.round(3).to_csv(OUT / "recording_flags.csv", index=False)

    print("Thresholds (from pooled correct reps):", {k: round(float(v), 3) for k, v in thr.items()})
    print("\nMedian of each measure per class (correct row is the reference):")
    print(df.groupby("label")[["depth", "lean", "valg", "varus", "heel"]].median().round(3))
    print("\nShare of reps showing the error, per class:")
    print(err.groupby("label").shows_error.mean().round(3))
    print(f"\nFlagged recordings (<{REC_MIN_FRAC:.0%} reps show the error): {int(rec.flag_recording.sum())} of {len(rec)}")
    print(rec[rec.flag_recording].sort_values(["label", "recording"]).to_string(index=False))


if __name__ == "__main__":
    main()
