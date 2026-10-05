"""
pair_videos.py -- match front-camera and side-camera clips of the same take, and write a labelling sheet.

Both cameras name files by their own clock (DJI_YYYYMMDDHHMMSS_NNNN_D.MP4). The two clocks differ by a
roughly constant amount, so we (1) estimate that constant from clips whose durations agree, then
(2) pair each front clip with the side clip it overlaps most in time, then (3) optionally measure the
clap offset for each pair.

    python two_view/pair_videos.py --front-dir <dir>/front --side-dir <dir>/side --out pairs.csv --clap

Open pairs.csv in Excel and fill the `subject` and `label` columns (e.g. correct / knees_in / forward_lean).
Rows with a non-empty `check` column need a manual look before you trust them.
"""
import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path
from statistics import median

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

_NAME = re.compile(r"_(\d{14})_")


def clip_info(path):
    m = _NAME.search(path.name)
    start = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return {"path": path, "start": start, "dur": (n / fps) if fps else 0.0, "fps": fps}


def overlap_s(f, s, clock_offset):
    """Seconds the two clips overlap, given side_clock = front_clock - clock_offset."""
    f0 = f["start"]
    f1 = f0 + dt.timedelta(seconds=f["dur"])
    s0 = s["start"] + dt.timedelta(seconds=clock_offset)
    s1 = s0 + dt.timedelta(seconds=s["dur"])
    return (min(f1, s1) - max(f0, s0)).total_seconds()


def estimate_clock_offset(fronts, sides, max_dur_diff=3.0, min_dur=20.0):
    cands = [(f["start"] - s["start"]).total_seconds()
             for f in fronts for s in sides
             if f["dur"] >= min_dur and abs(f["dur"] - s["dur"]) <= max_dur_diff]
    if not cands:
        raise RuntimeError("No front/side clips with matching durations to estimate the camera clock difference.")
    return median(cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front-dir", required=True)
    ap.add_argument("--side-dir", required=True)
    ap.add_argument("--out", default="pairs.csv")
    ap.add_argument("--clap", action="store_true", help="also measure the clap offset for each pair")
    a = ap.parse_args()

    ext = {".mp4", ".mov"}
    fronts = [clip_info(p) for p in sorted(Path(a.front_dir).iterdir()) if p.suffix.lower() in ext]
    sides = [clip_info(p) for p in sorted(Path(a.side_dir).iterdir()) if p.suffix.lower() in ext]
    clock_offset = estimate_clock_offset(fronts, sides)
    print(f"camera clock difference (front - side): {clock_offset:.0f} s")

    rows, used = [], set()
    for k, f in enumerate(fronts, start=1):
        scored = sorted(((overlap_s(f, s, clock_offset), j) for j, s in enumerate(sides)), reverse=True)
        ov, j = scored[0]
        s = sides[j]
        check = []
        if ov <= 0:
            check.append("no overlapping side clip")
        if j in used:
            check.append("side clip already used by another pair")
        if abs(f["dur"] - s["dur"]) > 5:
            check.append(f"durations differ by {abs(f['dur'] - s['dur']):.0f}s (retake or early stop?)")
        if s["fps"] and f["fps"] and abs(s["fps"] - f["fps"]) > 1:
            check.append(f"fps differ ({f['fps']:.0f} vs {s['fps']:.0f})")
        used.add(j)
        row = {"pair_id": k, "front_file": f["path"].name, "side_file": s["path"].name,
               "front_dur_s": round(f["dur"], 1), "side_dur_s": round(s["dur"], 1), "overlap_s": round(ov, 1),
               "clap_offset_s": "", "clap_conf_front": "", "clap_conf_side": "",
               "check": "; ".join(check), "subject": "", "label": ""}
        if a.clap:
            from sync_clap import sync_pair
            try:
                r = sync_pair(f["path"], s["path"])
                row.update(clap_offset_s=r["offset_s"], clap_conf_front=r["confidence_front"],
                           clap_conf_side=r["confidence_side"])
                if not r["reliable"]:
                    row["check"] = (row["check"] + "; weak clap").lstrip("; ")
            except Exception as e:
                row["check"] = (row["check"] + f"; clap failed: {e}").lstrip("; ")
        rows.append(row)
        print(f"{k:>2} {row['front_file'][-18:-6]} <-> {row['side_file'][-18:-6]}  dur {row['front_dur_s']}/{row['side_dur_s']}  "
              f"clap {row['clap_offset_s']}  {row['check']}")

    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
