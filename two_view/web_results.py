"""
web_results.py -- packages one analysis for the website.

Writes into the analysis folder (outputs/two_view/<name>/):
    results.json       all curves, reps, per-rep table, sync info (the website reads this)
    front.mp4/side.mp4 browser-playable copies of the overlay videos (H.264, 540p)
    front_angles.csv / side_angles.csv   copies of each view's extracted_data.csv

Why the re-encode: the pipeline writes 'mp4v' overlay videos, which most browsers cannot play.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from rep_fusion import ANGLE_COLS


def make_proxy(src, dst, height=540):
    """Re-encode `src` to a small H.264 file a browser can play. Returns True on success."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("[web] imageio-ffmpeg missing -> cannot make browser-playable videos (pip install imageio-ffmpeg)")
        return False
    cmd = [exe, "-v", "error", "-y", "-i", str(src), "-vf", f"scale=-2:{height}", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not Path(dst).exists():
        print(f"[web] video re-encode failed: {res.stderr.strip()[-300:]}")
        return False
    return True


def _clean(arr):
    return [None if not np.isfinite(v) else round(float(v), 2) for v in arr]


def _view_block(df, fps):
    block = {"t": [round(f / fps, 3) for f in df["frame"]], "n_frames": int(len(df))}
    block["m"] = {c: _clean(df[c].to_numpy(dtype=float)) for c in ANGLE_COLS}
    return block


def build_results(fused, fps_front, fps_side, offset_s, sync_info, out_dir, meta=None,
                  front_overlay=None, side_overlay=None, front_csv=None, side_csv=None):
    out_dir = Path(out_dir)
    table = fused["table"]
    reps = []
    for n, (i, j) in enumerate(fused["pairs"], start=1):
        fr, sr = fused["front_reps"][i], fused["side_reps"][j]
        reps.append({"id": n,
                     "front": [round(fr[k] / fps_front, 3) for k in ("start", "bottom", "end")],
                     "side": [round(sr[k] / fps_side, 3) for k in ("start", "bottom", "end")]})
    rows = [{k: (None if (isinstance(v, float) and not np.isfinite(v)) else (round(v, 3) if isinstance(v, float) else v))
             for k, v in r.items()} for r in table.to_dict("records")]

    media = {}
    for view, src in (("front", front_overlay), ("side", side_overlay)):
        if src and Path(src).exists() and make_proxy(src, out_dir / f"{view}.mp4"):
            media[view] = f"{view}.mp4"
    downloads = {"rep_features.csv": "Per-rep table (CSV)"}
    for view, src in (("front", front_csv), ("side", side_csv)):
        if src and Path(src).exists():
            shutil.copyfile(src, out_dir / f"{view}_angles.csv")
            downloads[f"{view}_angles.csv"] = f"{view.title()} angles per frame (CSV)"
    if (out_dir / "report.html").exists():
        downloads["report.html"] = "Static report (HTML)"

    results = {
        "sync_only": False, "meta": meta or {},
        "offset_s": offset_s, "sync": sync_info,
        "fps": {"front": fps_front, "side": fps_side},
        "front": _view_block(fused["front"], fps_front), "side": _view_block(fused["side"], fps_side),
        "reps": reps, "table": rows, "media": media, "downloads": downloads,
        "counts": {"front_reps": len(fused["front_reps"]), "side_reps": len(fused["side_reps"]), "matched": len(reps)},
    }
    (out_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    return results


def build_sync_only(offset_s, sync_info, out_dir, meta=None):
    out_dir = Path(out_dir)
    (out_dir / "results.json").write_text(json.dumps(
        {"sync_only": True, "meta": meta or {}, "offset_s": offset_s, "sync": sync_info}), encoding="utf-8")
