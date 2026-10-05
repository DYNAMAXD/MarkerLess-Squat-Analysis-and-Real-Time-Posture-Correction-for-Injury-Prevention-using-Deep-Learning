"""
run_two_view.py -- two-camera preprocessing in one command.

    python two_view/run_two_view.py --front front.mp4 --side side.mp4 --name subj01_set1

Steps
  1. clap sync        -> offset between the two videos (no video editing)
  2. existing pipeline on EACH video (YOLO -> AlphaPose -> MotionBERT -> angles)
  3. per-rep fusion   -> rep_features.csv (front + side features, one row per rep)
  4. results          -> report.html, results.json + browser-playable videos (the website reads these)

Already ran the pipeline? Skip step 2:
    python two_view/run_two_view.py --front-csv A/extracted_data.csv --side-csv B/extracted_data.csv \
        --fps-front 29.97 --fps-side 29.97 --offset 1.84 --name test

The website (two_view/web_app.py) calls this script and shows its progress using the "[stage]" lines below.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rep_fusion import fuse_views            # noqa: E402
from two_view_report import build_report     # noqa: E402
from web_results import build_results, build_sync_only  # noqa: E402


def stage(name):
    print(f"[stage] {name}", flush=True)


def video_fps(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    cap.release()
    return fps


def run_pipeline(video, out_name, device, det_conf):
    """Runs the existing single-view pipeline. Returns (extracted_data.csv, overlay video)."""
    import keypoint_pipeline as kp  # lazy: needs the full model environment
    kp.set_device(device)
    _, overlay, _ = kp.process_video(str(video), use_3d=True, use_mesh=False, det_conf=det_conf, output_name=out_name)
    csv_path = kp.OUTPUT_DIR / out_name / "extracted_data.csv"
    if not csv_path.exists():
        raise RuntimeError(f"No extracted_data.csv for {video} (3D lift failed? see log above).")
    return csv_path, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front"), ap.add_argument("--side")
    ap.add_argument("--front-csv"), ap.add_argument("--side-csv")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fps-front", type=float), ap.add_argument("--fps-side", type=float)
    ap.add_argument("--offset", type=float, help="clap offset in seconds (side minus front); skips audio sync")
    ap.add_argument("--no-sync", action="store_true", help="match reps by order, no clap sync")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--sync-only", action="store_true", help="only measure the clap offset, skip the pose models")
    a = ap.parse_args()

    out_dir = ROOT / "outputs" / "two_view" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"  # optional subject/label notes written by the website
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("name", a.name)

    offset, sync_info = a.offset, None
    if offset is None and not a.no_sync and a.front and a.side:
        stage("sync")
        from sync_clap import sync_pair
        try:
            sync_info = sync_pair(a.front, a.side)
            offset = sync_info["offset_s"]
            print(f"[sync] {sync_info}")
            if not sync_info["reliable"]:
                print("[sync] WARNING: weak clap peak -- verify the offset on the report's depth overlay.")
        except Exception as e:
            print(f"[sync] failed ({e}); falling back to matching reps by order.")

    if a.sync_only:
        (out_dir / "sync.json").write_text(json.dumps({"offset_s": offset, "sync": sync_info}, indent=2))
        build_sync_only(offset, sync_info, out_dir, meta)
        if offset is None:
            print("[sync] no offset could be measured.")
        else:
            print(f"[sync] DONE. offset = {offset:+.3f} s (side minus front). Details saved in {out_dir / 'sync.json'}")
        stage("done")
        return

    f_overlay = s_overlay = None
    if a.front_csv and a.side_csv:
        f_csv, s_csv = Path(a.front_csv), Path(a.side_csv)
    else:
        if not (a.front and a.side):
            ap.error("give --front and --side videos, or --front-csv and --side-csv")
        stage("front")
        f_csv, f_overlay = run_pipeline(a.front, f"{a.name}_front", a.device, a.det_conf)
        stage("side")
        s_csv, s_overlay = run_pipeline(a.side, f"{a.name}_side", a.device, a.det_conf)

    fps_f = a.fps_front or (video_fps(a.front) if a.front else 29.97)
    fps_s = a.fps_side or (video_fps(a.side) if a.side else 29.97)

    stage("fusion")
    fused = fuse_views(f_csv, s_csv, fps_f, fps_s, offset)
    fused["table"].to_csv(out_dir / "rep_features.csv", index=False)
    report = build_report(fused, fps_f, fps_s, offset, out_dir / "report.html", title=f"Two-view squat analysis: {a.name}")
    (out_dir / "sync.json").write_text(json.dumps({"offset_s": offset, "sync": sync_info,
                                                   "fps_front": fps_f, "fps_side": fps_s}, indent=2))
    stage("packaging")
    build_results(fused, fps_f, fps_s, offset, sync_info, out_dir, meta,
                  front_overlay=f_overlay, side_overlay=s_overlay, front_csv=f_csv, side_csv=s_csv)
    print(f"\nreps: front={len(fused['front_reps'])} side={len(fused['side_reps'])} matched={len(fused['pairs'])}")
    print(f"rep table : {out_dir / 'rep_features.csv'}")
    print(f"report    : {report}")
    stage("done")


if __name__ == "__main__":
    main()
