"""
run_pose2sim.py -- calibrated two-camera (side + front) 3D squat analysis with Pose2Sim. CPU, or GPU with --device.

Recording modes
    Mode 1: intrinsic videos + full-length videos [board -> clap -> squats]       (default)
    Mode 2: intrinsic videos + extrinsic videos [board + one clap] (--mode extrinsic, once) +
            squat videos [clap -> squats -> clap]  (--calib-from <that calibration> --squat-claps 2)
    batch_preprocess.py runs a whole dataset (mode 1: intrinsics once via --mode intrinsic, then --intr-from
    per person; mode 2: --calib-from per person).

Intrinsics only (no sync, no board search):
    python two_view/run_pose2sim.py --mode intrinsic --name intr01 --intr-side s.mp4 --intr-front f.mp4

Full analysis:
    python two_view/run_pose2sim.py --name subj01 --side side_full.mp4 --front front_full.mp4 \
        --intr-side side_intrinsic.mp4 --intr-front front_intrinsic.mp4 \
        --intr-corners 6,7 --intr-square 60 --ext-corners 6,7 --ext-square 60

Extrinsic calibration only (board + one clap anywhere; intrinsics from videos or from an earlier calibration):
    python two_view/run_pose2sim.py --mode extrinsic --name calib01 --side side.mp4 --front front.mp4 \
        --intr-from outputs/two_view/<earlier run>/calibration.toml

Reuse a complete calibration (no board needed in the squat videos), squats between two claps:
    python two_view/run_pose2sim.py --name subj02 --side s.mp4 --front f.mp4 --squat-claps 2 \
        --calib-from outputs/two_view/calib01/calibration.toml

Mode 1 full-length videos: [checkerboard held still, seen by BOTH cameras] -> [clap] -> [squats]

Steps (each prints a "[stage] <name>" line; the website shows them as a progress stepper)
  setup          Pose2Sim project folder outputs/pose2sim/<name>/ + Config.toml
  sync           clap in both full-length videos -> clap frame per camera + offset (front - side)
                 (--squat-claps 2: start and end clap in the whole videos + clock drift between them)
  calibration    intrinsics: Pose2Sim.calibration() on the intrinsic videos (or read from an earlier calibration)
                 extrinsics: walk back frame by frame from the clap until both cameras see the board, then
                 solvePnP (p2s_extrinsics.py)
  extract        cut both videos to start 0.5 s after the clap (and end 0.5 s before the end clap)
                 -> frame N is the same instant in both
  pose           Pose2Sim.poseEstimation()   RTMPose 2D keypoints (HALPE_26)
  triangulation  Pose2Sim.personAssociation() + triangulation() + filtering()   -> 3D .trc
  angles         angles.csv (3D joints) + advancedangles.csv (angles + CoM), see p2s_biomech.py
  packaging      results.json + browser-playable videos for the website

Everything printed is also saved to <out-dir>/log.txt (default outputs/two_view/<name>/log.txt).
Pose2Sim runs in its own environment (C:\\Pose2SimVenv) -- see README.md.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib
import numpy as np

# Pose2Sim switches matplotlib to Qt and can call plt.show(block=True), which would open a window and stall a
# website run. Pin the non-GUI backend before Pose2Sim is imported.
matplotlib.use("Agg", force=True)
matplotlib.use = lambda *args, **kwargs: None
import matplotlib.pyplot as plt  # noqa: E402
plt.show = lambda *args, **kwargs: None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from p2s_biomech import EXTRACTED_ANGLE_COLUMNS, write_outputs  # noqa: E402
from rep_fusion import detect_reps, rep_features                # noqa: E402
from p2s_sync import sync_side_front                            # noqa: E402
from sync_clap import _ffmpeg_exe                               # noqa: E402
from web_results import make_proxy                              # noqa: E402

CLAP_MARGIN_S = 0.5                                     # synced videos start this long after the clap
CAMS = {"side": "cam01_side", "front": "cam02_front"}   # Pose2Sim camera folder / video names (sorted = this order)


# --------------------------------------------------------------------------- logging
class Tee:
    """Copies everything written to stdout/stderr into log.txt."""
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False

    def __getattr__(self, k):
        return getattr(self.stream, k)


def clean_log(path):
    """Progress bars rewrite their line with '\\r'; keep only the final state of each line."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    Path(path).write_text("\n".join(ln.split("\r")[-1] for ln in txt.split("\n")), encoding="utf-8")


def stage(name):
    print(f"\n[stage] {name}", flush=True)


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- video helpers
def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "dur": n / fps if fps else 0.0}


def ffmpeg(args, what):
    res = subprocess.run([_ffmpeg_exe(), "-v", "error", "-y", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while {what}: {res.stderr.strip()[-500:]}")


def extract_intrinsic_frames(src, dst_dir, max_images=60, min_step_s=0.15):
    """Intrinsic video -> PNG frames in calibration/intrinsics/<cam>/.
    (Pose2Sim 0.10.49's own video extraction is broken: extract_frames() calls .stem on a bool and the error is
    swallowed, so it would calibrate from a single frame.) Returns the number of images written."""
    import cv2
    info = video_info(src)
    step = max(min_step_s, info["dur"] / max_images)
    cap = cv2.VideoCapture(str(src))
    n = 0
    for t in np.arange(0, info["dur"], step):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(str(Path(dst_dir) / f"intr_{n:04d}.png"), img)
        n += 1
    cap.release()
    return n


def trim(src, dst, start_s, dur_s, fps, stretch=None):
    """Frame-accurate cut, re-encoded at a common fps so frame N is the same instant in both cameras.
    stretch: time-scale factor (output duration / input duration) that removes clock drift between the cameras."""
    vf = ["-vf", f"setpts={stretch:.8f}*PTS"] if stretch else []
    ffmpeg(["-ss", f"{start_s:.3f}", "-i", str(src), *vf, "-t", f"{dur_s:.3f}", "-r", f"{fps:.5f}", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(dst)],
           f"cutting {Path(src).name}")


def resolve_device(want):
    """'auto' -> 'cuda' when onnxruntime can run on the GPU, else 'cpu'. 'cuda' that does not work is an error."""
    if want == "cpu":
        return "cpu"
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):  # CUDA/cuDNN DLLs from the nvidia-* pip packages (onnxruntime-gpu[cuda,cudnn])
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            log(f"[setup] note: onnxruntime.preload_dlls() failed: {e}")
    ok = "CUDAExecutionProvider" in ort.get_available_providers()
    # The provider is listed whenever onnxruntime-gpu is installed, even if CUDA/cuDNN cannot be loaded. If a pose
    # model is already downloaded, open it on the GPU to be sure.
    models = sorted(Path.home().glob(".cache/rtmlib/hub/checkpoints/*.onnx"))
    if ok and models:
        try:
            sess = ort.InferenceSession(str(models[0]), providers=["CUDAExecutionProvider"])
            ok = sess.get_providers()[0] == "CUDAExecutionProvider"
        except Exception:  # noqa: BLE001
            ok = False
    if want == "cuda" and not ok:
        raise RuntimeError("--device cuda, but onnxruntime cannot use the GPU (install onnxruntime-gpu[cuda,cudnn] "
                           f"and an NVIDIA driver). Providers: {ort.get_available_providers()}")
    return "cuda" if ok else "cpu"


# --------------------------------------------------------------------------- Pose2Sim project
def write_config(project, a, fps):
    """Start from Pose2Sim's own demo Config.toml (so every key the installed version expects exists),
    then set what this setup needs."""
    import rtoml
    import Pose2Sim as p2s_pkg
    cfg = rtoml.load(Path(p2s_pkg.__file__).parent / "Demo_SinglePerson" / "Config.toml")

    def put(path, value):
        d = cfg
        for k in path[:-1]:
            d = d.setdefault(k, {})
        if path[-1] not in d:
            log(f"[setup] note: Config key {'.'.join(path)} not in this Pose2Sim version's template; adding it")
        d[path[-1]] = value

    put(("project", "multi_person"), False)
    put(("project", "frame_rate"), round(fps, 3))
    put(("project", "frame_range"), "auto")
    put(("pose", "pose_model"), "Body_with_feet")      # HALPE_26: includes Head, Neck and Hip
    put(("pose", "mode"), a.pose_mode)
    put(("pose", "device"), a.device)                  # resolved by resolve_device(): 'cpu' or 'cuda'
    put(("pose", "backend"), "onnxruntime")            # explicit: with 'auto', Pose2Sim picks the device itself
    put(("pose", "display_detection"), False)
    put(("pose", "save_video"), "to_video")
    put(("pose", "overwrite_pose"), True)
    put(("calibration", "calibration_type"), "calculate")
    cal = ("calibration", "calculate")
    put(cal + ("intrinsics", "overwrite_intrinsics"), True)
    put(cal + ("intrinsics", "show_detection_intrinsics"), False)
    put(cal + ("intrinsics", "intrinsics_extension"), "png")  # frames extracted by extract_intrinsic_frames()
    put(cal + ("intrinsics", "intrinsics_corners_nb"), a.intr_corners)
    put(cal + ("intrinsics", "intrinsics_square_size"), a.intr_square)
    put(cal + ("extrinsics", "calculate_extrinsics"), False)  # done by p2s_extrinsics.py
    put(cal + ("extrinsics", "extrinsics_method"), "board")   # -> file name Calib_board.toml
    put(("filtering", "display_figures"), False)
    put(("filtering", "make_c3d"), False)
    put(("triangulation", "make_c3d"), False)
    put(("triangulation", "show_interp_indices"), False)
    if "synchronization" in cfg:
        cfg["synchronization"]["display_sync_plots"] = False
    (project / "Config.toml").write_text(rtoml.dumps(cfg), encoding="utf-8")
    log(f"[setup] wrote {project / 'Config.toml'}")


# --------------------------------------------------------------------------- main
def corners_arg(s):
    """'6,7' / '6x7' / '6 7' -> [6, 7]. Anything else is an error (an empty field used to crash Pose2Sim)."""
    try:
        v = [int(x) for x in re.split(r"[,xX ;]+", (s or "").strip()) if x]
    except ValueError:
        v = []
    if len(v) != 2 or min(v) < 2:
        raise argparse.ArgumentTypeError(f"board corners must be two numbers of INNER corners, e.g. 6,7 (got '{s}')")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "extrinsic", "intrinsic"],
                    help="full = calibration + 3D analysis; extrinsic = only sync + calibration; "
                         "intrinsic = only the intrinsics from --intr-side/--intr-front (reuse with --intr-from)")
    ap.add_argument("--side", help="side camera full-length video (board -> clap -> squats)")
    ap.add_argument("--front", help="front camera full-length video")
    ap.add_argument("--intr-side", help="intrinsic checkerboard video, side camera")
    ap.add_argument("--intr-front", help="intrinsic checkerboard video, front camera")
    ap.add_argument("--intr-from", help="take the intrinsics from this earlier calibration.toml instead of videos")
    ap.add_argument("--calib-from", help="(full mode) use this complete calibration.toml: no board search")
    ap.add_argument("--intr-corners", default="6,7", type=corners_arg, help="inner corners of the intrinsic board, e.g. 6,7")
    ap.add_argument("--intr-square", type=float, default=60.0, help="square size of the intrinsic board in mm")
    ap.add_argument("--ext-corners", default="6,7", type=corners_arg, help="inner corners of the extrinsic board")
    ap.add_argument("--ext-square", type=float, default=60.0, help="square size of the extrinsic board in mm")
    ap.add_argument("--board-position", default="vertical", choices=["vertical", "horizontal"],
                    help="vertical = held/standing board (world up from the level cameras); "
                         "horizontal = board flat on the floor (world up = board normal)")
    ap.add_argument("--clap-window", type=float, default=90.0, help="search the clap in the first N seconds")
    ap.add_argument("--squat-claps", type=int, default=1, choices=[1, 2],
                    help="1 = one clap before the squats (analyse to the end of the video); "
                         "2 = a clap at the start AND the end (analyse between them; whole video searched)")
    ap.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"], help="pose estimation device")
    ap.add_argument("--pose-mode", default="balanced", choices=["lightweight", "balanced", "performance"],
                    help="RTMPose model size (performance = most accurate, for a GPU)")
    ap.add_argument("--out-dir", help="output folder (default outputs/two_view/<name>)")
    ap.add_argument("--work-dir", help="Pose2Sim working folder (default outputs/pose2sim/<name>); must not be shared")
    a = ap.parse_args()
    if a.mode == "intrinsic":
        if not (a.intr_side and a.intr_front):
            ap.error("--mode intrinsic needs --intr-side and --intr-front")
        a.side, a.front, a.intr_from, a.calib_from = a.intr_side, a.intr_front, None, None
    elif not (a.side and a.front):
        ap.error("give --side and --front videos")
    for k in ("side", "front", "intr_side", "intr_front", "intr_from", "calib_from"):  # Pose2Sim changes the cwd
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).resolve()))
    if not a.calib_from and not a.intr_from and not (a.intr_side and a.intr_front):
        ap.error("give --intr-side and --intr-front videos, or --intr-from / --calib-from an earlier calibration.toml")

    out_dir = Path(a.out_dir).resolve() if a.out_dir else ROOT / "outputs" / "two_view" / a.name  # website reads this
    # The Pose2Sim project folder. It sits alone inside its own parent: Pose2Sim treats the parent as a "session"
    # folder whenever anything in it has "calib" in its name (e.g. another run called calib01) and then looks
    # for the calibration in the wrong place.
    project = (Path(a.work_dir).resolve() if a.work_dir else ROOT / "outputs" / "pose2sim" / a.name) / "trial"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    log_fh = open(log_path, "w", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_fh), Tee(sys.__stderr__, log_fh)
    t_start = time.time()
    code = 0
    try:
        run(a, out_dir, project)
        log(f"\n[done] total time {time.time() - t_start:.0f} s. Outputs in {out_dir}")
        stage("done")
    except Exception:
        log("\n[error] the analysis stopped:\n" + traceback.format_exc())
        code = 1
    finally:
        sys.stdout.flush()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_fh.close()
        clean_log(log_path)
    # Everything is written. Skip interpreter teardown: native libraries loaded by Pose2Sim occasionally
    # segfault while unloading, which would turn a finished run into a failed exit code.
    sys.stdout.flush()
    os._exit(code)


def intrinsics_from_videos(a, project, calib_dir, intr_vids):
    """Intrinsic videos -> PNG frames -> Pose2Sim.calibration() -> calibration/Calib_board.toml (no extrinsics)."""
    for cam, src in intr_vids.items():
        d = calib_dir / "intrinsics" / CAMS[cam]
        d.mkdir(parents=True)
        n = extract_intrinsic_frames(src, d)
        log(f"[calibration] {cam}: {n} frames from the intrinsic video -> calibration/intrinsics/{CAMS[cam]}/")
        if video_info(src)["dur"] < 15:
            log(f"[calibration] WARNING: the {cam} intrinsic video is only {video_info(src)['dur']:.1f} s. For good "
                "intrinsics record ~30 s, moving the board over the whole frame (corners and edges) at different tilts.")
    os.chdir(project)
    from Pose2Sim import Pose2Sim
    Pose2Sim.calibration(str(project / "Config.toml"))  # intrinsics only (calculate_extrinsics = false)
    if not (calib_dir / "Calib_board.toml").exists():
        raise RuntimeError("Pose2Sim did not write calibration/Calib_board.toml (see log above).")


def run(a, out_dir, project):
    import cv2
    from p2s_extrinsics import find_board, has_extrinsics, load_intrinsics, solve_extrinsics

    # ------------------------------------------------------------- setup
    stage("setup")
    log(f"[setup] mode: {({'extrinsic': 'extrinsic calibration only', 'intrinsic': 'intrinsic calibration only'}).get(a.mode, 'full analysis')}")
    log(f"[setup] Pose2Sim project folder: {project}")
    if project.exists():
        shutil.rmtree(project)
    calib_dir = project / "calibration"
    calib_dir.mkdir(parents=True)
    (project / "videos").mkdir()
    vids = {"side": a.side, "front": a.front}
    info = {cam: video_info(v) for cam, v in vids.items()}
    intr_vids = {"side": a.intr_side, "front": a.intr_front} if not (a.calib_from or a.intr_from) else {}
    for cam, v in (list(vids.items()) if a.mode != "intrinsic" else []) + [(f"{c} intrinsic", p) for c, p in intr_vids.items()]:
        i = video_info(v)
        log(f"[setup] {cam:16s} {Path(v).name}  {i['w']}x{i['h']}  {i['fps']:.2f} fps  {i['dur']:.1f} s")
        base = cam.split()[0]
        if (i["w"], i["h"]) != (info[base]["w"], info[base]["h"]):
            raise RuntimeError(f"The {base} full video is {info[base]['w']}x{info[base]['h']} but its intrinsic video is "
                               f"{i['w']}x{i['h']}. Intrinsics only hold at the same resolution/zoom.")
    fps = {cam: info[cam]["fps"] for cam in vids}
    if abs(fps["side"] - fps["front"]) > 0.5:
        log(f"[setup] note: the cameras have different frame rates; frames are paired by time")
    import Pose2Sim as p2s_pkg
    log(f"[setup] Pose2Sim {getattr(p2s_pkg, '__version__', '')} from {Path(p2s_pkg.__file__).parent}")
    if a.mode == "full":
        want = a.device
        a.device = resolve_device(want)
        log(f"[setup] pose estimation device: {a.device} (asked: {want}), RTMPose mode: {a.pose_mode}")
    write_config(project, a, fps["side"])

    if a.mode == "intrinsic":
        stage("calibration")
        intrinsics_from_videos(a, project, calib_dir, intr_vids)
        shutil.copyfile(calib_dir / "Calib_board.toml", out_dir / "calibration.toml")
        stage("packaging")
        meta = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {"name": a.name}
        (out_dir / "results.json").write_text(json.dumps({
            "kind": "p2s_intr", "sync_only": False, "meta": meta,
            "downloads": {"calibration.toml": "Intrinsics (TOML, no extrinsics)", "log.txt": "Run log (TXT)"},
        }), encoding="utf-8")
        log("[packaging] results.json written")
        return

    # ------------------------------------------------------------- sync
    stage("sync")
    two_claps = a.mode == "full" and a.squat_claps == 2
    log("[sync] looking for the start and end claps in the whole videos ..." if two_claps else
        f"[sync] looking for the clap in the first {a.clap_window:.0f} s of each video ...")
    sync = sync_side_front(a.side, a.front, seconds=a.clap_window, log=log, claps=2 if two_claps else 1)
    clap = {cam: int(round(sync[f"t_clap_{cam}_s"] * fps[cam])) for cam in vids}
    sync.update(clap_frame_side=clap["side"], clap_frame_front=clap["front"])
    log(f"[sync] clap (strongest sound heard by both): side {sync['t_clap_side_s']:.3f} s (frame {clap['side']}), "
        f"front {sync['t_clap_front_s']:.3f} s (frame {clap['front']}) -> offset (front - side) {sync['offset_s']:+.3f} s; "
        f"clap strength side {sync['confidence_side']}, front {sync['confidence_front']}")

    # ------------------------------------------------------------- calibration
    stage("calibration")
    calib_toml = calib_dir / "Calib_board.toml"
    errors, view_angle = None, None
    if a.calib_from and a.mode == "full":
        if not has_extrinsics(a.calib_from):
            raise RuntimeError(f"{a.calib_from} has no extrinsics (all translations are zero).")
        shutil.copyfile(a.calib_from, calib_toml)
        log(f"[calibration] using the complete calibration from {a.calib_from} (no board search)")
        for cam, cal in zip(("side", "front"), load_intrinsics(calib_toml)):
            if cal["size"] != [info[cam]["w"], info[cam]["h"]]:
                raise RuntimeError(f"The {cam} video is {info[cam]['w']}x{info[cam]['h']} but that calibration is for "
                                   f"{cal['size'][0]}x{cal['size'][1]}.")
    else:
        intr_src = a.intr_from or a.calib_from
        if intr_src:
            shutil.copyfile(intr_src, calib_toml)
            log(f"[calibration] intrinsics taken from {intr_src}")
        else:
            intrinsics_from_videos(a, project, calib_dir, intr_vids)
        intr = load_intrinsics(calib_toml)
        for cam, cal in zip(("side", "front"), intr):
            if cal["size"] != [info[cam]["w"], info[cam]["h"]]:
                raise RuntimeError(f"The {cam} video is {info[cam]['w']}x{info[cam]['h']} but its intrinsics are for "
                                   f"{cal['size'][0]}x{cal['size'][1]}.")
        log(f"[calibration] looking for the {a.ext_corners[0]}x{a.ext_corners[1]} extrinsic board, going back frame "
            f"by frame from the clap (side frame {clap['side']}, front frame {clap['front']}) ...")
        # board -> clap -> squats videos: the board is before the clap. Extrinsic-only videos: anywhere.
        found = find_board(vids, fps, clap, a.ext_corners, log=log, before_clap=a.mode == "full")
        errors, view_angle = solve_extrinsics(vids, found, intr, a.ext_corners, a.ext_square, a.board_position,
                                              calib_toml, out_dir, CAMS.values(), log=log)
        sync.update(board_frame_side=found["side"][0], board_frame_front=found["front"][0])
    shutil.copyfile(calib_toml, out_dir / "calibration.toml")

    meta = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {"name": a.name}
    calib_info = {f"{c} reprojection": f"{v} px" for c, v in errors.items()} if errors else None
    if calib_info and view_angle is not None:
        calib_info["camera angle"] = f"{view_angle:.0f}°"
    if a.calib_from and a.mode == "full":
        calib_info = {"reused from": Path(a.calib_from).parent.name}
    boards = [f for f in ("board_side.jpg", "board_front.jpg") if (out_dir / f).exists()]
    if a.mode == "extrinsic":
        stage("packaging")
        (out_dir / "results.json").write_text(json.dumps({
            "kind": "p2s_calib", "sync_only": False, "meta": meta, "offset_s": sync["offset_s"], "sync": sync,
            "calib": calib_info, "boards": boards,
            "downloads": {"calibration.toml": "Camera calibration (TOML)", "log.txt": "Run log (TXT)"},
        }), encoding="utf-8")
        log("[packaging] results.json written")
        return

    # ------------------------------------------------------------- extract
    stage("extract")
    start = {cam: sync[f"t_clap_{cam}_s"] + CLAP_MARGIN_S for cam in vids}
    out_fps = fps["side"]
    stretch = None
    if two_claps and sync.get("t_end_side_s"):
        # squats = between the claps. The front span is stretched to the side span if the camera clocks drifted
        # by more than half a frame, so frame N stays the same instant in both cameras until the end.
        span = {cam: sync[f"t_end_{cam}_s"] - CLAP_MARGIN_S - start[cam] for cam in vids}
        dur = span["side"]
        if abs(span["front"] - span["side"]) > 0.5 / out_fps:
            stretch = span["side"] / span["front"]
            log(f"[extract] the camera clocks drifted {1000 * (span['front'] - span['side']):+.0f} ms between the claps; "
                f"stretching the front video by {stretch:.6f}")
        what = "between the start and end claps (0.5 s margins)"
    else:
        dur = min(info[cam]["dur"] - start[cam] for cam in vids)
        what = "from 0.5 s after the clap to the end"
    if dur < 2:
        raise RuntimeError(f"Only {dur:.1f} s of squat video after the clap in one of the cameras.")
    sync.update(squat_start_side_s=round(start["side"], 3), squat_start_front_s=round(start["front"], 3),
                squat_dur_s=round(dur, 3))
    log(f"[extract] cutting both videos {what}: side from {start['side']:.3f} s, front from {start['front']:.3f} s, "
        f"{dur:.2f} s each, {out_fps:.2f} fps")
    for cam in vids:
        trim(vids[cam], project / "videos" / f"{CAMS[cam]}.mp4", start[cam], dur, out_fps,
             stretch if cam == "front" else None)
        log(f"[extract] wrote videos/{CAMS[cam]}.mp4")

    # ------------------------------------------------------------- Pose2Sim
    os.chdir(project)
    from Pose2Sim import Pose2Sim
    config = str(project / "Config.toml")

    stage("pose")
    log(f"[pose] RTMPose (HALPE_26, {a.pose_mode}) on the {a.device.upper()}")
    Pose2Sim.poseEstimation(config)

    stage("triangulation")
    Pose2Sim.personAssociation(config)
    Pose2Sim.triangulation(config)
    Pose2Sim.filtering(config)
    trcs = sorted(glob.glob(str(project / "pose-3d" / "*.trc")), key=os.path.getmtime)
    filt = [t for t in trcs if "filt" in Path(t).name]
    if not trcs:
        raise RuntimeError("Pose2Sim triangulation did not write a .trc file (see log above).")
    trc = (filt or trcs)[-1]
    shutil.copyfile(trc, out_dir / "points3d.trc")
    log(f"[triangulation] using {Path(trc).name}")

    # ------------------------------------------------------------- angles
    stage("angles")
    df, _ = write_outputs(trc, out_dir / "angles.csv", out_dir / "advancedangles.csv", log=log)
    # The TRC only covers frames Pose2Sim could triangulate; give every video frame a row so row index == frame.
    df = df.set_index("frame").reindex(range(int(df["frame"].max()) + 1)).rename_axis("frame").reset_index()
    reps = detect_reps(df["squat_depth_knee_deg"], out_fps)
    log(f"[angles] {len(reps)} squat reps detected")
    table = []
    for n, rep in enumerate(reps, start=1):
        row = {"rep_id": n, "bottom_s": round(rep["bottom"] / out_fps, 3)}
        row.update(rep_features(df, rep, out_fps, ""))
        table.append(row)
    import pandas as pd
    pd.DataFrame(table).to_csv(out_dir / "rep_features.csv", index=False)
    # rep boundaries, in the same frame numbering as angles.csv / advancedangles.csv (row index == frame here)
    pd.DataFrame([{"rep_no": n, "start_frame": int(df["frame"].iloc[rep["start"]]), "end_frame": int(df["frame"].iloc[rep["end"]])}
                  for n, rep in enumerate(reps, start=1)],
                 columns=["rep_no", "start_frame", "end_frame"]).to_csv(out_dir / "reps.csv", index=False)
    log(f"[angles] wrote {out_dir / 'reps.csv'}")

    # ------------------------------------------------------------- packaging
    stage("packaging")
    media = {}
    for cam in vids:
        overlay = sorted(glob.glob(str(project / "pose" / f"{CAMS[cam]}*.mp4")))
        src = overlay[0] if overlay else project / "videos" / f"{CAMS[cam]}.mp4"
        if make_proxy(src, out_dir / f"{cam}.mp4"):
            media[cam] = f"{cam}.mp4"
    clean = lambda v: None if (isinstance(v, float) and not np.isfinite(v)) else (round(v, 3) if isinstance(v, float) else v)  # noqa: E731
    results = {
        "kind": "pose2sim", "sync_only": False, "meta": meta,
        "offset_s": sync["offset_s"], "sync": sync, "device": a.device,
        "fps": {"p3d": out_fps},
        "p3d": {"t": [round(float(f) / out_fps, 3) for f in df["frame"]], "n_frames": int(len(df)),
                "m": {c: [None if not np.isfinite(v) else round(float(v), 2) for v in df[c]] for c in EXTRACTED_ANGLE_COLUMNS}},
        "reps": [{"id": n, "t": [round(rep[k] / out_fps, 3) for k in ("start", "bottom", "end")]} for n, rep in enumerate(reps, start=1)],
        "table": [{k: clean(v) for k, v in row.items()} for row in table],
        "media": media, "calib": calib_info, "boards": boards,
        "downloads": {"angles.csv": "3D joints per frame (angles.csv)", "advancedangles.csv": "Angles + CoM per frame (advancedangles.csv)",
                      "reps.csv": "Rep start/end frames (reps.csv)", "rep_features.csv": "Per-rep table (CSV)", "points3d.trc": "Pose2Sim 3D markers (TRC)",
                      "calibration.toml": "Camera calibration (TOML)", "log.txt": "Run log (TXT)"},
        "counts": {"reps": len(reps), "frames": int(len(df))},
    }
    (out_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    log("[packaging] results.json written")


if __name__ == "__main__":
    main()
