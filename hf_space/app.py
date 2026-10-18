"""
app.py -- Hugging Face ZeroGPU Space (Gradio): the front and side video of a squat set -> the error in every rep, on
the 3D body. The same pipeline as the final product (final_product/), in one Python environment:

  calibration  the dataset cameras (preset), a calibration .toml, or intrinsic videos (run_pose2sim --mode intrinsic)
  prepare      run_pose2sim --phase prepare: clap sync, board -> extrinsics, videos cut after the clap        CPU
  pose         RTMPose 2D keypoints: run_pose2sim.pose_phase() inside @spaces.GPU (CPU if no GPU is usable)   GPU
  finish       run_pose2sim --phase finish: triangulation, joint angles, reps                                 CPU
  mesh         p2s_mesh.py: MotionBERT 3D body from the side camera's 2D keypoints                            CPU
  model        squat_research.predict: CNN-LSTM error per rep + attention over the frames                     CPU
Only the pose step uses the GPU, so one analysis costs about a minute of the visitor's daily ZeroGPU quota.

The results open in the product's own results page (final_product/web/product.html, static viewer mode) inside an
iframe: the 3D body coloured with the painted template of each rep's error x the model's attention, the keypoint
videos, errors and fixes, the timeline. Its files are served by Gradio (/gradio_api/file=...).

Local run (CPU, Pose2Sim in its own environment):
    set POSE2SIM_PY=C:\\Pose2SimVenv\\Scripts\\python.exe & python app.py      (run from the built Space folder)
"""
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

APP = Path(__file__).resolve().parent
PY = sys.executable
os.environ.setdefault("PRODUCT_DATA", str(APP / "outputs" / "product"))
os.environ.setdefault("POSE2SIM_PY", PY)
os.environ.setdefault("SQUAT_MODEL_PY", PY)
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path[:0] = [str(APP / "final_product"), str(APP / "two_view")]


def ensure_onnxruntime_gpu():
    """rtmlib depends on the CPU package 'onnxruntime', which can overwrite onnxruntime-gpu's files when the Space
    installs its requirements. On the Space, put the GPU build back once (before anything imports onnxruntime)."""
    if not os.environ.get("SPACE_ID"):
        return
    probe = "import onnxruntime as o; print('CUDAExecutionProvider' in o.get_available_providers())"
    r = subprocess.run([PY, "-c", probe], capture_output=True, text=True)
    if r.stdout.strip() == "True":
        return
    spec = os.environ.get("ORT_GPU_SPEC", "onnxruntime-gpu==1.30.0")
    print(f"onnxruntime has no CUDA provider ({(r.stdout + r.stderr).strip()[-200:]}); reinstalling {spec}", flush=True)
    for extra in ([], ["--user"]):
        if subprocess.run([PY, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--no-cache-dir", spec, *extra]).returncode == 0:
            break


ensure_onnxruntime_gpu()

# --------------------------------------------------------------------------- the error model (swappable)
# models/squat_model/ (model.pt, config.json, norm_stats.json - the export of research/notebooks/04) or the same files
# as models/squat_model.zip. To change the model: rename the old one (e.g. squat_model_old) and upload the new one as
# squat_model; the Space restarts and uses it. Without a squat_model, the newest model in models/ is used.
MODELS = APP / "models"
ACTIVE_NAME = "squat_model"
MODEL_FILES = ("model.pt", "config.json", "norm_stats.json")


def _model_ok(p):
    return p.is_dir() and all((p / f).is_file() for f in MODEL_FILES)


def _unzip_model(z):
    """A model uploaded as a .zip: the three files at its top or in one folder inside it."""
    dst = APP / "outputs" / "model_cache" / f"{z.stem}_{z.stat().st_size}"
    if not _model_ok(dst):
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        with zipfile.ZipFile(z) as f:
            for m in f.infolist():
                if not m.is_dir() and Path(m.filename).name in MODEL_FILES + ("model_card.md",):
                    (dst / Path(m.filename).name).write_bytes(f.read(m))
    return dst if _model_ok(dst) else None


def find_model():
    """(model folder or None, what was used / what is wrong)."""
    def load(p):
        if p.is_dir():
            return p if _model_ok(p) else None
        return _unzip_model(p) if p.suffix.lower() == ".zip" else None

    if not MODELS.is_dir():
        return None, "there is no models/ folder"
    for p in (MODELS / ACTIVE_NAME, MODELS / f"{ACTIVE_NAME}.zip"):
        if p.exists():
            d = load(p)
            return (d, f"models/{p.name}") if d else (None, f"models/{p.name} is incomplete (it needs {', '.join(MODEL_FILES)})")
    found = []
    for p in sorted(MODELS.iterdir()):
        d = load(p)
        if d:
            created = (json.loads((d / "config.json").read_text(encoding="utf-8")) or {}).get("created", "")
            found.append((created, p.name, d))
    if found:
        _, name, d = max(found)
        return d, f"models/{name} (the newest: none is named {ACTIVE_NAME})"
    return None, "there is no model in models/"


MODEL_DIR, MODEL_FROM = find_model()
MODEL_CFG = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8")) if MODEL_DIR else {}
_code_fv = re.search(r"FEATURE_VERSION\s*=\s*(\d+)", (APP / "research" / "squat_research" / "__init__.py").read_text(encoding="utf-8"))
MODEL_PROBLEM = None
if MODEL_DIR and _code_fv and MODEL_CFG.get("feature_version") != int(_code_fv.group(1)):
    MODEL_PROBLEM = (f"{MODEL_FROM} was trained on feature version {MODEL_CFG.get('feature_version')}, but this Space's "
                     f"research/squat_research computes version {_code_fv.group(1)}: upload the matching research/ code "
                     "too. Reps are shown without errors until then.")
if MODEL_DIR and not MODEL_PROBLEM:
    os.environ["SQUAT_MODEL_DIR"] = str(MODEL_DIR)
else:
    os.environ["SQUAT_MODEL_DIR"] = str(APP / "models" / "_none")
print(f"error model: {MODEL_FROM}" + (f" - {MODEL_PROBLEM}" if MODEL_PROBLEM else ""), flush=True)

import spaces  # noqa: E402  (ZeroGPU: before torch / CUDA)
import gradio as gr  # noqa: E402
import product_pipeline as PL  # noqa: E402
import product_predictions as PP  # noqa: E402

try:  # Pose2Sim in this environment (the Space): the pose step runs in-process, inside the GPU function
    import Pose2Sim  # noqa: E402,F401
    import run_pose2sim as RP  # noqa: E402
except ImportError:  # local test with Pose2Sim in another environment (POSE2SIM_PY): subprocesses only
    RP = None

DATA = PL.PRODUCT
SESSIONS, VIEW, WORK = DATA / "sessions", DATA / "viewer", PL.WORK
WEB = APP / "final_product" / "web"
TEMPLATES = APP / "saliency_maps"
PRESET = APP / "presets" / "cameras" / "Dataset_cameras"
SMPL_PKL = APP / "MotionBERT" / "data" / "mesh" / "SMPL_NEUTRAL.pkl"
POSE_MODE = os.environ.get("PRODUCT_POSE_MODE", "performance")
MAX_GPU_S = int(os.environ.get("GPU_MAX_SECONDS", "240"))
ZIP_FILES = ["rep_results.csv", "predictions.json", "angles.csv", "advancedangles.csv", "points3d.trc", "front.mp4",
             "side.mp4", "calibration.toml", "log.txt"]
CALIB = ["Dataset cameras (DJI pair, 1920x1080)", "Calibration file (.toml)", "Intrinsic videos (one per camera)"]
CONTENT = json.loads((APP / "final_product" / "product_content.json").read_text(encoding="utf-8"))
for _d in (SESSIONS, VIEW, WORK):
    _d.mkdir(parents=True, exist_ok=True)


class StepError(Exception):
    """A failure with a message for the user."""


# --------------------------------------------------------------------------- static files of the results viewer
def furl(p):
    """URL under which Gradio serves a file of an allowed folder."""
    return "/gradio_api/file=" + quote(Path(p).resolve().as_posix(), safe="/:")


def tkey(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def session_rows():
    rows = []
    for d in SESSIONS.iterdir():
        s = PP.load_json(d / "session.json", {}) or {}
        if (d / "viewer_data.json").exists():
            rows.append({"id": d.name, "name": s.get("name", d.name), "created": s.get("created", ""), "state": "done",
                         "summary": s.get("summary"), "camera": s.get("camera")})
    return sorted(rows, key=lambda r: r["created"], reverse=True)


def write_viewer_index():
    """viewer_index.json: where the viewer page finds the texts, SMPL faces, templates and sessions."""
    shutil.copyfile(APP / "final_product" / "product_content.json", VIEW / "product_content.json")
    if SMPL_PKL.exists() and not (VIEW / "faces.bin").exists():
        import numpy as np
        import smplx
        faces = smplx.create(str(SMPL_PKL), model_type="smpl", gender="neutral").faces
        np.asarray(faces, dtype="<u2").tofile(VIEW / "faces.bin")
    tpl = {}
    for t in sorted(TEMPLATES.glob("*.json")):
        try:
            j = json.loads(t.read_text(encoding="utf-8"))
            if len(j.get("saliency") or []) == 6890:
                tpl[tkey(j.get("mesh_name") or re.sub(r"_saliency$", "", t.stem))] = furl(t)
        except (OSError, ValueError):
            pass
    idx = {"info": {"model_active": (PL.MODEL_DIR / "model.pt").exists(), "allow_paths": False, "viewer": True,
                    "smpl": SMPL_PKL.exists(), "pose_mode": POSE_MODE},
           "content": furl(VIEW / "product_content.json"), "faces": furl(VIEW / "faces.bin"), "templates": tpl,
           "root": furl(SESSIONS) + "/", "sessions": furl(VIEW / "sessions.json")}
    (VIEW / "viewer_index.json").write_text(json.dumps(idx), encoding="utf-8")
    (VIEW / "sessions.json").write_text(json.dumps(session_rows()), encoding="utf-8")


def viewer_html(jid):
    src = (furl(WEB / "product.html") + "?viewer=" + quote(furl(VIEW / "viewer_index.json"), safe="") +
           "#/session/" + quote(jid))
    return (f'<iframe src="{src}" title="Results" allow="fullscreen" '
            'style="width:100%;height:920px;border:0;border-radius:14px;background:#f3f5fb"></iframe>')


def prefetch_pose_models():
    """Download the RTMPose / YOLOX weights at start-up, so the GPU time is not spent downloading."""
    try:
        from rtmlib import BodyWithFeet
        BodyWithFeet(mode=POSE_MODE, backend="onnxruntime", device="cpu")
        print(f"pose models ({POSE_MODE}) ready", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"pose model prefetch skipped: {e}", flush=True)


# --------------------------------------------------------------------------- GPU step
def _gpu_seconds(project, n_frames, mode, log_path):
    per_frame = {"lightweight": 0.006, "balanced": 0.012, "performance": 0.03}.get(mode, 0.03)
    return int(max(60, min(MAX_GPU_S, 40 + n_frames * per_frame)))


@spaces.GPU(duration=_gpu_seconds)
def pose_on_gpu(project, n_frames, mode, log_path):
    """RTMPose on the GPU (onnxruntime-gpu). Returns 'cpu' at once when onnxruntime cannot use the GPU."""
    import torch  # noqa: F401  (loads the CUDA / cuDNN libraries onnxruntime-gpu runs on)
    with open(log_path, "a", encoding="utf-8") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
        dev = RP.resolve_device("auto")
        if dev != "cuda":
            print("[pose] onnxruntime cannot use the GPU here - the pose step runs on the CPU", flush=True)
            return "cpu"
        RP.pose_phase(project, dev, mode)
    return "cuda"


# --------------------------------------------------------------------------- helpers
class Run:
    def __init__(self, d):
        self.d, self.t0, self.lines, self.warnings = d, time.time(), [], []

    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.lines.append(line)
        with open(self.d / "pipeline_log.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def text(self, live=""):
        el = time.time() - self.t0
        return "\n".join(self.lines[-14:] + ([f"    {live}"] if live else []) + [f"({el // 60:.0f} min {el % 60:.0f} s)"])


def progress_line(text):
    stage, prog, last = None, None, ""
    for line in text.splitlines():
        if line.startswith("[stage] "):
            stage = line[8:].strip()
        m = re.search(r"Processed (\d+)/(\d+) frames", line) or re.search(r"\| *(\d+)/(\d+) \[", line)
        if m:
            prog = f"{m.group(1)}/{m.group(2)}"
        if line.strip():
            last = line.strip()[:140]
    return " · ".join(x for x in (stage, prog) if x) or last


def child(run, cmd, console, timeout_s, env=None):
    """Runs a step in a subprocess; yields its progress line while it runs. Returns (exit code, output)."""
    run.say("$ " + " ".join(Path(str(c)).name if i < 3 else str(c) for i, c in enumerate(cmd))[:300])
    with open(console, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=str(APP),
                                env={**os.environ, **env} if env else None)
        t0 = time.time()
        try:
            while proc.poll() is None:
                time.sleep(1.5)
                yield progress_line(PL.read_log(console))
                if time.time() - t0 > timeout_s:
                    raise StepError(f"A step took longer than {timeout_s / 60:.0f} minutes and was stopped.")
        finally:
            if proc.poll() is None:
                PL.kill_tree(proc)
    return proc.returncode, PL.read_log(console)


def board_from(intr_c, intr_s, ext_c, ext_s, pos):
    board = {}
    for key, val in (("intr_corners", intr_c), ("ext_corners", ext_c)):
        nums = re.findall(r"\d+", str(val or ""))
        if len(nums) != 2 or min(map(int, nums)) < 2:
            raise StepError("Board corners: give two numbers of INNER corners, e.g. 6,7.")
        board[key] = f"{nums[0]},{nums[1]}"
    for key, val in (("intr_square", intr_s), ("ext_square", ext_s)):
        if not val or float(val) <= 0:
            raise StepError("Board square size: give it in millimetres, e.g. 41.")
        board[key] = float(val)
    board["board_position"] = "horizontal" if str(pos).lower().startswith("lying") else "vertical"
    return board


def new_id(name):
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "Session").strip()).strip("_")[:40] or "Session"
    jid, n = f"{base}_{time.strftime('%Y%m%d-%H%M%S')}", 2
    while (SESSIONS / jid).exists():
        jid, n = f"{base}_{time.strftime('%Y%m%d-%H%M%S')}_{n}", n + 1
    return jid


def keep(src, dst_dir, stem):
    dst = dst_dir / f"{stem}{Path(src).suffix.lower() or '.mp4'}"
    shutil.copyfile(src, dst)
    return dst


def results_outputs(d):
    """Summary markdown, rep table and download files of a finished session."""
    sess = PP.load_json(d / "session.json", {}) or {}
    pred = PP.load_json(d / "predictions.json", {}) or {}
    s = sess.get("summary") or {}
    name = lambda c: (CONTENT["errors"].get(c) or {}).get("name", c or "not analysed")  # noqa: E731
    top = s.get("top_error")
    fix = ((CONTENT["errors"].get(top) or {}).get("fix") or [""])[0] if top else ""
    md = (f"### {sess.get('name', d.name)}\n**{s.get('reps', 0)} reps** · **{s.get('pct_correct', '–')}% correct**"
          + (f" · most frequent error: **{name(top)}**" if top else "")
          + (f"\n\nFocus next time: **{fix}**" if fix else "")
          + ("" if s.get("model_active", True) else "\n\n_The error model is not installed: reps are shown without errors._"))
    rows = [[r["rep"], round(r["start_s"], 2), round(r["end_s"], 2), name(r.get("label")),
             f"{round(100 * r['confidence'])}%" if r.get("confidence") is not None else "–"] for r in pred.get("reps", [])]
    files = [str(p) for p in (d / "download" / "results.zip", d / "download" / "reps.csv", d / "rep_results.csv") if p.exists()]
    return md, rows, files


def model_md():
    if MODEL_PROBLEM or not MODEL_DIR:
        return f"**Error model:** not active - {MODEL_PROBLEM or MODEL_FROM}. Reps are shown without errors."
    c, f1 = MODEL_CFG, MODEL_CFG.get("cv_macro_f1")
    unknown = [k for k in c.get("classes", []) if k not in CONTENT["errors"]]
    return (f"**Error model:** {c.get('model_name', c.get('model', '?'))} · trained {c.get('created', '?')}"
            + (f" · cross-validated macro-F1 {f1:.2f}" if isinstance(f1, (int, float)) else "")
            + f" · {len(c.get('classes', []))} classes · `{MODEL_FROM}`"
            + (f"  \n_Classes without a name / colour / template in product_content.json and saliency_maps/: "
               f"{', '.join(unknown)}_" if unknown else ""))


def past_choices():
    return [(f"{r['name']} · {r['created'][5:16]}", r["id"]) for r in session_rows()]


# --------------------------------------------------------------------------- the analysis
def analyse(name, calib_mode, calib_file, intr_front, intr_side, intr_c, intr_s, ext_c, ext_s, pos, front, side):
    def out(status, viewer=gr.skip(), md=gr.skip(), table=gr.skip(), files=gr.skip(), past=gr.skip()):
        return status, viewer, md, table, files, past

    if not front or not side:
        yield out("Add both squat videos (front and side) in step 2.")
        return
    jid = new_id(name)
    d = SESSIONS / jid
    up = d / "uploads"
    up.mkdir(parents=True)
    run = Run(d)
    try:
        board = board_from(intr_c, intr_s, ext_c, ext_s, pos)
        vids = {"front": keep(front, up, "front_in"), "side": keep(side, up, "side_in")}
        sess = {"id": jid, "name": (name or "").strip() or time.strftime("Session %d %b %H:%M"),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "front": str(vids["front"]), "side": str(vids["side"])}
        PL.write_json(d / "session.json", sess)
        (d / "meta.json").write_text(json.dumps({"name": sess["name"], "kind": "pose2sim", "person_id": jid,
                                                 "subject": sess["name"], "created": sess["created"]}), encoding="utf-8")
        run.say(f"session {jid}")
        yield out(run.text())

        # ---- calibration (intrinsics)
        if calib_mode == CALIB[1]:
            if not calib_file:
                raise StepError("Add the calibration .toml file (step 1).")
            intr = d / "intrinsics.toml"
            shutil.copyfile(calib_file, intr)
            try:
                if not PL.calib_sizes(intr):
                    raise ValueError
            except Exception:  # noqa: BLE001
                raise StepError("That file has no camera intrinsics (expected a Pose2Sim calibration .toml).") from None
            sess["camera"] = "uploaded calibration"
        elif calib_mode == CALIB[2]:
            if not intr_front or not intr_side:
                raise StepError("Add both intrinsic videos (front and side camera) in step 1.")
            iv = {"front": keep(intr_front, up, "intr_front"), "side": keep(intr_side, up, "intr_side")}
            run.say("intrinsic calibration from the two videos (CPU, a few minutes) ...")
            code, outp = yield from _wrap(run, child(
                run, [PL.P2S_PY, "-u", PL.RUN, "--mode", "intrinsic", "--name", f"intr_{jid}", "--intr-side", iv["side"],
                      "--intr-front", iv["front"], *PL.board_args(board), "--out-dir", d / "intrinsic",
                      "--work-dir", WORK / f"{jid}_intr"], d / "intrinsic_console.txt", 2 * 3600), out)
            intr = d / "intrinsic" / "calibration.toml"
            if code != 0 or not intr.exists():
                raise StepError("Intrinsic calibration failed: " + PL.friendly(PL.error_line(PL.read_log(d / "intrinsic" / "log.txt") or outp)))
            sess["camera"] = "intrinsic videos"
        else:
            intr = PRESET / "intrinsics.toml"
            sess["camera"] = "dataset cameras"
        PL.write_json(d / "session.json", sess)

        # ---- check the videos against the calibration
        sizes = list(PL.calib_sizes(intr).items())
        for cam, p in vids.items():
            info = PL.video_info(p)
            if not info or not info["readable"]:
                raise StepError(f"The {cam} video cannot be read.")
            if info["dur"] < 5:
                raise StepError(f"The {cam} video is only {info['dur']} s long.")
            want = next((s for k, s in sizes if cam in k), None)
            if want and [info["w"], info["h"]] != [int(x) for x in want]:
                raise StepError(f"The {cam} video is {info['w']}x{info['h']} but the calibration is for "
                                f"{int(want[0])}x{int(want[1])}. Use videos at the calibration's resolution and zoom.")
            run.say(f"{cam}: {info['w']}x{info['h']} {info['fps']} fps {info['dur']} s")
        yield out(run.text())

        # ---- prepare: sync, extrinsics, cut (CPU)
        run.say("finding the clap, calibrating the cameras from the board, cutting the videos ...")
        base = [PL.P2S_PY, "-u", PL.RUN, "--name", jid, "--out-dir", d, "--work-dir", WORK / jid, "--pose-mode", POSE_MODE]
        code, outp = yield from _wrap(run, child(
            run, base + ["--phase", "prepare", "--side", vids["side"], "--front", vids["front"], "--intr-from", intr,
                         "--squat-claps", "1", *PL.board_args(board), "--device", "cpu"],
            d / "preprocess_console_1.txt", 3600), out)
        if code != 0 or not (d / "phase_state.json").exists():
            raise StepError(PL.friendly(PL.error_line(PL.read_log(d / "log.txt") or outp)))

        # ---- pose: GPU (ZeroGPU), else CPU
        project = WORK / jid / "trial"
        n_frames = 0
        for v in (project / "videos").glob("*.mp4"):
            i = PL.video_info(v) or {}
            n_frames += int((i.get("dur") or 0) * (i.get("fps") or 0))
        dev = "cpu"
        if RP is not None:
            run.say(f"body keypoints on the GPU (RTMPose {POSE_MODE}, {n_frames} frames; "
                    f"up to {_gpu_seconds(None, n_frames, POSE_MODE, None)} s of GPU time) ...")
            yield out(run.text())
            try:
                dev = pose_on_gpu(str(project), n_frames, POSE_MODE, str(d / "log.txt"))
            except Exception as e:  # noqa: BLE001  (quota used up, no GPU free, time limit, ...)
                run.say(f"the GPU step did not finish ({str(e)[:200]}) - running it on the CPU instead")
                dev = "cpu"
        if dev != "cuda":
            run.say(f"body keypoints on the CPU (RTMPose {POSE_MODE}, {n_frames} frames; this is the slow part) ...")
            code, outp = yield from _wrap(run, child(run, base + ["--phase", "pose", "--device", "cpu"],
                                                     d / "preprocess_console_2.txt", 3 * 3600), out)
            if code != 0:
                raise StepError(PL.friendly(PL.error_line(PL.read_log(d / "log.txt") or outp)))
        run.say(f"keypoints done on the {dev.upper()}")

        # ---- finish: triangulation, angles, reps (CPU)
        code, outp = yield from _wrap(run, child(run, base + ["--phase", "finish"], d / "preprocess_console_3.txt", 3600), out)
        if code != 0 or not (d / "results.json").exists():
            raise StepError(PL.friendly(PL.error_line(PL.read_log(d / "log.txt") or outp)))
        res = PP.load_json(d / "results.json", {}) or {}
        if not (res.get("sync") or {}).get("reliable", True):
            run.warnings.append("The clap was not clearly heard by both cameras; the two videos may be slightly out of sync.")
        if not res.get("reps"):
            run.warnings.append("No squat reps were found in the video.")
        run.say(f"3D skeleton and joint angles done: {len(res.get('reps') or [])} reps")

        # ---- mesh (CPU)
        run.say("building the 3D body (MotionBERT) ...")
        yield from _wrap(run, child(run, [PY, "-u", APP / "two_view" / "p2s_mesh.py", d, "--method", "motionbert",
                                          "--camera", "side"], d / "mesh_console.txt", 1800), out)
        if not (d / "mesh" / "done.txt").exists():
            run.warnings.append("The 3D body could not be built (" + PL.error_line(PL.read_log(d / "mesh" / "log.txt")) +
                                "). Everything else is available.")

        # ---- error model (CPU)
        pred = None
        if (PL.MODEL_DIR / "model.pt").exists():
            run.say("detecting the error in every rep ...")
            code, outp = yield from _wrap(run, child(run, [PY, "-u", "-m", "squat_research.predict", d, "--model", PL.MODEL_DIR],
                                                     d / "model_console.txt", 1200,
                                                     env={"PYTHONPATH": str(APP / "research")}), out)
            pred = PP.load_json(d / "predictions.json") if code == 0 else None
            if pred is None:
                run.warnings.append("The error model failed (" + PL.error_line(outp) + "). Showing the reps without errors.")
        if pred is None:
            pred = PP.inactive_predictions(d)
        pred.setdefault("model_active", True)

        # ---- results
        sess["summary"] = PP.write_all(d, pred)
        PL.write_json(d / "session.json", sess)
        dl = d / "download"   # reps.csv here is the user's file (seconds); d/reps.csv is Pose2Sim's (frames)
        dl.mkdir(exist_ok=True)
        (dl / "reps.csv").write_text(PP.user_reps_csv(pred), encoding="utf-8")
        with zipfile.ZipFile(dl / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(dl / "reps.csv", "reps.csv")
            for f in ZIP_FILES:
                if (d / f).is_file():
                    z.write(d / f, f, compress_type=zipfile.ZIP_STORED if f.endswith(".mp4") else zipfile.ZIP_DEFLATED)
        (d / "viewer_data.json").write_text(json.dumps(PP.session_view(d, SMPL_PKL.exists(), run.warnings)), encoding="utf-8")
        write_viewer_index()
        for w in run.warnings:
            run.say("note: " + w)
        run.say("done")
        md, rows, files = results_outputs(d)
        yield out(run.text(), viewer_html(jid), md, rows, files, gr.update(choices=past_choices(), value=jid))
    except StepError as e:
        run.say(f"STOPPED: {e}")
        yield out(run.text())
    except Exception as e:  # noqa: BLE001
        import traceback
        (d / "pipeline_log.txt").open("a", encoding="utf-8").write(traceback.format_exc())
        run.say(f"STOPPED: unexpected error: {e}")
        yield out(run.text())


def _wrap(run, gen, out):
    """Forwards a child() generator's progress to the page; returns its result."""
    try:
        while True:
            yield out(run.text(next(gen)))
    except StopIteration as stop:
        return stop.value


def open_past(jid):
    if not jid or not (SESSIONS / jid / "viewer_data.json").exists():
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    md, rows, files = results_outputs(SESSIONS / jid)
    return viewer_html(jid), md, rows, files


# --------------------------------------------------------------------------- page
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Poppins"), "system-ui", "sans-serif"]).set(
    button_primary_background_fill="#001b94", button_primary_background_fill_hover="#1f3bb8",
    button_primary_text_color="#ffffff", color_accent_soft="#e9edfb")
# the results page inside is light (IndiGo palette): keep the Gradio page light too
LIGHT_JS = """() => { const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); } }"""
CSS = """
.sq-head{display:flex;align-items:center;gap:14px;padding:18px 22px;border-radius:16px;color:#fff;
  background:linear-gradient(115deg,#001b94 0%,#0a1a7a 55%,#0a1150 100%)}
.sq-head svg{width:46px;height:46px;flex:0 0 auto}
.sq-head h1{margin:0;font-size:24px;color:#fff} .sq-head p{margin:2px 0 0;color:rgba(255,255,255,.82);font-size:14px}
.sq-step{display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px;color:#111a3d;margin:8px 0 -2px}
.sq-step b{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;
  background:#001b94;color:#fff;font-size:13px;flex:0 0 auto}
.sq-step span{font-weight:400;font-size:13px;color:#5a6386}
"""
LOGO = ('<svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="#fff"/><g transform="translate(3.6 2.6)">'
        '<circle cx="12.2" cy="4" r="2.3" fill="#001b94"/><path d="M11.2 7.6L6.8 13.6L14.2 14.2L12.4 20.6H16.2M11.2 8.4H19" '
        'fill="none" stroke="#001b94" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></g>'
        '<path d="M5 25.6h22" stroke="#5f7cff" stroke-width="1.8" stroke-linecap="round"/></svg>')

with gr.Blocks(title="Squat Analysis") as demo:
    gr.HTML(f'<div class="sq-head">{LOGO}<div><h1>Squat Analysis</h1><p>Front and side video of a squat set &rarr; the '
            'error in every rep, on a 3D body coloured where the model looked.</p></div></div>')
    gr.Markdown(model_md())
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            name = gr.Textbox(label="Name (optional)", placeholder="e.g. Monday morning, set 2")
            gr.HTML('<div class="sq-step"><b>1</b>Camera calibration</div>')
            calib_mode = gr.Radio(CALIB, value=CALIB[0], show_label=False)
            calib_file = gr.File(label="Pose2Sim calibration (.toml); only its intrinsics are used", file_types=[".toml"],
                                 type="filepath", visible=False)
            with gr.Row(visible=False) as intr_row:
                intr_front = gr.File(label="Intrinsic video, front camera", file_types=["video"], type="filepath")
                intr_side = gr.File(label="Intrinsic video, side camera", file_types=["video"], type="filepath")
            with gr.Accordion("Calibration board: 6×7 inner corners, 41 mm squares, held upright", open=False):
                with gr.Row():
                    intr_c = gr.Textbox("6,7", label="Inner corners (intrinsic videos)")
                    intr_s = gr.Number(41, label="Square size, mm (intrinsic videos)")
                with gr.Row():
                    ext_c = gr.Textbox("6,7", label="Inner corners (in the squat videos)")
                    ext_s = gr.Number(41, label="Square size, mm (in the squat videos)")
                pos = gr.Radio(["Held upright", "Lying flat on the floor"], value="Held upright", label="Board before the clap")
            gr.HTML('<div class="sq-step"><b>2</b>Squat videos<span>board held still where both cameras see it, one '
                    'clap, then the squats</span></div>')
            with gr.Row():
                front = gr.File(label="Front video", file_types=["video"], type="filepath")
                side = gr.File(label="Side video", file_types=["video"], type="filepath")
            gr.HTML('<div class="sq-step"><b>3</b>Analyse<span>keypoints on the GPU (about a minute of your '
                    'ZeroGPU quota), the rest on the CPU</span></div>')
            with gr.Row():
                start = gr.Button("Start analysis", variant="primary", scale=3)
                stop = gr.Button("Stop", variant="stop", scale=1)
            status = gr.Textbox(label="Progress", lines=12, max_lines=12, interactive=False, autoscroll=True)
        with gr.Column(scale=5):
            summary = gr.Markdown("Results appear here and in the 3D viewer below.")
            table = gr.Dataframe(headers=["Rep", "Start (s)", "End (s)", "Error", "Confidence"], interactive=False,
                                 label="Reps")
            files = gr.File(label="Downloads", file_count="multiple", interactive=False)
            past = gr.Dropdown(label="Earlier sessions (until the Space restarts)", choices=[], interactive=True)
    viewer = gr.HTML()

    calib_mode.change(lambda m: (gr.update(visible=m == CALIB[1]), gr.update(visible=m == CALIB[2])),
                      calib_mode, [calib_file, intr_row])
    ev = start.click(analyse, [name, calib_mode, calib_file, intr_front, intr_side, intr_c, intr_s, ext_c, ext_s, pos,
                               front, side], [status, viewer, summary, table, files, past])
    stop.click(None, cancels=[ev])
    past.change(open_past, past, [viewer, summary, table, files])
    demo.load(lambda: gr.update(choices=past_choices()), None, past)

write_viewer_index()
if RP is not None:
    threading.Thread(target=prefetch_pose_models, daemon=True).start()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=THEME, css=CSS, js=LIGHT_JS, allowed_paths=[str(WEB), str(DATA), str(TEMPLATES)],
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"))
