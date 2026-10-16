"""
product_pipeline.py -- the background jobs of the final product (product_app.py). One process per job; the page reads
the job's status.json. Runs with the project environment (C:\\Proj1Venv); Pose2Sim steps run in C:\\Pose2SimVenv.

    python final_product/product_pipeline.py session <outputs/product/sessions/<id>>
    python final_product/product_pipeline.py camera  <outputs/product/cameras/<id>>

Session (front + side squat video -> results page):
  check        both videos readable, same resolution as the camera profile's intrinsics
  preprocess   run_pose2sim.py with the profile's intrinsics (own board before the clap); if the board is not
               found and the profile has a fallback calibration, again with --calib-from it.
               Its own stages (sync, calibration, extract, pose, triangulation, angles, packaging) show up live.
  mesh         p2s_mesh.py --method motionbert --camera side   (failure is not fatal: page without the mesh)
  model        trained model -> predictions.json; without one: the reps without errors (model_active = false)
  finish       rep_results.csv + summary in session.json
Finished steps are skipped when a session is run again (Retry).

Camera profile (the user's own cameras + board):
  intrinsic    run_pose2sim.py --mode intrinsic on the two intrinsic videos -> intrinsics.toml
  extrinsic    (optional) run_pose2sim.py --mode extrinsic on extrinsic videos -> fallback.toml

A file named CANCEL in the job folder stops the job.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import product_predictions as PP  # noqa: E402

PRODUCT = ROOT / "outputs" / "product"
WORK = PRODUCT / "_work"
P2S_PY = next((str(p) for p in (Path("C:/Pose2SimVenv/Scripts/python.exe"),) if p.exists()), None)
TWO_VIEW = ROOT / "two_view"   # the preprocessing scripts (run_pose2sim.py, p2s_mesh.py)
RUN = TWO_VIEW / "run_pose2sim.py"
MODEL_PY = Path(os.environ.get("SQUAT_MODEL_PY", "C:/SquatModelVenv/Scripts/python.exe"))
MODEL_DIR = Path(os.environ.get("SQUAT_MODEL_DIR", HERE / "squat_model_export"))   # model.pt, config.json, ...
MODEL_CODE = Path(os.environ.get("SQUAT_MODEL_CODE", TWO_VIEW))  # folder that holds the squat_model package
POSE_MODE = os.environ.get("PRODUCT_POSE_MODE", "performance")   # same default as the dataset batch
DEVICE = os.environ.get("PRODUCT_DEVICE", "auto")                # falls back to the CPU without a usable GPU
SESSION_STAGES = ["check", "sync", "calibration", "extract", "pose", "triangulation", "angles", "mesh", "model", "finish"]
CAMERA_STAGES = ["check", "intrinsic", "extrinsic", "finish"]
P2S_TO_STAGE = {"setup": "check", "sync": "sync", "calibration": "calibration", "extract": "extract", "pose": "pose",
                "triangulation": "triangulation", "angles": "angles", "packaging": "angles", "done": "angles"}


class Cancelled(Exception):
    pass


class JobError(Exception):
    """A failure with a message meant for the user."""


# --------------------------------------------------------------------------- helpers
def read_json(p, default=None):
    return PP.load_json(p, default)


def write_json(p, data):
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    for _ in range(20):  # Windows: the page may be reading the file right now
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, p)


def read_log(path):
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(ln.split("\r")[-1] for ln in txt.replace("\r\n", "\n").split("\n"))


def error_line(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"^[A-Za-z0-9_.]*(Error|Exception|BoardNotFound)\b", ln):
            return ln[:400]
    return lines[-1][:400] if lines else "no log written"


def friendly(err):
    """Turn the pipeline's last error line into advice."""
    e = err.lower()
    if "boardnotfound" in e:
        return ("The calibration board was not found in both videos before the clap. Hold the board still where both "
                "cameras can see it for a few seconds before clapping, or add fallback extrinsic videos to the camera "
                "profile.")
    if "clap" in e:
        return f"The clap could not be found in both videos: {err}"
    if "intrinsics are for" in e or "calibration is for" in e:
        return f"The videos do not match the camera profile (resolution / zoom): {err}"
    return err


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        ok, _ = cap.read()
        return {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 3), "dur": round(n / fps, 2) if fps else 0, "readable": ok}
    finally:
        cap.release()


def calib_sizes(toml_path):
    import tomllib
    d = tomllib.loads(Path(toml_path).read_text(encoding="utf-8"))
    return {k: v["size"] for k, v in d.items() if isinstance(v, dict) and "size" in v}


class Job:
    def __init__(self, folder, stages):
        self.dir = Path(folder).resolve()
        self.status_path = self.dir / "status.json"
        old = read_json(self.status_path, {}) or {}
        self.st = {"state": "running", "stage": None, "stages": stages, "done": [], "progress": None,
                   "started": time.time(), "updated": time.time(), "error": None,
                   "warnings": old.get("warnings", []) if old.get("state") != "done" else [], "pid": os.getpid()}
        self.log_fh = open(self.dir / "pipeline_log.txt", "a", encoding="utf-8")
        self.save()

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_fh.write(line + "\n")
        self.log_fh.flush()

    def save(self):
        self.st["updated"] = time.time()
        write_json(self.status_path, self.st)

    def stage(self, name):
        if self.st["stage"] and self.st["stage"] != name and self.st["stage"] not in self.st["done"]:
            self.st["done"].append(self.st["stage"])
        self.st["stage"], self.st["progress"] = name, None
        self.log(f"stage: {name}")
        self.save()

    def warn(self, msg):
        if msg not in self.st["warnings"]:
            self.st["warnings"].append(msg)
        self.log("WARNING: " + msg)
        self.save()

    def check_cancel(self):
        if (self.dir / "CANCEL").exists():
            raise Cancelled()

    def run_child(self, cmd, console, timeout_s, on_tick=None, env=None):
        """Run a child process (output -> console file); poll for cancel / timeout. Returns (exit code, output)."""
        self.log("$ " + " ".join(str(c) for c in cmd))
        with open(console, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                    env={**os.environ, **env} if env else None)
            t0 = time.time()
            try:
                while proc.poll() is None:
                    time.sleep(2)
                    if on_tick:
                        on_tick(read_log(console))
                    self.check_cancel()
                    if time.time() - t0 > timeout_s:
                        kill_tree(proc)
                        raise JobError(f"Step took longer than {timeout_s / 60:.0f} minutes and was stopped.")
            except BaseException:
                kill_tree(proc)
                raise
        return proc.returncode, read_log(console)

    def finish(self, state, error=None):
        if self.st["stage"] and state == "done" and self.st["stage"] not in self.st["done"]:
            self.st["done"].append(self.st["stage"])
        self.st.update(state=state, error=error, finished=time.time(), progress=None)
        (self.dir / "CANCEL").unlink(missing_ok=True)
        self.save()
        self.log(f"job {state}" + (f": {error}" if error else ""))
        self.log_fh.close()


def p2s_ticker(job):
    """Follows run_pose2sim's "[stage] x" lines and its frame counters while it runs."""
    def tick(text):
        stage, prog = None, None
        for line in text.splitlines():
            if line.startswith("[stage] "):
                stage, prog = line[8:].strip(), None
            m = re.search(r"Processed (\d+)/(\d+) frames", line) or re.search(r"\| *(\d+)/(\d+) \[", line)
            if m and int(m.group(2)) > 0:
                prog = {"done": int(m.group(1)), "total": int(m.group(2))}
        s = P2S_TO_STAGE.get(stage)
        if s and s in job.st["stages"] and s != job.st["stage"]:
            job.stage(s)
        if prog != job.st["progress"]:
            job.st["progress"] = prog
            job.save()
    return tick


def board_args(board):
    return ["--intr-corners", board.get("intr_corners", "6,7"), "--intr-square", str(board.get("intr_square", 41)),
            "--ext-corners", board.get("ext_corners", "6,7"), "--ext-square", str(board.get("ext_square", 41)),
            "--board-position", board.get("board_position", "vertical"),
            "--clap-window", str(board.get("clap_window", 90))]


# --------------------------------------------------------------------------- session
def run_session(folder):
    job = Job(folder, SESSION_STAGES)
    d = job.dir
    try:
        sess = read_json(d / "session.json")
        if not sess:
            raise JobError("session.json is missing.")
        cam_dir = PRODUCT / "cameras" / sess["camera"]
        profile = read_json(cam_dir / "profile.json")
        intr = cam_dir / "intrinsics.toml"
        if not profile or not intr.exists():
            raise JobError(f"Camera profile '{sess['camera']}' is missing or not calibrated yet.")
        if not P2S_PY:
            raise JobError("The Pose2Sim environment C:\\Pose2SimVenv is missing. Run two_view\\start_batch.ps1 once to create it.")
        vids = {"front": Path(sess["front"]), "side": Path(sess["side"])}

        # ---- check
        if not (d / "results.json").exists():
            job.stage("check")
            sizes = list(calib_sizes(intr).items())
            for cam, p in vids.items():
                if not p.is_file():
                    raise JobError(f"The {cam} video is missing: {p}")
                info = video_info(p)
                if not info or not info["readable"]:
                    raise JobError(f"The {cam} video cannot be read: {p.name}")
                if info["dur"] < 5:
                    raise JobError(f"The {cam} video is only {info['dur']} s long.")
                want = next((s for k, s in sizes if cam in k), None)
                if want and [info["w"], info["h"]] != list(want):
                    raise JobError(f"The {cam} video is {info['w']}x{info['h']} but the camera profile was calibrated at "
                                   f"{want[0]}x{want[1]}. Record at the same resolution and zoom as the calibration.")
                job.log(f"{cam}: {p.name} {info['w']}x{info['h']} {info['fps']} fps {info['dur']} s")
            job.check_cancel()

            # ---- preprocess (run_pose2sim)
            (d / "meta.json").write_text(json.dumps({"name": sess["name"], "kind": "pose2sim", "person_id": sess["id"],
                                                     "subject": sess["name"], "created": sess["created"]}), encoding="utf-8")
            attempts = [("own board", ["--intr-from", str(intr)])]
            if (cam_dir / "fallback.toml").exists():
                attempts.append(("fallback calibration", ["--calib-from", str(cam_dir / "fallback.toml")]))
            WORK.mkdir(parents=True, exist_ok=True)
            for i, (label, calib) in enumerate(attempts):
                cmd = [P2S_PY, "-u", RUN, "--name", sess["id"], "--side", vids["side"], "--front", vids["front"], *calib,
                       "--squat-claps", "1", *board_args(profile.get("board", {})), "--device", DEVICE,
                       "--pose-mode", POSE_MODE, "--out-dir", d, "--work-dir", WORK / sess["id"]]
                code, out = job.run_child(cmd, d / f"preprocess_console_{i + 1}.txt", 3 * 3600, p2s_ticker(job))
                if code == 0 and (d / "results.json").exists():
                    if i:
                        job.warn("The calibration board was not found in these videos, so the camera profile's "
                                 "fallback calibration was used. Results are only right if the cameras did not move.")
                    break
                err = error_line(read_log(d / "log.txt") or out)
                if "BoardNotFound" in err and i + 1 < len(attempts):
                    shutil.copyfile(d / "log.txt", d / "log_own_board.txt")
                    job.log("board not found -> trying the fallback calibration")
                    continue
                raise JobError(friendly(err))
            res = read_json(d / "results.json", {})
            if not (res.get("sync") or {}).get("reliable", True):
                job.warn("The clap was not clearly heard by both cameras; the two videos may be slightly out of sync.")
            if not res.get("reps"):
                job.warn("No squat reps were found in the video.")
        else:
            job.log("preprocessing already done - skipped")
            job.st["done"] += [s for s in SESSION_STAGES[:7] if s not in job.st["done"]]
        job.check_cancel()

        # ---- mesh
        job.stage("mesh")
        if (d / "mesh" / "done.txt").exists():
            job.log("mesh already built - skipped")
        else:
            code, out = job.run_child([sys.executable, "-u", TWO_VIEW / "p2s_mesh.py", d, "--method", "motionbert",
                                       "--camera", "side"], d / "mesh_console.txt", 30 * 60)
            if not (d / "mesh" / "done.txt").exists():
                job.warn("The 3D body mesh could not be built (" + error_line(read_log(d / "mesh" / "log.txt") or out) +
                         "). Everything else is available.")
        job.check_cancel()

        # ---- model
        job.stage("model")
        pred = None
        if (MODEL_DIR / "model.pt").exists() and MODEL_PY.exists():
            code, out = job.run_child([MODEL_PY, "-u", "-m", "squat_model.predict", d, "--model", MODEL_DIR],
                                      d / "model_console.txt", 20 * 60, env={"PYTHONPATH": str(MODEL_CODE)})
            pred = read_json(d / "predictions.json") if code == 0 else None
            if pred is None:
                job.warn("The error-detection model failed (" + error_line(out) + "). Showing the reps without errors.")
        else:
            job.log(f"no trained model at {MODEL_DIR} - reps are shown without errors")
        if pred is None:
            pred = PP.inactive_predictions(d)
        pred.setdefault("model_active", True)

        # ---- finish
        job.stage("finish")
        summ = PP.write_all(d, pred)
        sess["summary"] = summ
        write_json(d / "session.json", sess)
        job.finish("done")
    except Cancelled:
        job.finish("cancelled", "Stopped by the user.")
    except JobError as e:
        job.finish("error", str(e))
    except Exception as e:  # noqa: BLE001
        job.log(traceback.format_exc())
        job.finish("error", f"Unexpected error: {e}")


# --------------------------------------------------------------------------- camera profile
def run_camera(folder):
    job = Job(folder, CAMERA_STAGES)
    d = job.dir
    try:
        prof = read_json(d / "profile.json")
        if not P2S_PY:
            raise JobError("The Pose2Sim environment C:\\Pose2SimVenv is missing. Run two_view\\start_batch.ps1 once to create it.")
        board, vids = prof.get("board", {}), prof.get("videos", {})
        job.stage("check")
        for k in ("intr_front", "intr_side", "ext_front", "ext_side"):
            if vids.get(k):
                info = video_info(vids[k])
                if not info or not info["readable"]:
                    raise JobError(f"Video cannot be read: {Path(vids[k]).name}")
                job.log(f"{k}: {Path(vids[k]).name} {info['w']}x{info['h']} {info['fps']} fps {info['dur']} s")
        WORK.mkdir(parents=True, exist_ok=True)
        if not (d / "intrinsics.toml").exists():
            job.stage("intrinsic")
            out_dir = d / "intrinsic"
            cmd = [P2S_PY, "-u", RUN, "--mode", "intrinsic", "--name", f"intr_{d.name}", "--intr-side", vids["intr_side"],
                   "--intr-front", vids["intr_front"], *board_args(board), "--out-dir", out_dir,
                   "--work-dir", WORK / f"cam_{d.name}_intr"]
            code, out = job.run_child(cmd, d / "intrinsic_console.txt", 2 * 3600, p2s_ticker(job))
            if code != 0 or not (out_dir / "calibration.toml").exists():
                raise JobError("Intrinsic calibration failed: " + friendly(error_line(read_log(out_dir / "log.txt") or out)))
            shutil.copyfile(out_dir / "calibration.toml", d / "intrinsics.toml")
        job.check_cancel()
        if vids.get("ext_front") and vids.get("ext_side") and not (d / "fallback.toml").exists():
            job.stage("extrinsic")
            out_dir = d / "extrinsic"
            cmd = [P2S_PY, "-u", RUN, "--mode", "extrinsic", "--name", f"ext_{d.name}", "--side", vids["ext_side"],
                   "--front", vids["ext_front"], "--intr-from", d / "intrinsics.toml", *board_args(board),
                   "--out-dir", out_dir, "--work-dir", WORK / f"cam_{d.name}_ext"]
            code, out = job.run_child(cmd, d / "extrinsic_console.txt", 2 * 3600, p2s_ticker(job))
            if code != 0 or not (out_dir / "calibration.toml").exists():
                job.warn("The fallback (extrinsic) calibration failed: " +
                         friendly(error_line(read_log(out_dir / "log.txt") or out)) + " The profile works without it.")
            else:
                shutil.copyfile(out_dir / "calibration.toml", d / "fallback.toml")
        job.stage("finish")
        job.finish("done")
    except Cancelled:
        job.finish("cancelled", "Stopped by the user.")
    except JobError as e:
        job.finish("error", str(e))
    except Exception as e:  # noqa: BLE001
        job.log(traceback.format_exc())
        job.finish("error", f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("session", "camera"):
        sys.exit(__doc__)
    (run_session if sys.argv[1] == "session" else run_camera)(sys.argv[2])
