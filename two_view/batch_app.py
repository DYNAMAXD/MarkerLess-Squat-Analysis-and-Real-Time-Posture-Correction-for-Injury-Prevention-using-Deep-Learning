"""
batch_app.py -- local web page that starts and watches batch_preprocess.py (whole-dataset preprocessing).

    C:\\Pose2SimVenv\\Scripts\\python.exe two_view/batch_app.py      then open http://127.0.0.1:5050
    (or just run two_view\\start_batch.ps1, which also sets up the environment)

1. Scan: give the dataset folder -> shows the calibration videos, the mapping CSV and which squat videos were found.
2. Settings: board sizes, board position, device, pose model.
3. Start: runs the batch in the background (it keeps running if the page is closed) and shows its progress,
   the calibration result, a per-person table, GPU load and the live logs. Starting again resumes.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from batch_preprocess import CALIB_KEYS, discover, video_info  # noqa: E402

LAST = ROOT / "outputs" / "batch_last.json"   # remembers the last batch so a reopened page finds it
app = Flask(__name__)
BATCH = {"proc": None, "output": None}
_gpu_cache = {"t": 0, "v": None}


def _remember(output):
    LAST.parent.mkdir(parents=True, exist_ok=True)
    LAST.write_text(json.dumps({"output": str(output)}), encoding="utf-8")


def _last_output():
    if BATCH["output"]:
        return BATCH["output"]
    try:
        return json.loads(LAST.read_text(encoding="utf-8"))["output"]
    except (OSError, ValueError, KeyError):
        return None


def _running():
    p = BATCH["proc"]
    return p is not None and p.poll() is None


def _pid_alive(pid):
    """A batch started by an earlier run of this page (still running after a restart)?"""
    if not pid:
        return False
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _tail(path, n):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            txt = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    # progress bars rewrite their line with '\r': keep the last state of each line (logs end lines with \r\n)
    lines = [ln.split("\r")[-1] for ln in txt.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines[-n:])


def _gpu():
    """nvidia-smi summary (cached 3 s), or None without an NVIDIA GPU."""
    if time.time() - _gpu_cache["t"] < 3:
        return _gpu_cache["v"]
    v = None
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            r = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            name, util, used, total, temp = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            v = {"name": name, "util": int(util), "mem_used": int(used), "mem_total": int(total), "temp": int(temp)}
        except Exception:  # noqa: BLE001
            v = None
    _gpu_cache.update(t=time.time(), v=v)
    return v


def _ext_mode(f):
    return "shared" if f.get("extrinsics") == "shared" else "per-video"


@app.get("/")
def index():
    return send_from_directory(HERE / "web", "batch.html")


@app.get("/api/device")
def device():
    info = {"gpu": _gpu(), "python": sys.executable}
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass
        info["onnxruntime"] = ort.__version__
        info["providers"] = ort.get_available_providers()
        info["cuda"] = "CUDAExecutionProvider" in info["providers"]
    except ImportError:
        info.update(onnxruntime=None, providers=[], cuda=False)
    return jsonify(info)


@app.post("/api/scan")
def scan():
    f = request.get_json(force=True)
    inp = (f.get("input") or "").strip().strip('"')
    if not inp:
        return jsonify({"error": "Give the dataset folder."}), 400
    plan = discover(inp, (f.get("output") or "").strip().strip('"') or None, (f.get("csv") or "").strip().strip('"') or None,
                    _ext_mode(f))
    plan["calib_info"] = {}
    for k in CALIB_KEYS:
        p = plan["calib"].get(k)
        if p:
            plan["calib_info"][k] = {"path": p, **video_info(p)}
    sizes = {(v["w"], v["h"]) for v in plan["calib_info"].values()}
    if len(sizes) > 1:
        plan["warnings"].append(f"The calibration videos have different resolutions {sorted(sizes)}; all videos must "
                                "be recorded at the same resolution and lens mode.")
    for k, v in plan["calib_info"].items():
        if k.startswith("intrinsic") and v["dur"] and v["dur"] < 15:
            plan["warnings"].append(f"{k} is only {v['dur']} s; ~40-60 s moving the board over the whole frame is better.")
    root = Path(plan["input"])
    for r in plan["rows"]:
        for cam in ("front", "side"):
            if r.get(cam):
                r[cam + "_rel"] = str(Path(r[cam]).relative_to(root))
    out = Path(plan["output"])
    plan["already_done"] = sum(1 for r in plan["rows"] if (out / r["folder"] / "results.json").exists())
    return jsonify(plan)


@app.post("/api/start")
def start():
    if _running():
        return jsonify({"error": "A batch is already running."}), 409
    f = request.get_json(force=True)
    inp = (f.get("input") or "").strip().strip('"')
    plan = discover(inp, (f.get("output") or "").strip().strip('"') or None, (f.get("csv") or "").strip().strip('"') or None,
                    _ext_mode(f))
    if plan["errors"]:
        return jsonify({"error": " ".join(plan["errors"])}), 400
    out = Path(plan["output"])
    st = out / "batch_status.json"
    if st.exists():
        try:
            old = json.loads(st.read_text(encoding="utf-8"))
            if old.get("state") in ("starting", "calibrating", "running") and _pid_alive(old.get("pid")):
                return jsonify({"error": f"A batch is already running on this output folder (process {old['pid']})."}), 409
        except ValueError:
            pass
    for key, what in (("intr_corners", "Intrinsic corners"), ("ext_corners", "Extrinsic corners")):
        if len(re.findall(r"\d+", str(f.get(key, "")))) != 2:
            return jsonify({"error": f"{what}: two numbers of INNER corners, e.g. 6,7"}), 400
    try:
        squares = [float(f.get("intr_square")), float(f.get("ext_square")), float(f.get("timeout_min", 90))]
        assert min(squares) > 0
    except (TypeError, ValueError, AssertionError):
        return jsonify({"error": "Square sizes (mm) and the timeout must be positive numbers."}), 400
    cmd = [sys.executable, "-u", str(HERE / "batch_preprocess.py"), "--input", plan["input"], "--output", str(out),
           "--intr-corners", ",".join(re.findall(r"\d+", f["intr_corners"])), "--intr-square", str(squares[0]),
           "--ext-corners", ",".join(re.findall(r"\d+", f["ext_corners"])), "--ext-square", str(squares[1]),
           "--board-position", "horizontal" if f.get("board_position") == "horizontal" else "vertical",
           "--extrinsics", _ext_mode(f), "--squat-claps", "2" if str(f.get("squat_claps")) == "2" else "1",
           "--device", f.get("device") if f.get("device") in ("auto", "cuda", "cpu") else "auto",
           "--pose-mode", f.get("pose_mode") if f.get("pose_mode") in ("lightweight", "balanced", "performance") else "performance",
           "--timeout-min", str(squares[2])]
    if plan["csv"]:
        cmd += ["--csv", plan["csv"]]
    for flag in ("redo", "redo_calibration", "keep_work"):
        if f.get(flag):
            cmd.append("--" + flag.replace("_", "-"))
    out.mkdir(parents=True, exist_ok=True)
    console = open(out / "batch_console.txt", "w", encoding="utf-8")
    console.write("$ " + " ".join(cmd) + "\n\n")
    console.flush()
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    BATCH.update(proc=subprocess.Popen(cmd, stdout=console, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                       creationflags=flags), output=str(out))
    _remember(out)
    return jsonify({"ok": True, "output": str(out)})


@app.post("/api/stop")
def stop():
    f = request.get_json(force=True)
    out = Path(f.get("output") or _last_output() or ".")
    if f.get("now"):
        pid = BATCH["proc"].pid if _running() else None
        if pid is None:
            try:
                pid = json.loads((out / "batch_status.json").read_text(encoding="utf-8")).get("pid")
            except (OSError, ValueError):
                pid = None
        if pid and _pid_alive(pid):
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, 9)
            st = out / "batch_status.json"
            try:
                s = json.loads(st.read_text(encoding="utf-8"))
                s["state"] = "stopped"
                for p in s.get("persons", []):
                    if p.get("state") == "running":
                        p.update(state="error", problem="stopped by the user", stage=None)
                st.write_text(json.dumps(s, indent=1), encoding="utf-8")
            except (OSError, ValueError):
                pass
        return jsonify({"ok": True})
    out.mkdir(parents=True, exist_ok=True)
    (out / "STOP").write_text("stop after the current person\n", encoding="utf-8")
    return jsonify({"ok": True, "after_current": True})


@app.get("/api/status")
def status():
    out = request.args.get("output") or _last_output()
    if not out:
        return jsonify({"status": None})
    out = Path(out)
    st = None
    try:
        st = json.loads((out / "batch_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    alive = _running() if BATCH["output"] == str(out) else _pid_alive(st and st.get("pid"))
    if st and not alive and st.get("state") in ("starting", "calibrating", "running"):
        st["state"] = "interrupted"  # the process is gone without writing a final state
    cur_log = ""
    if st:
        cur = st.get("current")
        if st.get("state") == "calibrating":
            cur_log = _tail(out / "_calibration" / "log.txt", 40)
        elif cur:
            cur_log = _tail(out / cur / "log.txt", 40)
    console = _tail(out / "batch_console.txt", 30) if (not st or st.get("state") == "error" or not alive) else ""
    return jsonify({"output": str(out), "alive": alive, "status": st, "batch_log": _tail(out / "batch_log.txt", 80),
                    "current_log": cur_log, "console": console, "gpu": _gpu(), "stop_requested": (out / "STOP").exists(),
                    "now": time.time()})


@app.get("/api/file")
def file():
    out = Path(request.args.get("output") or _last_output() or "").resolve()
    rel = request.args.get("path", "")
    p = (out / rel).resolve()
    if not rel or out not in p.parents or not p.is_file() or p.suffix.lower() not in (".jpg", ".png", ".mp4", ".csv", ".txt", ".json", ".toml"):
        abort(404)
    return send_file(p, conditional=True)


if __name__ == "__main__":
    port = int(os.environ.get("BATCH_PORT", "5050"))
    import socket
    with socket.socket() as s:  # on Windows a second server can bind the same port and old code keeps answering
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"The batch page is already running at http://127.0.0.1:{port} - close that window first "
                  "(or just open the address).")
            sys.exit(1)
    print(f"Batch preprocessing page: http://127.0.0.1:{port}   (Ctrl+C or close this window to stop the page;"
          " a running batch keeps going only while this window is open)")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
