"""
sync_clap.py -- time-align a front and a side video using the clap sound.

No video editing needed: we only measure WHEN the clap happens in each file.

    offset_s = t_clap_side - t_clap_front

so a moment at time t in the front video is at  t + offset_s  in the side video.

Needs ffmpeg to read the audio:  pip install imageio-ffmpeg
Usage:
    python two_view/sync_clap.py front.mp4 side.mp4
"""
import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

LOW_CONFIDENCE = 8.0  # peak / median-noise ratio below this -> probably not a clap


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        import shutil
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError("ffmpeg not found. Run:  pip install imageio-ffmpeg")


def extract_audio(video, sr=16000, seconds=30.0):
    """First `seconds` of the video's audio as a mono float32 array."""
    exe = _ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        cmd = [exe, "-v", "error", "-i", str(video), "-t", str(seconds),
               "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "-y", str(wav_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not wav_path.exists():
            raise RuntimeError(f"Could not read audio from {video} (no audio track?): {res.stderr.strip()}")
        with wave.open(str(wav_path)) as w:
            raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) == 0:
        raise RuntimeError(f"{video} has no audio samples.")
    return x, sr


def detect_clap(x, sr, hop_ms=5, win_ms=10):
    """Returns (clap_time_s, confidence). Confidence = loudest sharp sound / typical noise level."""
    hp = np.diff(x, prepend=x[0])  # crude high-pass: claps are sharp and broadband
    hop, win = int(sr * hop_ms / 1000), int(sr * win_ms / 1000)
    n = (len(hp) - win) // hop
    env = np.array([np.sqrt(np.mean(hp[i * hop:i * hop + win] ** 2)) for i in range(n)])
    peak_i = int(np.argmax(env))
    conf = float(env[peak_i] / (np.median(env) + 1e-9))
    i = peak_i
    while i > 0 and env[i - 1] > 0.2 * env[peak_i]:  # walk back to the start of the bang
        i -= 1
    return i * hop / sr, conf


def sync_pair(front_video, side_video, seconds=30.0):
    xf, sr = extract_audio(front_video, seconds=seconds)
    xs, _ = extract_audio(side_video, seconds=seconds)
    t_f, c_f = detect_clap(xf, sr)
    t_s, c_s = detect_clap(xs, sr)
    return {
        "t_clap_front_s": round(t_f, 3),
        "t_clap_side_s": round(t_s, 3),
        "offset_s": round(t_s - t_f, 3),
        "confidence_front": round(c_f, 1),
        "confidence_side": round(c_s, 1),
        "reliable": bool(c_f >= LOW_CONFIDENCE and c_s >= LOW_CONFIDENCE),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("front")
    ap.add_argument("side")
    ap.add_argument("--seconds", type=float, default=30.0, help="how much audio from the start to search")
    a = ap.parse_args()
    r = sync_pair(a.front, a.side, a.seconds)
    for k, v in r.items():
        print(f"{k:>18}: {v}")
    if not r["reliable"]:
        print("WARNING: weak clap peak in at least one video -- check the audio / clap loudness.")
