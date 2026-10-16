"""
p2s_sync.py -- clap sync of the side + front videos that does not trust "the loudest sound in each file".

Two microphones hear things differently: the loudest sharp sound in one video can be a different event from the
loudest in the other (or happen before the other camera started). So instead of picking one peak per video we
cross-correlate the whole pattern of sharp sounds (claps, steps, the board being handled...) of both audio tracks.
The lag where most events line up is the offset. The clap is then the strongest event heard in BOTH videos.

    offset_s = t_front - t_side      (a moment at time t in the side video is at t + offset_s in the front video)
"""
import numpy as np

from sync_clap import extract_audio

HOP_S = 0.005          # 5 ms envelope resolution
EVENT_RATIO = 4.0      # a "sharp sound" is at least this many times the median background level
MATCH_TOL_S = 0.03     # events closer than this (after the offset) count as the same sound
CLAP_REL_STRENGTH = 0.15  # two-clap videos: the second clap is at least this fraction of the loudest (strength product)
MIN_CLAP_GAP_S = 3.0      # ... and at least this far from it


def onset_envelope(video, seconds, sr=16000):
    """Sharp-sound envelope (high-passed RMS in 10 ms windows, 5 ms hop), relative to the median background."""
    x, sr = extract_audio(video, sr=sr, seconds=seconds)
    hp = np.diff(x, prepend=x[0])  # claps are sharp and broadband
    hop, win = int(sr * HOP_S), int(sr * 2 * HOP_S)
    env = np.sqrt(np.convolve(hp ** 2, np.ones(win) / win, mode="valid")[::hop])
    return env / (np.median(env) + 1e-12)


def _events(env, min_gap_s=0.25):
    from scipy.signal import find_peaks
    p, _ = find_peaks(env, height=EVENT_RATIO, distance=max(1, int(min_gap_s / HOP_S)))
    return p


def _onset_start(env, i, max_back_s=0.05):
    """Walk back from a peak to where the bang starts (rule of sync_clap.detect_clap), at most max_back_s, so a
    noisy background cannot drag the clap seconds earlier."""
    lo, peak = max(0, i - int(max_back_s / HOP_S)), env[i]
    while i > lo and env[i - 1] > 0.2 * peak:
        i -= 1
    return i


def sync_side_front(side, front, seconds=90.0, log=print, claps=1):
    """claps=1: one clap in the first `seconds` (= the strongest sound heard in both videos).
    claps=2: a clap at the start AND at the end of the squats. The whole videos are searched; the two claps are the
    strongest sound heard in both and the strongest other loud one at least MIN_CLAP_GAP_S away (earlier = start).
    Also returns t_end_side_s / t_end_front_s and drift_s (how much the front clock ran ahead of the side clock
    between the two claps); the end values are None if no end clap was heard."""
    if claps == 2:
        seconds = None
    es, ef = onset_envelope(side, seconds), onset_envelope(front, seconds)
    ev_s, ev_f = _events(es), _events(ef)
    # one vote per sharp sound (a tiny loudness term only breaks ties): a single loud, echoing bang must not
    # outvote several sounds that line up
    cs, cf = np.zeros(len(es)), np.zeros(len(ef))
    cs[ev_s] = 1 + 0.01 * np.log(es[ev_s])
    cf[ev_f] = 1 + 0.01 * np.log(ef[ev_f])
    tol = max(1, int(round(MATCH_TOL_S / HOP_S)))
    k = np.ones(2 * tol + 1)
    cs_s, cf_s = np.convolve(cs, k, "same"), np.convolve(cf, k, "same")  # tolerate a few ms of jitter
    n = len(cs_s) + len(cf_s)
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(cf_s, nfft) * np.conj(np.fft.rfft(cs_s, nfft)), nfft)
    lags = np.concatenate([np.arange(0, len(cf_s)), np.arange(-len(cs_s) + 1, 0)])
    corr = np.concatenate([corr[:len(cf_s)], corr[nfft - len(cs_s) + 1:]])
    best = int(np.argmax(corr))
    lag = int(lags[best])
    offset = lag * HOP_S

    # which sharp sounds line up at this offset?
    matches = []
    for i in ev_s:
        j_exp = i + lag
        near = ev_f[np.abs(ev_f - j_exp) <= tol] if len(ev_f) else []
        if len(near):
            j = int(near[np.argmin(np.abs(near - j_exp))])
            matches.append((int(i), j, float(es[i] * ef[j])))
    others = [c for c, l in zip(corr, lags) if abs(l - lag) * HOP_S > 0.2]
    peak_ratio = float(corr[best] / (max(others) + 1e-9)) if others else np.inf
    log(f"[sync] sharp sounds: {len(ev_s)} in side, {len(ev_f)} in front; {len(matches)} line up at offset "
        f"{offset:+.3f} s (front - side); correlation peak {peak_ratio:.1f}x the next best")
    for i, j, _ in matches[:12]:
        log(f"[sync]   side {i * HOP_S:7.2f} s  <->  front {j * HOP_S:7.2f} s   "
            f"(strength side {es[i]:.0f}, front {ef[j]:.0f})")
    if not matches:
        raise RuntimeError("No sharp sound (clap) is heard in both videos. Check that both have audio and a clear clap.")

    i, j, _ = max(matches, key=lambda m: m[2])  # the clap = strongest sound heard in both videos
    end_clap = None
    if claps == 2:  # the other clap: the strongest remaining clap-like sound at least MIN_CLAP_GAP_S away
        loud = [m for m in matches if m[2] >= CLAP_REL_STRENGTH * max(x[2] for x in matches)
                and abs(m[0] - i) * HOP_S >= MIN_CLAP_GAP_S]
        if loud:
            other = max(loud, key=lambda m: m[2])
            (i, j, _), end_clap = sorted([(i, j, 0), other])  # earlier = start clap, later = end clap
    t0 = _onset_start(es, i) * HOP_S  # front clap = side clap + offset (consistent by construction)
    reliable = ((len(matches) >= 3 and peak_ratio >= 1.3) or                       # several sounds agree
                (len(ev_s) <= 2 and len(ev_f) <= 2 and es[i] >= 8 and ef[j] >= 8))  # quiet videos, one clear clap
    if not reliable:
        log("[sync] WARNING: the offset is not clearly determined (few matching sounds) -- check the result.")
    out = {"t_clap_side_s": round(t0, 3), "t_clap_front_s": round(t0 + offset, 3),
           "offset_s": round(offset, 3), "confidence_side": round(float(es[i]), 1),
           "confidence_front": round(float(ef[j]), 1), "matched_sounds": len(matches),
           "reliable": bool(reliable)}
    if claps == 2:
        out.update(t_end_side_s=None, t_end_front_s=None, drift_s=0.0)
        if end_clap is None:
            log(f"[sync] WARNING: only one clap heard in both videos (at side {t0:.2f} s); using it as the start clap, "
                "the squats run to the end of the video.")
            return out
        i2, j2, _ = end_clap
        # drift = change of the per-clap offset between the claps (each measured the same way, so mic bias cancels)
        drift = ((_onset_start(ef, j2) - _onset_start(es, i2)) - (_onset_start(ef, j) - _onset_start(es, i))) * HOP_S
        t1 = _onset_start(es, i2) * HOP_S
        out.update(t_end_side_s=round(t1, 3), t_end_front_s=round(t1 + offset + drift, 3), drift_s=round(drift, 3),
                   confidence_end_side=round(float(es[i2]), 1), confidence_end_front=round(float(ef[j2]), 1))
        log(f"[sync] start clap: side {t0:.3f} s / front {t0 + offset:.3f} s; end clap: side {t1:.3f} s / front "
            f"{t1 + offset + drift:.3f} s (strength side {es[i2]:.0f}, front {ef[j2]:.0f}); clock drift between the "
            f"claps {drift * 1000:+.0f} ms")
    return out
