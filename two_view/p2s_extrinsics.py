"""
p2s_extrinsics.py -- extrinsic calibration of the side + front cameras from the checkerboard held before the clap.

Why not Pose2Sim's own extrinsic step: it re-detects the board with the plain OpenCV detector on one image per
camera, which fails on boards seen at a steep angle (typical with cameras at 90 deg). Here we

  1. scan both videos up to the clap (board -> clap -> squats videos; the whole videos only if the board is not
     there) or the whole of short extrinsic-only videos (coarse pass on a few CPU processes), then check frame by
     frame -- walking back from the clap, then forward, pairing the cameras with the clap offset
     (side 549 <-> front 449, 548 <-> 448, ...) -- where both cameras may see the board, until enough still pairs
     are found; use the pair where the board is still (same place in the neighbouring frame, so a +-1 frame sync
     error does not matter) and largest;
  2. detect with the classic detector, then the more robust sector-based one (findChessboardCornersSB);
  3. put the corners in the SAME physical order in both views (handedness + which squares are dark --
     needs one even and one odd corner count, e.g. 6x7);
  4. solve each camera's pose with cv2.solvePnP using the intrinsics, and write a Pose2Sim-format Calib toml.

World frame: origin at the board's corner 0 (marked "origin" on board_side.jpg / board_front.jpg), Z up.
"Up" = the average up direction of the two cameras (assumes level tripods; works for a tilted, hand-held board)
or, for a board flat on the floor, the board's normal. Pose2Sim's .trc output turns this into Y up.
"""
import os
from pathlib import Path

import cv2
import numpy as np

DETECT_MAX_SIDE = 1600   # board search runs on frames downscaled to this size (speed); corners refined at full res
BLOCK = 60               # frames decoded per block in the frame-by-frame pass
COARSE_STEP_S = 0.25     # coarse pass: check one frame every this many seconds (whole video)
MAX_SCAN_WORKERS = 4     # coarse-pass processes (12 exhausted a 16 GB machine's page file)
STILL_PX = 1.5          # max mean corner motion (full-res px) between neighbouring frames to call the board still
ENOUGH_STILL = 45        # frame-by-frame pass stops once this many still pairs were found (~1.5 s at 30 fps)


# --------------------------------------------------------------------------- detection
def detect(gray, pattern):
    """Returns (corners Nx2, detector name) or (None, None)."""
    ok, c = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                                      | cv2.CALIB_CB_FAST_CHECK)
    if ok:
        return c.reshape(-1, 2), "classic"
    ok, c = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        return c.reshape(-1, 2), "sb"
    return None, None


def _sample(gray, p, r=3):
    h, w = gray.shape
    x, y = int(np.clip(round(p[0]), r, w - r - 1)), int(np.clip(round(p[1]), r, h - r - 1))
    return float(gray[y - r:y + r + 1, x - r:x + r + 1].mean())


def canonical_order(corners, pattern, gray):
    """Reorder detected corners so index 0 is the same PHYSICAL corner in every view.
    The detectors may return the grid starting from any of the 4 ends. We keep the orderings with a fixed image
    handedness (both cameras see the same face, so no mirroring); the remaining two differ by 180 deg. With one even
    and one odd corner count, a 180 deg turn swaps the colour of every inner square's parity class, so we pick the
    ordering in which the squares with (row + col) even are the DARK ones -- measured over all inner squares, which
    stays reliable when the board is small or blurred in one view."""
    nx, ny = pattern
    g = corners.reshape(ny, nx, 2)
    cands = [g, g[::-1], g[:, ::-1], g[::-1, ::-1]]

    def cross(c):
        a, b = c[0, 1] - c[0, 0], c[1, 0] - c[0, 0]
        return a[0] * b[1] - a[1] * b[0]

    cands = [c for c in cands if cross(c) > 0]
    side_px = np.median(np.linalg.norm(np.diff(g, axis=1), axis=2))
    r = int(np.clip(side_px / 5, 1, 4))

    def even_minus_odd(c):  # brightness of (row+col)-even inner squares minus odd ones; negative = even are dark
        centers = 0.25 * (c[:-1, :-1] + c[1:, :-1] + c[:-1, 1:] + c[1:, 1:])
        ii, jj = np.indices(centers.shape[:2])
        vals = np.array([[_sample(gray, centers[i, j], r) for j in range(centers.shape[1])] for i in range(centers.shape[0])])
        return vals[(ii + jj) % 2 == 0].mean() - vals[(ii + jj) % 2 == 1].mean()

    return min(cands, key=even_minus_odd).reshape(-1, 2)


def _small(frame, scale):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _read_range(cap, a, b, scale):
    """Frames a..b (inclusive) as {index: downscaled gray}. One seek, then sequential decoding."""
    out = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    for i in range(a, b + 1):
        ok, f = cap.read()
        if not ok:
            break
        out[i] = _small(f, scale)
    return out


def _read_full(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx} of {path}")
    return f


# --------------------------------------------------------------------------- whole-video search
def _ranges_s(idxs, fps):
    """[3,4,5,90,91] frames -> '0.1-0.2 s, 3.0-3.0 s' (merging gaps under 1 s)."""
    if not idxs:
        return "never"
    idxs, out, a = sorted(idxs), [], None
    for k, i in enumerate(idxs):
        if a is None:
            a = i
        if k == len(idxs) - 1 or idxs[k + 1] - i > fps:
            out.append(f"{a / fps:.1f}-{i / fps:.1f} s")
            a = None
    return ", ".join(out)


def _scan_chunk(args):
    """Worker: decode frames a..b-1 sequentially, run the detector on every `step`-th one."""
    path, a, b, step, scale, pattern = args
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    hits = []
    for i in range(a, b):
        if not cap.grab():
            break
        if (i - a) % step:
            continue
        ok, f = cap.retrieve()
        if ok:
            c, _ = detect(_small(f, scale), pattern)
            if c is not None:
                hits.append((i, c))
    cap.release()
    return path, hits


def _detect_near(gray, ref, pattern):
    """Detector on a crop around where the board was last seen (much faster than the whole frame)."""
    h, w = gray.shape
    (x0, y0), (x1, y1) = ref.min(0), ref.max(0)
    m = 0.8 * max(x1 - x0, y1 - y0) + 20
    xa, ya, xb, yb = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(w, x1 + m)), int(min(h, y1 + m))
    c, _ = detect(np.ascontiguousarray(gray[ya:yb, xa:xb]), pattern)
    return None if c is None else c + [xa, ya]


class BoardNotFound(RuntimeError):
    pass


def find_board(videos, fps, clap_frames, pattern, log=print, step_s=None, before_clap=False):
    """Find a frame pair where both cameras see the board, still, at the same synced instant.
    before_clap=True (board -> clap -> squats videos): only the part up to the clap is searched; the whole videos
    only if the board is not found there. before_clap=False (short extrinsic-only videos): the whole videos."""
    if before_clap:
        try:
            return _find_board(videos, fps, clap_frames, pattern, log, step_s, before_clap=True)
        except BoardNotFound as e:
            log(f"[calibration] {e}\n[calibration] not found before the clap; searching the whole videos ...")
    return _find_board(videos, fps, clap_frames, pattern, log, step_s, before_clap=False)


def _find_board(videos, fps, clap_frames, pattern, log, step_s, before_clap):
    """Search both videos (up to the clap, or whole) for frame pairs where both cameras see the board at the same
    synced instant, then pick the best pair: the board must be still (found at the same place in the neighbouring
    frame pair, so a +-1 frame sync error does not matter), and among those the one where the board is largest in
    the camera that sees it smallest.

      1. coarse pass: every `step_s` of each video, in parallel on a few CPU processes
      2. frame-by-frame pass: every frame in both cameras wherever both saw the board (paired via the clap sync),
         walking back from the clap and then forward from it; stops once ENOUGH_STILL still pairs were found
    Returns {'side': (frame_idx, corners_fullres), 'front': (...)}."""
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    step_s = step_s or COARSE_STEP_S
    caps = {k: cv2.VideoCapture(str(v)) for k, v in videos.items()}
    info = {k: (int(c.get(cv2.CAP_PROP_FRAME_COUNT)), int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for k, c in caps.items()}
    scale = {k: min(1.0, DETECT_MAX_SIDE / max(v[1:])) for k, v in info.items()}
    cs, cf = clap_frames["side"], clap_frames["front"]
    # frames searched per camera: up to the clap (+ a few frames of slack) or the whole video
    end = {k: min(info[k][0], clap_frames[k] + 2) if before_clap else info[k][0] for k in videos}

    def front_of(i):  # side frame -> front frame at the same instant
        return int(round((i - cs) / fps["side"] * fps["front"] + cf))

    def side_of(j):
        return int(round((j - cf) / fps["front"] * fps["side"] + cs))

    # ---- 1. coarse pass over both whole videos
    step = {k: max(1, int(round(step_s * fps[k]))) for k in videos}
    # each spawned worker re-imports run_pose2sim + cv2 and decodes 1080p, several hundred MB each: keep the pool small
    workers = max(1, min(MAX_SCAN_WORKERS, (os.cpu_count() or 2) - 1))
    jobs = []
    for cam, path in videos.items():
        n = end[cam]
        size = max(step[cam], (n // (2 * workers) // step[cam] + 1) * step[cam])
        jobs += [(str(path), a, min(n, a + size), step[cam], scale[cam], tuple(pattern)) for a in range(0, n, size)]
    log(f"[calibration] scanning {'up to the clap' if before_clap else 'the whole videos'}: side {end['side']} "
        f"frames, front {end['front']} frames, for the board, every {step_s:g} s, on {workers} CPU processes ...")
    coarse = {"side": {}, "front": {}}
    by_path = {str(p): c for c, p in videos.items()}
    try:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_scan_chunk, jobs))
    except BrokenProcessPool:  # a worker died, usually out of memory ("paging file is too small")
        log("[calibration] a scan process ran out of memory; scanning again in this process (slower) ...")
        results = [_scan_chunk(j) for j in jobs]
    for path, hits in results:
        for i, c in hits:
            coarse[by_path[path]][i] = c
    for cam in coarse:
        log(f"[calibration] {cam} camera sees the board at {_ranges_s(list(coarse[cam]), fps[cam])}")

    # ---- 2. frame-by-frame wherever both cameras saw it (in synced time)
    near_s = {i for h in coarse["side"] for i in range(h - step["side"], h + step["side"] + 1)}
    near_f = {side_of(j) for h in coarse["front"] for j in range(h - step["front"], h + step["front"] + 1)}
    cand = sorted(i for i in near_s & near_f if 0 <= i < end["side"] and 0 <= front_of(i) < end["front"])
    if not cand:
        raise BoardNotFound(
            f"The two cameras never see the {pattern[0]}x{pattern[1]} board at the same moment. Side sees it at "
            f"{_ranges_s(list(coarse['side']), fps['side'])}; front at {_ranges_s(list(coarse['front']), fps['front'])} "
            f"(front time = side time {(cf / fps['front'] - cs / fps['side']):+.2f} s). If both cameras do show the "
            "board together, the clap sync is wrong; otherwise check the corner count (inner corners) and that the "
            "whole board is visible in both views.")
    # walk back from the clap first, then forward from it (every candidate is checked either way)
    order = [i for i in reversed(cand) if i < cs] + [i for i in cand if i >= cs]
    log(f"[calibration] checking {len(cand)} frame pairs one by one (back from the clap at side frame {cs} / front "
        f"frame {cf}, then forward) ...")

    def nearest(cam, idx):
        keys = np.fromiter(coarse[cam], int)
        return coarse[cam][int(keys[np.argmin(np.abs(keys - idx))])]

    frames = {"side": {}, "front": {}}
    pairs = {}

    def stillness(i):  # worst-camera corner motion to the nearest neighbouring pair (inf if no neighbour)
        m = [max(_motion(pairs[i][c][1], pairs[n][c][1], pattern) for c in ("side", "front"))
             for n in (i - 1, i + 1) if n in pairs]
        return min(m) if m else np.inf

    def board_px(i):  # board size in the camera where it looks smallest
        return min(np.ptp(pairs[i][c][1], axis=0).max() for c in ("side", "front"))

    try:
        for k, i in enumerate(order, 1):
            j = front_of(i)
            found = {}
            for cam, idx in (("side", i), ("front", j)):
                if idx not in frames[cam]:  # decode a block of frames at once, in the walking direction
                    lo, hi = (max(0, idx - BLOCK + 1), idx) if i < cs else (idx, min(info[cam][0] - 1, idx + BLOCK - 1))
                    frames[cam] = _read_range(caps[cam], lo, hi, scale[cam])
                g = frames[cam].get(idx)
                c = None if g is None else _detect_near(g, nearest(cam, idx), pattern)
                if c is None:
                    break
                found[cam] = c / scale[cam]
            if k % 25 == 0 or k == len(order):
                log(f"[calibration] checked {k}/{len(order)} frame pairs ({'back' if i < cs else 'fwd '} "
                    f"{abs(i - cs)} frames from the clap); board in both cameras in {len(pairs) + (len(found) == 2)}")
            if len(found) == 2:
                pairs[i] = {"side": (i, found["side"]), "front": (j, found["front"])}
                if len(pairs) % 15 == 0 and sum(stillness(p) < STILL_PX for p in pairs) >= ENOUGH_STILL:
                    log(f"[calibration] {ENOUGH_STILL}+ still frame pairs found; stopping the search")
                    break
    finally:
        for c in caps.values():
            c.release()
    log(f"[calibration] board seen by both cameras in {len(pairs)} of {k} frame pairs checked")
    if not pairs:
        raise BoardNotFound("The board was never detected in both cameras in the same frame pair. Check the clap "
                            "sync and that the whole board is visible in both views.")

    still = [i for i in pairs if stillness(i) < STILL_PX]
    if still:
        best = max(still, key=board_px)
        log(f"[calibration] {len(still)} frame pairs with the board still in both cameras; using the one where it is "
            f"largest: side frame {best} / front frame {pairs[best]['front'][0]} (~{board_px(best):.0f} px)")
    else:
        best = min(pairs, key=lambda i: (stillness(i), -board_px(i)))
        log(f"[calibration] note: the board was never still in two neighbouring frames; using the steadiest one: "
            f"side frame {best} / front frame {pairs[best]['front'][0]}")
    return pairs[best]


def _motion(c1, c2, pattern):
    """Mean corner displacement ignoring ordering differences (match the 2 handedness-preserving orderings)."""
    nx, ny = pattern
    a, b = c1.reshape(ny, nx, 2), c2.reshape(ny, nx, 2)
    return min(np.linalg.norm(a - b, axis=2).mean(), np.linalg.norm(a - b[::-1, ::-1], axis=2).mean())


# --------------------------------------------------------------------------- solve
def load_intrinsics(calib_toml):
    """Pose2Sim Calib toml -> list of dicts (name, size, K, D) in camera order (side first, then front)."""
    import rtoml
    data = rtoml.load(Path(calib_toml))
    cams = [v for k, v in sorted(data.items()) if isinstance(v, dict) and "matrix" in v]
    if len(cams) != 2:
        raise RuntimeError(f"{calib_toml} has {len(cams)} cameras, expected 2 (side, front)")
    return [{"name": c.get("name"), "size": [int(round(s)) for s in c["size"]],
             "K": np.array(c["matrix"], float), "D": np.array(c["distortions"], float),
             "R": np.array(c.get("rotation", [0, 0, 0]), float), "T": np.array(c.get("translation", [0, 0, 0]), float)}
            for c in cams]


def has_extrinsics(calib_toml):
    return all(np.linalg.norm(c["T"]) > 0 for c in load_intrinsics(calib_toml))


def _object_points(pattern, square_m, axes):
    nx, ny = pattern
    U, V = axes
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    return (jj.reshape(-1, 1) * square_m * U + ii.reshape(-1, 1) * square_m * V).astype(np.float32)


def solve_extrinsics(videos, found, intr, pattern, square_mm, board_position, out_toml, debug_dir, cam_names, log=print):
    """found: output of find_board_backwards. intr: [side, front] from load_intrinsics. Writes out_toml.
    Returns {'side': rms_px, 'front': rms_px}."""
    from Pose2Sim.calibration import toml_write
    X, Y, Z = np.eye(3)
    sq = square_mm / 1000.0
    views = {}
    for cam, cal in zip(("side", "front"), intr):
        idx, approx = found[cam]
        full = _read_full(videos[cam], idx)
        h, w = full.shape[:2]
        if [w, h] != cal["size"]:
            raise RuntimeError(f"{cam} video is {w}x{h} but its intrinsics were computed at "
                               f"{cal['size'][0]}x{cal['size'][1]}; they must match.")
        gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        c, det = detect(gray, pattern)  # full-res re-detection (the SB detector is already sub-pixel accurate)
        if c is None or det == "classic":
            # refine; the search window must stay inside one square or corners jump to their neighbours
            base = approx if c is None else c
            sq_px = np.median(np.linalg.norm(np.diff(base.reshape(pattern[1], pattern[0], 2), axis=1), axis=2))
            win = int(np.clip(sq_px * 0.3, 2, 11))
            c = cv2.cornerSubPix(gray, base.reshape(-1, 1, 2).astype(np.float32), (win, win), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)).reshape(-1, 2)
        views[cam] = {"img": full, "imgp": canonical_order(c, pattern, gray), "idx": idx}

    # 1) each camera's pose in the BOARD frame (board plane = z 0, corner 0 = origin)
    objp = _object_points(pattern, sq, (X, Y))
    res = {}
    for cam, cal in zip(("side", "front"), intr):
        imgp = views[cam]["imgp"].astype(np.float32)
        ok, r, t = cv2.solvePnP(objp, imgp, cal["K"], cal["D"], flags=cv2.SOLVEPNP_ITERATIVE)
        proj = cv2.projectPoints(objp, r, t, cal["K"], cal["D"])[0].reshape(-1, 2)
        Rm = cv2.Rodrigues(r)[0]
        res[cam] = {"R": Rm, "t": t.ravel(), "proj": proj, "center": (-Rm.T @ t).ravel(),
                    "rms": float(np.sqrt(np.mean(np.sum((proj - imgp) ** 2, axis=1))))}
    if np.sign(res["side"]["center"][2]) != np.sign(res["front"]["center"][2]):
        raise RuntimeError("The two cameras ended up on opposite sides of the board, so the corner order differs "
                           "between the views. Use a board with one even and one odd inner-corner count (e.g. 6x7).")

    # 2) level the world: +Z up. Floor board -> its normal (towards the cameras). Held/standing board -> the mean
    #    "up" of the two cameras (tripods are close to level), since a hand-held board is usually tilted.
    if board_position == "horizontal":
        up = Z * np.sign(res["side"]["center"][2])
    else:
        up = sum(-res[c]["R"].T @ np.array([0.0, 1.0, 0.0]) for c in res)  # image y points down
        up /= np.linalg.norm(up)
    x_new = X - np.dot(X, up) * up
    if np.linalg.norm(x_new) < 1e-3:
        x_new = Y - np.dot(Y, up) * up
    x_new /= np.linalg.norm(x_new)
    W = np.vstack([x_new, np.cross(up, x_new), up])  # rows = new world axes, in board coordinates
    for c in res:  # x_cam = R X_board + t = (R W^T) X_world + t
        res[c]["R"] = res[c]["R"] @ W.T
        res[c]["r"] = cv2.Rodrigues(res[c]["R"])[0].ravel()
        res[c]["center"] = W @ res[c]["center"]

    c_s, c_f = res["side"]["center"], res["front"]["center"]
    view_angle = np.degrees(np.arccos(np.clip(np.dot(res["side"]["R"][2], res["front"]["R"][2]), -1, 1)))
    log(f"[calibration] extrinsics: reprojection error side {res['side']['rms']:.2f} px, front {res['front']['rms']:.2f} px")
    for cam in res:
        span = np.ptp(views[cam]["imgp"], axis=0).max()
        if span < 200:
            log(f"[calibration] WARNING: the board is only ~{span:.0f} px across in the {cam} view. Hold it closer to "
                f"that camera or use a bigger board for more accurate extrinsics.")
        if res[cam]["rms"] > 2:
            log(f"[calibration] WARNING: high reprojection error in the {cam} view ({res[cam]['rms']:.1f} px): check "
                f"that camera's intrinsic video (long, board over the whole frame) and that the board is flat.")
    log(f"[calibration] cameras are {np.linalg.norm(c_s - c_f):.2f} m apart; angle between their viewing directions "
        f"{view_angle:.0f} deg; heights above the board origin: side {c_s[2]:+.2f} m, front {c_f[2]:+.2f} m")

    toml_write(out_toml, list(cam_names), [cal["size"] for cal in intr], [cal["D"] for cal in intr],
               [cal["K"] for cal in intr], [res[c]["r"] for c in ("side", "front")],
               [res[c]["t"] for c in ("side", "front")])
    log(f"[calibration] wrote {out_toml}")

    for cam in ("side", "front"):  # picture of what was detected, for the website
        img = views[cam]["img"].copy()
        cv2.drawChessboardCorners(img, tuple(pattern), views[cam]["imgp"].reshape(-1, 1, 2).astype(np.float32), True)
        for p in res[cam]["proj"]:
            cv2.circle(img, tuple(int(v) for v in p), 6, (255, 0, 255), 2)
        p0 = tuple(int(v) for v in views[cam]["imgp"][0])
        cv2.circle(img, p0, 18, (0, 255, 255), 4)
        cv2.putText(img, "origin", (p0[0] + 20, p0[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(img, f"{cam} frame {views[cam]['idx']}  reproj {res[cam]['rms']:.2f}px", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        s = 1280 / max(img.shape[:2])
        cv2.imwrite(str(Path(debug_dir) / f"board_{cam}.jpg"), cv2.resize(img, None, fx=min(1, s), fy=min(1, s)))
    return {c: round(res[c]["rms"], 3) for c in ("side", "front")}, round(float(view_angle), 1)
