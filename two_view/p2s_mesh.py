"""
p2s_mesh.py -- SMPL body mesh fitted to the Pose2Sim 3D points (points3d.trc), for a MotionBERT-style mesh view.

    C:\\Proj1Venv\\Scripts\\python.exe two_view/p2s_mesh.py <run or person folder>

The folder is a Pose2Sim run (outputs/two_view/<run>/) or a batch person folder (preprocessed_data/<person_id>/):
it must hold points3d.trc. Writes into <folder>/mesh/:

    smpl_params.npz   betas (10), global_orient (T,3), body_pose (T,69), transl (T,3), frames (T,)  - compact
    verts.bin         int16 vertex positions in millimetres, (T, 6890, 3), for the mesh viewer (mesh_app.py)
    mesh.json         frame numbers, fps, fit error per marker, settings
    log.txt
and fills com_mesh / com_final in <folder>/advancedangles.csv (same rules as keypoint_pipeline.py: com_mesh =
centroid of the vertices, com_final = midpoint of com_keypoints and com_mesh).

How: Pose2Sim's 22 HALPE markers are matched to SMPL joints (knees, ankles, shoulders, elbows, wrists, hips,
pelvis, neck) and SMPL surface vertices (nose, top of head, big/small toes, heels). One body shape for the whole
video plus a pose and position per frame are optimised (Adam) to put those points on the markers, with a robust
(Geman-McClure) marker loss, small pose / shape regularisation, hinge limits for the knees and temporal smoothness.
The fit only evaluates the ~11 vertices it needs (much faster than full SMPL); the full mesh is computed once at
the end. Needs the SMPL files in MotionBERT/data/mesh/ (SMPL_NEUTRAL.pkl), torch and smplx (the C:\\Proj1Venv env).
Coordinates: metres, Y up, the same world frame as angles.csv.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SMPL_PKL = ROOT / "MotionBERT" / "data" / "mesh" / "SMPL_NEUTRAL.pkl"
sys.path.insert(0, str(HERE))
from p2s_biomech import read_trc  # noqa: E402

H36M_REGRESSOR = ROOT / "MotionBERT" / "data" / "mesh" / "J_regressor_h36m_correct.npy"
# marker -> ("j", SMPL joint index) / ("v", SMPL vertex index) / ("r", H36M joint regressed from the mesh surface,
# the convention MotionBERT uses - closer to where 2D detectors put hips and neck than SMPL's internal joints), weight
MARKERS = {
    "RKnee": ("j", 5, 1.0), "LKnee": ("j", 4, 1.0), "RAnkle": ("j", 8, 1.0), "LAnkle": ("j", 7, 1.0),
    "RShoulder": ("j", 17, 0.8), "LShoulder": ("j", 16, 0.8), "RElbow": ("j", 19, 1.0), "LElbow": ("j", 18, 1.0),
    "RWrist": ("j", 21, 1.0), "LWrist": ("j", 20, 1.0),
    "RHip": ("r", 1, 0.8), "LHip": ("r", 4, 0.8), "Hip": ("r", 0, 0.5), "Neck": ("r", 8, 0.5),
    "Nose": ("v", 332, 0.6), "Head": ("v", 411, 0.3),
    "RBigToe": ("v", 6617, 1.0), "RSmallToe": ("v", 6624, 0.7), "RHeel": ("v", 6787, 1.0),
    "LBigToe": ("v", 3216, 1.0), "LSmallToe": ("v", 3226, 0.7), "LHeel": ("v", 3387, 1.0),
}
KNEES = (4, 5)          # SMPL joints; flexion = +x rotation (checked: +x moves the ankle backwards)
SIGMA = 0.05            # Geman-McClure scale, metres
W_POSE, W_SHAPE, W_HINGE, W_SMOOTH, W_ACC = 2e-3, 5e-3, 1.0, 0.5, 5.0


def log(msg, fh=None):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def rotmat_to_aa(R):
    """(N,3,3) rotation matrices -> (N,3) axis-angle."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


class SubsetSMPL(torch.nn.Module):
    """SMPL evaluated only at its 24 joints + a few vertices (linear blend skinning, pose blend shapes included)."""

    def __init__(self, model, vert_ids):
        super().__init__()
        self.vid = torch.as_tensor(vert_ids, dtype=torch.long)
        self.v_template = model.v_template
        self.shapedirs = model.shapedirs
        self.J_regressor = model.J_regressor
        self.parents = model.parents
        P = model.posedirs.shape[0]
        self.posedirs_sub = model.posedirs.reshape(P, -1, 3)[:, self.vid, :].reshape(P, -1)
        self.weights_sub = model.lbs_weights[self.vid]

    def forward(self, betas, global_orient, body_pose, transl):
        from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes
        T = global_orient.shape[0]
        v_shaped = self.v_template + blend_shapes(betas[None], self.shapedirs)[0]       # (6890,3)
        J = torch.einsum("jv,vc->jc", self.J_regressor, v_shaped)                     # (24,3)
        pose = torch.cat([global_orient, body_pose], dim=1).reshape(T, 24, 3)
        rot = batch_rodrigues(pose.reshape(-1, 3)).reshape(T, 24, 3, 3)
        pose_feat = (rot[:, 1:] - torch.eye(3, device=rot.device)).reshape(T, -1)
        v = v_shaped[self.vid][None] + (pose_feat @ self.posedirs_sub).reshape(T, -1, 3)
        J_posed, A = batch_rigid_transform(rot, J[None].expand(T, -1, -1), self.parents)
        Tm = torch.einsum("vj,tjab->tvab", self.weights_sub, A)
        v_h = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
        v_posed = torch.einsum("tvab,tvb->tva", Tm, v_h)[..., :3]
        return J_posed + transl[:, None], v_posed + transl[:, None]


def gmof(sq, sigma=SIGMA):
    return sigma ** 2 * sq / (sq + sigma ** 2)


def _min_rotation(a, b):
    """(T,3) unit vectors a, b -> (T,3,3) smallest rotations taking a onto b."""
    v = np.cross(a, b)
    c = np.sum(a * b, axis=1)
    s2 = np.sum(v * v, axis=1)
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
    f = np.where(s2 > 1e-12, (1 - c) / np.maximum(s2, 1e-12), 0.5)
    return np.eye(3)[None] + K + (K @ K) * f[:, None, None]


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


# SMPL joint -> (child joint whose direction it sets, from marker(s), to marker(s))
IK_BONES = [
    (1, 4, ("LHip",), ("LKnee",)), (4, 7, ("LKnee",), ("LAnkle",)), (7, 10, ("LAnkle",), ("LBigToe", "LSmallToe")),
    (2, 5, ("RHip",), ("RKnee",)), (5, 8, ("RKnee",), ("RAnkle",)), (8, 11, ("RAnkle",), ("RBigToe", "RSmallToe")),
    (16, 18, ("LShoulder",), ("LElbow",)), (18, 20, ("LElbow",), ("LWrist",)),
    (17, 19, ("RShoulder",), ("RElbow",)), (19, 21, ("RElbow",), ("RWrist",)),
]


def ik_init(mk, J_rest, parents):
    """Pose guess per frame: pelvis from hips (left) + trunk (up); each limb bone rotated (smallest rotation) to
    point along its markers. Starts the optimiser near the answer, so a deep squat does not end in a curled-spine
    local minimum. Returns global_orient (T,3), body_pose (T,69) as axis-angle."""
    T = len(mk["Hip"])
    left = _unit(mk["LHip"] - mk["RHip"])
    up = 0.5 * (mk["LShoulder"] + mk["RShoulder"]) - mk["Hip"]
    up = _unit(up - np.sum(up * left, axis=1, keepdims=True) * left)
    G = {0: np.stack([left, up, np.cross(left, up)], axis=2)}            # global rotation of each joint
    L = np.repeat(np.eye(3)[None, None], 24, axis=1).repeat(T, axis=0)  # local rotations (T,24,3,3)
    bones = {j: (c, a, b) for j, c, a, b in IK_BONES}
    for j in range(1, 24):  # parents come before children in SMPL's joint order
        Gp = G[int(parents[j])]
        if j in bones:
            c, a, b = bones[j]
            rest = _unit(np.repeat((J_rest[c] - J_rest[j])[None], T, axis=0))
            cur = np.einsum("tab,tb->ta", Gp, rest)                      # bone direction with a rest local rotation
            tgt = _unit(np.mean([mk[n] for n in b], axis=0) - np.mean([mk[n] for n in a], axis=0))
            Gj = _min_rotation(cur, tgt) @ Gp
            L[:, j] = np.einsum("tba,tbc->tac", Gp, Gj)                   # Gp^T Gj
        else:
            Gj = Gp
        G[j] = Gj
    aa = rotmat_to_aa(L.reshape(-1, 3, 3)).reshape(T, 24, 3)
    return rotmat_to_aa(G[0]), aa[:, 1:].reshape(T, 69)


def fill_nans(a):
    """Linear interpolation over time of NaN rows (T,3); all-NaN -> zeros."""
    a = a.copy()
    t = np.arange(len(a))
    ok = np.isfinite(a).all(1)
    if not ok.any():
        return np.zeros_like(a)
    for c in range(3):
        a[~ok, c] = np.interp(t[~ok], t[ok], a[ok, c])
    return a


def fit(trc_path, iters=(150, 1500), device="cpu", logf=None):
    import smplx
    frames, times, markers = read_trc(trc_path)
    names = [n for n in MARKERS if n in markers]
    missing = [n for n in MARKERS if n not in markers]
    if missing:
        log(f"note: markers not in the TRC (ignored): {missing}", logf)
    T = len(frames)
    target = np.stack([markers[n] for n in names], axis=1)                    # (T, M, 3)
    valid = np.isfinite(target).all(-1)
    log(f"{T} frames, {len(names)} markers, {100 * (1 - valid.mean()):.1f} % of marker values missing", logf)

    model = smplx.create(str(SMPL_PKL), model_type="smpl", gender="neutral")
    reg = np.load(H36M_REGRESSOR)
    r_ids = sorted({MARKERS[n][1] for n in names if MARKERS[n][0] == "r"})
    vert_ids = sorted({MARKERS[n][1] for n in names if MARKERS[n][0] == "v"}
                      | {int(v) for r in r_ids for v in np.nonzero(np.abs(reg[r]) > 1e-8)[0]})
    sub = SubsetSMPL(model, vert_ids).to(device)
    reg_sub = torch.tensor(reg[r_ids][:, vert_ids], dtype=torch.float32, device=device)   # (R, Nv)
    kind = {"j": [], "v": [], "r": []}
    for m, n in enumerate(names):
        k, i, _ = MARKERS[n]
        kind[k].append((m, i if k == "j" else (vert_ids.index(i) if k == "v" else r_ids.index(i))))
    order = [m for k in "jvr" for m, _ in kind[k]]
    idx = {k: torch.tensor([i for _, i in kind[k]], dtype=torch.long) for k in "jvr"}
    tgt = torch.tensor(np.nan_to_num(target[:, order]), dtype=torch.float32, device=device)
    msk = torch.tensor(valid[:, order], dtype=torch.float32, device=device)
    wts = torch.tensor([MARKERS[names[m]][2] for m in order], dtype=torch.float32, device=device)

    mk = {n: fill_nans(markers[n]) for n in names}
    J_rest = (model.J_regressor @ model.v_template).numpy()
    go0, bp0 = ik_init(mk, J_rest, model.parents.numpy())
    go = torch.tensor(go0, dtype=torch.float32, device=device, requires_grad=True)
    body = torch.tensor(bp0, dtype=torch.float32, device=device, requires_grad=True)
    tr = torch.tensor(mk["Hip"] - J_rest[0], dtype=torch.float32, device=device, requires_grad=True)
    betas = torch.zeros(10, device=device, requires_grad=True)

    def predict():
        J, V = sub(betas, go, body, tr)
        R = torch.einsum("rv,tvc->trc", reg_sub, V)
        return torch.cat([J[:, idx["j"]], V[:, idx["v"]], R[:, idx["r"]]], dim=1), J

    def loss_fn(stage, sigma):
        pred, J = predict()
        sq = ((pred - tgt) ** 2).sum(-1)
        data = (gmof(sq, sigma) * msk * wts).sum() / msk.sum().clamp(min=1)
        terms = {"markers": data * 1e3}
        terms["shape"] = W_SHAPE * (betas ** 2).sum()
        if stage >= 2:
            bp = body.reshape(T, 23, 3)
            terms["pose"] = W_POSE * (body ** 2).sum() / T
            kn = bp[:, [k - 1 for k in KNEES]]
            terms["hinge"] = W_HINGE * ((torch.relu(-kn[..., 0]) ** 2) + kn[..., 1] ** 2 + kn[..., 2] ** 2).sum() / T
            if T > 2:
                terms["smooth"] = W_SMOOTH * ((body[1:] - body[:-1]) ** 2).sum() / T
                terms["acc"] = W_ACC * ((J[2:] - 2 * J[1:-1] + J[:-2]) ** 2).sum() / T * 1e2
        return sum(terms.values()), terms

    def report(tag):
        with torch.no_grad():
            pred, _ = predict()
            err = torch.sqrt(((pred - tgt) ** 2).sum(-1)) * msk
            per = {names[order[m]]: round(float(err[:, m].sum() / msk[:, m].sum().clamp(min=1)) * 1000, 1)
                   for m in range(len(order))}
            mean = float(err.sum() / msk.sum()) * 1000
        log(f"{tag}: mean marker distance {mean:.1f} mm; per marker (mm): {per}", logf)
        return mean, per

    report("initial guess (inverse kinematics)")
    t0 = time.time()
    # 1: position / shape (pose fixed at the IK guess), 2: everything with a wide robust scale (~ least squares),
    # 3: refinement with a tight scale (outlier markers stop pulling)
    n2 = int(iters[1] * 2 / 3)
    schedule = [(iters[0], [tr, betas], 0.02, 0.3),
                (n2, [go, tr, betas, body], 0.02, 0.3),
                (iters[1] - n2, [go, tr, betas, body], 0.005, SIGMA)]
    for stage, (n_it, params, lr, sigma) in enumerate(schedule, start=1):
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(n_it):
            opt.zero_grad()
            loss, terms = loss_fn(stage, sigma)
            loss.backward()
            opt.step()
            if it % 100 == 0 or it == n_it - 1:
                log(f"stage {stage} iter {it:4d}  loss {loss.item():.4f}  "
                    + "  ".join(f"{k} {v.item():.4f}" for k, v in terms.items()), logf)
    log(f"fit done in {time.time() - t0:.0f} s", logf)
    mean_err, per_marker = report("fitted")
    params = {"betas": betas.detach().cpu().numpy(), "global_orient": go.detach().cpu().numpy(),
              "body_pose": body.detach().cpu().numpy(), "transl": tr.detach().cpu().numpy(), "frames": frames}
    return model, params, {"mean_err_mm": round(mean_err, 1), "per_marker_mm": per_marker}


def full_vertices(model, p, chunk=200):
    out = []
    with torch.no_grad():
        for a in range(0, len(p["frames"]), chunk):
            b = a + chunk
            n = len(p["frames"][a:b])
            o = model(betas=torch.tensor(p["betas"])[None].expand(n, -1), global_orient=torch.tensor(p["global_orient"][a:b]),
                      body_pose=torch.tensor(p["body_pose"][a:b]), transl=torch.tensor(p["transl"][a:b]))
            out.append(o.vertices.numpy())
    return np.concatenate(out)


def fill_com_mesh(adv_csv, frames, verts):
    """com_mesh = vertex centroid, com_final = midpoint with com_keypoints (keypoint_pipeline.py rules)."""
    if not Path(adv_csv).exists():
        return False
    com = {int(f): verts[k].mean(0) for k, f in enumerate(frames)}
    with open(adv_csv, newline="") as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    if not {"frame", "com_keypoints", "com_mesh", "com_final"} <= set(head):
        return False
    i_f, i_k, i_m, i_c = (head.index(c) for c in ("frame", "com_keypoints", "com_mesh", "com_final"))
    fmt = lambda c: f"{c[0]:.6f},{c[1]:.6f},{c[2]:.6f}"  # noqa: E731
    for r in rows[1:]:
        c = com.get(int(float(r[i_f])))
        if c is None or not np.all(np.isfinite(c)):
            continue
        r[i_m] = fmt(c)
        try:
            kp = np.array([float(x) for x in r[i_k].split(",")])
            r[i_c] = fmt(0.5 * (kp + c)) if kp.size == 3 and np.all(np.isfinite(kp)) else fmt(c)
        except ValueError:
            r[i_c] = fmt(c)
    with open(adv_csv, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return True


def process(folder, a):
    """Fit + write the mesh files of one run / person folder. Returns True on success."""
    trc = folder / "points3d.trc"
    out = folder / "mesh"
    out.mkdir(exist_ok=True)
    (out / "done.txt").unlink(missing_ok=True)
    logf = open(out / "log.txt", "w", encoding="utf-8")
    try:
        if not trc.exists():
            raise FileNotFoundError(f"{trc} not found (run the Pose2Sim analysis first)")
        if not SMPL_PKL.exists():
            raise FileNotFoundError(f"SMPL model not found at {SMPL_PKL} (see keypoint_pipeline.check_smpl_assets)")
        model, p, quality = fit(trc, tuple(int(x) for x in a.iters.split(",")), logf=logf)
        if quality["mean_err_mm"] > 60:
            log("WARNING: the body fits the 3D points badly (good recordings: ~20-35 mm). The 3D points probably do "
                "not look like a body - check this recording's calibration and clap sync.", logf)
        np.savez_compressed(out / "smpl_params.npz", **p)
        verts = full_vertices(model, p)
        np.clip(np.round(verts * 1000), -32767, 32767).astype("<i2").tofile(out / "verts.bin")
        fps = 30.0
        try:
            fps = float(json.loads((folder / "results.json").read_text(encoding="utf-8"))["fps"]["p3d"])
        except Exception:  # noqa: BLE001
            pass
        floor = float(np.percentile(verts[:, :, 1].min(axis=1), 5))  # lowest point of the body ~ the floor
        # facing direction (horizontal) in the most upright frame (straightest knees): the body's +Z axis
        from scipy.spatial.transform import Rotation
        k = int(np.argmin(np.abs(p["body_pose"][:, 9]) + np.abs(p["body_pose"][:, 12])))
        fwd = Rotation.from_rotvec(p["global_orient"][k]).as_matrix()[:, 2]
        fwd_xz = (fwd[[0, 2]] / max(np.linalg.norm(fwd[[0, 2]]), 1e-9)).round(4).tolist()
        (out / "mesh.json").write_text(json.dumps({
            "frames": [int(f) for f in p["frames"]], "fps": fps, "n_frames": int(len(p["frames"])), "n_verts": 6890,
            "units": "mm", "up": "y", "floor_y_mm": round(floor * 1000, 1), "facing_xz": fwd_xz, **quality,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        if not a.no_com and fill_com_mesh(folder / "advancedangles.csv", p["frames"], verts):
            log("filled com_mesh / com_final in advancedangles.csv", logf)
        log(f"wrote {out / 'verts.bin'} ({verts.shape[0]} frames, {(out / 'verts.bin').stat().st_size / 1e6:.0f} MB)", logf)
        (out / "done.txt").write_text("ok", encoding="utf-8")
        return True
    except Exception as e:  # noqa: BLE001
        import traceback
        log("ERROR: " + "".join(traceback.format_exception(e)), logf)
        return False
    finally:
        logf.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="run / person folder holding points3d.trc, or a folder of them "
                                   "(e.g. preprocessed_data: every person without a mesh yet)")
    ap.add_argument("--iters", default="150,1500", help="iterations of stage 1 (position, shape) and stages 2+3 (all)")
    ap.add_argument("--no-com", action="store_true", help="do not fill com_mesh / com_final in advancedangles.csv")
    ap.add_argument("--redo", action="store_true", help="folder of persons: also refit those that have a mesh")
    a = ap.parse_args()
    folder = Path(a.folder).resolve()
    if (folder / "points3d.trc").exists():
        sys.exit(0 if process(folder, a) else 1)
    todo = sorted(d for d in folder.iterdir() if d.is_dir() and (d / "points3d.trc").exists()
                  and (a.redo or not (d / "mesh" / "done.txt").exists()))
    print(f"{len(todo)} folders to fit in {folder}", flush=True)
    failed = [d.name for k, d in enumerate(todo, 1) if not (print(f"[{k}/{len(todo)}] {d.name}", flush=True) or process(d, a))]
    print(f"done; {len(todo) - len(failed)} fitted" + (f", failed: {failed}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
