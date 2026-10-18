"""
engine.py - experiments.

    run_experiment(models, cfg, out_dir)   person-wise CV (folds x seeds) of each model; writes per model
                                           summary.json, preds.csv, alpha.npy, per-fold metrics/history, and
                                           out_dir/progress.json + log.txt while running (see monitor.py)
    comparison_table(out_dir)              one row per model: mean +- std over folds x seeds
    select_best(table, rule)               the model-selection rule of the study
    train_final / export / predict_recording   final model on all persons -> export folder -> predictions.json
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import FEATURE_VERSION
from .common import nice, read_json, write_json
from .dataset import Normaliser, apply_switches, make_folds, switches
from .models import TRANSFORMERS, TREE_MODELS, build_model, n_params


# ---------------------------------------------------------------------------------------------- helpers
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(y, pred, n):
    labels = list(range(n))
    return dict(macro_f1=float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
                accuracy=float(accuracy_score(y, pred)),
                per_class_f1=f1_score(y, pred, average=None, labels=labels, zero_division=0).tolist(),
                confusion=confusion_matrix(y, pred, labels=labels).tolist())


def recording_weights(rec):
    _, inv, cnt = np.unique(rec, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.mean()


class Progress:
    """progress.json + log.txt in the experiment folder, read by monitor.py while the run goes on."""

    def __init__(self, out_dir, models, cfg):
        self.out, self.t0 = Path(out_dir), time.time()
        self.state = dict(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), models=models,
                          seeds=cfg["cv"]["seeds"], folds=cfg["cv"]["n_splits"],
                          total_folds=len(models) * len(cfg["cv"]["seeds"]) * cfg["cv"]["n_splits"],
                          done=[], current=None, fold_seconds={})
        self.save()

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.out / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def epoch(self, model, seed, fold, ep, max_ep, train_loss, val_f1, best_ep):
        self.state["current"] = dict(model=model, seed=seed, fold=fold, epoch=ep, max_epochs=max_ep, train_loss=train_loss,
                                     val_f1=val_f1, best_epoch=best_ep)
        self.save()

    def fold_done(self, model, seed, fold, f1, seconds):
        self.state["done"].append(dict(model=model, seed=seed, fold=fold, f1=f1, seconds=seconds))
        self.state["fold_seconds"].setdefault(model, []).append(seconds)
        self.save()

    def eta(self):
        """Seconds left: remaining folds x the mean fold time of their model (unknown models: the mean of all)."""
        st = self.state
        per = {m: float(np.mean(v)) for m, v in st["fold_seconds"].items()}
        allm = float(np.mean([s for v in st["fold_seconds"].values() for s in v])) if per else None
        if allm is None:
            return None
        n = len(st["seeds"]) * st["folds"]
        left = 0.0
        for m in st["models"]:
            done = sum(1 for d in st["done"] if d["model"] == m)
            left += (n - done) * per.get(m, allm)
        return left

    def save(self):
        self.state.update(updated=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=time.time() - self.t0)
        self.state["eta_s"] = self.eta() if self.state.get("fold_seconds") else None
        write_json(self.out / "progress.json", self.state)

    def finish(self, state="done"):
        self.state.update(state=state, current=None)
        self.save()


# ---------------------------------------------------------------------------------------------- neural
def _prep(ds, idx, norm, sw, dev):
    j, f, t = norm(ds["joints"][idx], ds["frame"][idx], ds["tempo"][idx])
    j, f, t = apply_switches(j, f, t, **sw)
    return [torch.from_numpy(x).to(dev) for x in (j, f, t)]


@torch.no_grad()
def predict_tensors(model, X, amp=False, bs=256):
    model.eval()
    P, A = [], []
    for i in range(0, len(X[0]), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lo, al = model(*[x[i:i + bs] for x in X])
        P.append(torch.softmax(lo.float(), 1).cpu())
        A.append(al.float().cpu() if al is not None else None)
    return torch.cat(P).numpy(), (torch.cat(A).numpy() if A and A[0] is not None else None)


def deletion_test(model, X, y, amp, frac, seed):
    """Drop of the true-class probability: top-frac alpha frames set to the rep mean vs random frames (3 repeats)."""
    rng = np.random.default_rng(seed)
    p0, alpha = predict_tensors(model, X, amp)
    if alpha is None or not len(y):
        return None
    base = p0[np.arange(len(y)), y]
    T = X[0].shape[1]
    k = max(1, int(round(frac * T)))

    def masked(sel):
        j, f = X[0].clone(), X[1].clone()
        idx = torch.as_tensor(sel, device=j.device)
        rows = torch.arange(len(j), device=j.device)[:, None]
        j[rows, idx] = j.mean(1, keepdim=True).expand(-1, k, -1, -1)
        f[rows, idx] = f.mean(1, keepdim=True).expand(-1, k, -1)
        return predict_tensors(model, [j, f, X[2]], amp)[0][np.arange(len(y)), y]
    top = np.argsort(-alpha, axis=1)[:, :k]
    d_top = float(np.mean(base - masked(top)))
    d_rnd = float(np.mean([np.mean(base - masked(np.stack([rng.choice(T, k, replace=False) for _ in range(len(y))])))
                           for _ in range(3)]))
    return dict(alpha_drop=d_top, random_drop=d_rnd)


def fit_neural(name, ds, tr, va, te, cfg, seed, dev, fixed_epochs=None, progress=None, fold=None):
    set_seed(seed)
    tc, n_cls = cfg["train"], len(ds["classes"])
    sw = switches(cfg, ds["meta"]["joint_names"])
    norm = Normaliser().fit(ds["joints"][tr], ds["frame"][tr], ds["tempo"][tr])
    Xtr = _prep(ds, tr, norm, sw, dev)
    Xva = _prep(ds, va, norm, sw, dev) if len(va) else None
    Xte = _prep(ds, te, norm, sw, dev) if te is not None and len(te) else None
    ytr = torch.from_numpy(ds["y"][tr]).to(dev)
    cls_w = len(tr) / (n_cls * np.maximum(np.bincount(ds["y"][tr], minlength=n_cls), 1))
    w = cls_w[ds["y"][tr]] * recording_weights(ds["recording"][tr]) if tc.get("weighting", True) else np.ones(len(tr))
    w = torch.from_numpy((w / w.mean()).astype(np.float32)).to(dev)
    T, J, F = ds["joints"].shape[1], ds["joints"].shape[2], ds["frame"].shape[2]
    model = build_model(name, J, F, T, n_cls, ds["meta"]["joint_names"], cfg["model"]).to(dev)
    amp = bool(tc.get("bf16", True)) and name in TRANSFORMERS and dev.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = fixed_epochs or tc["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1, best_ep, best_state, bad = -1.0, 0, None, 0
    hist = dict(train_loss=[], val_f1=[], val_loss=[])
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tot = 0.0
        for i in range(0, len(tr), tc["batch_size"]):
            b = perm[i:i + tc["batch_size"]]
            if len(b) < 2:
                continue
            j, f, t = Xtr[0][b], Xtr[1][b], Xtr[2][b]
            if tc["noise_std"] > 0:
                j = j + tc["noise_std"] * torch.randn_like(j) * (j != 0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(j, f, t)
            loss = (Fn.cross_entropy(logits.float(), ytr[b], reduction="none", label_smoothing=tc["label_smoothing"]) * w[b]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        hist["train_loss"].append(tot / len(tr))
        vf1 = None
        if Xva is not None and fixed_epochs is None:
            pv, _ = predict_tensors(model, Xva, amp)
            vf1 = float(f1_score(ds["y"][va], pv.argmax(1), average="macro", labels=list(range(n_cls)), zero_division=0))
            hist["val_f1"].append(vf1)
            hist["val_loss"].append(float(-np.mean(np.log(np.clip(pv[np.arange(len(va)), ds["y"][va]], 1e-9, 1)))))
            if vf1 > best_f1:
                best_f1, best_ep, bad, best_state = vf1, ep, 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
        if progress:
            progress.epoch(name, seed, fold, ep, epochs, hist["train_loss"][-1], vf1, best_ep)
        if Xva is not None and fixed_epochs is None and bad >= tc["patience"]:
            break
    if best_state is None:
        best_state, best_ep = copy.deepcopy(model.state_dict()), epochs
    model.load_state_dict(best_state)
    probs = alpha = dele = None
    if Xte is not None:
        probs, alpha = predict_tensors(model, Xte, amp)
        dele = deletion_test(model, Xte, ds["y"][te], amp, cfg.get("alpha_check", {}).get("frac", 0.1), seed)
    return dict(probs=probs, alpha=alpha, deletion=dele, history=hist, best_epoch=best_ep, state=best_state,
                norm=norm, n_params=n_params(model), model=model, best_val_f1=best_f1 if best_f1 >= 0 else None,
                epochs_run=len(hist["train_loss"]), val_persons=sorted(set(ds["person"][va].tolist())) if len(va) else [])


# ---------------------------------------------------------------------------------------------- trees
def tree_features(ds, cfg):
    return np.concatenate([ds["summary"], ds["tempo"]], axis=1) if cfg["data"]["tempo"] else ds["summary"]


def make_tree(name, seed, n_cls):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)), "logisticregression"
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=4)), "randomforestclassifier"
    if name == "xgboost":
        from xgboost import XGBClassifier   # only the baselines need it (not the product's predict step)
        return make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, random_state=seed, n_jobs=4, eval_metric="mlogloss", objective="multi:softprob")), "xgbclassifier"
    raise ValueError(name)


def fit_tree(name, ds, tr, te, cfg, seed):
    X, n_cls = tree_features(ds, cfg), len(ds["classes"])
    model, step = make_tree(name, seed, n_cls)
    w = recording_weights(ds["recording"][tr]) if cfg["train"].get("weighting", True) else np.ones(len(tr))
    model.fit(X[tr], ds["y"][tr], **{f"{step}__sample_weight": w})
    full = np.zeros((len(te), n_cls))
    full[:, model[-1].classes_] = model.predict_proba(X[te])
    return dict(probs=full, model=model)


# ---------------------------------------------------------------------------------------------- experiment
def run_experiment(models, ds, cfg, out_dir, save_models=False):
    """All folds x seeds of each model. Returns {model: summary}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config_used.json", cfg)
    dev, n_cls = device(), len(ds["classes"])
    prog = Progress(out, list(models), cfg)
    prog.log(f"device {dev}; {len(ds['y'])} reps, {len(set(ds['person']))} persons, {n_cls} classes; models {', '.join(models)}")
    summaries = {}
    try:
        for name in models:
            mdir = out / name
            mdir.mkdir(exist_ok=True)
            rows, alphas, fms, dels, params, t_model = [], [], [], [], None, time.time()
            prog.log(f"=== {nice(name)}")
            for seed in cfg["cv"]["seeds"]:
                for k, (tr, va, te) in enumerate(make_folds(ds["y"], ds["person"], cfg["cv"]["n_splits"], seed, cfg["cv"]["val_persons"])):
                    tr = subsample_persons(tr, ds["person"], cfg["cv"].get("train_person_frac", 1.0), seed * 100 + k)
                    t0 = time.time()
                    fdir = mdir / f"fold_{seed}_{k}"
                    fdir.mkdir(exist_ok=True)
                    extra = {}
                    if name in TREE_MODELS:
                        r = fit_tree(name, ds, np.concatenate([tr, va]), te, cfg, seed)
                    else:
                        r = fit_neural(name, ds, tr, va, te, cfg, seed, dev, progress=prog, fold=k)
                        params = r["n_params"]
                        write_json(fdir / "history.json", r["history"])
                        extra.update(best_epoch=r["best_epoch"], best_val_f1=r["best_val_f1"], epochs_run=r["epochs_run"],
                                     val_persons=r["val_persons"])
                        if r["alpha"] is not None:
                            alphas.append(r["alpha"])
                        if r["deletion"]:
                            dels.append({"seed": seed, "fold": k, **r["deletion"]})
                            extra["deletion"] = r["deletion"]
                    pred = r["probs"].argmax(1)
                    m = metrics(ds["y"][te], pred, n_cls)
                    m.update(seed=seed, fold=k, n_test=int(len(te)), test_persons=sorted(set(ds["person"][te].tolist())), **extra)
                    write_json(fdir / "metrics.json", m)
                    if save_models:                      # checkpoint of the best epoch (by validation macro-F1) of this fold
                        if name in TREE_MODELS:
                            import joblib
                            joblib.dump(r["model"], fdir / "model.joblib")
                        else:
                            torch.save(dict(model=name, state=r["state"], norm=r["norm"].to_dict(), best_epoch=r["best_epoch"],
                                            best_val_f1=r["best_val_f1"], epochs_run=r["epochs_run"], seed=seed, fold=k,
                                            test_persons=m["test_persons"], val_persons=r["val_persons"], test_macro_f1=m["macro_f1"],
                                            classes=ds["classes"], model_cfg=cfg["model"], data_cfg=cfg["data"], train_cfg=cfg["train"],
                                            joint_names=ds["meta"]["joint_names"]), fdir / "best.pt")
                    fms.append(m)
                    df = pd.DataFrame(r["probs"], columns=[f"p{i}" for i in range(n_cls)])
                    for col, val in (("pred", pred), ("y", ds["y"][te]), ("rep_no", ds["rep_no"][te]), ("person", ds["person"][te]),
                                     ("recording", ds["recording"][te]), ("fold", k), ("seed", seed)):
                        df.insert(0, col, val)
                    rows.append(df)
                    prog.fold_done(name, seed, k, m["macro_f1"], time.time() - t0)
                    prog.log(f"  {nice(name)} seed {seed} fold {k}: macro-F1 {m['macro_f1']:.3f} acc {m['accuracy']:.3f} "
                             f"persons {m['test_persons']}" + (f" best epoch {extra['best_epoch']}" if "best_epoch" in extra else ""))
            preds = pd.concat(rows, ignore_index=True)
            preds.to_csv(mdir / "preds.csv", index=False)
            if alphas:
                np.save(mdir / "alpha.npy", np.concatenate(alphas))
            s = summarise(name, preds, fms, n_cls, ds["classes"], params, dels, time.time() - t_model)
            write_json(mdir / "summary.json", s)
            summaries[name] = s
            prog.log(f"  {nice(name)} DONE macro-F1 {s['macro_f1_mean']:.3f} +- {s['macro_f1_std']:.3f}, recording-level "
                     f"{s['rec_macro_f1_mean']:.3f}" + (f", alpha drop {s['alpha_drop']:.3f} vs random {s['random_drop']:.3f}" if s.get("alpha_drop") is not None else ""))
        prog.finish("done")
    except BaseException:
        prog.finish("error")
        raise
    comparison_table(out).to_csv(out / "comparison.csv", index=False)
    return summaries


def subsample_persons(idx, person, frac, seed):
    """Keep the reps of a random fraction of the persons in idx (learning-curve ablation); frac >= 1 keeps all."""
    if frac >= 1.0:
        return idx
    ps = np.unique(person[idx])
    keep = np.random.default_rng(seed).choice(ps, size=max(1, int(round(frac * len(ps)))), replace=False)
    return idx[np.isin(person[idx], keep)]


def summarise(name, preds, fms, n_cls, classes, params, dels, seconds):
    g = preds.groupby(["seed", "fold", "recording"])
    rec = g[[f"p{i}" for i in range(n_cls)]].mean()
    rec["y"] = g["y"].first()
    rec["pred"] = rec[[f"p{i}" for i in range(n_cls)]].to_numpy().argmax(1)
    rec = rec.reset_index()
    rf1 = [f1_score(x["y"], x["pred"], average="macro", labels=list(range(n_cls)), zero_division=0) for _, x in rec.groupby(["seed", "fold"])]
    pooled = metrics(preds["y"], preds["pred"], n_cls)
    f1s = [m["macro_f1"] for m in fms]
    s = dict(model=name, name=nice(name), classes=classes, n_params=params, n_runs=len(fms),
             macro_f1_mean=float(np.mean(f1s)), macro_f1_std=float(np.std(f1s)),
             accuracy_mean=float(np.mean([m["accuracy"] for m in fms])), accuracy_std=float(np.std([m["accuracy"] for m in fms])),
             rec_macro_f1_mean=float(np.mean(rf1)), rec_macro_f1_std=float(np.std(rf1)),
             per_class_f1=pooled["per_class_f1"], confusion=pooled["confusion"], seconds=round(seconds, 1),
             best_epochs=[m["best_epoch"] for m in fms if "best_epoch" in m], fold_f1=f1s)
    if dels:
        s["alpha_drop"] = float(np.mean([d["alpha_drop"] for d in dels]))
        s["random_drop"] = float(np.mean([d["random_drop"] for d in dels]))
    return s


def comparison_table(out_dir):
    rows = []
    for f in sorted(Path(out_dir).glob("*/summary.json")):
        s = read_json(f)
        rows.append(dict(model=s["model"], name=s["name"], macro_f1=s["macro_f1_mean"], macro_f1_std=s["macro_f1_std"],
                         accuracy=s["accuracy_mean"], rec_macro_f1=s["rec_macro_f1_mean"], params=s["n_params"],
                         alpha_drop=s.get("alpha_drop"), random_drop=s.get("random_drop"),
                         median_best_epoch=float(np.median(s["best_epochs"])) if s["best_epochs"] else None,
                         minutes=round(s["seconds"] / 60, 1), runs=s["n_runs"]))
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def fold_scores(run_dir, model):
    """Macro-F1 of every (seed, fold) in run order; runs with the same seeds / folds / dataset are paired."""
    s = read_json(Path(run_dir) / model / "summary.json")
    return np.array(s["fold_f1"]) if s else None


def paired_test(a, b):
    """Mean difference a - b and the two-sided Wilcoxon signed-rank p-value over paired folds x seeds."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) and len(d) >= 5 else 1.0
    return dict(mean_diff=float(d.mean()), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p_value=p)


def select_best(table, alpha_ratio=2.0):
    """The selection rule:
    1. candidates = models with attention pooling whose alpha passes the deletion test
       (masking the top-alpha frames lowers the true-class probability at least alpha_ratio x more than random frames)
    2. the highest mean macro-F1 among them; models within one standard deviation of it are 'tied'
    3. among tied models: the higher recording-level macro-F1, then the fewer parameters.
    Returns (best model name, explanation table)."""
    t = table.copy()
    t["alpha_ok"] = t.apply(lambda r: r["alpha_drop"] is not None and pd.notna(r["alpha_drop"]) and r["alpha_drop"] > 0
                            and r["alpha_drop"] >= alpha_ratio * max(r["random_drop"], 1e-3), axis=1)
    cand = t[t["alpha_ok"]]
    if cand.empty:
        cand = t[t["alpha_drop"].notna()]
    top = cand.iloc[cand["macro_f1"].argmax()]
    tied = cand[cand["macro_f1"] >= top["macro_f1"] - top["macro_f1_std"]].copy()
    tied = tied.sort_values(["rec_macro_f1", "params"], ascending=[False, True])
    best = tied.iloc[0]["model"]
    t["tied_with_best"] = t["model"].isin(tied["model"])
    t["chosen"] = t["model"] == best
    return best, t


# ---------------------------------------------------------------------------------------------- final model
def train_final(name, ds, cfg, epochs, log=print):
    dev = device()
    idx = np.arange(len(ds["y"]))
    seed = cfg["cv"]["seeds"][0]
    log(f"final {nice(name)}: all {len(idx)} reps, {len(set(ds['person']))} persons, {epochs} epochs on {dev}")
    r = fit_neural(name, ds, idx, np.array([], dtype=int), None, cfg, seed, dev, fixed_epochs=epochs)
    return r


def export(r, name, ds, cfg, out_dir, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(r["state"], out / "model.pt")
    write_json(out / "norm_stats.json", r["norm"].to_dict())
    write_json(out / "config.json", dict(model=name, model_name=nice(name), feature_version=FEATURE_VERSION, classes=ds["classes"],
                                         model_cfg=cfg["model"], data_cfg=cfg["data"], train_cfg=cfg["train"],
                                         joint_names=ds["meta"]["joint_names"], T=int(ds["joints"].shape[1]),
                                         F=int(ds["frame"].shape[2]), n_params=r["n_params"], created=time.strftime("%Y-%m-%d %H:%M:%S"),
                                         **(extra or {})))
    return out


def load_export(export_dir):
    export_dir = Path(export_dir)
    conf = read_json(export_dir / "config.json")
    if conf["feature_version"] != FEATURE_VERSION:
        raise RuntimeError("export made with another feature version")
    model = build_model(conf["model"], len(conf["joint_names"]), conf["F"], conf["T"], len(conf["classes"]),
                        conf["joint_names"], conf["model_cfg"])
    model.load_state_dict(torch.load(export_dir / "model.pt", map_location="cpu", weights_only=True))
    return model.eval(), Normaliser.from_dict(read_json(export_dir / "norm_stats.json")), conf


def alpha_to_frames(alpha, n, window=10, sigma=5.0):
    """alpha (T values) -> one value per real frame: interpolate, 10-frame window means, Gaussian sigma 5, peak 1
    (the final product's contract)."""
    a = np.asarray(alpha, float)
    x = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    for i in range(0, n, window):
        x[i:i + window] = x[i:i + window].mean()
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    x = np.convolve(np.pad(x, r, mode="edge"), k / k.sum(), mode="valid")
    return x / x.max() if x.max() > 0 else x


@torch.no_grad()
def predict_recording(rec_dir, export_dir, uncertain_below=0.5):
    """A preprocessed recording folder -> predictions dict (per rep: label, confidence, probs, alpha, alpha_frames;
    intensity over the whole timeline), the format the final product reads as predictions.json."""
    from .features import recording_to_reps
    model, norm, conf = load_export(export_dir)
    reps, info = recording_to_reps(rec_dir, conf["data_cfg"])
    classes, fps = conf["classes"], info["fps"]
    intensity = np.zeros(info["n_frames"])
    out = []
    for it in reps:
        j, f, t = norm(it["joints"][None], it["frame"][None], it["tempo"][None])
        j, f, t = apply_switches(j, f, t, **switches({"data": conf["data_cfg"]}, conf["joint_names"]))
        lo, al = model(torch.from_numpy(j), torch.from_numpy(f), torch.from_numpy(t))
        p = torch.softmax(lo, 1)[0].numpy()
        c = int(p.argmax())
        n = it["end"] - it["start"] + 1
        af = alpha_to_frames(al[0].numpy(), n) if al is not None else np.zeros(n)
        if classes[c] != "correct":
            intensity[it["start"]:it["end"] + 1] = af[: max(0, min(n, len(intensity) - it["start"]))] * p[c]
        out.append(dict(rep=it["rep_no"], start_frame=it["start"], end_frame=it["end"], bottom_frame=it["bottom"],
                        start_s=round(it["start"] / fps, 3), end_s=round(it["end"] / fps, 3), valid=True, label=classes[c],
                        confidence=round(float(p[c]), 4), uncertain=bool(p[c] < uncertain_below), probs=[round(float(v), 4) for v in p],
                        alpha=[round(float(v), 5) for v in (al[0].numpy() if al is not None else [])],
                        alpha_frames=[round(float(v), 4) for v in af], flex=it["flex"].tolist()))
    # reps the feature extraction could not use (too short / long, no knee flexion): kept on the timeline, without a label
    import pandas as pd
    rows = pd.read_csv(Path(rec_dir) / "reps.csv")
    for rep_no, reason in info["dropped"]:
        r = rows[rows["rep_no"] == rep_no]
        if len(r):
            a, b = int(r["start_frame"].iloc[0]), int(r["end_frame"].iloc[0])
            out.append(dict(rep=int(rep_no), start_frame=a, end_frame=b, bottom_frame=(a + b) // 2, start_s=round(a / fps, 3),
                            end_s=round(b / fps, 3), valid=False, label=None, uncertain=False, reason=reason))
    out.sort(key=lambda r: r["rep"])
    return dict(model=conf["model"], model_name=conf["model_name"], model_version=f"{conf['model']} {conf.get('created', '')}".strip(),
                model_active=True, demo=False, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                feature_version=FEATURE_VERSION, classes=classes, fps=fps,
                n_frames=info["n_frames"], uncertain_below=uncertain_below, reps=out,
                intensity=[round(float(v), 4) for v in intensity])
