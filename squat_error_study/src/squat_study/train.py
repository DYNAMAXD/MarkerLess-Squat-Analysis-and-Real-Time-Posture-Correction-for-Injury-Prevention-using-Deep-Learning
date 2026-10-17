"""
train.py   (MODULE - used by scripts/run_cv.py, scripts/run_ablation.py)
=========================================================================
Person-wise cross-validation for every model, with metrics, saved models and predictions.

Sections
--------
 1. Helpers        seeds, metrics, recording-level majority vote
 2. Neural models  fit_neural(): one fold of one neural model (train / validate / test)
 3. Tree models    fit_tree():   decision tree, logistic regression, random forest, XGBoost
 4. Experiment     run_experiment(): all folds x seeds of one model -> folder with results
 5. Final model    train_final(): fit on ALL persons and save for the product
"""

import copy
import json
import time
from pathlib import Path

import joblib
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
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import Normaliser, apply_switches, make_folds
from .models import NEURAL_MODELS, build_model, n_params

TREE_MODELS = ["decision_tree", "logreg", "random_forest", "xgboost"]
ALL_MODELS = TREE_MODELS + NEURAL_MODELS


# ----------------------------------------------------------------------------
# 1. Helpers
# ----------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(y, pred, n_classes):
    labels = list(range(n_classes))
    return dict(macro_f1=float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
                accuracy=float(accuracy_score(y, pred)),
                per_class_f1=f1_score(y, pred, average=None, labels=labels, zero_division=0).tolist(),
                confusion=confusion_matrix(y, pred, labels=labels).tolist())


def recording_vote(df, n_classes):
    """Average the class probabilities of all reps of a recording -> one prediction per recording."""
    pcols = [f"p{i}" for i in range(n_classes)]
    g = df.groupby(["seed", "fold", "recording"])
    out = g[pcols].mean()
    out["y"] = g["y"].first()
    out["pred"] = out[pcols].to_numpy().argmax(1)
    return out.reset_index()


def recording_weights(rec_ids):
    """Each recording counts equally, whatever its number of reps (9, 10, 12, ...). Mean = 1."""
    _, inv, cnt = np.unique(rec_ids, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.mean()


# ----------------------------------------------------------------------------
# 2. Neural models
# ----------------------------------------------------------------------------
def _prep(ds, idx, norm, sw, device):
    j, f, t = norm(ds["joints"][idx], ds["frame"][idx], ds["tempo"][idx])
    j, f, t = apply_switches(j, f, t, **sw)
    return (torch.from_numpy(j).to(device), torch.from_numpy(f).to(device),
            torch.from_numpy(t).to(device))


@torch.no_grad()
def _predict(model, tensors, bs=256):
    model.eval()
    probs, alphas = [], []
    for i in range(0, len(tensors[0]), bs):
        logits, alpha = model(*[x[i:i + bs] for x in tensors])
        probs.append(torch.softmax(logits, 1).cpu())
        if alpha is not None:
            alphas.append(alpha.cpu())
    return torch.cat(probs).numpy(), (torch.cat(alphas).numpy() if alphas else None)


def fit_neural(name, ds, tr, va, te, cfg, seed, device, fixed_epochs=None, log=print):
    """
    Trains one model on `tr`, picks the best epoch on `va` (validation persons), evaluates on `te`.
    fixed_epochs: train exactly that many epochs, no validation (used for the final model).
    Returns dict(probs, alpha, history, best_epoch, state, norm, n_params).
    """
    set_seed(seed)
    tc, mc = cfg["train"], cfg["model"]
    d = cfg["data"]
    sw = dict(inputs=d["inputs"], velocity=d["velocity"], tempo_on=d["tempo"], view=d["view"])
    n_cls = len(ds["classes"])

    norm = Normaliser().fit(ds["joints"][tr], ds["frame"][tr], ds["tempo"][tr])
    Xtr = _prep(ds, tr, norm, sw, device)
    Xva = _prep(ds, va, norm, sw, device) if len(va) else None
    Xte = _prep(ds, te, norm, sw, device) if te is not None and len(te) else None
    ytr = torch.from_numpy(ds["y"][tr]).to(device)

    # sample weights: classes balanced, each recording counted equally
    cls_w = len(tr) / (n_cls * np.maximum(np.bincount(ds["y"][tr], minlength=n_cls), 1))
    w = cls_w[ds["y"][tr]] * recording_weights(ds["recording"][tr])
    w = torch.from_numpy((w / w.mean()).astype(np.float32)).to(device)

    T, J, F = ds["joints"].shape[1], ds["joints"].shape[2], ds["frame"].shape[2]
    model = build_model(name, J, F, T, n_cls, ds["meta"]["joint_names"], mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = fixed_epochs or tc["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_ep, best_state, bad = -1.0, 0, None, 0
    hist = dict(train_loss=[], val_f1=[])
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), device=device)
        tot = 0.0
        for i in range(0, len(tr), tc["batch_size"]):
            b = perm[i:i + tc["batch_size"]]
            if len(b) < 2:                                   # BatchNorm needs more than one sample
                continue
            j, f, t = Xtr[0][b], Xtr[1][b], Xtr[2][b]
            if tc["noise_std"] > 0:
                j = j + tc["noise_std"] * torch.randn_like(j) * (j != 0)      # blanked (ablated) channels stay exactly 0
            logits, _ = model(j, f, t)
            loss = (Fn.cross_entropy(logits, ytr[b], reduction="none",
                                     label_smoothing=tc["label_smoothing"]) * w[b]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        hist["train_loss"].append(tot / len(tr))

        if Xva is not None and fixed_epochs is None:
            pv, _ = _predict(model, Xva)
            f1 = f1_score(ds["y"][va], pv.argmax(1), average="macro", labels=list(range(n_cls)),
                          zero_division=0)
            hist["val_f1"].append(float(f1))
            if f1 > best_f1:
                best_f1, best_ep, bad = f1, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= tc["patience"]:
                    break
    if best_state is None:                                   # no validation -> last epoch
        best_state, best_ep = copy.deepcopy(model.state_dict()), epochs
    model.load_state_dict(best_state)

    probs = alpha = None
    if Xte is not None:
        probs, alpha = _predict(model, Xte)
    return dict(probs=probs, alpha=alpha, history=hist, best_epoch=best_ep, state=best_state,
                norm=norm, n_params=n_params(model))


# ----------------------------------------------------------------------------
# 3. Tree / linear models (use the per-rep summary features + tempo)
# ----------------------------------------------------------------------------
def tree_features(ds, tempo_on=True):
    x = ds["summary"]
    return np.concatenate([x, ds["tempo"]], axis=1) if tempo_on else x


def make_tree_model(name, seed, n_cls):
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=4, random_state=seed), "decisiontreeclassifier"
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0)), "logisticregression"
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)), \
            "randomforestclassifier"
    if name == "xgboost":
        return make_pipeline(SimpleImputer(strategy="median"),
                             XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                           subsample=0.9, colsample_bytree=0.9, random_state=seed,
                                           n_jobs=4, eval_metric="mlogloss", num_class=n_cls,
                                           objective="multi:softprob")), "xgbclassifier"
    raise ValueError(name)


def fit_tree(name, ds, tr, te, cfg, seed):
    X = tree_features(ds, cfg["data"]["tempo"])
    n_cls = len(ds["classes"])
    model, step = make_tree_model(name, seed, n_cls)
    w = recording_weights(ds["recording"][tr])
    if name == "decision_tree":
        model.fit(np.nan_to_num(X[tr]), ds["y"][tr], sample_weight=w)
        probs = model.predict_proba(np.nan_to_num(X[te])) if te is not None else None
    else:
        model.fit(X[tr], ds["y"][tr], **{f"{step}__sample_weight": w})
        probs = model.predict_proba(X[te]) if te is not None else None
    full = np.zeros((len(te), n_cls)) if te is not None else None
    if probs is not None:                                    # a class missing in training -> zero column
        cls = model.classes_ if name == "decision_tree" else model[-1].classes_
        full[:, cls] = probs
    return dict(probs=full, model=model)


# ----------------------------------------------------------------------------
# 4. One experiment = one model (or ablation variant), all folds and seeds
# ----------------------------------------------------------------------------
def run_experiment(name, exp_name, ds, cfg, out_dir, device, save_models=True, log=print):
    """
    Writes into out_dir/exp_name/:
        summary.json     mean +- std over folds and seeds (rep level and recording level)
        preds.csv        one row per test rep (seed, fold, recording, person, y, pred, probabilities)
        alpha.npy        frame attention of every test rep (only models with attention pooling)
        fold_<s>_<k>/    metrics.json, history.json, best.pt / model.joblib
    """
    out = Path(out_dir) / exp_name
    out.mkdir(parents=True, exist_ok=True)
    n_cls = len(ds["classes"])
    t0 = time.time()
    rows, alphas, fold_metrics, params = [], [], [], None

    for seed in cfg["cv"]["seeds"]:
        folds = make_folds(ds["y"], ds["person"], cfg["cv"]["n_splits"], seed, cfg["cv"]["val_persons"])
        for k, (tr, va, te) in enumerate(folds):
            fdir = out / f"fold_{seed}_{k}"
            fdir.mkdir(exist_ok=True)
            if name in TREE_MODELS:
                r = fit_tree(name, ds, np.concatenate([tr, va]), te, cfg, seed)
                if save_models:
                    joblib.dump(r["model"], fdir / "model.joblib")
                extra = {}
            else:
                r = fit_neural(name, ds, tr, va, te, cfg, seed, device, log=log)
                params = r["n_params"]
                if save_models:
                    torch.save(dict(state=r["state"], norm=r["norm"].to_dict(),
                                    best_epoch=r["best_epoch"]), fdir / "best.pt")
                json.dump(r["history"], open(fdir / "history.json", "w"))
                extra = dict(best_epoch=r["best_epoch"])
                if r["alpha"] is not None:
                    alphas.append(r["alpha"])

            pred = r["probs"].argmax(1)
            m = metrics(ds["y"][te], pred, n_cls)
            m.update(seed=seed, fold=k, n_test_reps=int(len(te)),
                     test_persons=sorted(set(ds["person"][te].tolist())), **extra)
            json.dump(m, open(fdir / "metrics.json", "w"), indent=1)
            fold_metrics.append(m)
            log(f"  [{exp_name}] seed {seed} fold {k}: macro-F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}"
                f"  ({len(te)} reps, persons {m['test_persons']})")

            df = pd.DataFrame(r["probs"], columns=[f"p{i}" for i in range(n_cls)])
            df.insert(0, "pred", pred)
            df.insert(0, "y", ds["y"][te])
            df.insert(0, "rep_no", ds["rep_no"][te])
            df.insert(0, "person", ds["person"][te])
            df.insert(0, "recording", ds["recording"][te])
            df.insert(0, "fold", k)
            df.insert(0, "seed", seed)
            rows.append(df)

    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(out / "preds.csv", index=False)
    if alphas:
        np.save(out / "alpha.npy", np.concatenate(alphas))

    rec = recording_vote(preds, n_cls)
    rec_f1 = [f1_score(g["y"], g["pred"], average="macro", labels=list(range(n_cls)), zero_division=0)
              for _, g in rec.groupby(["seed", "fold"])]
    rec_acc = [accuracy_score(g["y"], g["pred"]) for _, g in rec.groupby(["seed", "fold"])]
    f1s = [m["macro_f1"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]
    pooled = metrics(preds["y"], preds["pred"], n_cls)
    summary = dict(
        experiment=exp_name, model=name, classes=ds["classes"], n_params=params,
        macro_f1_mean=float(np.mean(f1s)), macro_f1_std=float(np.std(f1s)),
        accuracy_mean=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
        rec_macro_f1_mean=float(np.mean(rec_f1)), rec_macro_f1_std=float(np.std(rec_f1)),
        rec_accuracy_mean=float(np.mean(rec_acc)),
        per_class_f1_pooled=pooled["per_class_f1"], confusion_pooled=pooled["confusion"],
        seconds=round(time.time() - t0, 1), data_settings=cfg["data"], model_settings=cfg["model"])
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    log(f"  [{exp_name}] DONE  macro-F1 {summary['macro_f1_mean']:.3f} +- {summary['macro_f1_std']:.3f}"
        f"   recording-level {summary['rec_macro_f1_mean']:.3f}   {summary['seconds']}s")
    return summary


# ----------------------------------------------------------------------------
# 5. Final model for the product (trained on ALL persons)
# ----------------------------------------------------------------------------
def train_final(name, ds, cfg, out_dir, device, epochs=None, log=print):
    """Neural: trains `epochs` epochs on everything (default: the config epochs). Tree: fits on everything."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    allidx = np.arange(len(ds["y"]))
    seed = cfg["cv"]["seeds"][0]
    if name in TREE_MODELS:
        r = fit_tree(name, ds, allidx, None, cfg, seed)
        joblib.dump(r["model"], out / "model.joblib")
    else:
        r = fit_neural(name, ds, allidx, np.array([], dtype=int), None, cfg, seed, device,
                       fixed_epochs=epochs or cfg["train"]["epochs"], log=log)
        torch.save(dict(state=r["state"], norm=r["norm"].to_dict()), out / "model.pt")
    json.dump(dict(model=name, classes=ds["classes"], data=cfg["data"], model_cfg=cfg["model"],
                   T=int(ds["joints"].shape[1])), open(out / "config.json", "w"), indent=1)
    log(f"final model saved to {out}")
