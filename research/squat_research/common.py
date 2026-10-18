"""common.py - config loading, paths and small helpers shared by every module and notebook."""
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # research_folders/


def load_config(path=None, overrides=None):
    """Read config/default.yaml (or `path`) and apply {"section.key": value} overrides."""
    cfg = yaml.safe_load(open(path or ROOT / "config" / "default.yaml", encoding="utf-8"))
    return apply_overrides(cfg, overrides or {})


def apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        sec, key = k.split(".", 1)
        cfg[sec][key] = v
    return cfg


def path(cfg, key):
    """A path from the config's `paths` section, made absolute (relative paths are relative to research_folders/)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def results_dir(cfg, *parts):
    d = path(cfg, "results").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


MODEL_NAMES = {
    "logreg": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost",
    "cnn1d": "1D-CNN", "bilstm": "BiLSTM", "bigru": "BiGRU", "cnn_lstm": "CNN-LSTM", "tcn": "TCN",
    "st_transformer": "ST-Transformer",
}


def nice(name):
    return MODEL_NAMES.get(name, name)
