"""Konfiguration: YAML-Datei wird über die Defaults gelegt (deep merge).

Alle Parameter, die Ergebnisse beeinflussen, stehen in der Config und werden pro Lauf
mit abgespeichert (results/<run>/config_used.yaml) -> Reproduzierbarkeit.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "seed": 42,
    "n_jobs": -1,
    "data": {
        "manifest": "data_synth/manifest.csv",
        "cache_dir": "cache",
        "fs": 50.0,                     # Ziel-Samplingrate nach Synchronisation (Hz)
        "sensors": ["S01", "S02", "S05", "S06"],
        "positions": {"S01": "Handgelenk rechts", "S02": "Handgelenk links",
                      "S05": "Rumpf", "S06": "Bein"},
        "channels": ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
        "time_unit": "s",               # s | ms | us | ns | datetime
        "column_map": {},               # optional: {"acc_x": "Ax", ...} falls Auto-Erkennung nicht reicht
        "max_gap_s": 0.1,               # längere Lücken -> NaN (werden NICHT still interpoliert)
        "trim_s": 1.0,                  # Anfang/Ende jedes Trials abschneiden (Übergänge)
        "require_all_sensors": True,
        "acc_range_g": 16.0,            # für Clipping-Check (AX6 konfigurierbar: 2/4/8/16 g)
        "gyr_range_dps": 2000.0,        # für Clipping-Check (AX6: 125...2000 dps)
    },
    "windows": {
        "length_s": 3.0,
        "step_s": 1.5,                  # 50 % Überlappung (nur INNERHALB eines Split-Teils)
        "gap_s": 3.0,                   # Abstand Train/Test bei Block-Split (>= 1 Fensterlänge)
    },
    "split": {
        "primary": "auto",              # auto | trial | block | session
        "n_blocks": 4,                  # für Block-Split
        "max_folds": 6,                 # für Trial-Split
        "random_folds": 5,              # nur für die (unzulässige) Random-Window-Demo
    },
    "features": {"default_set": "extended"},
    "models": {
        "rf": {"n_estimators": 500},
        "svm": {"C": 10.0},
        "knn": {"n_neighbors": 5},
        "minirocket": {"n_kernels": 10000},
        "cnn": {"epochs": 60, "batch_size": 64, "patience": 10, "rot_deg": 15.0,
                "width": 64, "dropout": 0.3, "lr": 1e-3},
    },
    "aggregation": "mean_log",          # mean_log | mean_proba | vote
    "experiments": {
        "main_activity": "Gehen",
        "activities": None,             # None = alle vorhandenen
        "E1_models": ["majority", "stratified", "dt", "knn", "svm", "rf", "minirocket", "cnn"],
        "E2_model": "rf",
        "E2_sensor_sets": {
            "S01 (re. Hand)": ["S01"], "S02 (li. Hand)": ["S02"], "S05 (Rumpf)": ["S05"],
            "S06 (Bein)": ["S06"], "Hände (S01+S02)": ["S01", "S02"],
            "Rumpf+Bein (S05+S06)": ["S05", "S06"], "Alle 4": ["S01", "S02", "S05", "S06"],
        },
        "E4_window_lengths": [1.0, 2.0, 3.0, 4.0, 6.0],
        "E7_n_permutations": 100,
        "E7_n_bootstrap": 2000,
        "E8_n_repeats": 5,
    },
    "results_dir": "results",
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "E2_sensor_sets":
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Lädt die Config. Relative Pfade im YAML werden relativ zur YAML-Datei aufgelöst."""
    cfg = copy.deepcopy(DEFAULTS)
    base_dir = Path.cwd()
    if path is not None:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
        base_dir = path.resolve().parent.parent if path.parent.name == "configs" else path.resolve().parent
    if overrides:
        _deep_update(cfg, overrides)
    for key in ("manifest", "cache_dir"):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            cfg["data"][key] = str((base_dir / p).resolve())
    rd = Path(cfg["results_dir"])
    if not rd.is_absolute():
        cfg["results_dir"] = str((base_dir / rd).resolve())
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
