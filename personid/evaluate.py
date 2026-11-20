"""Cross-Validation, Aggregation auf Clip-/Trial-Ebene, Metriken und Statistik.

Zwei Ebenen werden IMMER getrennt berichtet:
* Fenster-Ebene: jedes Fenster einzeln klassifiziert (viele, aber abhängige Beispiele)
* Einheiten-Ebene (Clip bzw. Testblock): Fensterwahrscheinlichkeiten werden aggregiert
  (Default: Mittelwert der Log-Wahrscheinlichkeiten) -> "Wem gehört dieser Clip?"

Statistik:
* Mittelwert ± SD über Folds (Projektpipeline) + gepoolte Werte über alle Testeinheiten
* 95 %-Bootstrap-KI mit Resampling von PERSONEN (Cluster-Bootstrap) – Clips derselben
  Person sind nicht unabhängig
* Permutationstest mit Label-Permutation auf TRIAL-Ebene (je Aktivität)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .features import FeatureStore
from .models import is_raw, make_model
from .splits import check_leakage
from .windows import channel_names, extract


def metrics(y_true, y_pred) -> dict:
    labels = np.unique(y_true)
    return {"acc": accuracy_score(y_true, y_pred),
            "bacc": balanced_accuracy_score(y_true, y_pred) if len(labels) > 1 else np.nan,
            "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)}


def aggregate(units, logp, classes, method="mean_log"):
    """Fenster -> Einheiten. ``logp``: (n, K) Log-Wahrscheinlichkeiten. Rückgabe: DataFrame."""
    df = pd.DataFrame({"unit": units})
    rows = []
    for u, idx in df.groupby("unit", sort=False).indices.items():
        lp = logp[idx]
        if method == "vote":
            score = np.bincount(lp.argmax(1), minlength=len(classes)).astype(float)
        elif method == "mean_proba":
            score = np.exp(lp).mean(0)
        else:
            score = lp.mean(0)
        order = np.argsort(-score)
        rows.append({"unit": u, "pred": classes[order[0]], "top3": list(classes[order[:3]]),
                     "n_windows": len(idx), "margin": float(score[order[0]] - score[order[1]]) if len(order) > 1 else np.nan})
    return pd.DataFrame(rows)


@dataclass
class CVResult:
    name: str
    windows: pd.DataFrame           # fold, rec, start, unit, y, pred
    units: pd.DataFrame             # fold, unit, y, pred, top3, n_windows, ...
    folds: pd.DataFrame             # Metriken pro Fold
    leakage: pd.DataFrame
    classes: np.ndarray
    logp: np.ndarray | None = None  # (n_test_windows, K) – für Evidenzkurven
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        f = self.folds
        out = {"name": self.name, "n_folds": len(f), "n_test_units": len(self.units),
               "n_test_windows": len(self.windows)}
        for lvl in ("win", "unit"):
            for m in ("acc", "bacc", "f1"):
                col = f"{lvl}_{m}"
                out[f"{col}_mean"] = f[col].mean()
                out[f"{col}_sd"] = f[col].std(ddof=1) if len(f) > 1 else np.nan
        pw = metrics(self.windows["y"], self.windows["pred"])
        pu = metrics(self.units["y"], self.units["pred"])
        out.update({f"pooled_win_{k}": v for k, v in pw.items()})
        out.update({f"pooled_unit_{k}": v for k, v in pu.items()})
        out["pooled_unit_top3"] = float(np.mean([y in t for y, t in zip(self.units["y"], self.units["top3"])]))
        out["chance"] = 1.0 / max(1, self.units["y"].nunique())
        out["leakage_ok"] = bool(self.leakage["ok"].all()) if len(self.leakage) else True
        out.update(self.info)
        return out

    def confusion(self, level="unit"):
        d = self.units if level == "unit" else self.windows
        labels = np.array(sorted(set(d["y"]) | set(d["pred"])))
        return confusion_matrix(d["y"], d["pred"], labels=labels), labels


def run_cv(recs, folds, model_name: str, cfg: dict, sensors=None, feature_set="extended",
           L_s=None, step_s=None, labels: dict | None = None, store: FeatureStore | None = None,
           keep_logp=False, seed=None, name=None, required_gap_s=None, verbose=False) -> CVResult:
    """Trainiert/testet ``model_name`` auf allen Folds. ``labels``: rec_idx -> Label (Default: Person)."""
    fs = recs[0].fs
    L = int(round((L_s or cfg["windows"]["length_s"]) * fs))
    step = int(round((step_s or cfg["windows"]["step_s"]) * fs))
    sensors = sensors or cfg["data"]["sensors"]
    seed = cfg.get("seed", 0) if seed is None else seed
    store = store or FeatureStore(recs, fs)
    lab = labels or {i: r.person for i, r in enumerate(recs)}
    used = {f[0] for fo in folds for f in (fo.train + fo.test)} | \
           {int(i) for fo in folds if fo.train_w is not None for i in np.r_[fo.train_w.rec, fo.test_w.rec]}
    classes = np.array(sorted({lab[i] for i in used}))
    cidx = {c: k for k, c in enumerate(classes)}
    wrows, urows, frows, lrows, logps = [], [], [], [], []
    t_start = time.time()
    for fo in folds:
        Wtr, Wte = fo.windows(recs, L, step)
        gap_req = (required_gap_s if required_gap_s is not None else 0.0)
        lrows.append(check_leakage(fo, recs, L, step, gap_req if not fo.leaky else 0.0) | {"leaky_by_design": fo.leaky})
        if len(Wtr) == 0 or len(Wte) == 0:
            continue
        ytr = np.array([lab[i] for i in Wtr.rec])
        yte = np.array([lab[i] for i in Wte.rec])
        if is_raw(model_name):
            Xtr, Xte = extract(recs, Wtr, sensors), extract(recs, Wte, sensors)
        else:
            Xtr, _ = store.get(Wtr, sensors, feature_set)
            Xte, _ = store.get(Wte, sensors, feature_set)
        model = make_model(model_name, cfg, seed=seed, sensors_channels=channel_names(recs, sensors))
        model.fit(Xtr, ytr)
        P = model.predict_proba(Xte)
        full = np.full((len(Wte), len(classes)), 1e-9)
        for j, c in enumerate(model.classes_):
            full[:, cidx[c]] = np.maximum(P[:, j], 1e-9)
        full /= full.sum(1, keepdims=True)
        logp = np.log(full)
        pred = classes[logp.argmax(1)]
        agg = aggregate(Wte.unit, logp, classes, cfg.get("aggregation", "mean_log"))
        unit_y = pd.Series(yte, index=Wte.unit).groupby(level=0).first()
        agg["y"] = agg["unit"].map(unit_y)
        agg["fold"] = fo.name
        wm, um = metrics(yte, pred), metrics(agg["y"], agg["pred"])
        frows.append({"fold": fo.name, "n_train_win": len(Wtr), "n_test_win": len(Wte), "n_test_units": len(agg),
                      **{f"win_{k}": v for k, v in wm.items()}, **{f"unit_{k}": v for k, v in um.items()}})
        wrows.append(pd.DataFrame({"fold": fo.name, "rec": Wte.rec, "start": Wte.start, "unit": Wte.unit,
                                   "y": yte, "pred": pred}))
        urows.append(agg)
        if keep_logp:
            logps.append(logp)
        if verbose:
            print(f"   {fo.name}: win-F1 {wm['f1']:.3f}  unit-acc {um['acc']:.3f}  ({len(Wtr)}/{len(Wte)} Fenster)")
    res = CVResult(name or f"{model_name}/{feature_set}",
                   pd.concat(wrows, ignore_index=True) if wrows else pd.DataFrame(columns=["y", "pred"]),
                   pd.concat(urows, ignore_index=True) if urows else pd.DataFrame(columns=["y", "pred", "top3"]),
                   pd.DataFrame(frows), pd.DataFrame(lrows), classes,
                   np.vstack(logps) if keep_logp and logps else None,
                   {"model": model_name, "feature_set": feature_set if not is_raw(model_name) else "raw",
                    "sensors": "+".join(sensors), "L_s": L / fs, "step_s": step / fs,
                    "protocol": folds[0].protocol if folds else "", "runtime_s": round(time.time() - t_start, 1)})
    return res


# --------------------------------------------------------------------------------------
# Evidenzkurve: Wie viele Sekunden Bewegung braucht man?
# --------------------------------------------------------------------------------------
def evidence_curve(res: CVResult, step_s: float, L_s: float, max_m: int = 12, n_rep: int = 20, seed: int = 0):
    """Für jede Testeinheit zufällige zusammenhängende Läufe aus m Fenstern aggregieren."""
    assert res.logp is not None, "run_cv(..., keep_logp=True) nötig"
    rng = np.random.default_rng(seed)
    w = res.windows.reset_index(drop=True)
    rows = []
    for (fold, unit), g in w.groupby(["fold", "unit"], sort=False):
        g = g.sort_values("start")
        idx, y = g.index.to_numpy(), g["y"].iloc[0]
        for m in range(1, min(max_m, len(idx)) + 1):
            starts = np.arange(len(idx) - m + 1)
            if len(starts) > n_rep:
                starts = rng.choice(starts, n_rep, replace=False)
            hits = [res.classes[res.logp[idx[s:s + m]].mean(0).argmax()] == y for s in starts]
            rows.append({"unit": unit, "m": m, "seconds": L_s + (m - 1) * step_s, "acc": float(np.mean(hits))})
    df = pd.DataFrame(rows)
    out = df.groupby(["m", "seconds"]).agg(acc=("acc", "mean"), n_units=("unit", "nunique")).reset_index()
    return out


# --------------------------------------------------------------------------------------
# Unsicherheit & Signifikanz
# --------------------------------------------------------------------------------------
def bootstrap_ci(units: pd.DataFrame, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Cluster-Bootstrap über PERSONEN (alle Einheiten einer Person werden gemeinsam gezogen)."""
    rng = np.random.default_rng(seed)
    persons = units["y"].unique()
    by_p = {p: units.index[units["y"] == p].to_numpy() for p in persons}
    accs, f1s = [], []
    for _ in range(n_boot):
        pick = rng.choice(persons, len(persons), replace=True)
        idx = np.concatenate([by_p[p] for p in pick])
        y, yp = units.loc[idx, "y"].to_numpy(), units.loc[idx, "pred"].to_numpy()
        accs.append(np.mean(y == yp))
        f1s.append(f1_score(y, yp, labels=np.unique(y), average="macro", zero_division=0))
    a = (1 - level) / 2
    return {"acc": float(np.mean(units["y"] == units["pred"])),
            "acc_lo": float(np.quantile(accs, a)), "acc_hi": float(np.quantile(accs, 1 - a)),
            "f1_lo": float(np.quantile(f1s, a)), "f1_hi": float(np.quantile(f1s, 1 - a))}


def trial_permutation_labels(recs, idx, rng) -> dict:
    """Personen-Labels zwischen den TRIALS derselben Aktivität permutieren.
    Alle Clips eines Trials behalten ein gemeinsames (falsches) Label.

    Interpretation: Bleibt die Accuracy unter Permutation hoch, misst das Protokoll nur die
    Wiedererkennung der AUFNAHME (nicht der Person). Fällt sie auf Zufallsniveau, nutzt das
    Modell Information, die über Aufnahmen hinweg zur Person gehört.
    """
    by_act = {}
    for i in idx:
        by_act.setdefault(recs[i].activity, {}).setdefault(recs[i].trial_key, []).append(i)
    lab = {}
    for act, trials in by_act.items():
        keys = list(trials)
        persons = [k[0] for k in keys]
        perm = rng.permutation(persons)
        for k, p in zip(keys, perm):
            for i in trials[k]:
                lab[i] = p
    return lab


def permutation_test(recs, idx, folds, model_name, cfg, n_perm=50, seed=0, metric="unit_acc", **kw) -> dict:
    rng = np.random.default_rng(seed)
    store = kw.pop("store", None) or FeatureStore(recs, recs[0].fs)
    obs = run_cv(recs, folds, model_name, cfg, store=store, **kw).summary()[f"pooled_{metric}"]
    null = []
    for b in range(n_perm):
        lab = {i: r.person for i, r in enumerate(recs)}
        lab.update(trial_permutation_labels(recs, idx, rng))
        null.append(run_cv(recs, folds, model_name, cfg, labels=lab, store=store, **kw).summary()[f"pooled_{metric}"])
    null = np.array(null)
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, .95)),
            "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm)), "null": null}
