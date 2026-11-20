"""Experimente E0–E8. Jedes Experiment schreibt CSVs + PNGs nach results/<run>/E*/.

E0  Inventar & QC              Datenabdeckung, Warnflags, Geräte-Konfundierung, Leakage-Selbsttest
E1  Modellvergleich (Gehen)    Baselines, DT, KNN, SVM, RF, MiniRocket, 1D-CNN – gleiche Folds
E2  Sensorposition             S01/S02/S05/S06 einzeln & kombiniert, je Aktivität
E3  Aktivitäten & Transfer     Welche Aktivität ist am personenspezifischsten? Train A -> Test B
E4  Datenmenge                 Wie viele Sekunden braucht man? Einfluss der Fensterlänge
E5  Evaluationsprotokolle      Wie stark überschätzen naive Splits? ("Validitätsleiter")
E6  Shortcut-Diagnostik        Static-Probe vs. rotationsinvariante Features; Einbettung
E7  Statistik                  Cluster-Bootstrap-KIs, Permutationstest auf Trial-Ebene
E8  Erklärbarkeit              Gruppierte Permutation Importance (Sensor x Feature-Familie)
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import save_config
from .evaluate import bootstrap_ci, evidence_curve, metrics, permutation_test, run_cv
from .features import FeatureStore, compute_features, feature_meta
from .inventory import run_inventory
from .io import load_manifest, load_recordings, recordings_table
from .models import MODEL_LABELS, make_model
from .splits import (PROTOCOL_LABELS, block_folds, check_leakage, choose_primary, cross_fold,
                     leave_one_activity_out, random_clip_folds, random_window_folds, trial_folds)
from .windows import channel_names, extract

# ----------------------------------------------------------------------------- Kontext


@dataclass
class Ctx:
    cfg: dict
    recs: list
    manifest: pd.DataFrame
    qc: pd.DataFrame
    log: pd.DataFrame
    out: Path
    store: FeatureStore
    rt: pd.DataFrame
    main_session: str
    results: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    def idx(self, activity=None, session="main", exclude_activity=None):
        sess = self.main_session if session == "main" else session
        return [i for i, r in enumerate(self.recs)
                if (activity is None or r.activity == activity) and (sess is None or r.session == sess)
                and (exclude_activity is None or r.activity != exclude_activity)]

    @property
    def L_s(self):
        return float(self.cfg["windows"]["length_s"])

    @property
    def gap_s(self):
        return max(float(self.cfg["windows"]["gap_s"]), self.L_s)

    def primary(self, idx, gap_s=None, L_s=None):
        sp = self.cfg["split"]
        proto = sp["primary"] if sp["primary"] != "auto" else choose_primary(self.recs, idx)
        if proto == "trial":
            f = trial_folds(self.recs, idx, sp.get("max_folds", 6))
            if f:
                return f
        g = gap_s if gap_s is not None else max(self.gap_s, L_s or 0)
        return block_folds(self.recs, idx, sp.get("n_blocks", 5), g, sp.get("gap_clips", 0))

    def activities(self):
        acts = self.cfg["experiments"].get("activities")
        present = list(dict.fromkeys(r.activity for r in self.recs if r.session == self.main_session))
        return [a for a in (acts or present) if a in present]

    def retest_idx(self, activity):
        other = [i for i, r in enumerate(self.recs) if r.session != self.main_session and r.activity == activity]
        return other

    def dir(self, name):
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def setup(cfg: dict, run_name: str | None = None) -> Ctx:
    t = time.strftime("%Y%m%d-%H%M%S")
    out = Path(cfg["results_dir"]) / (run_name or t)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config_used.yaml")
    man = load_manifest(cfg["data"]["manifest"])
    recs, qc, log = load_recordings(man, cfg["data"])
    rt = recordings_table(recs)
    if rt.empty:
        raise ValueError("Keine Recordings geladen – Manifest/Spalten prüfen (siehe load_log und qc_files).")
    act = cfg["experiments"]["main_activity"]
    if act not in set(rt["activity"]):
        raise ValueError(f"experiments.main_activity '{act}' nicht im Datensatz. Vorhanden: {sorted(set(rt['activity']))}")
    med = float(rt["duration_s"].median())
    if med < float(cfg["windows"]["length_s"]) * 2:
        print(f"[WARNUNG] Recordings/Clips sind im Median nur {med:.1f} s lang – Fensterlänge "
              f"{cfg['windows']['length_s']} s ist dafür zu groß (weniger als 2 Fenster pro Clip).")
    main_session = rt.groupby("session")["person"].nunique().idxmax()
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "aeon", "keras"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "nicht installiert"
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    return Ctx(cfg, recs, man, qc, log, out, FeatureStore(recs, recs[0].fs), rt, main_session)


def _row(res, extra=None):
    s = res.summary()
    if extra:
        s.update(extra)
    return s


def _ci(res, ctx, seed=0):
    return bootstrap_ci(res.units, n_boot=ctx.cfg["experiments"].get("E7_n_bootstrap", 2000), seed=seed)


# ----------------------------------------------------------------------------- E0
def e0_inventory(ctx: Ctx):
    d = ctx.dir("E0_inventar")
    inv = run_inventory(ctx.manifest, ctx.recs, ctx.qc, ctx.log)
    ctx.qc.to_csv(d / "qc_files.csv", index=False)
    inv["flagged"].to_csv(d / "qc_flagged.csv", index=False)
    inv["recordings"].to_csv(d / "recordings.csv", index=False)
    inv["coverage_seconds"].to_csv(d / "coverage_seconds.csv")
    inv["coverage_units"].to_csv(d / "coverage_units.csv")
    inv["device_combos"].to_csv(d / "device_check.csv", index=False)
    ctx.log.to_csv(d / "load_log.csv", index=False)
    cov = inv["coverage_seconds"].replace(0, np.nan)
    plots.heatmap(cov, d / "coverage.png", title="Datenabdeckung: Sekunden pro Person und Aktivität",
                  cbar_label="Sekunden", vmin=0, vmax=float(np.nanmax(cov.to_numpy())), annotate=False,
                  figsize=(7.5, 0.22 * len(cov) + 1.8))
    plots.split_diagram(d / "split_schema.png")
    # Leakage-Selbsttest des Hauptprotokolls (Gehen)
    idx = ctx.idx(ctx.cfg["experiments"]["main_activity"])
    fs = ctx.recs[0].fs
    L, st = int(ctx.L_s * fs), int(ctx.cfg["windows"]["step_s"] * fs)
    checks = pd.DataFrame([check_leakage(f, ctx.recs, L, st, ctx.gap_s) for f in ctx.primary(idx)])
    leaky = pd.DataFrame([check_leakage(f, ctx.recs, L, st, 0) for f in random_window_folds(ctx.recs, idx, L, st, 3)])
    checks.to_csv(d / "leakage_check_primary.csv", index=False)
    leaky.to_csv(d / "leakage_check_random_windows.csv", index=False)
    inv["summary"]["primary_protocol_leakage_ok"] = bool(checks["ok"].all())
    inv["summary"]["random_windows_overlapping_pairs"] = int(leaky["overlapping_pairs"].sum())
    inv["summary"]["main_session"] = ctx.main_session
    ctx.report["E0"] = inv["summary"]
    print("[E0]", json.dumps(inv["summary"], ensure_ascii=False, default=str)[:600])


# ----------------------------------------------------------------------------- E1
def e1_models(ctx: Ctx):
    d = ctx.dir("E1_modelle")
    act = ctx.cfg["experiments"]["main_activity"]
    idx = ctx.idx(act)
    folds = ctx.primary(idx)
    rows = []
    for m in ctx.cfg["experiments"]["E1_models"]:
        print(f"[E1] {m} ...")
        res = run_cv(ctx.recs, folds, m, ctx.cfg, store=ctx.store, keep_logp=True, name=m,
                     required_gap_s=ctx.gap_s)
        ctx.results[f"E1:{m}"] = res
        ci = _ci(res, ctx)
        rows.append(_row(res, {"model": m, "label": MODEL_LABELS[m], **{f"ci_{k}": v for k, v in ci.items()}}))
        res.units.assign(top3=res.units["top3"].astype(str)).to_csv(d / f"units_{m}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(d / "E1_models.csv", index=False)
    base = {"majority", "stratified"}
    plots.bars(df, "pooled_unit_acc", "label", d / "E1_clip_accuracy.png", err_lo="ci_acc_lo", err_hi="ci_acc_hi",
               gray=[MODEL_LABELS[b] for b in base], chance=df["chance"].iloc[0],
               title=f"Wem gehört der Clip? ({act}, {len(set(ctx.recs[i].person for i in idx))} Personen)",
               xlabel="Clip-Accuracy (95 %-KI, Bootstrap über Personen)",
               note=f"Protokoll: {PROTOCOL_LABELS.get(folds[0].protocol)} (Gap {ctx.gap_s:.0f} s), "
                    f"{len(folds)} Folds, Fenster {ctx.L_s:.0f} s")
    plots.bars(df, "win_f1_mean", "label", d / "E1_window_macroF1.png", err_sd="win_f1_sd",
               gray=[MODEL_LABELS[b] for b in base], chance=df["chance"].iloc[0],
               title="Fenster-Ebene: Macro-F1 (Mittelwert ± SD über Folds)", xlabel="Macro-F1")
    best = df[~df["model"].isin(base)].sort_values("pooled_unit_acc", ascending=False).iloc[0]["model"]
    res = ctx.results[f"E1:{best}"]
    cm, labels = res.confusion("unit")
    plots.confusion(cm, labels, d / f"E1_confusion_{best}.png",
                    title=f"Konfusionsmatrix (Clips) – {MODEL_LABELS[best]}")
    per_person = res.units.assign(correct=res.units["y"] == res.units["pred"]).groupby("y")["correct"].mean()
    per_person.rename("clip_acc").to_csv(d / f"per_person_{best}.csv")
    ctx.report["E1"] = {"best": best, "table": df[["label", "pooled_unit_acc", "ci_acc_lo", "ci_acc_hi",
                                                    "pooled_unit_f1", "win_f1_mean", "win_f1_sd", "pooled_unit_top3",
                                                    "chance", "leakage_ok", "runtime_s"]].round(3).to_dict("records"),
                        "protocol": folds[0].protocol, "n_folds": len(folds)}


# ----------------------------------------------------------------------------- E2 + E3a
def e2_sensors(ctx: Ctx):
    d = ctx.dir("E2_sensorposition")
    cfg2 = json.loads(json.dumps(ctx.cfg))
    cfg2["models"]["rf"]["n_estimators"] = min(300, cfg2["models"]["rf"]["n_estimators"])
    model = ctx.cfg["experiments"]["E2_model"]
    rows = []
    for act in ctx.activities():
        idx = ctx.idx(act)
        folds = ctx.primary(idx)
        for set_name, sensors in ctx.cfg["experiments"]["E2_sensor_sets"].items():
            res = run_cv(ctx.recs, folds, model, cfg2, sensors=sensors, store=ctx.store, name=set_name)
            rows.append(_row(res, {"activity": act, "sensor_set": set_name}))
        print(f"[E2] {act} fertig")
    df = pd.DataFrame(rows)
    df.to_csv(d / "E2_sensors.csv", index=False)
    mat = df.pivot(index="sensor_set", columns="activity", values="pooled_unit_acc")
    mat = mat.reindex(index=list(ctx.cfg["experiments"]["E2_sensor_sets"]), columns=ctx.activities())
    lo = float(np.nanmin(mat.to_numpy(dtype=float)))
    vmin = np.floor(lo * 10) / 10 if lo > 0.5 else 0.0  # Farbskala an den Wertebereich anpassen (Skala steht dran)
    plots.heatmap(mat, d / "E2_sensor_x_activity.png", title="Clip-Accuracy nach Sensorposition und Aktivität",
                  cbar_label=f"Clip-Accuracy (Farbskala ab {vmin:.0%})", vmin=vmin)
    ctx.report["E2"] = {"matrix": mat.round(3).to_dict()}
    allrow = df[df["sensor_set"] == list(ctx.cfg["experiments"]["E2_sensor_sets"])[-1]]
    ctx.results["E3:per_activity"] = allrow


# ----------------------------------------------------------------------------- E3
def e3_activities(ctx: Ctx):
    d = ctx.dir("E3_aktivitaeten")
    acts = ctx.activities()
    cfg2 = json.loads(json.dumps(ctx.cfg))
    cfg2["models"]["rf"]["n_estimators"] = min(300, cfg2["models"]["rf"]["n_estimators"])
    model = ctx.cfg["experiments"]["E2_model"]
    if "E3:per_activity" in ctx.results:
        per = ctx.results["E3:per_activity"][["activity", "pooled_unit_acc"]].copy()
    else:
        per = pd.DataFrame([{"activity": a, "pooled_unit_acc": run_cv(ctx.recs, ctx.primary(ctx.idx(a)), model, cfg2,
                                                                       store=ctx.store).summary()["pooled_unit_acc"]}
                            for a in acts])
    # Leave-one-activity-out (neue Aufnahme, gleicher Tag)
    loao = []
    for fo in leave_one_activity_out(ctx.recs, ctx.idx(None), acts):
        for fset in ("extended", "invariant"):
            res = run_cv(ctx.recs, [fo], model, cfg2, feature_set=fset, store=ctx.store)
            loao.append(_row(res, {"activity": fo.name.split(":", 1)[1], "feature_set": fset}))
    loao = pd.DataFrame(loao)
    loao.to_csv(d / "E3_leave_one_activity_out.csv", index=False)
    # Transfer: nur A trainieren -> B testen
    T = pd.DataFrame(index=acts, columns=acts, dtype=float)
    for a in acts:
        for b in acts:
            if a == b:
                T.loc[a, b] = per.set_index("activity").loc[a, "pooled_unit_acc"]
                continue
            fo = cross_fold(ctx.recs, ctx.idx(a), ctx.idx(b), f"{a}->{b}", "cross_activity")
            T.loc[a, b] = run_cv(ctx.recs, [fo], model, cfg2, store=ctx.store).summary()["pooled_unit_acc"]
        print(f"[E3] Transfer von {a} fertig")
    T.to_csv(d / "E3_transfer_matrix.csv")
    plots.heatmap(T, d / "E3_transfer.png", title="Transfer: trainiert auf Zeile -> getestet auf Spalte",
                  cbar_label="Clip-Accuracy", xlabel="Test-Aktivität", ylabel="Trainings-Aktivität")
    comb = per.rename(columns={"pooled_unit_acc": "Block-Split (gleiche Aufnahme)"}).merge(
        loao[loao["feature_set"] == "extended"][["activity", "pooled_unit_acc"]].rename(
            columns={"pooled_unit_acc": "Leave-one-activity-out"}), on="activity")
    long = comb.melt(id_vars="activity", var_name="Protokoll", value_name="acc")
    plots.grouped_bars(long, "activity", "Protokoll", "acc", d / "E3_per_activity.png",
                       title="Personenerkennung je Aktivität", ylabel="Clip-Accuracy",
                       chance=1 / ctx.rt[ctx.rt.session == ctx.main_session]["person"].nunique(),
                       cat_order=acts)
    ctx.report["E3"] = {"per_activity": comb.round(3).to_dict("records"),
                        "loao": loao[["activity", "feature_set", "pooled_unit_acc"]].round(3).to_dict("records")}


# ----------------------------------------------------------------------------- E4
def e4_amount(ctx: Ctx):
    d = ctx.dir("E4_datenmenge")
    st = float(ctx.cfg["windows"]["step_s"])
    act = ctx.cfg["experiments"]["main_activity"]
    # Evidenzkurven je Modell (aus E1, nur CSV) ...
    for m in ("rf", "minirocket", "cnn"):
        if f"E1:{m}" in ctx.results:
            evidence_curve(ctx.results[f"E1:{m}"], st, ctx.L_s).to_csv(d / f"E4_evidence_model_{m}.csv", index=False)
    # ... und je Protokoll (Plot): Wie viele Sekunden braucht man bei gleicher vs. neuer Aufnahme?
    series = {}
    for key, folds, gap in _protocols(ctx, act):
        if key not in ("block", "cross_activity", "cross_session") or not folds:
            continue
        res = run_cv(ctx.recs, folds, "rf", ctx.cfg, store=ctx.store, keep_logp=True, required_gap_s=gap)
        ev = evidence_curve(res, st, ctx.L_s)
        ev.to_csv(d / f"E4_evidence_{key}.csv", index=False)
        series[PROTOCOL_LABELS[key]] = ev
    if series:
        n_p = len({ctx.recs[i].person for i in ctx.idx(act)})
        plots.lines(series, d / "E4_evidence.png", title=f"Wie viele Sekunden {act} braucht man? (Random Forest)",
                    xlabel="Sekunden Bewegung (aggregierte Fenster eines Clips)", ylabel="Accuracy", chance=1 / n_p)
    act = ctx.cfg["experiments"]["main_activity"]
    idx = ctx.idx(act)
    rows = []
    for L in ctx.cfg["experiments"]["E4_window_lengths"]:
        folds = ctx.primary(idx, gap_s=max(ctx.gap_s, L))
        res = run_cv(ctx.recs, folds, "rf", ctx.cfg, L_s=L, step_s=L / 2, store=ctx.store,
                     required_gap_s=max(ctx.gap_s, L))
        rows.append(_row(res, {"window_s": L}))
    df = pd.DataFrame(rows)
    df.to_csv(d / "E4_window_length.csv", index=False)
    plots.lines({"Fenster (Macro-F1)": df.assign(x=df["window_s"], y=df["win_f1_mean"]),
                 "Clip (Accuracy)": df.assign(x=df["window_s"], y=df["pooled_unit_acc"])},
                d / "E4_window_length.png", x="x", y="y", title="Einfluss der Fensterlänge (Random Forest)",
                xlabel="Fensterlänge [s] (Schrittweite = 50 %)", ylabel="Score", chance=df["chance"].iloc[0])
    ctx.report["E4"] = {"window_length": df[["window_s", "win_f1_mean", "pooled_unit_acc"]].round(3).to_dict("records"),
                        "evidence": {k: v[["seconds", "acc"]].round(3).to_dict("records") for k, v in series.items()}}


# ----------------------------------------------------------------------------- E5
def _protocols(ctx: Ctx, act):
    """Alle Protokolle für Aktivität ``act`` als (key, label, folds, required_gap)."""
    fs = ctx.recs[0].fs
    L, st = int(ctx.L_s * fs), int(ctx.cfg["windows"]["step_s"] * fs)
    idx = ctx.idx(act)
    sp = ctx.cfg["split"]
    out = [("random_windows", random_window_folds(ctx.recs, idx, L, st, sp.get("random_folds", 5), ctx.cfg["seed"]), 0),
           ("random_clips", random_clip_folds(ctx.recs, idx, sp.get("random_folds", 5), ctx.cfg["seed"]), 0),
           ("block_nogap", block_folds(ctx.recs, idx, sp.get("n_blocks", 5), 0.0, 0), 0),
           ("block", ctx.primary(idx), ctx.gap_s)]
    tf = trial_folds(ctx.recs, idx, sp.get("max_folds", 6))
    if tf:
        out.append(("trial", tf, 0))
    others = ctx.idx(None, exclude_activity=act)
    if others:
        out.append(("cross_activity", [cross_fold(ctx.recs, others, idx, f"andere->{act}", "cross_activity")], 0))
    re = ctx.retest_idx(act)
    if re:
        out.append(("cross_session", [cross_fold(ctx.recs, idx, re, f"Tag1->Tag2 {act}", "cross_session")], 0))
    return out


def e5_protocols(ctx: Ctx):
    d = ctx.dir("E5_protokolle")
    act = ctx.cfg["experiments"]["main_activity"]
    rows = []
    for key, folds, gap in _protocols(ctx, act):
        if not folds:
            continue
        res = run_cv(ctx.recs, folds, "rf", ctx.cfg, store=ctx.store, required_gap_s=gap, name=key)
        ctx.results[f"E5:{key}"] = res
        ci = _ci(res, ctx)
        rows.append(_row(res, {"protocol_key": key, "label": PROTOCOL_LABELS[key],
                               **{f"ci_{k}": v for k, v in ci.items()},
                               "overlapping_pairs": int(res.leakage["overlapping_pairs"].sum()),
                               "min_gap_s": res.leakage["min_gap_s"].min()}))
        print(f"[E5] {key}: Clip-Acc {rows[-1]['pooled_unit_acc']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(d / "E5_protocols.csv", index=False)
    plots.bars(df, "pooled_unit_acc", "label", d / "E5_validity_ladder.png", err_lo="ci_acc_lo", err_hi="ci_acc_hi",
               gray=[PROTOCOL_LABELS["random_windows"], PROTOCOL_LABELS["random_clips"], PROTOCOL_LABELS["block_nogap"]],
               chance=df["chance"].min(), title=f"Wie ehrlich ist die Accuracy? ({act}, Random Forest)",
               xlabel="Clip-Accuracy (95 %-KI)",
               note="Grau = unzulässig/naiv (Train & Test aus benachbarten bzw. überlappenden Rohdaten). "
                    "Von oben nach unten: zunehmend strenger.")
    ctx.report["E5"] = df[["label", "pooled_unit_acc", "ci_acc_lo", "ci_acc_hi", "pooled_win_f1", "n_test_units",
                           "overlapping_pairs", "min_gap_s"]].round(3).to_dict("records")


# ----------------------------------------------------------------------------- E6
def e6_shortcuts(ctx: Ctx):
    d = ctx.dir("E6_shortcuts")
    act = ctx.cfg["experiments"]["main_activity"]
    prots = {k: (f, g) for k, f, g in _protocols(ctx, act) if k in ("block", "cross_activity", "cross_session") and f}
    names = {"static": "Static-Probe", "basic": "Basis (Vorgabe)", "invariant": "Rotations-invariant",
             "extended": "Erweitert"}
    rows = []
    for key, (folds, gap) in prots.items():
        for fset in ("static", "basic", "invariant", "extended"):
            res = run_cv(ctx.recs, folds, "rf", ctx.cfg, feature_set=fset, store=ctx.store, required_gap_s=gap)
            rows.append(_row(res, {"protocol_key": key, "Protokoll": PROTOCOL_LABELS[key],
                                   "feature_set": fset, "Feature-Set": names[fset]}))
        print(f"[E6] {key} fertig")
    df = pd.DataFrame(rows)
    df.to_csv(d / "E6_feature_sets.csv", index=False)
    plots.grouped_bars(df, "Feature-Set", "Protokoll", "pooled_unit_acc", d / "E6_shortcut_probe.png",
                       title="Lernt das Modell die Person – oder den Sensorsitz?", ylabel="Clip-Accuracy",
                       chance=df["chance"].min(), cat_order=list(names.values()),
                       note="Static-Probe: nur Achsen-Mittelwerte (Schwerkraftrichtung = Sensorsitz/Haltung, Gyro-Bias). "
                            "Rotations-invariant: nur Beträge |acc|, |gyr|.")
    # Einbettung (t-SNE) der Gehen-Fenster
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from .windows import make_windows
    idx = ctx.idx(act)
    fs = ctx.recs[0].fs
    W = make_windows(ctx.recs, [(i, 0, ctx.recs[i].n, ctx.recs[i].rec_id) for i in idx], int(ctx.L_s * fs),
                     int(ctx.L_s * fs))
    F, _ = ctx.store.get(W, ctx.cfg["data"]["sensors"], "invariant")
    emb = TSNE(2, random_state=ctx.cfg["seed"], perplexity=min(30, max(5, len(F) // 50))).fit_transform(
        StandardScaler().fit_transform(F))
    persons = np.array([ctx.recs[i].person for i in W.rec])
    plots.embedding(emb, persons, d / "E6_tsne_invariant.png",
                    title="t-SNE der Gehen-Fenster (rotationsinvariante Features)")
    ctx.report["E6"] = df[["Protokoll", "Feature-Set", "pooled_unit_acc"]].round(3).to_dict("records")


# ----------------------------------------------------------------------------- E7
def e7_stats(ctx: Ctx):
    d = ctx.dir("E7_statistik")
    act = ctx.cfg["experiments"]["main_activity"]
    n_perm = int(ctx.cfg["experiments"].get("E7_n_permutations", 50))
    cfg2 = json.loads(json.dumps(ctx.cfg))
    cfg2["models"]["rf"]["n_estimators"] = 150
    prots = {k: (f, g) for k, f, g in _protocols(ctx, act) if k in ("block", "cross_activity", "cross_session") and f}
    rows = []
    for key, (folds, gap) in prots.items():
        idx = sorted({t[0] for f in folds for t in f.train + f.test})
        r = permutation_test(ctx.recs, idx, folds, "rf", cfg2, n_perm=n_perm, seed=ctx.cfg["seed"], store=ctx.store)
        r.update({"label": PROTOCOL_LABELS[key], "protocol_key": key})
        rows.append(r)
        print(f"[E7] {key}: beobachtet {r['observed']:.3f}, Null-Mittel {r['null_mean']:.3f}, p={r['p_value']:.3f}")
    pd.DataFrame([{k: v for k, v in r.items() if k != "null"} for r in rows]).to_csv(d / "E7_permutation.csv", index=False)
    n_p = len({ctx.recs[i].person for i in ctx.idx(act)})
    plots.permutation_strip(rows, d / "E7_permutation.png", chance=1 / n_p,
                            title="Permutationstest: Personen-Labels zwischen Aufnahmen vertauscht")
    ctx.report["E7"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "null"}
                        for r in rows]


# ----------------------------------------------------------------------------- E8
def e8_importance(ctx: Ctx):
    """Gruppierte Permutation Importance: alle Features einer Gruppe (Sensor x Familie)
    werden gemeinsam über Testfenster permutiert; gemessen wird der Abfall der Clip-Accuracy."""
    d = ctx.dir("E8_erklaerbarkeit")
    act = ctx.cfg["experiments"]["main_activity"]
    idx = ctx.idx(act)
    folds = ctx.primary(idx)
    fs = ctx.recs[0].fs
    L, st = int(ctx.L_s * fs), int(ctx.cfg["windows"]["step_s"] * fs)
    sensors = ctx.cfg["data"]["sensors"]
    rng = np.random.default_rng(ctx.cfg["seed"])
    n_rep = int(ctx.cfg["experiments"].get("E8_n_repeats", 5))
    rows, mdi = [], []
    from .evaluate import aggregate
    for fo in folds:
        Wtr, Wte = fo.windows(ctx.recs, L, st)
        Xtr, names = ctx.store.get(Wtr, sensors, "extended")
        Xte, _ = ctx.store.get(Wte, sensors, "extended")
        ytr = np.array([ctx.recs[i].person for i in Wtr.rec])
        yte = np.array([ctx.recs[i].person for i in Wte.rec])
        model = make_model("rf", ctx.cfg, seed=ctx.cfg["seed"]).fit(Xtr, ytr)
        meta = pd.DataFrame([feature_meta(n) for n in names])
        meta["group"] = meta["sensor"] + " · " + meta["family"]
        mdi.append(pd.Series(model.feature_importances_, index=names))

        def clip_acc(X):
            lp = np.log(np.maximum(model.predict_proba(X), 1e-9))
            agg = aggregate(Wte.unit, lp, model.classes_)
            truth = pd.Series(yte, index=Wte.unit).groupby(level=0).first()
            return float((agg["pred"].to_numpy() == agg["unit"].map(truth).to_numpy()).mean())

        base = clip_acc(Xte)
        for g, cols in meta.groupby("group").indices.items():
            drops = []
            for _ in range(n_rep):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[:, cols] = Xp[perm][:, cols]
                drops.append(base - clip_acc(Xp))
            rows.append({"fold": fo.name, "group": g, "sensor": meta.loc[cols[0], "sensor"],
                         "family": meta.loc[cols[0], "family"], "drop": float(np.mean(drops))})
    imp = pd.DataFrame(rows).groupby(["group", "sensor", "family"])["drop"].agg(["mean", "std"]).reset_index()
    imp = imp.rename(columns={"mean": "importance", "std": "importance_sd", "group": "feature"})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(d / "E8_group_importance.csv", index=False)
    plots.importance(imp, d / "E8_group_importance.png", top=16,
                     title="Welche Sensor-/Merkmalsgruppen tragen die Identität? (Gehen, RF)")
    mdi = pd.concat(mdi, axis=1).mean(1).sort_values(ascending=False)
    top = mdi.head(20).rename("importance").reset_index().rename(columns={"index": "feature"})
    top["sensor"] = [feature_meta(f)["sensor"] for f in top["feature"]]
    top.to_csv(d / "E8_top20_mdi.csv", index=False)
    plots.importance(top, d / "E8_top20_mdi.png", top=20, title="Top-20 Einzelfeatures (RF, Gini-Importance)")
    ctx.report["E8"] = imp.head(10).round(4).to_dict("records")


# ----------------------------------------------------------------------------- Bericht
def write_report(ctx: Ctx):
    rep = ctx.report
    (ctx.out / "summary.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    L = ["# Ergebnisbericht Personenerkennung", "",
         f"Lauf: `{ctx.out.name}` · Hauptsession: {ctx.main_session} · Fenster {ctx.L_s:g} s, "
         f"Schritt {ctx.cfg['windows']['step_s']:g} s, Gap {ctx.gap_s:g} s", ""]
    if "E0" in rep:
        e = rep["E0"]
        L += ["## E0 Inventar & QC", f"- {e['n_persons']} Personen, {e['n_recordings']} Recordings "
              f"({'Clips' if e['clips'] else 'Trials'}), {e['total_hours']:.2f} h, Aktivitäten: {', '.join(e['activities'])}",
              f"- {e['n_files_flagged']} von {e['n_files_manifest']} Dateien mit Warnflag -> `E0_inventar/qc_flagged.csv`",
              f"- Geräte-Check: {e['device_check']}",
              f"- Leakage-Selbsttest Hauptprotokoll: {'OK' if e['primary_protocol_leakage_ok'] else 'FEHLER'}; "
              f"Random-Fenster-Split: {e['random_windows_overlapping_pairs']} überlappende Train/Test-Fensterpaare", ""]
    if "E1" in rep:
        L += ["## E1 Modellvergleich", "",
              "| Modell | Clip-Acc | 95 %-KI | Clip-Macro-F1 | Fenster-Macro-F1 (±SD über Folds) | Top-3 | Laufzeit [s] |",
              "|---|---|---|---|---|---|---|"]
        for r in rep["E1"]["table"]:
            L.append(f"| {r['label']} | {r['pooled_unit_acc']:.1%} | {r['ci_acc_lo']:.1%}–{r['ci_acc_hi']:.1%} | "
                     f"{r['pooled_unit_f1']:.3f} | {r['win_f1_mean']:.3f} ± {r['win_f1_sd']:.3f} | "
                     f"{r['pooled_unit_top3']:.1%} | {r['runtime_s']:.0f} |")
        L += ["", f"Zufallsniveau: {rep['E1']['table'][0]['chance']:.1%} · Protokoll: {rep['E1']['protocol']} "
              f"({rep['E1']['n_folds']} Folds) · Bestes Modell: {rep['E1']['best']}", ""]
    if "E5" in rep:
        L += ["## E5 Validitätsleiter (Gehen, RF)", "", "| Protokoll | Clip-Acc | 95 %-KI | überlappende Fensterpaare | min. Abstand [s] |",
              "|---|---|---|---|---|"]
        for r in rep["E5"]:
            L.append(f"| {r['label']} | {r['pooled_unit_acc']:.1%} | {r['ci_acc_lo']:.1%}–{r['ci_acc_hi']:.1%} | "
                     f"{r['overlapping_pairs']} | {r['min_gap_s']} |")
        L.append("")
    if "E7" in rep:
        L += ["## E7 Permutationstest (Trial-Ebene)", ""]
        for r in rep["E7"]:
            L.append(f"- {r['label']}: beobachtet {r['observed']:.1%}, permutiert Ø {r['null_mean']:.1%} "
                     f"(95. Perzentil {r['null_p95']:.1%}), p = {r['p_value']:.3f}")
        L.append("")
    if "E2" in rep:
        mat = pd.DataFrame(rep["E2"]["matrix"])
        L += ["## E2 Sensorposition (Clip-Accuracy, RF)", "", _md(mat.map(lambda v: f"{v:.0%}"), index=True), ""]
    if "E3" in rep:
        per = pd.DataFrame(rep["E3"]["per_activity"]).set_index("activity")
        lo = pd.DataFrame(rep["E3"]["loao"]).pivot(index="activity", columns="feature_set", values="pooled_unit_acc")
        per["LOAO invariant"] = lo.get("invariant")
        L += ["## E3 Aktivitäten (Clip-Accuracy, RF)", "", _md(per.map(lambda v: f"{v:.0%}"), index=True), ""]
    if "E4" in rep:
        wl = pd.DataFrame(rep["E4"]["window_length"])
        L += ["## E4 Fensterlänge (RF)", "", _md(wl.round(3)), ""]
        for name, ev in rep["E4"]["evidence"].items():
            ev = pd.DataFrame(ev)
            hit = ev[ev["acc"] >= 0.9]
            L.append(f"- {name}: ≥ 90 % Accuracy ab {hit['seconds'].iloc[0]:.1f} s Bewegung" if len(hit)
                     else f"- {name}: 90 % nicht erreicht (max {ev['acc'].max():.0%})")
        L.append("")
    if "E6" in rep:
        e6 = pd.DataFrame(rep["E6"]).pivot(index="Feature-Set", columns="Protokoll", values="pooled_unit_acc")
        L += ["## E6 Shortcut-Diagnostik (Clip-Accuracy, RF)", "", _md(e6.map(lambda v: f"{v:.0%}"), index=True), ""]
    if "E8" in rep:
        L += ["## E8 Wichtigste Merkmalsgruppen (Gehen, RF)", ""]
        L += [f"- {r['feature']}: Accuracy-Abfall {r['importance']:.3f}" for r in rep["E8"][:6]]
        L.append("")
    L += ["---", "Hinweis: Alle Zahlen gelten nur für die verwendeten Daten. Bei synthetischen Daten sind sie "
          "KEINE Ergebnisse, sondern ein Funktionstest der Pipeline."]
    (ctx.out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def _md(df: pd.DataFrame, index=False) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(out)


EXPERIMENTS = {"E0": e0_inventory, "E1": e1_models, "E2": e2_sensors, "E3": e3_activities, "E4": e4_amount,
               "E5": e5_protocols, "E6": e6_shortcuts, "E7": e7_stats, "E8": e8_importance}


def run(cfg: dict, only: list[str] | None = None, run_name: str | None = None) -> Ctx:
    ctx = setup(cfg, run_name)
    for key, fn in EXPERIMENTS.items():
        if only and key not in only:
            continue
        t = time.time()
        fn(ctx)
        print(f"[{key}] fertig in {time.time() - t:.0f} s")
        write_report(ctx)
    return ctx
