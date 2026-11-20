"""Dateninventar & Qualitätsprüfung (Projektpipeline, Abschnitt 3).

Erzeugt:
* qc_files.csv        Kennzahlen + Warnflags je Datei (Lücken, NaN, Einheiten, Clipping, Zeitstempel)
* recordings.csv      alle geladenen Trials/Clips
* coverage_*.csv      Datenabdeckung Person x Aktivität (Sekunden, Clips)
* device_check.csv    Geräte-Konfundierung: Ist die Geräte-Kombination pro Person (fast) eindeutig?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import recordings_table


def coverage(recs) -> tuple[pd.DataFrame, pd.DataFrame]:
    rt = recordings_table(recs)
    sec = rt.pivot_table(index="person", columns="activity", values="duration_s", aggfunc="sum", fill_value=0.0)
    n = rt.pivot_table(index="person", columns="activity", values="rec_id", aggfunc="count", fill_value=0)
    return sec.round(1), n


def device_check(manifest: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Prüft, ob Geräte (device_id) mit Personen konfundiert sind.

    Problem: Jedes AX6 hat eigene Kalibrier-Offsets/Gyro-Bias. Trägt jede Person eine
    *eigene* Gerätekombination (bei 7 Geräten und 4 Positionen gibt es 840 Kombinationen),
    kann ein Modell die Person teilweise am Geräte-"Fingerabdruck" erkennen.
    """
    if "device_id" not in manifest.columns or (manifest["device_id"] == "").all():
        return pd.DataFrame(), "Keine device_id im Manifest -> Geräte-Konfundierung nicht prüfbar (bitte ergänzen!)."
    m = manifest[manifest["device_id"] != ""].assign(pair=lambda d: d["sensor"] + ":" + d["device_id"])
    combo = (m.groupby(["person", "session"])["pair"].agg(lambda x: "|".join(sorted(set(x))))
             .rename("combo").reset_index())
    counts = combo["combo"].value_counts()
    combo["n_persons_with_same_combo"] = combo["combo"].map(counts)
    uniq = float((combo["n_persons_with_same_combo"] == 1).mean())
    per_pos = m.groupby("sensor")["device_id"].nunique()
    msg = (f"{uniq:.0%} der Person-Sessions haben eine EINZIGARTIGE Gerätekombination; "
           f"Geräte pro Position: {per_pos.to_dict()}. ")
    if uniq > 0.5:
        msg += ("-> RISIKO: Geräte-Fingerabdruck kann Personen verraten. Gegenmaßnahmen: feste Gerät-Position-"
                "Zuordnung für alle Personen, Bias-Korrektur, rotations-/offsetrobuste Features (Set 'invariant'), "
                "Static-Probe berichten.")
    elif per_pos.max() == 1:
        msg += "-> OK: jede Position nutzt immer dasselbe Gerät (kein Person-Gerät-Confound)."
    return combo, msg


def run_inventory(manifest, recs, qc, log) -> dict:
    sec, n = coverage(recs)
    combo, dmsg = device_check(manifest)
    flagged = qc[qc["flags"].fillna("") != ""] if "flags" in qc else pd.DataFrame()
    rt = recordings_table(recs)
    summary = {
        "n_files_manifest": int(manifest["path"].nunique()),
        "n_files_flagged": int(len(flagged)),
        "n_recordings": int(len(recs)),
        "n_persons": int(rt["person"].nunique()) if len(rt) else 0,
        "n_sessions_per_person_max": int(rt.groupby("person")["session"].nunique().max()) if len(rt) else 0,
        "activities": sorted(rt["activity"].unique()) if len(rt) else [],
        "clips": bool((rt["clip"] != "").any()) if len(rt) else False,
        "total_hours": float(rt["duration_s"].sum() / 3600) if len(rt) else 0.0,
        "device_check": dmsg,
    }
    return {"summary": summary, "coverage_seconds": sec, "coverage_units": n, "device_combos": combo,
            "flagged": flagged, "recordings": rt, "log": log}
