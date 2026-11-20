"""Manifest direkt aus der Namenskonvention des Kurs-Repos bauen.

Konvention (ZDDduesseldorf/human_activity_recognition_zdd2026, docs/naming_convention.md):

    <YYYYMMDD>_<Pxx>_<TYPE>-<activity>_<Txx>_<device>-<position>_<channel>_<stage>.<ext>
    z. B. 20261117_P03_HAR-walking_T01_ax6-ankle-right_imu_proc.csv

Abbildung auf das Manifest:
* ``session``  = Datum (YYYYMMDD) -> Aufnahmen an verschiedenen Tagen werden automatisch zu
  verschiedenen Sessions (Cross-Session-Test, sobald es Retest-Tage gibt)
* ``sensor``   = Position (``wrist-right`` ...); sitzen mehrere Geräte an derselben Position,
  ``<device>-<position>``
* ``channel``  = imu | acc | gyr (getrennte acc/gyr-Dateien werden beim Laden zusammengeführt)
* ``stage``    = proc bevorzugt, sonst raw (mit Hinweis)
* optional zentrales Log (CSV): Sensor-ID -> ``device_id``, ungültige Trials werden ausgeschlossen
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ACTIVITIES = {"walking": "Gehen", "stairs": "Treppen steigen", "jumping-jacks": "Hampelmänner",
              "ball": "Ball werfen/fangen", "squats": "Squats", "door-handle": "Türgriff öffnen",
              "hand-washing": "Hände waschen"}

PATTERN = re.compile(
    r"^(?P<date>\d{8})_(?P<person>P\d{2,})_(?P<type>HAR|EVT)-(?P<activity>[a-z0-9]+(?:-[a-z0-9]+)*)"
    r"_(?P<trial>T\d{2,})_(?P<device>[a-z0-9]+)-(?P<position>[a-z]+(?:-[a-z]+)*\d*)"
    r"_(?P<channel>acc|gyr|imu)_(?P<stage>raw|proc)\.(?P<ext>[A-Za-z0-9]+(?:\.gz)?)$")

_INVALID = re.compile(r"^(invalid|ungültig|ungueltig|nicht gültig|no|nein|false|0|x|verworfen)$", re.I)


def parse_name(name: str) -> dict | None:
    m = PATTERN.match(Path(name).name)
    return m.groupdict() if m else None


def _stem_key(name: str) -> str:
    """Dateiname ohne Stage/Endung – damit raw- und proc-Namen auf denselben Log-Eintrag passen."""
    return re.sub(r"_(raw|proc)\.[A-Za-z0-9.]+$", "", Path(str(name)).name)


def _merge_log(df: pd.DataFrame, log_csv: str | Path) -> pd.DataFrame:
    """Übernimmt Sensor-ID (-> device_id) und Gültigkeit aus dem zentralen Log (Spalten werden erraten)."""
    log = pd.read_csv(log_csv, dtype=str, keep_default_na=False, sep=None, engine="python")
    fcol = next((c for c in log.columns if re.search(r"file|datei", c, re.I) and not re.search(r"orig", c, re.I)), None)
    idcol = next((c for c in log.columns if re.search(r"sensor.?id|serial|serien|device.?id|ger(ä|ae)te.?id", c, re.I)), None)
    vcol = next((c for c in log.columns if re.search(r"valid|gültig|gueltig|status", c, re.I)), None)
    if fcol is None:
        print(f"[convention] Log {log_csv}: keine Dateinamen-Spalte gefunden – Log ignoriert. Spalten: {list(log.columns)}")
        return df
    log = pd.DataFrame({"_key": log[fcol].map(_stem_key),
                        "_devid": log[idcol] if idcol else "", "_valid": log[vcol] if vcol else ""})
    m = df.assign(_key=df["file"].map(_stem_key)).merge(log.drop_duplicates("_key"), on="_key", how="left",
                                                        indicator=True)
    n_missing = int((m["_merge"] == "left_only").sum())
    if n_missing:
        print(f"[convention] {n_missing} Dateien ohne Log-Eintrag")
    if idcol:
        m["device_id"] = m["_devid"].fillna("")
        print(f"[convention] Sensor-ID aus Log-Spalte '{idcol}' übernommen ({int((m['device_id'] != '').sum())} Dateien)")
    if vcol:
        bad = m["_valid"].fillna("").str.strip().str.match(_INVALID)
        if bad.any():
            print(f"[convention] {int(bad.sum())} Dateien laut Log-Spalte '{vcol}' ungültig -> ausgeschlossen")
        m = m[~bad]
    return m.drop(columns=["_key", "_devid", "_valid", "_merge"])


def manifest_from_convention(root: str | Path, stage: str = "proc", types=("HAR",), devices=None,
                             log_csv: str | Path | None = None, out_csv: str | Path | None = None) -> pd.DataFrame:
    """Sucht rekursiv unter ``root`` alle Dateien nach Namenskonvention und baut das Manifest."""
    rows, skipped = [], 0
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        d = parse_name(p.name)
        if d is None:
            skipped += p.suffix.lower() in (".csv", ".tsv", ".txt", ".parquet", ".cwa", ".gz")
            continue
        if d["type"] not in types or (devices and d["device"] not in devices):
            continue
        rows.append({"path": str(p.resolve()), "person": d["person"], "session": d["date"],
                     "activity": d["activity"], "trial": d["trial"], "device": d["device"],
                     "position": d["position"], "channel": d["channel"], "stage": d["stage"], "file": p.name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Keine Dateien nach Namenskonvention unter {root} gefunden ({skipped} andere Datendateien).")
    key = ["person", "session", "activity", "trial", "device", "position", "channel"]
    df["_pref"] = (df["stage"] != stage).astype(int)
    df = df.sort_values(key + ["_pref"]).drop_duplicates(key, keep="first").drop(columns="_pref")
    n_other = int((df["stage"] != stage).sum())
    if n_other:
        print(f"[convention] {n_other} Dateien nur als '{df.loc[df['stage'] != stage, 'stage'].iloc[0]}' vorhanden "
              f"(kein '{stage}') – diese werden verwendet.")
    n_dev = df.groupby("position")["device"].nunique()
    df["sensor"] = [pos if n_dev[pos] == 1 else f"{dev}-{pos}" for dev, pos in zip(df["device"], df["position"])]
    df["device_id"] = ""
    if log_csv:
        df = _merge_log(df, log_csv)
    if skipped:
        print(f"[convention] {skipped} Datendateien passten nicht zur Namenskonvention (ignoriert)")
    print(f"[convention] {len(df)} Dateien, {df['person'].nunique()} Personen, Sessions/Tage: "
          f"{sorted(df['session'].unique())}, Sensoren: {sorted(df['sensor'].unique())}, "
          f"Aktivitäten: {sorted(df['activity'].unique())}")
    df = df.reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
