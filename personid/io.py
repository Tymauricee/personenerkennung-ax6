"""Daten laden: Manifest -> Sensordateien -> synchronisierte Recordings.

Ein *Recording* ist ein zusammenhängendes Stück Messung einer Person bei einer Aktivität,
alle Sensoren (S01, S02, S05, S06) auf einem gemeinsamen Zeitraster:

* entweder ein ganzer Trial (kontinuierlich), oder
* ein Clip aus einem Trial (euer Fall: "1 Trial, aufgeteilt in mehrere Clips").

Das Manifest (CSV) ist die Schnittstelle zu euren echten Daten – eine Zeile pro
(Trial bzw. Clip) x Sensor:

    path,person,session,activity,trial,sensor[,clip,start_s,end_s,device_id,position]

* ``path``      Datei (relativ zum Manifest oder absolut); .csv/.tsv/.txt/.parquet/.cwa
* ``session``   optional (Default "S1"): unterschiedliche Tage / neu angelegte Sensoren
* ``clip``      optional: Clip-ID (z. B. "C03"). Clips desselben Trials werden bei den
                Splits als zeitlich geordnete Teile EINER Aufnahme behandelt.
* ``start_s``/``end_s`` optional: Ausschnitt einer längeren Datei (Zeitbasis der Datei),
                z. B. Clip-Grenzen aus euren Annotationen.

Clips mit *absoluten* Zeitstempeln werden über die Zeit angeordnet (auch überlappende
Clips werden korrekt erkannt). Haben alle Clips eines Trials relative Zeit (Start ~0),
wird die Reihenfolge aus der Clip-ID abgeleitet (lückenlos angenommen) – mit Warnung.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

CANONICAL = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
REQUIRED_MANIFEST_COLS = ["path", "person", "activity", "trial", "sensor"]

# Regex auf "normalisierte" Spaltennamen (klein, nur a-z0-9). Präfix-Match, damit auch
# Einheiten-Suffixe wie "Acc_X (g)" -> "accxg" erkannt werden.
_COL_PATTERNS = {
    "time": [r"^(time|timestamp|zeit|datetime|unixtime|timesec|times|epoch|elapsed)", r"^t$"],
    "acc_x": [r"^(acc|accel|accelerometer|acceleration|a)x"],
    "acc_y": [r"^(acc|accel|accelerometer|acceleration|a)y"],
    "acc_z": [r"^(acc|accel|accelerometer|acceleration|a)z"],
    "gyr_x": [r"^(gyr|gyro|gyroscope|angularvelocity|g|w)x"],
    "gyr_y": [r"^(gyr|gyro|gyroscope|angularvelocity|g|w)y"],
    "gyr_z": [r"^(gyr|gyro|gyroscope|angularvelocity|g|w)z"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _natural_key(s: str):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(s))]


# --------------------------------------------------------------------------------------
# Datenstruktur
# --------------------------------------------------------------------------------------
@dataclass
class Recording:
    """Zusammenhängendes, synchronisiertes Messstück (Trial oder Clip)."""

    rec_id: str
    person: str
    session: str
    activity: str
    trial: str
    fs: float
    channels: list[str]            # z. B. ["S01_acc_x", ..., "S06_gyr_z"]
    X: np.ndarray                  # (n_samples, n_channels), float32, NaN = Lücke
    clip: str = ""
    t0: float = 0.0                # absolute Zeit [s] von Sample 0 (Zeitbasis der Session)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n / self.fs

    @property
    def t1(self) -> float:
        return self.t0 + self.n / self.fs

    @property
    def trial_key(self) -> tuple:
        return (self.person, self.session, self.activity, self.trial)

    @property
    def sensors(self) -> list[str]:
        return list(dict.fromkeys(c.split("_", 1)[0] for c in self.channels))

    def channel_idx(self, sensors: list[str] | None = None) -> np.ndarray:
        if sensors is None:
            return np.arange(len(self.channels))
        return np.array([i for i, c in enumerate(self.channels) if c.split("_", 1)[0] in sensors])


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} fehlen Spalten: {missing}")
    for c, default in (("session", "S1"), ("clip", ""), ("device_id", ""), ("position", ""), ("channel", "imu")):
        if c not in df.columns:
            df[c] = default
    df["session"] = df["session"].replace("", "S1")
    df["channel"] = df["channel"].replace("", "imu").str.lower()
    for c in ("start_s", "end_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    base = path.resolve().parent
    df["path"] = [str(p) if Path(p).is_absolute() else str((base / p).resolve()) for p in df["path"]]
    return df


def build_manifest_from_pattern(root: str | Path, pattern: str,
                                extensions=(".csv", ".tsv", ".txt", ".parquet", ".cwa"),
                                out_csv: str | Path | None = None) -> pd.DataFrame:
    """Erzeugt ein Manifest aus Dateinamen.

    ``pattern`` ist ein Regex mit benannten Gruppen (person, activity, trial, sensor,
    optional session, clip), angewandt auf den Pfad relativ zu ``root`` (mit "/").
    Beispiel für ``P01/Gehen/clip_03/S05.csv``::

        r"(?P<person>P\\d+)/(?P<activity>[^/]+)/clip_(?P<clip>\\d+)/(?P<sensor>S\\d+)\\.csv"

    Fehlt ``trial`` im Pattern, wird "T1" gesetzt (ein Trial pro Person & Aktivität).
    """
    root = Path(root)
    rx = re.compile(pattern)
    rows, unmatched = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        rel = p.relative_to(root).as_posix()
        m = rx.fullmatch(rel) or rx.search(rel)
        if not m:
            unmatched.append(rel)
            continue
        row = {"path": str(p.resolve())}
        row.update({k: v for k, v in m.groupdict().items() if v is not None})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Kein Treffer für Pattern unter {root}. Beispiele: {unmatched[:5]}")
    for c, default in (("session", "S1"), ("trial", "T1")):
        if c not in df.columns:
            df[c] = default
    if unmatched:
        print(f"[manifest] {len(unmatched)} Dateien passten nicht zum Pattern, z. B. {unmatched[:3]}")
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------------------
# Einzelne Sensordatei lesen
# --------------------------------------------------------------------------------------
def _map_columns(cols: list[str], column_map: dict | None, want: list[str] | None = None) -> dict:
    """Ordnet Datei-Spalten den kanonischen Namen zu. Rückgabe: {kanonisch: dateispalte}."""
    want = list(want or CANONICAL)
    mapping = {}
    if column_map:
        mapping.update({k: v for k, v in column_map.items() if v in cols and (k == "time" or k in want)})
    normed = {c: _norm(c) for c in cols}
    for canon in ["time"] + want:
        if canon in mapping or canon not in _COL_PATTERNS:
            continue
        for c, n in normed.items():
            if c in mapping.values():
                continue
            if any(re.match(p, n) for p in _COL_PATTERNS[canon]):
                mapping[canon] = c
                break
    kinds = {w.split("_")[0] for w in want}
    kind = kinds.pop() if len(kinds) == 1 else "acc"
    for ax in "xyz":  # Fallback: "x,y,z" bzw. "x-axis (g)" ohne Präfix (AX-/Mbient-Exporte)
        if f"{kind}_{ax}" in want and f"{kind}_{ax}" not in mapping:
            cand = [c for c, n in normed.items() if re.match(rf"^{ax}(axis)?([a-z]*)$", n) and c not in mapping.values()]
            if cand:
                mapping[f"{kind}_{ax}"] = cand[0]
    return mapping


def _time_to_seconds(series: pd.Series, unit: str) -> np.ndarray:
    if unit == "datetime" or not pd.api.types.is_numeric_dtype(series):
        ts = pd.to_datetime(series, errors="coerce", utc=True)
        return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    t = series.to_numpy(dtype=float)
    if unit == "auto":
        dt = np.nanmedian(np.diff(t[: min(len(t), 5000)]))
        unit = "s" if dt < 0.5 else "ms" if dt < 500 else "us" if dt < 5e5 else "ns"
    return t / {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]


def _read_cwa(path: Path) -> pd.DataFrame:
    """AX6 .cwa über openmovement-python. UNGETESTET ohne echte Datei – bitte prüfen!"""
    try:
        from openmovement.load import CwaData
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Für .cwa-Dateien: pip install "
            "\"git+https://github.com/digitalinteraction/openmovement-python.git@master\""
        ) from e
    with CwaData(str(path), include_gyro=True, include_temperature=False) as cwa:
        return cwa.get_samples()


def channels_for(kind: str, cfg_data: dict) -> list[str]:
    """Welche kanonischen Kanäle eine Datei enthält: imu = alle, acc bzw. gyr = nur diese."""
    chans = list(cfg_data.get("channels", CANONICAL))
    if kind in ("acc", "gyr"):
        return [c for c in chans if c.startswith(kind)]
    return chans


def read_sensor_file(path: str | Path, cfg_data: dict, channels: list[str] | None = None):
    """Liest eine Sensordatei. Rückgabe: (t [s], Y [n, k] in g bzw. °/s, Kanalnamen)."""
    path = Path(path)
    suf = path.suffix.lower()
    unit = cfg_data.get("time_unit", "s")
    if suf == ".cwa":
        df = _read_cwa(path)
        unit = "datetime"
    elif suf == ".parquet":
        df = pd.read_parquet(path)
    elif suf in (".txt", ".tsv"):
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_csv(path)
    chans = list(channels or cfg_data.get("channels", CANONICAL))
    mapping = _map_columns(list(df.columns), cfg_data.get("column_map"), chans)
    miss = [c for c in ["time"] + chans if c not in mapping]
    if miss:
        raise ValueError(f"{path.name}: Spalten nicht erkannt: {miss}. Gefunden: {list(df.columns)}. "
                         f"-> data.column_map in der Config setzen.")
    t = _time_to_seconds(df[mapping["time"]], unit)
    Y = np.column_stack([pd.to_numeric(df[mapping[c]], errors="coerce").to_numpy(float) for c in chans])
    for j, c in enumerate(chans):  # Einheiten vereinheitlichen -> g und °/s
        if c.startswith("acc") and cfg_data.get("acc_unit", "g") in ("m/s2", "ms2", "m/s^2"):
            Y[:, j] /= 9.80665
        if c.startswith("gyr") and cfg_data.get("gyr_unit", "dps") in ("rad/s", "rads"):
            Y[:, j] = np.degrees(Y[:, j])
    return t, Y, chans


def clean_time(t: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sortiert, entfernt doppelte/NaN-Zeitstempel. Gibt Zähler für den QC-Bericht zurück."""
    info = {"n_nan_time": int(np.isnan(t).sum()), "n_nonmonotonic": int((np.diff(t) < 0).sum())}
    ok = ~np.isnan(t)
    t, Y = t[ok], Y[ok]
    order = np.argsort(t, kind="stable")
    t, Y = t[order], Y[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    info["n_dup_time"] = int((~keep).sum())
    return t[keep], Y[keep], info


def qc_stats(t: np.ndarray, Y: np.ndarray, cfg_data: dict, chans: list[str] | None = None) -> dict:
    """Qualitätskennzahlen einer Datei (vor Synchronisation)."""
    chans = chans or CANONICAL[: Y.shape[1]]
    ia = [j for j, c in enumerate(chans) if c.startswith("acc")]
    ig = [j for j, c in enumerate(chans) if c.startswith("gyr")]
    dt = np.diff(t)
    max_gap = float(cfg_data.get("max_gap_s", 0.1))
    s = {
        "channels": "+".join(sorted({c.split("_")[0] for c in chans})),
        "n_samples": int(len(t)),
        "t_start": float(t[0]) if len(t) else np.nan,
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "fs_est": float(1.0 / np.median(dt)) if len(dt) else np.nan,
        "dt_jitter_cv": float(np.std(dt) / np.mean(dt)) if len(dt) else np.nan,
        "n_gaps": int((dt > max_gap).sum()),
        "max_gap_s": float(dt.max()) if len(dt) else np.nan,
        "pct_nan": float(np.isnan(Y).any(axis=1).mean() * 100),
    }
    flags = []
    if ia:
        acc = Y[:, ia]
        an = np.linalg.norm(acc, axis=1)
        s.update({"acc_norm_median_g": float(np.nanmedian(an)), "acc_norm_p99_g": float(np.nanpercentile(an, 99)),
                  "acc_clip_pct": float((np.abs(acc) >= 0.98 * cfg_data.get("acc_range_g", 16)).any(axis=1).mean() * 100)})
        if not 0.6 <= s["acc_norm_median_g"] <= 1.6:
            flags.append("|acc| unplausibel (Einheit m/s²? Kalibrierung?)")
    if ig:
        gyr_abs = np.abs(Y[:, ig])
        s.update({"gyr_abs_p99_dps": float(np.nanpercentile(gyr_abs, 99)),
                  "gyr_clip_pct": float((gyr_abs >= 0.98 * cfg_data.get("gyr_range_dps", 2000)).any(axis=1).mean() * 100)})
        if s["gyr_abs_p99_dps"] < 5 and s.get("acc_norm_p99_g", 2.0) > 1.3:
            flags.append("Gyro sehr klein (rad/s statt °/s?)")
    if s["n_gaps"]:
        flags.insert(0, f"{s['n_gaps']} Lücke(n) > {max_gap}s (max {s['max_gap_s']:.2f}s)")
    if s["pct_nan"] > 0:
        flags.append("NaN-Werte")
    if s.get("acc_clip_pct", 0) > 0.1 or s.get("gyr_clip_pct", 0) > 0.1:
        flags.append("Clipping (Messbereich zu klein?)")
    s["flags"] = "; ".join(flags)
    return s


# --------------------------------------------------------------------------------------
# Synchronisation
# --------------------------------------------------------------------------------------
def synchronize(streams: dict[str, tuple[np.ndarray, np.ndarray]], fs: float, max_gap_s: float,
                lowpass: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bringt mehrere Sensorstreams auf ein gemeinsames Raster (Schnittmenge der Zeitbereiche).

    1. ggf. Anti-Aliasing (Butterworth 4. Ordnung, zero-phase, fc = 0.4*fs) auf einem
       gleichmäßigen Raster der Original-Rate, 2. lineare Interpolation aufs Zielraster,
    3. Samples innerhalb echter Lücken (> max_gap_s) werden NaN – NICHT still interpoliert.
    """
    t0 = max(t[0] for t, _ in streams.values())
    t1 = min(t[-1] for t, _ in streams.values())
    if t1 - t0 < 1.0:
        raise ValueError(f"Zeitbereiche der Sensoren überlappen kaum ({t1 - t0:.2f}s)")
    n = int(np.floor((t1 - t0) * fs + 1e-9)) + 1
    grid = t0 + np.arange(n) / fs
    blocks, info = [], {}
    for sensor, (t, Y) in streams.items():
        dt = np.diff(t)
        fs_in = 1.0 / np.median(dt)
        gaps = np.where(dt > max_gap_s)[0]
        tt, YY = t, Y
        if lowpass and fs_in > 1.05 * fs and len(t) > 50:
            m = int(np.floor((t[-1] - t[0]) * fs_in)) + 1
            tt = t[0] + np.arange(m) / fs_in
            YY = np.column_stack([np.interp(tt, t, Y[:, j]) for j in range(Y.shape[1])])
            nan_mask = np.isnan(YY)
            YY = np.where(nan_mask, 0.0, YY)
            sos = butter(4, 0.4 * fs, btype="low", fs=fs_in, output="sos")
            YY = sosfiltfilt(sos, YY, axis=0)
            YY[nan_mask] = np.nan
        Xs = np.column_stack([np.interp(grid, tt, YY[:, j]) for j in range(YY.shape[1])])
        for g in gaps:
            Xs[(grid > t[g]) & (grid < t[g + 1])] = np.nan
        blocks.append(Xs)
        info[sensor] = {"fs_in": float(fs_in), "n_gaps": int(len(gaps)),
                        "offset_start_s": float(t[0] - t0), "offset_end_s": float(t[-1] - t1)}
    return grid, np.hstack(blocks).astype(np.float32), info


# --------------------------------------------------------------------------------------
# Alles laden (mit Cache)
# --------------------------------------------------------------------------------------
def _cache_key(manifest: pd.DataFrame, cfg_data: dict) -> str:
    h = hashlib.sha1()
    for p in sorted(manifest["path"].unique()):
        st = Path(p).stat() if Path(p).exists() else None
        h.update(f"{p}|{st.st_size if st else -1}|{st.st_mtime_ns if st else -1}".encode())
    h.update(pd.util.hash_pandas_object(manifest.drop(columns=["path"]).astype(str), index=False).values.tobytes())
    keys = ["fs", "sensors", "channels", "time_unit", "column_map", "max_gap_s", "trim_s",
            "require_all_sensors", "acc_unit", "gyr_unit"]
    h.update(json.dumps({k: cfg_data.get(k) for k in keys}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _infer_clip_times(recs: list["Recording"]) -> list[str]:
    """Clips mit relativen Zeitstempeln (alle starten ~gleich) -> Reihenfolge aus Clip-ID."""
    warnings, by_trial = [], {}
    for r in recs:
        if r.clip:
            by_trial.setdefault(r.trial_key, []).append(r)
    for key, clips in by_trial.items():
        if len(clips) < 2:
            continue
        starts = np.array([c.t0 for c in clips])
        if np.ptp(starts) < 0.5:  # alle Clips beginnen "gleichzeitig" -> relative Zeit
            clips.sort(key=lambda c: _natural_key(c.clip))
            t = 0.0
            for c in clips:
                c.t0 = t
                c.meta["t0_inferred"] = True
                t += c.duration_s
            warnings.append(f"{'|'.join(key)}: Clip-Zeitstempel relativ -> Reihenfolge aus Clip-ID "
                            f"angenommen (lückenlos, ohne Überlappung). Bitte prüfen; sonst split.gap_clips erhöhen!")
    return warnings


def sensor_list(manifest: pd.DataFrame, cfg_data: dict) -> list[str]:
    """Sensoren aus der Config – oder ``auto``: alle im Manifest vorkommenden (sortiert)."""
    s = cfg_data.get("sensors", "auto")
    if s in (None, "auto", "all") or s == ["auto"]:
        return sorted(manifest["sensor"].unique())
    return list(s)


def load_recordings(manifest: pd.DataFrame, cfg_data: dict, use_cache: bool = True,
                    verbose: bool = True) -> tuple[list[Recording], pd.DataFrame, pd.DataFrame]:
    """Lädt alle Trials/Clips. Rückgabe: (recordings, qc_pro_datei, log)."""
    cache_dir = Path(cfg_data.get("cache_dir", "cache"))
    cache_file = cache_dir / f"recordings_{_cache_key(manifest, cfg_data)}.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as f:
            out = pickle.load(f)
        if verbose:
            print(f"[io] {len(out[0])} Recordings aus Cache ({cache_file.name})")
        return out

    fs = float(cfg_data["fs"])
    sensors = sensor_list(manifest, cfg_data)
    channels = list(cfg_data.get("channels", CANONICAL))
    trim = int(round(cfg_data.get("trim_s", 0) * fs))
    max_gap = float(cfg_data.get("max_gap_s", 0.1))
    kind_order = {"imu": 0, "acc": 1, "gyr": 2}

    # Pass 1: jede Datei genau einmal lesen + QC (Clips können dieselbe Datei referenzieren)
    files, qc_rows = {}, []
    first = manifest.drop_duplicates("path").set_index("path")
    for path, row in first.iterrows():
        q = {"file": Path(path).name, "path": path, "person": row["person"], "session": row["session"],
             "activity": row["activity"], "trial": row["trial"], "sensor": row["sensor"],
             "channel": row.get("channel", "imu"), "device_id": row.get("device_id", "")}
        if not Path(path).exists():
            q["flags"] = "DATEI FEHLT"
        else:
            try:
                t, Y, chans = read_sensor_file(path, cfg_data, channels_for(row.get("channel", "imu"), cfg_data))
                t, Y, tinfo = clean_time(t, Y)
                q.update(qc_stats(t, Y, cfg_data, chans))
                q.update(tinfo)
                if tinfo["n_dup_time"] or tinfo["n_nonmonotonic"]:
                    q["flags"] = (q["flags"] + "; " if q["flags"] else "") + \
                        f"Zeitstempel: {tinfo['n_dup_time']} doppelt, {tinfo['n_nonmonotonic']} rückwärts"
                files[path] = (t, Y, chans)
            except Exception as e:  # kaputte Datei -> protokollieren, nicht abbrechen
                q["flags"] = f"LESEFEHLER: {e}"
        qc_rows.append(q)

    # Pass 2: pro Trial/Clip alle Streams (Sensor x imu|acc|gyr) synchronisieren
    recs, log = [], []
    gcols = ["person", "session", "activity", "trial", "clip"]
    for (person, session, activity, trial, clip), g in manifest.groupby(gcols, sort=True):
        rec_id = "|".join([person, session, activity, trial] + ([clip] if clip else []))
        streams, devices, have = {}, {}, {}
        rows = sorted(g.to_dict("records"), key=lambda r: (sensors.index(r["sensor"]) if r["sensor"] in sensors else 99,
                                                           kind_order.get(r.get("channel", "imu"), 9)))
        for row in rows:
            sen = row["sensor"]
            if sen not in sensors or row["path"] not in files:
                continue
            t, Y, chans = files[row["path"]]
            new_ch = [c for c in chans if c not in have.get(sen, set())]
            if not new_ch:          # z. B. imu-Datei UND getrennte acc/gyr-Dateien: imu gewinnt
                continue
            cols = [chans.index(c) for c in new_ch]
            if pd.notna(row.get("start_s")) or pd.notna(row.get("end_s")):
                lo = row["start_s"] if pd.notna(row.get("start_s")) else -np.inf
                hi = row["end_s"] if pd.notna(row.get("end_s")) else np.inf
                sel = (t >= lo) & (t < hi)
                t, Y = t[sel], Y[sel]
            if len(t) < 10:
                continue
            streams[f"{sen}|{row.get('channel', 'imu')}"] = (t, Y[:, cols], new_ch, sen)
            have.setdefault(sen, set()).update(new_ch)
            devices[sen] = row.get("device_id", "")
        missing = [f"{s} ({', '.join(c for c in channels if c not in have.get(s, set()))})"
                   for s in sensors if not set(channels) <= have.get(s, set())]
        if missing and (cfg_data.get("require_all_sensors", True) or not streams):
            log.append({"rec_id": rec_id, "reason": f"Sensor(en)/Kanäle fehlen oder unlesbar: {missing}"})
            continue
        try:
            grid, X, sinfo = synchronize({k: (v[0], v[1]) for k, v in streams.items()}, fs, max_gap)
        except ValueError as e:
            log.append({"rec_id": rec_id, "reason": f"Synchronisation: {e}"})
            continue
        t0 = float(grid[0])
        if trim and X.shape[0] > 2 * trim + 1:
            X = X[trim:-trim]
            t0 += trim / fs
        chans = [f"{v[3]}_{c}" for v in streams.values() for c in v[2]]
        recs.append(Recording(rec_id, str(person), str(session), str(activity), str(trial), fs, chans, X,
                              clip=str(clip), t0=t0,
                              meta={"devices": devices, "sync": sinfo,
                                    "pct_nan": float(np.isnan(X).any(axis=1).mean() * 100)}))
    for w in _infer_clip_times(recs):
        log.append({"rec_id": w.split(":")[0], "reason": "WARNUNG: " + w.split(": ", 1)[1]})
    qc = pd.DataFrame(qc_rows)
    logdf = pd.DataFrame(log, columns=["rec_id", "reason"])
    if verbose:
        print(f"[io] {len(recs)} Recordings geladen, {len(logdf)} Log-Einträge, {len(qc)} Dateien geprüft")
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((recs, qc, logdf), f, protocol=pickle.HIGHEST_PROTOCOL)
    return recs, qc, logdf


def recordings_table(recs: list[Recording]) -> pd.DataFrame:
    return pd.DataFrame([{"rec_idx": i, "rec_id": r.rec_id, "person": r.person, "session": r.session,
                          "activity": r.activity, "trial": r.trial, "clip": r.clip, "t0": r.t0,
                          "duration_s": r.duration_s, "pct_nan": r.meta.get("pct_nan", 0.0)}
                         for i, r in enumerate(recs)])
