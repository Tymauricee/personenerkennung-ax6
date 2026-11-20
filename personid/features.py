"""Handcrafted Features pro Fenster (vektorisiert, numpy).

Feature-Sets
------------
basic        Vorgabe der Projektpipeline: Mittelwert, Std, Min, Max, RMS, Energie –
             für alle Achsen + Beträge (|acc|, |gyr|) aller Sensoren.
extended     basic + Perzentile, IQR, Schiefe, Kurtosis, MAD, Nulldurchgänge, Jerk,
             Spektrum (Dominanzfrequenz, spektrale Entropie, Schwerpunkt, Bandenergien),
             Gang-Autokorrelation nach Moe-Nilssen & Helbostad (2004) auf den Beträgen
             (Schritt-/Doppelschritt-Regularität, Symmetrie, Schrittzeit),
             Achsen-Korrelationen je Sensor, Korrelationen zwischen Sensoren.
invariant    nur aus |acc| und |gyr| je Sensor -> unabhängig davon, wie der Sensor gedreht
             sitzt (Montage/Ausrichtung). Kontrolle gegen den "Sensorsitz-Shortcut".
static       nur Mittelwerte der Achsen (acc: Richtung der Schwerkraft = Sensorsitz/Haltung,
             gyr: Bias). DIAGNOSE-Probe: Wenn das allein Personen erkennt, kann das Modell
             über Sensorsitz/Gerät statt über Bewegung "abkürzen".
"""
from __future__ import annotations

import numpy as np

FEATURE_SETS = ("basic", "extended", "invariant", "static")
_BANDS = [(0.3, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 15)]


def _sensors(channels):
    return list(dict.fromkeys(c.split("_", 1)[0] for c in channels))


def _norm_channels(W, channels):
    arrs, names = [], []
    for s in _sensors(channels):
        for kind in ("acc", "gyr"):
            idx = [channels.index(f"{s}_{kind}_{a}") for a in "xyz" if f"{s}_{kind}_{a}" in channels]
            if len(idx) == 3:
                arrs.append(np.linalg.norm(W[:, :, idx], axis=2))
                names.append(f"{s}_{kind}norm")
    if not arrs:
        return np.zeros((W.shape[0], W.shape[1], 0), W.dtype), []
    return np.stack(arrs, axis=2), names


def _stats(X, fs, full=True):
    n, L, C = X.shape
    mean = X.mean(1)
    Xc = X - mean[:, None, :]
    std = Xc.std(1)
    out = {"mean": mean, "std": std, "min": X.min(1), "max": X.max(1),
           "rms": np.sqrt((X ** 2).mean(1)),
           "energy": (Xc ** 2).mean(1)}   # AC-Energie (Bao & Intille 2004, via Parseval) ~ Varianz
    if full:
        q10, q25, q50, q75, q90 = np.percentile(X, [10, 25, 50, 75, 90], axis=1)
        s = std + 1e-9
        out.update({"p10": q10, "median": q50, "p90": q90, "iqr": q75 - q25,
                    "skew": (Xc ** 3).mean(1) / s ** 3, "kurt": (Xc ** 4).mean(1) / s ** 4 - 3,
                    "mad": np.abs(Xc).mean(1),
                    "zcr": (np.diff(np.signbit(Xc), axis=1) != 0).mean(1),
                    "jerk": np.abs(np.diff(X, axis=1)).mean(1) * fs})
    return out


def _spectral(X, fs):
    n, L, C = X.shape
    Xc = (X - X.mean(1, keepdims=True)) * np.hanning(L)[None, :, None]
    nfft = max(256, 1 << int(np.ceil(np.log2(L))))
    P = np.abs(np.fft.rfft(Xc, n=nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= 0.3) & (f <= min(15.0, fs / 2))
    Pb, fb = P[:, band, :], f[band]
    tot = Pb.sum(1) + 1e-12
    p = Pb / tot[:, None, :]
    out = {"domfreq": fb[Pb.argmax(1)], "domfrac": Pb.max(1) / tot,
           "specent": -(p * np.log(p + 1e-12)).sum(1) / np.log(Pb.shape[1]),
           "speccent": (p * fb[None, :, None]).sum(1)}
    for lo, hi in _BANDS:
        if lo < fs / 2:
            m = (fb >= lo) & (fb < hi)
            out[f"band{lo:g}-{hi:g}"] = Pb[:, m, :].sum(1) / tot
    return out


def _acf(X, fs):
    """Unverzerrte Autokorrelation (Moe-Nilssen & Helbostad 2004) -> Gangregularität."""
    n, L, C = X.shape
    Xc = X - X.mean(1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    F = np.fft.rfft(Xc, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=1)[:, :L, :]
    ac = ac / (L - np.arange(L))[None, :, None]
    ac = ac / (ac[:, :1, :] + 1e-12)
    lags = np.arange(L) / fs

    def peak(lo, hi):
        m = (lags >= lo) & (lags <= hi)
        if m.sum() < 2:
            z = np.zeros((n, C))
            return z, z
        seg = ac[:, m, :]
        j = seg.argmax(1)
        return seg.max(1), lags[m][j]

    p1v, p1l = peak(0.25, 0.85)                       # Schritt
    p2v, p2l = peak(0.70, min(1.7, 0.6 * L / fs))     # Doppelschritt (Stride)
    return {"acf_step": p1v, "acf_steptime": p1l, "acf_stride": p2v, "acf_stridetime": p2l,
            "acf_sym": np.abs(p1v - p2v)}


def _corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def _flatten(d: dict, chan_names: list[str]):
    cols, names = [], []
    for fname, arr in d.items():
        for j, c in enumerate(chan_names):
            cols.append(arr[:, j])
            names.append(f"{c}__{fname}")
    return cols, names


def compute_features(W: np.ndarray, channels: list[str], fs: float, feature_set: str = "extended"):
    """W: (n, L, C) Rohfenster. Rückgabe: (F (n, d) float32, feature_names)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set muss in {FEATURE_SETS} sein")
    W = W.astype(np.float64, copy=False)
    sensors = _sensors(channels)
    cols, names = [], []
    if feature_set == "static":
        c, n = _flatten({"mean": W.mean(1)}, channels)
        return np.column_stack(c).astype(np.float32), n
    N, norm_names = _norm_channels(W, channels)
    if feature_set == "invariant":
        X, xn = N, norm_names
    else:
        X, xn = np.concatenate([W, N], axis=2), list(channels) + norm_names
    full = feature_set in ("extended", "invariant")
    c, n = _flatten(_stats(X, fs, full=full), xn)
    cols += c
    names += n
    if full:
        c, n = _flatten(_spectral(X, fs), xn)
        cols += c
        names += n
        c, n = _flatten(_acf(N, fs), norm_names)
        cols += c
        names += n
        for s in sensors:  # Kopplung Beschleunigung <-> Drehung je Sensor (rotationsinvariant)
            if f"{s}_accnorm" in norm_names and f"{s}_gyrnorm" in norm_names:
                ia, ig = norm_names.index(f"{s}_accnorm"), norm_names.index(f"{s}_gyrnorm")
                cols.append(_corr(N[:, :, ia], N[:, :, ig]))
                names.append(f"{s}_accnorm~{s}_gyrnorm__corr")
        for x in range(len(sensors)):  # Koordination zwischen Körperteilen
            for y in range(x + 1, len(sensors)):
                for kind in ("acc", "gyr"):
                    na, nb = f"{sensors[x]}_{kind}norm", f"{sensors[y]}_{kind}norm"
                    if na in norm_names and nb in norm_names:
                        cols.append(_corr(N[:, :, norm_names.index(na)], N[:, :, norm_names.index(nb)]))
                        names.append(f"{na}~{nb}__corr")
    if feature_set == "extended":  # Achsen-Korrelationen (orientierungsabhängig)
        for s in sensors:
            for kind in ("acc", "gyr"):
                for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
                    ca, cb = f"{s}_{kind}_{a}", f"{s}_{kind}_{b}"
                    if ca in channels and cb in channels:
                        cols.append(_corr(W[:, :, channels.index(ca)], W[:, :, channels.index(cb)]))
                        names.append(f"{ca}~{cb}__corr")
    F = np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return F, names


def feature_meta(name: str) -> dict:
    """Zerlegt einen Feature-Namen in Sensor, Kanal und Feature-Familie (für Auswertungen)."""
    chan, feat = name.split("__", 1)
    sens = {p.split("_", 1)[0] for p in chan.split("~")}
    sensor = sens.pop() if len(sens) == 1 else "mehrere"
    if feat.startswith("acf"):
        fam = "Gang-Autokorrelation"
    elif feat.startswith(("domfreq", "domfrac", "specent", "speccent", "band")):
        fam = "Spektrum"
    elif feat == "corr":
        fam = "Koordination (Korrelation)"
    elif feat in ("mean", "median", "p10", "p90", "min", "max") and "norm" not in chan:
        fam = "Lage/Ausrichtung (Achsenwerte)"
    else:
        fam = "Statistik"
    return {"sensor": sensor, "channel": chan, "family": fam}


class FeatureStore:
    """Cache: Features je (Sensoren, Feature-Set, Fensterlänge) und Fenster (rec, start)."""

    def __init__(self, recs, fs: float):
        self.recs, self.fs = recs, fs
        self._st: dict = {}

    def get(self, W, sensors, feature_set):
        from .windows import Windows, channel_names, extract
        key = (tuple(sensors), feature_set, W.L)
        st = self._st.setdefault(key, {"index": {}, "M": None, "names": None})
        ids = list(zip(W.rec.tolist(), W.start.tolist()))
        missing = [k for k in dict.fromkeys(ids) if k not in st["index"]]
        if missing:
            Wm = Windows(np.array([m[0] for m in missing]), np.array([m[1] for m in missing]),
                         np.zeros(len(missing), object), W.L)
            F, names = compute_features(extract(self.recs, Wm, sensors), channel_names(self.recs, sensors),
                                        self.fs, feature_set)
            base = 0 if st["M"] is None else st["M"].shape[0]
            st["M"] = F if st["M"] is None else np.vstack([st["M"], F])
            st["names"] = names
            for j, k in enumerate(missing):
                st["index"][k] = base + j
        if not ids:
            return np.zeros((0, 0), np.float32), st["names"] or []
        return st["M"][[st["index"][k] for k in ids]], st["names"]
