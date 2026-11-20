"""Fensterbildung – IMMER erst NACH der Train/Test-Aufteilung der Rohdaten.

Ein Split liefert Rohdaten-Intervalle ``(rec_idx, a, b, unit)`` (Samples [a, b) eines
Recordings). Fenster werden ausschließlich innerhalb dieser Intervalle gebildet; ein Fenster
kann also nie über eine Split-Grenze oder in den Gap ragen (Projektpipeline, Abschnitt 8).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Windows:
    rec: np.ndarray     # (n,) int  – Index des Recordings
    start: np.ndarray   # (n,) int  – Startsample
    unit: np.ndarray    # (n,) object – Auswerteeinheit (Clip / Testblock), für Aggregation
    L: int              # Fensterlänge in Samples

    def __len__(self) -> int:
        return len(self.rec)

    def take(self, idx) -> "Windows":
        return Windows(self.rec[idx], self.start[idx], self.unit[idx], self.L)


def empty(L: int) -> Windows:
    return Windows(np.zeros(0, int), np.zeros(0, int), np.zeros(0, object), L)


def concat(ws: list[Windows]) -> Windows:
    ws = [w for w in ws if len(w)]
    if not ws:
        return empty(0)
    return Windows(np.concatenate([w.rec for w in ws]), np.concatenate([w.start for w in ws]),
                   np.concatenate([w.unit for w in ws]), ws[0].L)


def _nan_cumsum(rec) -> np.ndarray:
    """Kumulierte Anzahl NaN-Samples (am Recording zwischengespeichert)."""
    cs = getattr(rec, "_nan_cs", None)
    if cs is None or len(cs) != rec.n + 1:
        bad = np.isnan(rec.X).any(axis=1)
        cs = np.concatenate([[0], np.cumsum(bad)])
        rec._nan_cs = cs
    return cs


def make_windows(recs, intervals, L: int, step: int, drop_nan: bool = True) -> Windows:
    """Fenster der Länge L (Samples) mit Schrittweite ``step`` innerhalb jedes Intervalls."""
    R, S, U = [], [], []
    for (i, a, b, unit) in intervals:
        a, b = max(0, int(a)), min(recs[i].n, int(b))
        if b - a < L:
            continue
        starts = np.arange(a, b - L + 1, step)
        if drop_nan:
            cs = _nan_cumsum(recs[i])
            starts = starts[(cs[starts + L] - cs[starts]) == 0]
        R.append(np.full(len(starts), i))
        S.append(starts)
        U.append(np.full(len(starts), unit, dtype=object))
    if not R:
        return empty(L)
    return Windows(np.concatenate(R), np.concatenate(S), np.concatenate(U), L)


def extract(recs, W: Windows, sensors: list[str] | None = None) -> np.ndarray:
    """Rohfenster als Array (n, L, C) für die gewählten Sensoren."""
    if len(W) == 0:
        return np.zeros((0, W.L, 0), np.float32)
    ch = recs[int(W.rec[0])].channel_idx(sensors)
    out = np.empty((len(W), W.L, len(ch)), np.float32)
    for k, (i, s) in enumerate(zip(W.rec, W.start)):
        out[k] = recs[i].X[s:s + W.L][:, ch]
    return out


def channel_names(recs, sensors: list[str] | None = None) -> list[str]:
    r = recs[0]
    return [r.channels[j] for j in r.channel_idx(sensors)]
