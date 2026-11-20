"""Leakage-sichere Train/Test-Splits für Personenerkennung.

Grundregel (Projektpipeline, Abschnitt 7/8): ERST werden zeitlich getrennte Rohdatenblöcke
für Training und Test festgelegt, DANN werden Fenster gebildet. Fenster, die eine
Split-Grenze schneiden oder im Gap liegen, entstehen gar nicht erst.

Protokolle (von "zu optimistisch" bis "streng"):

====================  ==============================================================
random_windows        UNZULÄSSIG (nur Demo): alle überlappenden Fenster zufällig verteilt
random_clips          naiv: ganze Clips zufällig verteilt -> Nachbar-Clips in Train & Test
block (gap=0)         zusammenhängende Zeitblöcke pro Trial, ohne Abstand
block (gap>0)         zusammenhängende Zeitblöcke + Gap  <- euer Hauptprotokoll
trial                 Leave-one-trial-out (braucht >= 2 Trials pro Person & Aktivität)
cross_activity        Train auf Aktivität(en) A, Test auf Aktivität B (neue Aufnahme)
cross_session         Train Tag 1, Test Tag 2 (Retest-Subset) – echte Wiedererkennung
====================  ==============================================================

Jede Fold-Definition kann mit :func:`check_leakage` automatisch geprüft werden.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .windows import Windows, concat, make_windows

PROTOCOL_LABELS = {
    "random_windows": "Random-Fenster (unzulässig)",
    "random_clips": "Random-Clips (naiv)",
    "block_nogap": "Block-Split ohne Gap",
    "block": "Block-Split mit Gap",
    "trial": "Leave-one-trial-out",
    "cross_activity": "Cross-Activity",
    "cross_session": "Cross-Session (Tag 2)",
}


@dataclass
class Fold:
    name: str
    protocol: str
    train: list = field(default_factory=list)   # [(rec_idx, a, b, unit)]
    test: list = field(default_factory=list)
    leaky: bool = False
    train_w: Windows | None = None               # explizite Fenster (nur random_windows)
    test_w: Windows | None = None
    info: dict = field(default_factory=dict)

    def windows(self, recs, L: int, step: int) -> tuple[Windows, Windows]:
        if self.train_w is not None:
            return self.train_w, self.test_w
        return make_windows(recs, self.train, L, step), make_windows(recs, self.test, L, step)


def _by_trial(recs, idx):
    groups = defaultdict(list)
    for i in idx:
        groups[recs[i].trial_key].append(i)
    for k in groups:
        groups[k].sort(key=lambda i: recs[i].t0)
    return groups


def _whole(recs, i, unit=None):
    return (i, 0, recs[i].n, unit or recs[i].rec_id)


# --------------------------------------------------------------------------------------
# Block-Split (euer Hauptprotokoll bei 1 Trial pro Person & Aktivität)
# --------------------------------------------------------------------------------------
def block_folds(recs, idx, n_blocks: int = 5, gap_s: float = 3.0, gap_clips: int = 0) -> list[Fold]:
    """Zeitlich zusammenhängende Blöcke pro Trial; Fold k testet Block k.

    * Kontinuierlicher Trial (1 Recording): Testblock = Samples in Block k (Auswerteeinheit
      = der Block), Training = alles außerhalb [Blockstart - gap, Blockende + gap).
    * Trial aus Clips: Testclips = Clips, deren Mitte in Block k liegt (ganze Clips sind die
      Auswerteeinheit). Training = alle Samples anderer Clips mit Zeitabstand >= gap_s zum
      Testbereich. Überlappende Clips werden dadurch korrekt ausgeschlossen. Zusätzlich
      werden ``gap_clips`` Nachbarclips komplett ausgeschlossen (automatisch 1, wenn die
      Clip-Zeiten nur aus der Clip-ID abgeleitet wurden).
    """
    folds = [Fold(f"block{k + 1}", "block" if gap_s > 0 else "block_nogap",
                  info={"gap_s": gap_s, "n_blocks": n_blocks}) for k in range(n_blocks)]
    for key, members in _by_trial(recs, idx).items():
        T0 = min(recs[i].t0 for i in members)
        T1 = max(recs[i].t1 for i in members)
        edges = np.linspace(T0, T1, n_blocks + 1)
        clip_mode = len(members) > 1 or any(recs[i].clip for i in members)
        inferred = any(recs[i].meta.get("t0_inferred") for i in members)
        gclips = max(gap_clips, 1 if inferred else 0)
        for k, fold in enumerate(folds):
            lo, hi = edges[k], edges[k + 1]
            if not clip_mode:
                i = members[0]
                r = recs[i]
                a = int(round((lo - r.t0) * r.fs))
                b = int(round((hi - r.t0) * r.fs))
                g = int(np.ceil(gap_s * r.fs))
                fold.test.append((i, a, b, f"{r.rec_id}#B{k + 1}"))
                if a - g > 0:
                    fold.train.append((i, 0, a - g, r.rec_id))
                if b + g < r.n:
                    fold.train.append((i, b + g, r.n, r.rec_id))
                continue
            mids = np.array([(recs[i].t0 + recs[i].t1) / 2 for i in members])
            last = k == n_blocks - 1
            in_block = (mids >= lo) & ((mids <= hi) if last else (mids < hi))
            test_pos = np.where(in_block)[0]
            if len(test_pos) == 0:
                continue
            ts = min(recs[members[p]].t0 for p in test_pos)
            te = max(recs[members[p]].t1 for p in test_pos)
            for p in test_pos:
                fold.test.append(_whole(recs, members[p]))
            for p, i in enumerate(members):
                if p in test_pos:
                    continue
                if gclips and np.min(np.abs(test_pos - p)) <= gclips:
                    continue
                r = recs[i]
                ex_a = int(np.floor((ts - gap_s - r.t0) * r.fs))
                ex_b = int(np.ceil((te + gap_s - r.t0) * r.fs))
                if ex_a > 0:
                    fold.train.append((i, 0, min(r.n, ex_a), r.rec_id))
                if ex_b < r.n:
                    fold.train.append((i, max(0, ex_b), r.n, r.rec_id))
    return [f for f in folds if f.test]


# --------------------------------------------------------------------------------------
# Weitere Protokolle
# --------------------------------------------------------------------------------------
def trial_folds(recs, idx, max_folds: int = 6) -> list[Fold]:
    """Leave-one-trial-out: alle Clips eines Trials gehen gemeinsam in Train ODER Test."""
    by_pa = defaultdict(list)
    for key, members in _by_trial(recs, idx).items():
        person, session, activity, trial = key
        by_pa[(person, activity)].append((session, trial, members))
    sizes = [len(v) for v in by_pa.values()]
    if not sizes or max(sizes) < 2:
        return []
    K = min(max_folds, min(s for s in sizes if s >= 2))
    folds = [Fold(f"trial{k + 1}", "trial") for k in range(K)]
    for (person, activity), trials in by_pa.items():
        trials.sort(key=lambda x: (x[0], x[1]))
        for rank, (_, _, members) in enumerate(trials):
            for k, fold in enumerate(folds):
                target = fold.test if (len(trials) >= 2 and rank % K == k) else fold.train
                target.extend(_whole(recs, i) for i in members)
    return [f for f in folds if f.test]


def cross_fold(recs, train_idx, test_idx, name: str, protocol: str) -> Fold:
    """Eine Fold: Training auf ganzen Recordings ``train_idx``, Test auf ``test_idx``.
    Closed-Set: Testpersonen, die im Training fehlen, werden entfernt."""
    train_persons = {recs[i].person for i in train_idx}
    test_idx = [i for i in test_idx if recs[i].person in train_persons]
    return Fold(name, protocol, [_whole(recs, i) for i in train_idx], [_whole(recs, i) for i in test_idx])


def leave_one_activity_out(recs, idx, activities=None) -> list[Fold]:
    acts = activities or sorted({recs[i].activity for i in idx})
    folds = []
    for a in acts:
        tr = [i for i in idx if recs[i].activity != a]
        te = [i for i in idx if recs[i].activity == a]
        if tr and te:
            folds.append(cross_fold(recs, tr, te, f"LOAO:{a}", "cross_activity"))
    return folds


def random_window_folds(recs, idx, L: int, step: int, n_folds: int = 5, seed: int = 0) -> list[Fold]:
    """UNZULÄSSIG – nur zur Demonstration: alle überlappenden Fenster erzeugen und zufällig
    verteilen. Nahezu identische Fenster landen in Train UND Test."""
    W = make_windows(recs, [_whole(recs, i) for i in idx], L, step)
    y = np.array([recs[i].person for i in W.rec])
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    return [Fold(f"rw{k + 1}", "random_windows", leaky=True, train_w=W.take(tr), test_w=W.take(te))
            for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))]


def random_clip_folds(recs, idx, n_folds: int = 5, seed: int = 0, clip_s: float = 10.0) -> list[Fold]:
    """Naiv: ganze Clips zufällig (stratifiziert nach Person) verteilen – ohne zeitliche Ordnung.
    Kontinuierliche Trials werden dafür in Pseudo-Clips à ``clip_s`` zerlegt."""
    units = []  # (rec_idx, a, b, unit)
    for i in idx:
        r = recs[i]
        if r.clip:
            units.append(_whole(recs, i))
        else:
            m = int(round(clip_s * r.fs))
            for c, a in enumerate(range(0, r.n - m + 1, m)):
                units.append((i, a, a + m, f"{r.rec_id}#c{c + 1}"))
    y = np.array([recs[u[0]].person for u in units])
    n_splits = min(n_folds, int(pd.Series(y).value_counts().min()))
    if n_splits < 2:
        return []
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds.append(Fold(f"rc{k + 1}", "random_clips", [units[j] for j in tr], [units[j] for j in te], leaky=True))
    return folds


def choose_primary(recs, idx) -> str:
    """'trial', wenn die meisten Person-Aktivität-Paare >= 2 Trials haben, sonst 'block'.
    (Cross-Session wird separat als eigene Stufe der Validitätsleiter ausgewertet.)"""
    pa_trials = defaultdict(set)
    for i in idx:
        r = recs[i]
        pa_trials[(r.person, r.activity)].add((r.session, r.trial))
    if pa_trials and np.median([len(v) for v in pa_trials.values()]) >= 2:
        return "trial"
    return "block"


# --------------------------------------------------------------------------------------
# Leakage-Check
# --------------------------------------------------------------------------------------
def check_leakage(fold: Fold, recs, L: int, step: int, required_gap_s: float = 0.0) -> dict:
    """Prüft auf Ebene der ROHDATEN-ZEIT: Überlappen Train- und Testfenster derselben Aufnahme
    (gleicher Trial) – oder liegen sie näher als ``required_gap_s`` beieinander?"""
    Wtr, Wte = fold.windows(recs, L, step)
    spans = defaultdict(lambda: ([], []))
    for W, side in ((Wtr, 0), (Wte, 1)):
        for i, s in zip(W.rec, W.start):
            r = recs[i]
            t = r.t0 + s / r.fs
            spans[r.trial_key][side].append((t, t + L / r.fs))
    n_overlap, min_dist = 0, np.inf
    for key, (tr, te) in spans.items():
        if not tr or not te:
            continue
        tr, te = np.array(tr), np.array(te)
        d = np.maximum(tr[:, None, 0], te[None, :, 0]) - np.minimum(tr[:, None, 1], te[None, :, 1])
        n_overlap += int((d < -1e-9).sum())
        min_dist = min(min_dist, float(d.min()))
    shared_trials = sum(1 for tr, te in spans.values() if tr and te)
    return {"fold": fold.name, "protocol": fold.protocol, "n_train_win": len(Wtr), "n_test_win": len(Wte),
            "n_test_units": len(set(Wte.unit)), "trials_in_both": shared_trials,
            "overlapping_pairs": n_overlap, "min_gap_s": (None if min_dist == np.inf else round(min_dist, 3)),
            "ok": bool(n_overlap == 0 and (min_dist == np.inf or min_dist >= required_gap_s - 1e-6))}
