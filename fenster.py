"""Pipeline-Schritt 3: Clips und Fenster bilden, Train/Test-Splits ohne Data Leakage."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config


def schneide_fenster(trials, fs=config.FS, n_clips=config.N_CLIPS, fenster_s=config.FENSTER_S,
                     schritt_s=config.SCHRITT_S):
    """Teilt jeden Trial in n_clips gleich lange Clips und jeden Clip in überlappende Fenster.

    Die Clip-Grenzen werden VOR den Fenstern gezogen (Pipeline Abschnitt 8). Jedes Fenster liegt
    deshalb ganz in einem Clip und reicht nie über eine Grenze.

    Rückgabe:
      X     Array (Anzahl Fenster, Samples pro Fenster, Anzahl Kanäle)
      info  Tabelle mit einer Zeile pro Fenster: person, aktivitaet, trial, clip_nr, clip_id,
            start_s und ende_s (Sekunden ab Trial-Beginn), clip_laenge_s
    """
    laenge = int(fenster_s * fs)          # 3 s * 50 Hz = 150 Samples
    schritt = int(schritt_s * fs)         # 1,5 s * 50 Hz = 75 Samples
    X, zeilen = [], []
    for trial in trials:
        daten = trial["daten"].to_numpy()
        clip_laenge = len(daten) // n_clips
        for clip in range(n_clips):
            clip_beginn = clip * clip_laenge
            for start in range(clip_beginn, clip_beginn + clip_laenge - laenge + 1, schritt):
                fenster = daten[start:start + laenge]
                if np.isnan(fenster).any():          # Fenster mit Datenlücke verwerfen
                    continue
                X.append(fenster)
                zeilen.append({"person": trial["person"], "aktivitaet": trial["aktivitaet"],
                               "trial": trial["id"], "clip_nr": clip, "clip_id": f"{trial['id']}#{clip}",
                               "start_s": start / fs, "ende_s": (start + laenge) / fs,
                               "clip_laenge_s": clip_laenge / fs})
    return np.array(X, dtype=np.float32), pd.DataFrame(zeilen)


def block_split(info, test_clip, gap_s=config.GAP_S):
    """Ein Fold des Block-Splits (Pipeline Abschnitt 7).

    Test:      der Clip test_clip aus jedem Trial
    Training:  alle anderen Clips, aber ohne Fenster, die näher als gap_s am Test-Clip liegen
    Rückgabe:  Zeilennummern (in info) der Trainings- und der Testfenster
    """
    test = info.clip_nr == test_clip
    beginn = test_clip * info.clip_laenge_s            # Beginn des Test-Clips im jeweiligen Trial
    ende = (test_clip + 1) * info.clip_laenge_s        # Ende des Test-Clips
    weit_genug_weg = (info.ende_s <= beginn - gap_s) | (info.start_s >= ende + gap_s)
    train = ~test & weit_genug_weg
    return np.flatnonzero(train), np.flatnonzero(test)


def block_folds(info, n_clips=config.N_CLIPS, gap_s=config.GAP_S):
    """Alle Folds des Block-Splits: Jeder Clip ist genau einmal Test."""
    return [block_split(info, clip, gap_s) for clip in range(n_clips)]


def cross_activity_split(info, test_aktivitaet=config.HAUPT_AKTIVITAET):
    """Training auf allen anderen Aktivitäten, Test auf test_aktivitaet (z. B. Gehen).

    Es zählen nur Personen, die auf beiden Seiten vorkommen: Das Modell kann nur Personen
    erkennen, die es im Training gesehen hat.
    """
    test = info.aktivitaet == test_aktivitaet
    personen = set(info.person[test]) & set(info.person[~test])
    bekannt = info.person.isin(personen)
    return np.flatnonzero(~test & bekannt), np.flatnonzero(test & bekannt)


def zufalls_folds(info, n_folds=config.N_CLIPS, seed=config.SEED):
    """FALSCH, nur zum Vergleich: Fenster zufällig auf die Folds verteilen.

    Die Pipeline verbietet das (Abschnitt 8), weil sich überlappende Fenster dann in Training
    und Test wiederfinden. Wir rechnen es nur, um zu zeigen, wie stark das Ergebnis dadurch steigt.
    """
    return list(KFold(n_folds, shuffle=True, random_state=seed).split(info))


def kleinster_abstand(info, train, test):
    """Kleinster zeitlicher Abstand (in s) zwischen einem Trainings- und einem Testfenster
    aus demselben Trial. Negativ heißt: Die Fenster überlappen, das ist Data Leakage.
    Unendlich heißt: Training und Test haben keinen gemeinsamen Trial.
    """
    tr = info.iloc[train][["trial", "start_s", "ende_s"]]
    te = info.iloc[test][["trial", "start_s", "ende_s"]]
    paare = tr.merge(te, on="trial", suffixes=("_train", "_test"))   # alle Paare im selben Trial
    if paare.empty:
        return np.inf
    abstand = np.maximum(paare.start_s_test - paare.ende_s_train,     # Test liegt nach Training
                         paare.start_s_train - paare.ende_s_test)     # Training liegt nach Test
    return abstand.min()
