"""Pipeline-Schritte 4 und 5: Modelle trainieren, Fenster zu einer Clip-Entscheidung
zusammenfassen und bewerten."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config


def erstelle_modell(name, seed=config.SEED):
    """Erstellt ein neues, untrainiertes Modell.

    KNN und SVM rechnen mit Abständen und brauchen skalierte Merkmale. Der StandardScaler steckt
    in einer Pipeline und lernt Mittelwert und Streuung deshalb nur aus den Trainingsdaten
    (Pipeline Abschnitt 4).
    """
    if name == "Mehrheit":   # lernt nichts: sagt immer die Person mit den meisten Trainingsfenstern
        return DummyClassifier(strategy="most_frequent")
    if name == "Entscheidungsbaum":
        return DecisionTreeClassifier(random_state=seed)
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "SVM":   # SVC liefert nur Abstände zur Trennfläche; CalibratedClassifierCV macht daraus
                        # Wahrscheinlichkeiten (Platt-Skalierung), die wir für die Clip-Entscheidung brauchen
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=10), ensemble=False))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    raise ValueError(f"Unbekanntes Modell: {name}")


def clip_entscheidung(wahrscheinlichkeiten, clip_ids, personen):
    """Fasst alle Fenster eines Clips zu einer Entscheidung zusammen.

    Für jeden Clip werden die Wahrscheinlichkeiten seiner Fenster gemittelt. Die Person mit dem
    höchsten Mittelwert gewinnt (eine "weiche" Mehrheitsentscheidung).
    Rückgabe: Series clip_id -> vorhergesagte Person
    """
    tabelle = pd.DataFrame(wahrscheinlichkeiten, columns=personen)
    mittelwerte = tabelle.groupby(np.asarray(clip_ids)).mean()
    return mittelwerte.idxmax(axis=1)


def bewerte(modell_name, merkmale, info, folds, seed=config.SEED):
    """Trainiert und testet ein Modell in jedem Fold.

    Rückgabe:
      ergebnis     eine Zeile pro Fold: Macro-F1 der Fenster, Macro-F1 und Accuracy der Clips
      vorhersagen  eine Zeile pro Test-Clip (für Confusion Matrix und Auswertung je Person)
    """
    ergebnis, vorhersagen = [], []
    for fold, (train, test) in enumerate(folds):
        modell = erstelle_modell(modell_name, seed)           # für jeden Fold ein neues Modell
        modell.fit(merkmale.iloc[train], info.person.iloc[train])
        p = modell.predict_proba(merkmale.iloc[test])         # (Testfenster, Personen)

        fenster_wahr = info.person.iloc[test]
        fenster_vorhersage = modell.classes_[p.argmax(axis=1)]
        clip_wahr = info.iloc[test].groupby("clip_id").person.first()
        clip_vorhersage = clip_entscheidung(p, info.clip_id.iloc[test], modell.classes_)[clip_wahr.index]

        ergebnis.append({
            "fold": fold,
            "f1_fenster": f1_score(fenster_wahr, fenster_vorhersage, average="macro", zero_division=0),
            "f1_clip": f1_score(clip_wahr, clip_vorhersage, average="macro", zero_division=0),
            "acc_clip": accuracy_score(clip_wahr, clip_vorhersage),
        })
        vorhersagen.append(pd.DataFrame({"fold": fold, "clip_id": clip_wahr.index,
                                         "wahr": clip_wahr.values, "vorhergesagt": clip_vorhersage.values}))
    return pd.DataFrame(ergebnis), pd.concat(vorhersagen, ignore_index=True)


def zusammenfassen(ergebnis):
    """Mittelwert und Standardabweichung über die Folds (Pipeline Abschnitt 5)."""
    werte = ergebnis.drop(columns="fold")
    return pd.concat([werte.mean().add_suffix("_mittel"), werte.std().add_suffix("_sd")])
