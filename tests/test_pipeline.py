"""Automatische Prüfungen: Läuft bei jedem Push auf GitHub (Actions) und lokal mit  pytest

Wichtigster Test: test_block_split_ohne_leakage. Er stellt sicher, dass zwischen jedem
Trainings- und jedem Testfenster desselben Trials mindestens GAP_S Sekunden liegen.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import beispieldaten
import config
from fenster import block_folds, cross_activity_split, kleinster_abstand, schneide_fenster, zufalls_folds
from laden import DATEINAME, lade_trials
from merkmale import berechne_merkmale
from modelle import bewerte


@pytest.fixture(scope="module")
def daten(tmp_path_factory):
    """Kleiner künstlicher Datensatz: 3 Personen, 2 Aktivitäten, 4 Sensoren."""
    ordner = tmp_path_factory.mktemp("daten")
    beispieldaten.erzeuge(ordner, n_personen=3, aktivitaeten=["walking", "squats"], seed=1)
    trials, bericht = lade_trials(ordner)
    X, info = schneide_fenster(trials)
    kanaele = list(trials[0]["daten"].columns)
    return trials, bericht, X, info, kanaele


def test_dateiname_wird_zerlegt():
    teile = DATEINAME.fullmatch("20261117_P03_HAR-jumping-jacks_T01_ax6-wrist-right_imu_proc.csv")
    assert teile["person"] == "P03"
    assert teile["aktivitaet"] == "jumping-jacks"
    assert teile["position"] == "wrist-right"
    assert DATEINAME.fullmatch("notizen.csv") is None


def test_laden_und_qualitaetsbericht(daten):
    trials, bericht, X, info, kanaele = daten
    assert len(trials) == 6                                    # 3 Personen x 2 Aktivitäten
    assert len(kanaele) == 4 * 6                               # 4 Sensoren x (3 acc + 3 gyr)
    # die Datei in m/s² wurde in g umgerechnet: alle Beträge liegen nahe 1 g
    assert bericht.acc_betrag_median_g.between(0.7, 1.4).all()
    # die eingebaute Lücke wurde gefunden und nicht still überbrückt
    assert bericht.luecken.sum() == 1
    assert any(trial["daten"].isna().any().any() for trial in trials)


def test_fenster_bleiben_im_clip(daten):
    _, _, X, info, kanaele = daten
    assert X.shape == (len(info), int(config.FENSTER_S * config.FS), len(kanaele))
    assert not np.isnan(X).any()                               # Fenster mit Lücke wurden verworfen
    clip_beginn = info.clip_nr * info.clip_laenge_s
    assert (info.start_s >= clip_beginn - 1e-9).all()
    assert (info.ende_s <= clip_beginn + info.clip_laenge_s + 1e-9).all()


def test_block_split_ohne_leakage(daten):
    _, _, X, info, _ = daten
    folds = block_folds(info)
    assert len(folds) == config.N_CLIPS
    for train, test in folds:
        assert len(set(train) & set(test)) == 0
        assert kleinster_abstand(info, train, test) >= config.GAP_S
        assert set(info.person.iloc[train]) == set(info.person.iloc[test])   # dieselben Personen


def test_zufallssplit_wird_als_leakage_erkannt(daten):
    _, _, X, info, _ = daten
    abstaende = [kleinster_abstand(info, train, test) for train, test in zufalls_folds(info)]
    assert min(abstaende) < 0                                  # überlappende Fenster


def test_cross_activity_trennt_aktivitaeten(daten):
    _, _, X, info, _ = daten
    train, test = cross_activity_split(info, "walking")
    assert (info.aktivitaet.iloc[test] == "walking").all()
    assert (info.aktivitaet.iloc[train] != "walking").all()


def test_betrag_haengt_nicht_von_der_drehung_ab(daten):
    _, _, X, info, kanaele = daten
    gedreht = X.copy()
    for j in range(0, X.shape[2], 3):                          # je drei Achsen = ein Sensor, acc oder gyr
        drehung = Rotation.random(random_state=j).as_matrix()
        gedreht[:, :, j:j + 3] = X[:, :, j:j + 3] @ drehung.T
    vorher = berechne_merkmale(X, kanaele, "nur_betraege")
    nachher = berechne_merkmale(gedreht, kanaele, "nur_betraege")
    assert np.allclose(vorher, nachher, atol=1e-3)
    # die Achsen-Mittelwerte ändern sich dagegen: Sie hängen von der Sensorlage ab
    assert not np.allclose(berechne_merkmale(X, kanaele, "nur_mittelwerte"),
                           berechne_merkmale(gedreht, kanaele, "nur_mittelwerte"), atol=1e-3)


def test_bewertung_liefert_gueltige_werte(daten):
    _, _, X, info, kanaele = daten
    merkmale = berechne_merkmale(X, kanaele)
    ergebnis, vorhersagen = bewerte("Random Forest", merkmale, info, block_folds(info))
    assert len(ergebnis) == config.N_CLIPS
    assert ergebnis[["f1_fenster", "f1_clip", "acc_clip"]].stack().between(0, 1).all()
    assert len(vorhersagen) == info.clip_id.nunique()          # jeder Clip genau einmal Test
