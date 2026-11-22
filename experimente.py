"""Führt die Experimente E1 bis E4 aus und speichert Tabellen und Grafiken.

    python experimente.py          echte Daten aus config.DATEN_ORDNER -> ergebnisse/echt/
    python experimente.py --demo   künstliche Daten (vorher: python beispieldaten.py) -> ergebnisse/demo/

E1  Welches Modell erkennt Personen am besten?             (Block-Split mit Gap)
E2  Wie stark hängt das Ergebnis vom Split ab?             (falscher Split bis Cross-Activity)
E3  Erkennt das Modell die Person oder nur die Sensorlage? (Merkmalssätze)
E4  Welche Sensorposition verrät die Person am besten?
"""
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # Grafiken nur als Datei speichern, kein Fenster öffnen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import config
from fenster import (block_folds, cross_activity_split, kleinster_abstand, schneide_fenster,
                     zufalls_folds)
from laden import lade_trials
from merkmale import berechne_merkmale
from modelle import bewerte, zusammenfassen

FARBEN = ["#2b6cb0", "#dd6b20", "#718096"]   # blau, orange, grau
CROSS = "andere Aktivitäten -> Gehen"


def pruefe_split(info, folds, gap_s):
    """Bricht ab, wenn Trainings- und Testfenster näher als gap_s beieinander liegen."""
    for fold, (train, test) in enumerate(folds):
        abstand = kleinster_abstand(info, train, test)
        assert abstand >= gap_s - 1e-9, f"Leakage in Fold {fold}: Abstand nur {abstand:.2f} s"


def balkendiagramm(tabelle, x, titel, datei, gruppen=None, zufall=None):
    """Macro-F1 pro Clip als Balken: Mittelwert über die Folds, Fehlerbalken = Standardabweichung."""
    if gruppen is None:
        mittel = tabelle.set_index(x)["f1_clip_mittel"]
        sd = tabelle.set_index(x)["f1_clip_sd"]
        farbe = FARBEN[0]
    else:
        zeilen, spalten = tabelle[x].unique(), tabelle[gruppen].unique()
        mittel = tabelle.pivot(index=x, columns=gruppen, values="f1_clip_mittel").loc[zeilen, spalten]
        sd = tabelle.pivot(index=x, columns=gruppen, values="f1_clip_sd").loc[zeilen, spalten]
        farbe = FARBEN[:len(spalten)]
    ax = mittel.plot.bar(yerr=sd, capsize=3, color=farbe, figsize=(9, 4.5), rot=0, width=0.75)
    if zufall is not None:
        ax.axhline(zufall, color="gray", linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[0] + 0.05, zufall + 0.015, "Zufall", color="gray", fontsize=9)
    ax.set_xticks(range(len(mittel)), [textwrap.fill(str(name), 18) for name in mittel.index])
    ax.set(ylim=(0, 1.05), ylabel="Macro-F1 pro Clip", xlabel="", title=titel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if gruppen is not None:
        ax.legend(title=None, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(spalten),
                  frameon=False)
    plt.tight_layout()
    plt.savefig(datei, dpi=150)
    plt.close()


def confusion_matrix(vorhersagen, titel, datei):
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(vorhersagen.wahr, vorhersagen.vorhergesagt, ax=ax,
                                            cmap="Blues", colorbar=False, xticks_rotation=90,
                                            include_values=vorhersagen.wahr.nunique() <= 15)
    ax.set(title=titel, xlabel="vorhergesagte Person", ylabel="wahre Person")
    plt.tight_layout()
    plt.savefig(datei, dpi=150)
    plt.close()


def main():
    demo = "--demo" in sys.argv
    ordner = config.DEMO_ORDNER if demo else config.DATEN_ORDNER
    ziel = Path(config.ERGEBNIS_ORDNER) / ("demo" if demo else "echt")
    ziel.mkdir(parents=True, exist_ok=True)

    # --- Schritte 1-3: laden, prüfen, Fenster, Merkmale ---------------------------------------
    trials, bericht = lade_trials(ordner)
    bericht.round(3).to_csv(ziel / "qualitaet.csv", index=False)
    X, info = schneide_fenster(trials)
    kanaele = list(trials[0]["daten"].columns)
    merkmale = berechne_merkmale(X, kanaele, "alle")
    gehen = (info.aktivitaet == config.HAUPT_AKTIVITAET).to_numpy()
    info_g = info[gehen].reset_index(drop=True)
    merkmale_g = merkmale[gehen].reset_index(drop=True)
    zufall = 1 / info.person.nunique()

    with open(ziel / "protokoll.txt", "w", encoding="utf-8") as f:     # Pipeline Abschnitt 9
        f.write(f"Lauf: {datetime.now():%Y-%m-%d %H:%M}   Daten: {ordner}\n")
        f.write(f"{len(trials)} Trials, {info.person.nunique()} Personen, {info.clip_id.nunique()} Clips, "
                f"{len(info)} Fenster, {merkmale.shape[1]} Merkmale\n")
        f.write(f"Sensoren: {sorted({k.split('_')[0] for k in kanaele})}\n")
        f.write(f"fs={config.FS} Hz, Fenster={config.FENSTER_S} s, Schritt={config.SCHRITT_S} s, "
                f"Gap={config.GAP_S} s, Clips pro Trial={config.N_CLIPS}, Rand={config.RAND_S} s, "
                f"Seed={config.SEED}\n\nFenster pro Person und Aktivität:\n")
        f.write(pd.crosstab(info.person, info.aktivitaet).to_string())
    print(f"{len(info)} Fenster, {info.clip_id.nunique()} Clips, {info.person.nunique()} Personen")

    # --- E1: Modellvergleich (Block-Split mit Gap) --------------------------------------------
    zeilen = []
    for daten_name, inf, mm in [("nur Gehen", info_g, merkmale_g), ("alle Aktivitäten", info, merkmale)]:
        folds = block_folds(inf)
        pruefe_split(inf, folds, config.GAP_S)
        for modell in config.MODELLE:
            ergebnis, vorhersagen = bewerte(modell, mm, inf, folds)
            zeilen.append({"daten": daten_name, "modell": modell, **zusammenfassen(ergebnis)})
            if modell == "Random Forest" and daten_name == "nur Gehen":
                confusion_matrix(vorhersagen, "Random Forest, Gehen, alle Folds", ziel / "e1_confusion_rf.png")
                bericht_pro_person = classification_report(vorhersagen.wahr, vorhersagen.vorhergesagt,
                                                            output_dict=True, zero_division=0)
                pd.DataFrame(bericht_pro_person).T.round(3).to_csv(ziel / "e1_pro_person_rf.csv")
            print(f"E1  {daten_name:17s} {modell:18s} F1 Clip {zeilen[-1]['f1_clip_mittel']:.3f}")
    e1 = pd.DataFrame(zeilen)
    e1.round(3).to_csv(ziel / "e1_modelle.csv", index=False)
    balkendiagramm(e1, "modell", "E1: Modellvergleich (Block-Split mit Gap)", ziel / "e1_modelle.png",
                   gruppen="daten", zufall=zufall)

    # --- E2: Wie stark hängt das Ergebnis vom Split ab? (Random Forest) ------------------------
    splits = {
        "Zufällige Fenster (falsch)": (info_g, merkmale_g, zufalls_folds(info_g)),
        "Block ohne Gap": (info_g, merkmale_g, block_folds(info_g, gap_s=0)),
        "Block mit Gap": (info_g, merkmale_g, block_folds(info_g)),
        CROSS: (info, merkmale, [cross_activity_split(info)]),
    }
    zeilen = []
    for name, (inf, mm, folds) in splits.items():
        abstand = min(kleinster_abstand(inf, train, test) for train, test in folds)
        ergebnis, _ = bewerte("Random Forest", mm, inf, folds)
        zeilen.append({"split": name, "kleinster_abstand_s": abstand, **zusammenfassen(ergebnis)})
        print(f"E2  {name:28s} Abstand {abstand:5.1f} s   F1 Fenster {zeilen[-1]['f1_fenster_mittel']:.3f}"
              f"   F1 Clip {zeilen[-1]['f1_clip_mittel']:.3f}")
    e2 = pd.DataFrame(zeilen)
    e2.round(3).to_csv(ziel / "e2_splits.csv", index=False)
    balkendiagramm(e2, "split", "E2: Gleiches Modell, verschiedene Splits (Random Forest)",
                   ziel / "e2_splits.png", zufall=zufall)

    # --- E3: Person oder Sensorlage? (Random Forest) -------------------------------------------
    zeilen = []
    for satz in ["alle", "nur_mittelwerte", "nur_betraege"]:
        mm = berechne_merkmale(X, kanaele, satz)
        mm_g = mm[gehen].reset_index(drop=True)
        for split_name, inf, m, folds in [("Block mit Gap", info_g, mm_g, block_folds(info_g)),
                                          (CROSS, info, mm, [cross_activity_split(info)])]:
            ergebnis, _ = bewerte("Random Forest", m, inf, folds)
            zeilen.append({"merkmale": satz, "split": split_name, "anzahl": mm.shape[1],
                           **zusammenfassen(ergebnis)})
            print(f"E3  {satz:16s} {split_name:28s} F1 Clip {zeilen[-1]['f1_clip_mittel']:.3f}")
    e3 = pd.DataFrame(zeilen)
    e3.round(3).to_csv(ziel / "e3_merkmale.csv", index=False)
    balkendiagramm(e3, "merkmale", "E3: Person oder Sensorlage? (Random Forest)", ziel / "e3_merkmale.png",
                   gruppen="split", zufall=zufall)

    # --- E4: Welche Sensorposition? (Random Forest, Gehen, Block-Split mit Gap) -----------------
    sensoren = sorted({kanal.split("_")[0] for kanal in kanaele})
    zeilen = []
    for auswahl in [[sensor] for sensor in sensoren] + [sensoren]:
        spalten = [j for j, kanal in enumerate(kanaele) if kanal.split("_")[0] in auswahl]
        mm = berechne_merkmale(X[gehen][:, :, spalten], [kanaele[j] for j in spalten], "alle")
        ergebnis, _ = bewerte("Random Forest", mm, info_g, block_folds(info_g))
        name = auswahl[0] if len(auswahl) == 1 else "alle zusammen"
        zeilen.append({"sensoren": name, **zusammenfassen(ergebnis)})
        print(f"E4  {name:20s} F1 Clip {zeilen[-1]['f1_clip_mittel']:.3f}")
    e4 = pd.DataFrame(zeilen)
    e4.round(3).to_csv(ziel / "e4_sensoren.csv", index=False)
    balkendiagramm(e4, "sensoren", "E4: Sensorposition (Random Forest, Gehen)", ziel / "e4_sensoren.png",
                   zufall=zufall)
    print(f"Fertig. Ergebnisse in {ziel}/")


if __name__ == "__main__":
    main()
