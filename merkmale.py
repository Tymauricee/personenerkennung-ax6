"""Merkmale pro Fenster (Pipeline Abschnitt 5): Mittelwert, Standardabweichung, Minimum,
Maximum, RMS und Energie, für jede Achse und für den Betrag jedes Sensors."""
import numpy as np
import pandas as pd

MERKMALSAETZE = ["alle", "nur_mittelwerte", "nur_betraege"]


def statistiken(x):
    """x: Array (Anzahl Fenster, Samples). Gibt die sechs Kennzahlen pro Fenster zurück."""
    return {
        "mean": x.mean(axis=1),
        "std": x.std(axis=1),
        "min": x.min(axis=1),
        "max": x.max(axis=1),
        "rms": np.sqrt((x ** 2).mean(axis=1)),   # Effektivwert
        "energie": (x ** 2).mean(axis=1),        # mittlere Energie = rms² (Definition wie UCI-HAR)
    }


def berechne_merkmale(X, kanaele, satz="alle"):
    """Berechnet eine Merkmalstabelle mit einer Zeile pro Fenster.

    X:       Array (Fenster, Samples, Kanäle)
    kanaele: Kanalnamen wie "ax6-hip_acc_x", in derselben Reihenfolge wie in X
    satz:    "alle"            Kennzahlen jeder Achse und des Betrags
             "nur_mittelwerte" nur der Mittelwert jeder Achse (Lage-Test, siehe README)
             "nur_betraege"    nur Kennzahlen des Betrags, unabhängig davon, wie der Sensor gedreht ist
    """
    if satz not in MERKMALSAETZE:
        raise ValueError(f"Unbekannter Merkmalssatz: {satz}")
    merkmale = {}
    if satz in ("alle", "nur_mittelwerte"):
        for j, kanal in enumerate(kanaele):
            werte = statistiken(X[:, :, j])
            if satz == "nur_mittelwerte":
                werte = {"mean": werte["mean"]}
            for name, spalte in werte.items():
                merkmale[f"{kanal}_{name}"] = spalte
    if satz in ("alle", "nur_betraege"):
        for gruppe in sorted({kanal[:-2] for kanal in kanaele}):       # "ax6-hip_acc_x" -> "ax6-hip_acc"
            achsen = [j for j, kanal in enumerate(kanaele) if kanal[:-2] == gruppe]
            betrag = np.linalg.norm(X[:, :, achsen], axis=2)          # Wurzel aus x² + y² + z²
            for name, spalte in statistiken(betrag).items():
                merkmale[f"{gruppe}_betrag_{name}"] = spalte
    return pd.DataFrame(merkmale)
