"""Erzeugt künstliche Sensordaten im Format des Kurs-Datensatzes.

Damit läuft der Code, bevor die echten Daten da sind, und die Tests haben etwas zum Prüfen.
Die Ergebnisse auf diesen Daten sagen NICHTS darüber, wie gut Personenerkennung wirklich klappt.

    python beispieldaten.py                 15 Personen nach config.DEMO_ORDNER
    python beispieldaten.py --personen 25

Jede künstliche Person hat ein eigenes Schritttempo, eine eigene Bewegungsform und eine eigene
Lage jedes Sensors (wie er am Körper sitzt). Eingebaut sind zwei typische Datenprobleme:
eine Lücke (P02, walking, ankle-right) und eine Datei in m/s² statt g (P03, squats, wrist-left).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import config

AKTIVITAETEN = {  # Dauer in s, Tempo relativ zum Gehen, Stärke der Bewegung
    "walking": (60, 1.0, 1.0),
    "stairs": (40, 0.85, 1.2),
    "jumping-jacks": (40, 0.75, 1.8),
    "squats": (40, 0.3, 0.8),
}
SENSOREN = ["wrist-right", "wrist-left", "hip", "ankle-right"]
FS_GERAET = 100           # die Geräte messen schneller als unsere Ziel-Rate von 50 Hz
START = 1794906000        # 17.11.2026, 9:00 Uhr (Unix-Zeit in Sekunden)


def langsame_schwankung(t, staerke, rng):
    """Faktor um 1, der sich über 20 bis 40 Sekunden langsam hebt und senkt (z. B. Ermüdung)."""
    periode = rng.uniform(20, 40)
    return 1 + staerke * np.sin(2 * np.pi * t / periode + rng.uniform(0, 2 * np.pi))


def erzeuge(ordner, n_personen=15, aktivitaeten=None, seed=config.SEED):
    ordner = Path(ordner)
    ordner.mkdir(parents=True, exist_ok=True)
    for alt in ordner.glob("*.csv"):
        alt.unlink()
    aktivitaeten = aktivitaeten or list(AKTIVITAETEN)
    rng = np.random.default_rng(seed)
    # Körperhaltung je Aktivität und Sensor: für alle Personen gleich
    haltung = {(a, s): Rotation.from_rotvec(np.random.default_rng([i, j]).normal(0, 0.4, 3))
               for i, a in enumerate(AKTIVITAETEN) for j, s in enumerate(SENSOREN)}

    for nr in range(1, n_personen + 1):
        person = f"P{nr:02d}"
        tempo = rng.uniform(1.75, 1.95)                    # Schritte pro Sekunde
        form = rng.uniform(0.4, 1.0, size=3)               # Anteil der 1., 2. und 3. Oberwelle
        kraft = rng.uniform(0.85, 1.15)                    # wie kräftig sich die Person bewegt
        lage = {s: Rotation.from_rotvec(rng.normal(0, 0.1, 3)) for s in SENSOREN}
        gyro_offset = {s: rng.normal(0, 1.0, 3) for s in SENSOREN}

        for a_nr, aktivitaet in enumerate(aktivitaeten):
            dauer, tempo_faktor, staerke = AKTIVITAETEN[aktivitaet]
            # Rhythmus und Kraft schwanken langsam innerhalb des Trials (alle Sensoren gemeinsam).
            # Deshalb ähneln sich zeitlich benachbarte Fenster mehr als weit entfernte.
            t = np.arange(dauer * FS_GERAET + 50) / FS_GERAET
            tempo_t = tempo * tempo_faktor * langsame_schwankung(t, 0.04, rng)
            phase = 2 * np.pi * np.cumsum(tempo_t) / FS_GERAET + np.cumsum(rng.normal(0, 0.03, len(t)))
            intensitaet = kraft * staerke * langsame_schwankung(t, 0.15, rng)
            for k, sensor in enumerate(SENSOREN):
                n = dauer * FS_GERAET + int(rng.integers(0, 50))
                ph, it = phase[:n], intensitaet[:n, None]
                bewegung = sum(form[h] * np.sin((h + 1) * ph + k) for h in range(3))
                acc = np.column_stack([0.35 * bewegung, 0.25 * np.cos(ph + k), 0.6 * bewegung]) * it
                gyr = np.column_stack([60 * np.cos(ph + k), 40 * np.sin(2 * ph + k), 25 * bewegung]) * it
                acc[:, 2] += 1.0                           # Schwerkraft: 1 g nach oben
                drehung = lage[sensor] * haltung[(aktivitaet, sensor)]
                acc = drehung.apply(acc) + rng.normal(0, 0.08, (n, 3))
                gyr = drehung.apply(gyr) + gyro_offset[sensor] + rng.normal(0, 5, (n, 3))

                zeit = START + nr * 600 + a_nr * 60 + rng.uniform(0, 0.4) + t[:n]
                zeit += rng.normal(0, 0.0005, n)           # kleine Unregelmäßigkeit der Uhr
                tabelle = pd.DataFrame({"time": zeit, "acc_x": acc[:, 0], "acc_y": acc[:, 1],
                                        "acc_z": acc[:, 2], "gyr_x": gyr[:, 0], "gyr_y": gyr[:, 1],
                                        "gyr_z": gyr[:, 2]})
                if (person, aktivitaet, sensor) == ("P02", "walking", "ankle-right"):
                    tabelle = tabelle[(t[:n] < 20) | (t[:n] > 21)]                  # 1 s Datenlücke
                if (person, aktivitaet, sensor) == ("P03", "squats", "wrist-left"):
                    tabelle[["acc_x", "acc_y", "acc_z"]] *= 9.81            # falsche Einheit
                name = f"20261117_{person}_HAR-{aktivitaet}_T01_ax6-{sensor}_imu_proc.csv"
                tabelle.to_csv(ordner / name, index=False, float_format="%.4f")
    print(f"{n_personen} Personen x {len(aktivitaeten)} Aktivitäten x {len(SENSOREN)} Sensoren -> {ordner}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--personen", type=int, default=15)
    parser.add_argument("--ordner", default=config.DEMO_ORDNER)
    args = parser.parse_args()
    erzeuge(args.ordner, args.personen)
