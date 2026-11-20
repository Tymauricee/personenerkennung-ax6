# Ergebnisbericht Personenerkennung

Lauf: `demo_synth` · Hauptsession: S1 · Fenster 3 s, Schritt 1.5 s, Gap 3 s

## E0 Inventar & QC
- 25 Personen, 513 Recordings (Clips), 1.43 h, Aktivitäten: Ball, Gehen, Haendewaschen, Hampelmann, Squats, Treppe, Tuergriff
- 4 von 732 Dateien mit Warnflag -> `E0_inventar/qc_flagged.csv`
- Geräte-Check: 100% der Person-Sessions haben eine EINZIGARTIGE Gerätekombination; Geräte pro Position: {'S01': 7, 'S02': 7, 'S05': 7, 'S06': 7}. -> RISIKO: Geräte-Fingerabdruck kann Personen verraten. Gegenmaßnahmen: feste Gerät-Position-Zuordnung für alle Personen, Bias-Korrektur, rotations-/offsetrobuste Features (Set 'invariant'), Static-Probe berichten.
- Leakage-Selbsttest Hauptprotokoll: OK; Random-Fenster-Split: 704 überlappende Train/Test-Fensterpaare

## E1 Modellvergleich

| Modell | Clip-Acc | 95 %-KI | Clip-Macro-F1 | Fenster-Macro-F1 (±SD über Folds) | Top-3 | Laufzeit [s] |
|---|---|---|---|---|---|---|
| Mehrheit (Baseline) | 4.0% | 0.0%–12.0% | 0.003 | 0.003 ± 0.000 | 12.0% | 1 |
| Zufall (Baseline) | 8.0% | 0.0%–20.0% | 0.038 | 0.048 ± 0.000 | 20.0% | 0 |
| Decision Tree | 89.6% | 84.0%–94.4% | 0.894 | 0.842 ± 0.060 | 97.6% | 1 |
| KNN | 99.2% | 97.6%–100.0% | 0.992 | 0.965 ± 0.020 | 100.0% | 0 |
| SVM (RBF) | 100.0% | 100.0%–100.0% | 1.000 | 0.980 ± 0.009 | 100.0% | 1 |
| Random Forest | 100.0% | 100.0%–100.0% | 1.000 | 1.000 ± 0.000 | 100.0% | 10 |
| MiniRocket | 100.0% | 100.0%–100.0% | 1.000 | 0.998 ± 0.004 | 100.0% | 4 |
| 1D-CNN | 100.0% | 100.0%–100.0% | 1.000 | 0.998 ± 0.004 | 100.0% | 80 |

Zufallsniveau: 4.0% · Protokoll: block (5 Folds) · Bestes Modell: rf

## E5 Validitätsleiter (Gehen, RF)

| Protokoll | Clip-Acc | 95 %-KI | überlappende Fensterpaare | min. Abstand [s] |
|---|---|---|---|---|
| Random-Fenster (unzulässig) | 100.0% | 100.0%–100.0% | 842 | -1.5 |
| Random-Clips (naiv) | 100.0% | 100.0%–100.0% | 0 | 1.0 |
| Block-Split ohne Gap | 100.0% | 100.0%–100.0% | 0 | 1.0 |
| Block-Split mit Gap | 100.0% | 100.0%–100.0% | 0 | 4.0 |
| Cross-Activity | 97.6% | 95.2%–100.0% | 0 | nan |
| Cross-Session (Tag 2) | 15.0% | 0.0%–40.0% | 0 | nan |

## E7 Permutationstest (Trial-Ebene)

- Block-Split mit Gap: beobachtet 100.0%, permutiert Ø 100.0% (95. Perzentil 100.0%), p = 1.000
- Cross-Activity: beobachtet 98.4%, permutiert Ø 4.6% (95. Perzentil 14.1%), p = 0.020
- Cross-Session (Tag 2): beobachtet 20.0%, permutiert Ø 1.7% (95. Perzentil 12.5%), p = 0.039

## E2 Sensorposition (Clip-Accuracy, RF)

| Sensoren | Ball | Gehen | Haendewaschen | Hampelmann | Squats | Treppe | Tuergriff |
|---|---|---|---|---|---|---|---|
| S01 (re. Hand) | 100% | 98% | 98% | 92% | 100% | 99% | 96% |
| S02 (li. Hand) | 100% | 99% | 96% | 96% | 98% | 93% | 98% |
| S05 (Rumpf) | 99% | 96% | 100% | 98% | 94% | 95% | 98% |
| S06 (Bein) | 99% | 94% | 94% | 90% | 92% | 92% | 92% |
| Hände (S01+S02) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Rumpf+Bein (S05+S06) | 100% | 99% | 98% | 98% | 100% | 100% | 100% |
| Alle 4 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

## E3 Aktivitäten (Clip-Accuracy, RF)

| Aktivität | Block-Split (gleiche Aufnahme) | Leave-one-activity-out | LOAO invariant |
|---|---|---|---|
| Ball | 100% | 93% | 47% |
| Gehen | 100% | 98% | 43% |
| Haendewaschen | 100% | 34% | 58% |
| Hampelmann | 100% | 28% | 4% |
| Squats | 100% | 46% | 46% |
| Treppe | 100% | 84% | 33% |
| Tuergriff | 100% | 84% | 66% |

## E4 Fensterlänge (RF)

| window_s | win_f1_mean | pooled_unit_acc |
|---|---|---|
| 1.0 | 1.0 | 1.0 |
| 2.0 | 1.0 | 1.0 |
| 3.0 | 1.0 | 1.0 |
| 4.0 | 0.996 | 0.992 |
| 6.0 | 0.951 | 0.968 |

- Block-Split mit Gap: ≥ 90 % Accuracy ab 3.0 s Bewegung
- Cross-Activity: ≥ 90 % Accuracy ab 3.0 s Bewegung
- Cross-Session (Tag 2): 90 % nicht erreicht (max 20%)

## E6 Shortcut-Diagnostik (Clip-Accuracy, RF)

| Feature-Set | Block-Split mit Gap | Cross-Activity | Cross-Session (Tag 2) |
|---|---|---|---|
| Basis (Vorgabe) | 100% | 99% | 0% |
| Erweitert | 100% | 98% | 15% |
| Rotations-invariant | 96% | 41% | 60% |
| Static-Probe | 100% | 100% | 5% |

## E8 Wichtigste Merkmalsgruppen (Gehen, RF)

- S05 · Lage/Ausrichtung (Achsenwerte): Log-Loss-Anstieg 0.425
- S01 · Lage/Ausrichtung (Achsenwerte): Log-Loss-Anstieg 0.393
- S02 · Lage/Ausrichtung (Achsenwerte): Log-Loss-Anstieg 0.369
- S06 · Lage/Ausrichtung (Achsenwerte): Log-Loss-Anstieg 0.196
- S05 · Statistik: Log-Loss-Anstieg 0.126
- S02 · Spektrum: Log-Loss-Anstieg 0.102

---
Hinweis: Alle Zahlen gelten nur für die verwendeten Daten. Bei synthetischen Daten sind sie KEINE Ergebnisse, sondern ein Funktionstest der Pipeline.