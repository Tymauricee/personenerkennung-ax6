# Personenerkennung aus Wearable-Bewegungsdaten (AX6, 4 Sensoren)

**PF 1.5 Digital Health – Data Science & AI Project · HSD**

Aufgabe: Ein Modell bekommt einen **Clip** einer Aktivität (z. B. Gehen, 4 IMUs: S01 rechte Hand,
S02 linke Hand, S05 Rumpf, S06 Bein) und sagt, **welche der N bekannten Personen** ihn aufgenommen
hat (Closed-Set-Identifikation, pseudonymisierte IDs).

Die Pipeline ist auf eure Datenstruktur zugeschnitten: **≈25 Personen, alles an einem Tag,
1 Trial pro Aktivität, der in mehrere Clips zerlegt ist**. Genau diese Struktur macht die
Auswertung heikel. Deshalb steht hier nicht die höchste Accuracy im Mittelpunkt, sondern eine
**ehrliche Evaluation**.

---

## Kernidee: die Validitätsleiter

| Stufe | Protokoll | Was es beweist |
|---|---|---|
| ✗ | Random-Fenster / Random-Clips | nichts: überlappende oder benachbarte Rohdaten liegen in Train **und** Test (Leakage) |
| 1 | **Block-Split mit Gap** innerhalb des Trials | Die *Aufnahme* wird wiedererkannt. Person und Aufnahme sind bei 1 Trial pro Person nicht trennbar. |
| 2 | **Cross-Activity** (Train: andere Aktivitäten → Test: Gehen) | Die Signatur überträgt sich auf eine *neue Aufnahme* desselben Tages. Sensorsitz und Tagesform können aber noch helfen. |
| 3 | **Cross-Session** (Train Tag 1 → Test Tag 2) | echte Wiedererkennung der *Person*. Dafür ist ein kleines Retest-Subset nötig. |

Dazu kommen **Shortcut-Kontrollen**:
- *Static-Probe*: Ein Modell nur auf Achsen-Mittelwerten erfasst Sensorsitz, Haltung und Gerät.
- *Rotationsinvariante Features*: Diese hängen nicht davon ab, wie der Sensor gedreht sitzt.
- *Permutationstest auf Trial-Ebene*: Er zeigt, ob ein Protokoll Personen oder nur Aufnahmen misst.

## Schnellstart

```bash
pip install -r requirements.txt          # Keras braucht EIN Backend: torch, tensorflow oder jax
python -m pytest -q tests                # 12 Tests, u. a. Leakage-Freiheit aller Splits

# Demo mit synthetischen Daten (gleiche Struktur wie euer Datensatz, KEINE echten Ergebnisse!)
python scripts/make_synthetic_data.py --out data_synth
python scripts/run_experiments.py --config configs/synth.yaml            # alles, ca. 20 min auf 2 CPU-Kernen
python scripts/run_experiments.py --config configs/synth.yaml --quick    # schneller Testlauf
```

Ergebnisse landen unter `results/<lauf>/`: pro Experiment CSV und PNG, dazu `REPORT.md` (Zusammenfassung),
`summary.json`, `config_used.yaml` und `environment.json`.

## Eigene Daten anbinden (3 Schritte)

**1. Manifest erstellen.** Eine CSV mit einer Zeile pro (Clip bzw. Trial) × Sensor:

```
path,person,session,activity,trial,sensor,clip,start_s,end_s,device_id
data/P01/Gehen/S01.csv,P01,S1,Gehen,T1,S01,C01,0.0,10.0,AX6-03
data/P01/Gehen/S01.csv,P01,S1,Gehen,T1,S01,C02,10.0,20.0,AX6-03
...
```

- `clip`, `start_s` und `end_s` sind optional. Sind eure Clips **eigene Dateien**, lasst `start_s`/`end_s` weg.
  Schneidet ihr Clips aus einer längeren Datei, tragt die Grenzen in deren Zeitbasis ein.
- **`device_id` unbedingt ausfüllen.** Daraus prüft E0, ob Geräte mit Personen konfundiert sind (siehe unten).
- Bei einem Namensschema erzeugt `scripts/build_manifest.py` das Manifest automatisch (Regex-Beispiele im Skript).

**2. Config anpassen:** `configs/ax6_beta.yaml` kopieren und alle `TODO` prüfen
(Zeiteinheit, Einheiten g vs. m/s² und °/s vs. rad/s, Spaltennamen, Messbereich, Aktivitätsname).
Spalten werden automatisch erkannt (`time`, `acc_x` … `gyr_z`, auch `Accel-X (g)`, `gx` u. ä.).
Falls nicht, hilft `data.column_map`.

**3. Zuerst nur Inventar und QC laufen lassen**, dann alles:

```bash
python scripts/run_experiments.py --config configs/ax6_beta.yaml --only E0
# -> results/<lauf>/E0_inventar/qc_flagged.csv, coverage.png, device_check.csv, load_log.csv ansehen!
python scripts/run_experiments.py --config configs/ax6_beta.yaml
```

> **Clip-Zeitstempel:** Haben Clips absolute Zeitstempel, ordnet die Pipeline sie über die Zeit
> an. Auch überlappende Clips werden dabei korrekt behandelt. Beginnen alle Clips eines Trials bei ~0
> (relative Zeit), wird die Reihenfolge aus der Clip-ID abgeleitet (Warnung in `load_log.csv`), und
> Nachbarclips des Testblocks werden zusätzlich ausgeschlossen.

## Pipeline

```
Manifest ─► Dateien lesen + QC ─► Einheiten (g, °/s) ─► Sync aller Sensoren (Anti-Aliasing, 50 Hz, Lücken = NaN)
        ─► SPLIT AUF ROHDATEN-EBENE (Block+Gap / Cross-Activity / Cross-Session)
        ─► erst JETZT Fenster (3 s, 50 % Überlappung, nur innerhalb der Split-Teile)
        ─► Features (basic | extended | invariant | static)  oder  Rohfenster (MiniRocket, 1D-CNN)
        ─► Modell (fit nur auf Train; Skalierung im Modell)  ─► Fenster-Wahrscheinlichkeiten
        ─► Aggregation pro Clip (Mittel der Log-Wahrscheinlichkeiten)  ─► Metriken + KIs + Leakage-Check
```

**Leakage-Regeln** (aus eurer Projektpipeline, Abschnitte 7–8), alle im Code umgesetzt:

1. Erst die Rohdaten zeitlich in Train und Test trennen, **dann** Fenster bilden (`splits.py`, `windows.py`).
2. Gap ≥ 1 Fensterlänge zwischen Train- und Testdaten (`windows.gap_s`, wird automatisch auf ≥ Fensterlänge gesetzt).
3. Skalierung und lernende Transformationen nur auf Trainingsdaten (sklearn-Pipelines, `ChannelScaler`).
4. **Automatischer Leakage-Check** für jede Fold (`check_leakage`): Er zählt überlappende Train/Test-Fensterpaare und
   misst den minimalen zeitlichen Abstand. Das Ergebnis steht in jeder Ergebnistabelle (`leakage_ok`) und in `E0_inventar/leakage_check_*.csv`.
5. Anzahl Fenster ≠ Anzahl unabhängiger Messungen: Bootstrap-KIs ziehen deshalb **Personen**, nicht Fenster.

## Experimente

| | Frage | Ausgabe (Poster-Kandidaten fett) |
|---|---|---|
| E0 | Was ist im Datensatz? Gibt es Lücken, Einheitenfehler, Geräte-Confounds? | `coverage.png`, `qc_flagged.csv`, `device_check.csv`, **`split_schema.png`** |
| E1 | Welches Modell erkennt Personen am besten (gleiche Folds)? | **`E1_clip_accuracy.png`**, `E1_window_macroF1.png`, Konfusionsmatrix |
| E2 | Welche Sensorposition trägt die meiste Identität, je Aktivität? | **`E2_sensor_x_activity.png`** |
| E3 | Welche Aktivität ist am personenspezifischsten? Überträgt sich die Identität (Train A → Test B)? | **`E3_per_activity.png`**, `E3_transfer.png` |
| E4 | Wie viele Sekunden braucht man (je Protokoll)? Welche Fensterlänge? | **`E4_evidence.png`**, `E4_window_length.png` |
| E5 | Wie stark überschätzen naive Splits die Leistung? | **`E5_validity_ladder.png`** |
| E6 | Lernt das Modell die Person oder den Sensorsitz? | **`E6_shortcut_probe.png`**, `E6_tsne_invariant.png` |
| E7 | Ist das Ergebnis mehr als Aufnahme-Wiedererkennung? | **`E7_permutation.png`** |
| E8 | Welche Sensor- und Merkmalsgruppen tragen die Identität? | `E8_group_importance.png`, `E8_top20_mdi.png` |

**Modelle (E1):** Mehrheits- und Zufalls-Baseline, Decision Tree, KNN, SVM (RBF), Random Forest
(Features), **MiniRocket** (Dempster et al., KDD 2021; aeon) und ein **1D-CNN** (Keras 3, backend-agnostisch)
mit Rotations-Augmentation. Alle Modelle laufen auf denselben Folds. Die Hyperparameter sind a priori fest
(kein Tuning auf Testdaten). Wer tunen will, braucht eine innere, ebenfalls gruppierte CV.

**Metriken:** Accuracy und Macro-F1 auf Fenster- und Clip-Ebene, Balanced Accuracy und Top-3.
Berichtet werden Mittelwert ± SD über Folds (Projektpipeline) und gepoolte Werte mit 95 %-Cluster-Bootstrap-KI.

## Wie man die Ergebnisse liest

- **Block-Split ≈ 100 %** heißt noch nicht „Personenerkennung funktioniert“. Es heißt nur, dass Clips derselben Aufnahme
  wiedererkannt werden. Bei 1 Trial pro Person ist das die Obergrenze, nicht der Beweis.
- **Static-Probe hoch** (vor allem bei Cross-Activity): Das Modell kann den *Sensorsitz* bzw. das *Gerät* nutzen.
  Rotationsinvariante Features sind dann die ehrlichere Variante.
- **Permutationstest:** Die Personen-Labels werden zwischen Aufnahmen vertauscht. Bleibt die Accuracy hoch (p ≈ 1),
  misst das Protokoll nur Aufnahme-Identität. Fällt sie auf Zufallsniveau, steckt personenbezogene Information drin.
- **Geräte-Check (E0):** Trägt jede Person eine *eigene* Kombination von AX6-Geräten (7 Geräte × 4 Positionen), können
  geräteeigene Offsets und Biases die Person verraten. Am besten nutzt jede Position immer dasselbe Gerät.

## Reproduzierbarkeit

Feste Seeds, `config_used.yaml` und `environment.json` pro Lauf, ein Cache für geladene Daten (`cache/`) und Tests
für die Split-Logik. Hinweis: Einige Abschnitte nutzen parallele Bäume (`n_jobs`). Für bitgenaue Wiederholung `n_jobs: 1` setzen.

## Projektstruktur

```
personid/      io.py · synth.py · inventory.py · windows.py · splits.py · features.py · models.py
               evaluate.py · plots.py · experiments.py · config.py
configs/       synth.yaml (Demo) · ax6_beta.yaml (Vorlage für eure Daten)
scripts/       make_synthetic_data.py · build_manifest.py · run_experiments.py
tests/         test_core.py
```

## Grenzen (ins Poster schreiben)

- 1 Session pro Person: Ohne Retest lässt sich nur Wiedererkennung innerhalb eines Tages zeigen (Session-Risiko).
- Etwa 25 Personen, kurze Trials: KIs berichten, keine Überinterpretation kleiner Unterschiede zwischen Modellen.
- Closed-Set: Das Modell kennt nur die Trainingspersonen. Es ist kein Identitäts- oder Sicherheitssystem.
- Synthetische Demo-Daten testen nur den Code. Ihre Zahlen sind **keine** Befunde.

## Datenschutz

Gangdaten können als biometrische Daten gelten (DSGVO Art. 4 Nr. 14; bei Verarbeitung zur eindeutigen Identifizierung
besondere Kategorie nach Art. 9). Deshalb gilt: nur pseudonymisierte IDs, keine Klarnamen in Code, Logs oder Grafiken,
Personenerkennung nur im Rahmen von Projektfreigabe und Einwilligung, und keine trainierten Identifikationsmodelle
weitergeben.
