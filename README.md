# Personenerkennung aus AX6-Bewegungsdaten

**PF 1.5 Digital Health: Data Science & AI Project · Hochschule Düsseldorf**
Team: _Name A_, _Name B_ (siehe [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md))

**Forschungsfrage.** Lässt sich aus einem Gehen-Clip (vier Sensoren: beide Handgelenke, Hüfte, Bein)
erkennen, welche der rund 25 bekannten Personen geht? Und wie viel davon ist wirklich die Person,
wie viel nur die Wiedererkennung derselben Aufnahme oder desselben Sensorsitzes?

Das ist die explorative Personenerkennung aus Abschnitt 7 der Projektpipeline des Kurses: eine
Machbarkeitsstudie mit pseudonymisierten IDs, kein Identitäts- oder Sicherheitssystem.

## Vorgehen

Der Code folgt Schritt für Schritt der Projektpipeline des Kurses.

| Schritt | Pipeline-Abschnitt | Datei |
|---|---|---|
| Dateien nach Namenskonvention finden, Qualitätsbericht | 3 | `laden.py` |
| Sensoren synchronisieren (50 Hz, Tiefpass, Lücken bleiben Lücken) | 4 | `laden.py` |
| Trial in 5 Clips, Clips in Fenster (3 s, 50 % Überlappung) | 8 | `fenster.py` |
| Block-Split mit 3 s Gap, Cross-Activity-Split, Leakage-Prüfung | 7, 8 | `fenster.py` |
| Merkmale: mean, std, min, max, rms, energie pro Achse und Betrag | 5 | `merkmale.py` |
| Modelle, Clip-Entscheidung, Macro-F1 (Mittelwert ± SD über Folds) | 5, 7 | `modelle.py` |
| Experimente E1 bis E4, Tabellen und Grafiken | 9 | `experimente.py` |

Alle Einstellungen (Fensterlänge, Gap, Anzahl Clips …) stehen in [`config.py`](config.py).

## Schnellstart

```bash
git clone https://github.com/Tymauricee/personenerkennung-ax6
cd personenerkennung-ax6
pip install -r requirements.txt

python beispieldaten.py          # künstliche Testdaten im Format des Kurs-Datensatzes
python experimente.py --demo     # alle Experimente auf den Testdaten, etwa 1 Minute
pytest                           # automatische Prüfungen, u. a. kein Data Leakage
jupyter lab lernen.ipynb         # der ganze Weg Schritt für Schritt, mit Kontrollfragen
```

**Echte Daten:** proc-Dateien aus dem Kurs-Datensatz nach `data/processed/` legen (der Ordner wird nie
committet) und `python experimente.py` starten. Ergebnisse landen in `ergebnisse/echt/` und bleiben lokal.
Vorher prüfen: Trialdauer (reicht sie für 5 Clips mit Fenstern?), Spaltennamen, Einheiten. Erst
`qualitaet.csv` ansehen, dann die Ergebnisse.

## Experimente

| | Frage | Ergebnis |
|---|---|---|
| E1 | Welches Modell erkennt Personen am besten? Mehrheits-Baseline, Entscheidungsbaum, KNN, SVM, Random Forest; nur Gehen und alle Aktivitäten | `e1_modelle.png`, `e1_confusion_rf.png`, `e1_pro_person_rf.csv` |
| E2 | Wie stark hängt das Ergebnis vom Split ab? Zufällige Fenster (falsch), Block ohne Gap, Block mit Gap, andere Aktivitäten → Gehen | `e2_splits.png` |
| E3 | Person oder Sensorsitz? Alle Merkmale, nur Achsen-Mittelwerte, nur Beträge | `e3_merkmale.png` |
| E4 | Welche Sensorposition verrät die Person am besten? | `e4_sensoren.png` |

Ein Beispiellauf auf den künstlichen Daten liegt in [`ergebnisse/demo/`](ergebnisse/demo/).
**Diese Zahlen sagen nichts über echte Personenerkennung**, sie zeigen nur, dass der Code läuft.

## Wichtigste Einschränkung: das Session-Risiko

Pro Person gibt es eine Aufnahme pro Aktivität, alle am selben Tag. Training und Test stammen deshalb
aus derselben Aufnahme. Ein Modell kann Personen dann auch am Sensorsitz, am Gerät oder an der Tagesform
unterscheiden statt am Gang (Pipeline, Abschnitt 7). Zwei Kontrollen machen das sichtbar:

- **Cross-Activity (E2, E3):** Training auf anderen Aktivitäten, Test auf Gehen. Das ist eine andere
  Aufnahme, aber derselbe Tag und derselbe Sensorsitz.
- **Lage-Test (E3):** Die Mittelwerte der Achsen hängen vor allem davon ab, wie der Sensor gedreht sitzt.
  Erkennt ein Modell die Personen schon damit, ist der Sensorsitz eine Abkürzung. Die Beträge
  (|a| = Wurzel aus x² + y² + z²) hängen dagegen nicht von der Drehung ab.

Echte Personenerkennung über Tage hinweg ließe sich erst mit einer zweiten Messung an einem anderen Tag prüfen.

## Datenschutz

> **Dieses Repository ist öffentlich.** Niemals Messdaten (auch keine proc-Dateien), das zentrale Log
> oder trainierte Modelle committen. `data/`, `ergebnisse/echt/` und Modelldateien stehen deshalb in
> der `.gitignore`. Das Notebook nach einem Lauf mit echten Daten nur ohne Ausgaben committen.

Gangdaten können biometrische Daten sein (DSGVO Art. 4 Nr. 14, bei Verarbeitung zur eindeutigen
Identifizierung Art. 9). Deshalb: nur pseudonymisierte IDs, Personenerkennung nur im Rahmen von
Projektfreigabe und Einwilligung (Pipeline, Abschnitt 10).

## Projektstruktur

```
config.py            alle Einstellungen
laden.py             Dateien finden, lesen, Qualitätsbericht, synchronisieren
fenster.py           Clips, Fenster, Block-Split, Cross-Activity-Split, Leakage-Prüfung
merkmale.py          Merkmale pro Fenster
modelle.py           Modelle, Clip-Entscheidung, Bewertung
experimente.py       E1 bis E4
beispieldaten.py     künstliche Testdaten
lernen.ipynb         Lernpfad mit Kontrollfragen
tests/               automatische Prüfungen (laufen bei jedem Push auf GitHub)
docs/FACHGESPRAECH.md   typische Fragen im Fachgespräch mit Kernpunkten und Code-Stellen
ergebnisse/demo/     Beispiellauf auf künstlichen Daten
```

## Entstehung und KI-Nutzung

Der Code wurde mit Unterstützung des KI-Assistenten Claude (Anthropic) erstellt. Commits mit
KI-Beteiligung sind in der Git-Historie mit „Co-Authored-By: Claude“ gekennzeichnet. Eine erste,
deutlich umfangreichere Fassung liegt im Branch `archiv/framework-v1`. Die aktuelle Fassung ist auf
die Methoden der Projektpipeline reduziert.

_TODO Team: nach Rücksprache mit der Prüferin bzw. dem Prüfer ergänzen, in der dort gewünschten Form:
wofür ihr KI genutzt habt und was ihr selbst gemacht, geprüft und entschieden habt._
