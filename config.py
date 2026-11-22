"""Alle Einstellungen an einer Stelle.

Wer einen Wert ändern will, ändert ihn nur hier. Die Werte sind Startpunkte aus der
Projektpipeline des Kurses (Abschnitte 4, 7 und 8). Endgültig festgelegt werden sie erst,
wenn wir die echten Daten gesehen haben (Dauer der Trials, Abtastrate).
"""

# --- Daten ----------------------------------------------------------------------------
DATEN_ORDNER = "data/processed"   # proc-Dateien aus dem Kurs-Datensatz (werden nie committet)
DEMO_ORDNER = "data_demo"         # hierhin schreibt beispieldaten.py künstliche Daten
STUFE = "proc"                    # "proc" = Standardformat, "raw" = unveränderter Export
SENSOREN = None                   # None = alle Sensoren, die in jedem Trial vorkommen,
                                  # sonst z. B. ["ax6-wrist-right", "ax6-hip"]

# --- Vorverarbeitung (Pipeline Abschnitt 4) -------------------------------------------
FS = 50              # gemeinsame Abtastrate in Hz
MAX_LUECKE_S = 0.1   # größere Lücken im Zeitstempel werden nicht überbrückt (werden NaN)
RAND_S = 1.0         # so viele Sekunden am Anfang und Ende jedes Trials abschneiden

# --- Clips, Fenster, Split (Pipeline Abschnitte 7 und 8) ------------------------------
N_CLIPS = 5          # jeder Trial wird in 5 gleich lange Clips geteilt -> 5 Folds
FENSTER_S = 3.0      # Fensterlänge in Sekunden
SCHRITT_S = 1.5      # Schrittweite -> 50 % Überlappung
GAP_S = 3.0          # Mindestabstand Training/Test, mindestens eine Fensterlänge

# --- Experimente ----------------------------------------------------------------------
HAUPT_AKTIVITAET = "walking"
MODELLE = ["Mehrheit", "Entscheidungsbaum", "KNN", "SVM", "Random Forest"]
SEED = 42            # fester Zufallswert, damit jeder Lauf dieselben Zahlen liefert
ERGEBNIS_ORDNER = "ergebnisse"
