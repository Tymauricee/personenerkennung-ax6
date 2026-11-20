"""personid – leakage-sichere Personenerkennung aus Wearable-IMU-Daten (AX6, 4 Sensoren).

Module
------
config      Konfiguration laden (YAML + Defaults)
io          Manifest, Sensordateien lesen (CSV/Parquet/CWA), Synchronisation -> Recording
synth       Synthetischer Testdatensatz im AX6-Format (nur zum Testen der Pipeline!)
inventory   Dateninventar, Qualitätsprüfung, Datenabdeckung
windows     Fensterbildung innerhalb vorher festgelegter Rohdaten-Intervalle
splits      Leakage-sichere Splits (Trial, Session, Block+Gap, Cross-Activity) + Leakage-Check
features    Handcrafted Features (Statistik, Spektrum, Gang-Autokorrelation, rotationsinvariant)
models      Baselines, DT, KNN, SVM, RF, MiniRocket, 1D-CNN (Keras 3)
evaluate    Cross-Validation, Trial-Aggregation, Metriken, Bootstrap-CI, Permutationstest
plots       Abbildungen (Poster-tauglich)
experiments Experimente E0–E8
"""

__version__ = "1.0.0"
