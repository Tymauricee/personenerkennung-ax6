"""Pipeline-Schritte 1 und 2: Dateien finden, lesen, prüfen und synchronisieren."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import config

# Namenskonvention des Kurs-Repos (docs/naming_convention.md), Beispiel:
# 20261117_P03_HAR-walking_T01_ax6-wrist-right_imu_proc.csv
DATEINAME = re.compile(
    r"(?P<datum>\d{8})_(?P<person>P\d+)_(?P<typ>HAR|EVT)-(?P<aktivitaet>[a-z0-9-]+)_(?P<trial>T\d+)"
    r"_(?P<geraet>[a-z0-9]+)-(?P<position>[a-z0-9-]+)_(?P<kanal>acc|gyr|imu)_(?P<stufe>raw|proc)"
    r"\.csv(\.gz)?"
)


def finde_dateien(ordner, stufe=config.STUFE):
    """Sucht alle Sensordateien im Ordner und liest Person, Aktivität usw. aus dem Dateinamen.

    Rückgabe: Tabelle mit einer Zeile pro Datei (nur HAR-Trials der gewünschten Stufe).
    """
    zeilen = []
    for pfad in sorted(Path(ordner).rglob("*.csv*")):
        treffer = DATEINAME.fullmatch(pfad.name)
        if treffer:                                   # Dateien mit anderem Namen ignorieren
            zeilen.append({**treffer.groupdict(), "pfad": str(pfad)})
    dateien = pd.DataFrame(zeilen)
    if dateien.empty:
        raise FileNotFoundError(f"Keine Dateien nach der Namenskonvention in '{ordner}' gefunden.")
    dateien = dateien[(dateien.typ == "HAR") & (dateien.stufe == stufe)].copy()
    dateien["sensor"] = dateien.geraet + "-" + dateien.position          # z. B. ax6-wrist-right
    dateien["trial_id"] = (dateien.datum + "_" + dateien.person + "_HAR-" + dateien.aktivitaet
                           + "_" + dateien.trial)
    return dateien.reset_index(drop=True)


def lies_datei(pfad, kanal):
    """Liest eine Sensordatei.

    Rückgabe: Tabelle mit der Spalte zeit (Sekunden) und den Messwerten acc_x ... gyr_z
    (Beschleunigung in g, Drehrate in Grad pro Sekunde).
    """
    roh = pd.read_csv(pfad)
    werte = pd.DataFrame({"zeit": _zeit_in_sekunden(roh[_finde_zeitspalte(roh.columns)])})
    arten = ["acc", "gyr"] if kanal == "imu" else [kanal]   # imu-Dateien enthalten beides
    for art in arten:
        for achse in "xyz":
            spalte = _finde_achse(roh.columns, art, achse, nur_eine_art=(kanal != "imu"))
            if spalte is None:
                raise ValueError(f"{Path(pfad).name}: keine Spalte für {art}_{achse}. "
                                 f"Vorhandene Spalten: {list(roh.columns)}")
            werte[f"{art}_{achse}"] = pd.to_numeric(roh[spalte], errors="coerce")

    # Einheit prüfen: Ein Beschleunigungssensor misst immer auch die Schwerkraft, der Betrag
    # liegt also im Mittel bei etwa 1 g. Liegt er bei etwa 9,81, sind die Werte in m/s².
    if "acc_x" in werte:
        betrag = np.linalg.norm(werte[["acc_x", "acc_y", "acc_z"]], axis=1)
        if np.nanmedian(betrag) > 5:
            print(f"  Hinweis: {Path(pfad).name} ist in m/s² und wird in g umgerechnet")
            werte[["acc_x", "acc_y", "acc_z"]] /= 9.81

    # Zeilen ohne Werte entfernen, nach Zeit sortieren, doppelte Zeitstempel entfernen
    return werte.dropna().sort_values("zeit").drop_duplicates("zeit").reset_index(drop=True)


def _finde_zeitspalte(spalten):
    # Absolute Zeitstempel bevorzugen: nur damit lassen sich mehrere Sensoren synchronisieren.
    # "elapsed" (Sekunden seit Start der Datei) ist nur die letzte Wahl.
    for stichwort in ["epoc", "unix", "timestamp", "time", "zeit", "elapsed"]:
        for spalte in spalten:
            if stichwort in spalte.lower():
                return spalte
    raise ValueError(f"Keine Zeitspalte gefunden. Vorhandene Spalten: {list(spalten)}")


def _zeit_in_sekunden(spalte):
    if not pd.api.types.is_numeric_dtype(spalte):      # Text wie "2026-11-17 10:00:00.010"
        zeit = pd.to_datetime(spalte, utc=True)
        return (zeit - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    t = spalte.to_numpy(dtype=float)
    abstand = np.median(np.diff(t))                    # typischer Abstand zweier Messungen
    if abstand < 0.5:
        return t                                       # schon Sekunden (z. B. 0.01)
    if abstand < 500:
        return t / 1e3                                 # Millisekunden (z. B. 10)
    if abstand < 5e5:
        return t / 1e6                                 # Mikrosekunden
    return t / 1e9                                     # Nanosekunden


def _finde_achse(spalten, art, achse, nur_eine_art):
    """Sucht die Spalte z. B. für art="acc", achse="x".

    Erkennt u. a. acc_x, accel_x, ax, "acc_x (g)" und bei Dateien mit nur einer Sensorart
    (z. B. Mbient-Export, getrennt in acc und gyr) auch "x-axis (g)".
    """
    for spalte in spalten:
        name = re.sub(r"\(.*?\)|[^a-z]", "", spalte.lower())   # "acc_x (g)" -> "accx"
        if name in (art + achse, art[0] + achse) or (name.startswith(art) and name.endswith(achse)):
            return spalte
        if nur_eine_art and name in (achse, achse + "axis"):
            return spalte
    return None


def qualitaet(werte, max_luecke_s=config.MAX_LUECKE_S):
    """Kennzahlen einer Datei für den Datenqualitätsbericht (Pipeline Abschnitt 3)."""
    abstaende = np.diff(werte.zeit)
    bericht = {
        "dauer_s": werte.zeit.iloc[-1] - werte.zeit.iloc[0],
        "fs_hz": 1 / np.median(abstaende),
        "luecken": int((abstaende > max_luecke_s).sum()),
        "groesste_luecke_s": abstaende.max(),
    }
    if "acc_x" in werte:   # sollte nahe 1 g liegen
        bericht["acc_betrag_median_g"] = np.median(np.linalg.norm(werte[["acc_x", "acc_y", "acc_z"]], axis=1))
    if "gyr_x" in werte:   # sehr kleine Werte -> vielleicht rad/s statt Grad/s
        bericht["gyr_p99_grad_s"] = np.percentile(np.abs(werte[["gyr_x", "gyr_y", "gyr_z"]]), 99)
    return bericht


def _tiefpass(t, werte, fs_ziel):
    """Anti-Aliasing vor dem Heruntertakten.

    Ein Signal mit 50 Hz kann nur Frequenzen bis 25 Hz darstellen (Nyquist). Schnellere Anteile
    würden sonst als falsche, langsame Schwingungen auftauchen. Darum vorher ein Tiefpass bei
    20 Hz (Butterworth 4. Ordnung). filtfilt filtert vorwärts und rückwärts, damit das Signal
    zeitlich nicht verschoben wird.
    """
    fs_ein = 1 / np.median(np.diff(t))
    if fs_ein < 1.2 * fs_ziel:                       # nicht schneller abgetastet -> nichts zu tun
        return t, werte
    t_gleich = np.arange(t[0], t[-1], 1 / fs_ein)    # filtfilt braucht gleichmäßige Abstände
    werte = np.column_stack([np.interp(t_gleich, t, werte[:, j]) for j in range(werte.shape[1])])
    b, a = butter(4, 0.4 * fs_ziel, fs=fs_ein)
    return t_gleich, filtfilt(b, a, werte, axis=0)


def synchronisiere(streams, fs=config.FS, max_luecke_s=config.MAX_LUECKE_S):
    """Bringt alle Sensoren eines Trials auf ein gemeinsames Zeitraster mit fs Hz.

    streams: Liste von (sensorname, Tabelle aus lies_datei)
    1. gemeinsamer Zeitbereich: nur die Zeit, in der alle Sensoren laufen
    2. Tiefpass, falls ein Sensor schneller abgetastet wurde
    3. lineare Interpolation auf das Raster
    4. echte Lücken werden NaN statt still überbrückt (Pipeline Abschnitt 4)
    Rückgabe: Tabelle mit einer Spalte pro Kanal, z. B. "ax6-hip_acc_x".
    """
    beginn = max(werte.zeit.iloc[0] for _, werte in streams)
    ende = min(werte.zeit.iloc[-1] for _, werte in streams)
    if ende - beginn < 10:
        raise ValueError(f"Sensoren laufen nur {ende - beginn:.1f} s gleichzeitig")
    raster = np.arange(beginn, ende, 1 / fs)

    ergebnis = {}
    for sensor, werte in streams:
        t = werte.zeit.to_numpy()
        kanaele = [k for k in werte.columns if k != "zeit"]
        t_gefiltert, gefiltert = _tiefpass(t, werte[kanaele].to_numpy(), fs)
        in_luecke = np.zeros(len(raster), dtype=bool)
        for i in np.flatnonzero(np.diff(t) > max_luecke_s):
            in_luecke |= (raster > t[i]) & (raster < t[i + 1])
        for j, kanal in enumerate(kanaele):
            spalte = np.interp(raster, t_gefiltert, gefiltert[:, j])
            spalte[in_luecke] = np.nan
            ergebnis[f"{sensor}_{kanal}"] = spalte
    return pd.DataFrame(ergebnis)


def lade_trials(ordner, sensoren=config.SENSOREN, stufe=config.STUFE, fs=config.FS,
                max_luecke_s=config.MAX_LUECKE_S, rand_s=config.RAND_S):
    """Lädt alle HAR-Trials eines Ordners.

    Rückgabe:
      trials   Liste mit einem Eintrag pro Trial: id, person, aktivitaet, datum und daten
               (Tabelle: eine Zeile pro 1/fs Sekunde, eine Spalte pro Kanal)
      bericht  Datenqualitätsbericht mit einer Zeile pro Datei
    """
    dateien = finde_dateien(ordner, stufe)
    if sensoren is None:
        # nur Sensoren, die in jedem Trial vorkommen (Pipeline: alle Streams müssen da sein)
        sensoren_pro_trial = dateien.groupby("trial_id").sensor.apply(set)
        sensoren = sorted(set.intersection(*sensoren_pro_trial))
    print(f"{dateien.trial_id.nunique()} Trials gefunden, verwendete Sensoren: {sensoren}")

    trials, bericht = [], []
    for trial_id, gruppe in dateien.groupby("trial_id"):
        gruppe = gruppe[gruppe.sensor.isin(sensoren)]
        fehlend = set(sensoren) - set(gruppe.sensor)
        if fehlend:
            print(f"  übersprungen: {trial_id} (fehlender Sensor: {sorted(fehlend)})")
            continue
        streams = []
        for _, datei in gruppe.iterrows():
            werte = lies_datei(datei.pfad, datei.kanal)
            bericht.append({"datei": Path(datei.pfad).name, **qualitaet(werte, max_luecke_s)})
            streams.append((datei.sensor, werte))
        try:
            daten = synchronisiere(streams, fs, max_luecke_s)
        except ValueError as fehler:
            print(f"  übersprungen: {trial_id} ({fehler})")
            continue
        rand = int(rand_s * fs)                       # Anfang und Ende abschneiden
        daten = daten.iloc[rand:len(daten) - rand].reset_index(drop=True)
        trials.append({"id": trial_id, "person": gruppe.person.iloc[0],
                       "aktivitaet": gruppe.aktivitaet.iloc[0], "datum": gruppe.datum.iloc[0],
                       "daten": daten})

    # Alle Trials brauchen dieselben Kanäle in derselben Reihenfolge
    kanaele = sorted(set.intersection(*(set(trial["daten"].columns) for trial in trials)))
    for trial in trials:
        trial["daten"] = trial["daten"][kanaele]
    return trials, pd.DataFrame(bericht)
