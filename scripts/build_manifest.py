"""Baut das Manifest für eure echten Daten.

(1) Empfohlen – Dateien nach der Namenskonvention des Kurs-Repos
    (z. B. 20261117_P03_HAR-walking_T01_ax6-ankle-right_imu_proc.csv):

    python scripts/build_manifest.py --convention --root data --out data/manifest.csv \
        [--stage proc] [--devices ax6] [--log data/metadata/<zentrales_log>.csv]

    Datum -> session, Position -> sensor, imu/acc/gyr -> channel; mit --log werden Sensor-IDs
    (device_id) übernommen und ungültige Trials ausgeschlossen.

(2) Beliebiges Schema per Regex mit benannten Gruppen (person, activity, sensor; optional
    trial, session, clip, channel), z. B. für  data/P01/Gehen/clip_03/S05.csv :

    python scripts/build_manifest.py --root data --out data/manifest.csv \
        --pattern "(?P<person>P\\d+)/(?P<activity>[^/]+)/clip_(?P<clip>\\d+)/(?P<sensor>S\\d+)\\.csv"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personid.convention import manifest_from_convention  # noqa: E402
from personid.io import build_manifest_from_pattern  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--convention", action="store_true", help="Namenskonvention des Kurs-Repos verwenden")
ap.add_argument("--stage", default="proc", help="proc (Standardformat) oder raw")
ap.add_argument("--devices", default=None, help="z. B. ax6 oder ax6,mbient")
ap.add_argument("--types", default="HAR", help="HAR und/oder EVT")
ap.add_argument("--log", default=None, help="zentrales Log (CSV) für Sensor-ID und Gültigkeit")
ap.add_argument("--pattern", default=None, help="Regex für eigenes Namensschema")
a = ap.parse_args()
if a.convention:
    df = manifest_from_convention(a.root, stage=a.stage, types=tuple(a.types.split(",")),
                                  devices=a.devices.split(",") if a.devices else None, log_csv=a.log, out_csv=a.out)
elif a.pattern:
    df = build_manifest_from_pattern(a.root, a.pattern, out_csv=a.out)
else:
    ap.error("--convention oder --pattern angeben")
print(df.head(8).to_string())
print(f"... {len(df)} Zeilen -> {a.out}")
