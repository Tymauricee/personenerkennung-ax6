"""Baut ein Manifest aus euren Dateinamen (Regex mit benannten Gruppen).

Beispiel für  data/P01/Gehen/clip_03/S05.csv :
    python scripts/build_manifest.py --root data --out data/manifest.csv \
        --pattern "(?P<person>P\\d+)/(?P<activity>[^/]+)/clip_(?P<clip>\\d+)/(?P<sensor>S\\d+)\\.csv"

Beispiel für  data/P01_Gehen_T1_S05.csv :
    --pattern "(?P<person>P\\d+)_(?P<activity>[^_]+)_(?P<trial>T\\d+)_(?P<sensor>S\\d+)\\.csv"

Gruppen: person, activity, sensor (Pflicht); trial, session, clip (optional).
Danach: Spalte device_id ergänzen (welches AX6 an welcher Position)!
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personid.io import build_manifest_from_pattern  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--pattern", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
df = build_manifest_from_pattern(a.root, a.pattern, out_csv=a.out)
print(df.head(10).to_string())
print(f"... {len(df)} Zeilen -> {a.out}")
