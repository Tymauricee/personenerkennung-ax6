"""Erzeugt den synthetischen Testdatensatz (nur zum Testen der Pipeline!).

    python scripts/make_synthetic_data.py --out data_synth
    python scripts/make_synthetic_data.py --out data_synth --persons 25 --retest 8 --clip 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personid.synth import generate_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data_synth")
ap.add_argument("--persons", type=int, default=25)
ap.add_argument("--retest", type=int, default=8, help="Personen mit 2. Tag (Gehen) – 0 = wie euer Datensatz")
ap.add_argument("--clip", type=float, default=10.0, help="Clip-Länge [s]")
ap.add_argument("--clip-step", type=float, default=None, help="Clip-Schrittweite [s] (< clip = überlappend)")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
m = generate_dataset(a.out, n_persons=a.persons, retest_persons=a.retest, clip_s=a.clip,
                     clip_step_s=a.clip_step, seed=a.seed)
print(f"{len(m)} Dateien im Manifest -> {a.out}/manifest.csv und {a.out}/manifest_clips.csv")
