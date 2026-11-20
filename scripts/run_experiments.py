"""Startet die Experimente.

    python scripts/run_experiments.py --config configs/synth.yaml               # alles (E0–E8)
    python scripts/run_experiments.py --config configs/ax6_beta.yaml --only E0  # erst nur Inventar/QC!
    python scripts/run_experiments.py --config configs/synth.yaml --only E1,E5 --quick
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personid.config import load_config  # noqa: E402
from personid.experiments import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--only", default=None, help="z. B. E0,E1,E5")
ap.add_argument("--name", default=None, help="Name des Ergebnisordners")
ap.add_argument("--quick", action="store_true", help="schneller Testlauf (weniger Bäume/Epochen/Permutationen)")
a = ap.parse_args()
over = None
if a.quick:
    over = {"models": {"rf": {"n_estimators": 100}, "minirocket": {"n_kernels": 2000}, "cnn": {"epochs": 8}},
            "experiments": {"E7_n_permutations": 10, "E7_n_bootstrap": 300, "E8_n_repeats": 2}}
cfg = load_config(a.config, over)
ctx = run(cfg, only=a.only.split(",") if a.only else None, run_name=a.name)
print(f"\nFertig. Ergebnisse: {ctx.out}\nBericht: {ctx.out / 'REPORT.md'}")
