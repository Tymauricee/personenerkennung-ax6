"""Tests: Leakage-Sicherheit der Splits, Clip-Logik, Rotationsinvarianz der Features, Laden.

    pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personid.features import compute_features  # noqa: E402
from personid.io import Recording, load_manifest, load_recordings  # noqa: E402
from personid.splits import (block_folds, check_leakage, cross_fold, random_clip_folds,  # noqa: E402
                             random_window_folds, trial_folds)

FS = 50.0
CH = [f"S{s}_{k}_{a}" for s in ("01", "05") for k in ("acc", "gyr") for a in "xyz"]


def _rec(person, activity, trial="T1", clip="", t0=0.0, seconds=60.0, session="S1", seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    X = rng.normal(size=(n, len(CH))).astype(np.float32)
    rid = "|".join([person, session, activity, trial] + ([clip] if clip else []))
    return Recording(rid, person, session, activity, trial, FS, CH, X, clip=clip, t0=t0)


def _clips(person, activity, n_clips=6, clip_s=10.0, step_s=10.0, relative=False, seed=0):
    return [_rec(person, activity, clip=f"C{k + 1:02d}", t0=0.0 if relative else k * step_s, seconds=clip_s,
                 seed=seed + k) for k in range(n_clips)]


L, STEP, GAP = int(3 * FS), int(1.5 * FS), 3.0


@pytest.mark.parametrize("gap", [3.0, 6.0])
def test_block_split_continuous_has_no_leakage(gap):
    recs = [_rec(f"P{p}", "Gehen", seed=p) for p in range(4)]
    folds = block_folds(recs, range(len(recs)), n_blocks=5, gap_s=gap)
    assert len(folds) == 5
    for f in folds:
        chk = check_leakage(f, recs, L, STEP, required_gap_s=gap)
        assert chk["ok"], chk
        assert chk["overlapping_pairs"] == 0 and chk["min_gap_s"] >= gap - 1e-6


@pytest.mark.parametrize("step_s", [10.0, 5.0])  # 5.0 = 50 % überlappende Clips
def test_block_split_clips_absolute_time(step_s):
    recs = []
    for p in range(3):
        recs += _clips(f"P{p}", "Gehen", n_clips=10, step_s=step_s, seed=10 * p)
    folds = block_folds(recs, range(len(recs)), n_blocks=5, gap_s=GAP)
    assert folds
    for f in folds:
        chk = check_leakage(f, recs, L, STEP, required_gap_s=GAP)
        assert chk["ok"], chk
        # Testeinheiten sind ganze Clips
        assert all(a == 0 and b == recs[i].n for i, a, b, _ in f.test)


def test_relative_clip_times_are_inferred_and_safe():
    from personid.io import _infer_clip_times
    recs = []
    for p in range(3):
        recs += _clips(f"P{p}", "Gehen", n_clips=6, relative=True, seed=10 * p)
    warn = _infer_clip_times(recs)
    assert len(warn) == 3
    assert sorted(r.t0 for r in recs if r.person == "P0") == [0, 10, 20, 30, 40, 50]
    for f in block_folds(recs, range(len(recs)), n_blocks=3, gap_s=GAP):
        assert check_leakage(f, recs, L, STEP, required_gap_s=GAP)["ok"]


def test_random_window_split_is_detected_as_leaky():
    recs = [_rec(f"P{p}", "Gehen", seed=p) for p in range(3)]
    folds = random_window_folds(recs, range(3), L, STEP, n_folds=3)
    assert sum(check_leakage(f, recs, L, STEP)["overlapping_pairs"] for f in folds) > 0


def test_random_clip_split_violates_gap():
    recs = []
    for p in range(3):
        recs += _clips(f"P{p}", "Gehen", n_clips=6, seed=10 * p)
    folds = random_clip_folds(recs, range(len(recs)), n_folds=3)
    assert any(not check_leakage(f, recs, L, STEP, required_gap_s=GAP)["ok"] for f in folds)


def test_trial_and_cross_folds_disjoint():
    recs = [_rec(f"P{p}", "Gehen", trial=f"T{t}", seed=10 * p + t) for p in range(3) for t in (1, 2, 3)]
    folds = trial_folds(recs, range(len(recs)))
    assert len(folds) == 3
    for f in folds:
        tr = {i for i, *_ in f.train}
        te = {i for i, *_ in f.test}
        assert not tr & te
    recs2 = [_rec(f"P{p}", a, seed=p) for p in range(3) for a in ("Gehen", "Treppe")]
    f = cross_fold(recs2, [0, 2, 4], [1, 3, 5], "x", "cross_activity")
    assert check_leakage(f, recs2, L, STEP)["trials_in_both"] == 0


def test_invariant_features_are_rotation_invariant():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(5, 150, len(CH)))
    W2 = W.copy()
    for s in ("S01", "S05"):
        R = Rot.random(random_state=int(s[1:])).as_matrix()
        for k in ("acc", "gyr"):
            idx = [CH.index(f"{s}_{k}_{a}") for a in "xyz"]
            W2[:, :, idx] = W[:, :, idx] @ R.T
    F1, n1 = compute_features(W, CH, FS, "invariant")
    F2, n2 = compute_features(W2, CH, FS, "invariant")
    assert n1 == n2
    assert np.allclose(F1, F2, atol=1e-4)
    Fe1, _ = compute_features(W, CH, FS, "extended")
    Fe2, _ = compute_features(W2, CH, FS, "extended")
    assert not np.allclose(Fe1, Fe2, atol=1e-4)  # Achsen-Features ändern sich (Kontrolle)


def test_feature_shapes_and_finite():
    W = np.random.default_rng(0).normal(size=(4, 150, len(CH)))
    for fs_name in ("basic", "extended", "invariant", "static"):
        F, names = compute_features(W, CH, FS, fs_name)
        assert F.shape == (4, len(names)) and np.isfinite(F).all()
        assert len(set(names)) == len(names)


def test_loader_roundtrip(tmp_path):
    from personid.synth import generate_dataset
    generate_dataset(tmp_path, n_persons=2, retest_persons=1, durations={"Gehen": 25, "Squats": 20}, verbose=False)
    cfg = {"fs": 50.0, "sensors": ["S01", "S02", "S05", "S06"], "trim_s": 0.0, "max_gap_s": 0.1,
           "cache_dir": str(tmp_path / "cache")}
    man = load_manifest(tmp_path / "manifest_clips.csv")
    recs, qc, log = load_recordings(man, cfg, use_cache=False, verbose=False)
    assert len(recs) > 0 and all(r.X.shape[1] == 24 for r in recs)
    assert abs(np.nanmedian(np.linalg.norm(recs[0].X[:, :3], axis=1)) - 1.0) < 0.2  # ~1 g


def test_features_without_gyro():
    ch = [c for c in CH if "_acc_" in c]
    W = np.random.default_rng(0).normal(size=(3, 150, len(ch)))
    for fs_name in ("basic", "extended", "invariant", "static"):
        F, names = compute_features(W, ch, FS, fs_name)
        assert F.shape[0] == 3 and np.isfinite(F).all() and not any("gyr" in n for n in names)
