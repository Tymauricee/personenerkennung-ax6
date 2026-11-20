"""Modelle: Baselines, klassische ML-Modelle auf Features, MiniRocket und 1D-CNN auf Rohfenstern.

Alle Modelle haben dieselbe Schnittstelle: ``fit(X, y)``, ``predict_proba(X)``, ``classes_``.
Skalierungen werden IMMER nur auf den Trainingsdaten gefittet (in Pipelines / im Modell).
Hyperparameter sind a priori fest gewählt (kein Tuning auf Testdaten). Wer tunen will:
nur mit innerer, ebenfalls gruppierter CV auf den Trainingsdaten (nested CV).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURE_MODELS = ("majority", "stratified", "dt", "knn", "svm", "rf", "logreg")
RAW_MODELS = ("minirocket", "cnn")
MODEL_LABELS = {"majority": "Mehrheit (Baseline)", "stratified": "Zufall (Baseline)", "dt": "Decision Tree",
                "knn": "KNN", "svm": "SVM (RBF)", "rf": "Random Forest", "logreg": "Log. Regression",
                "minirocket": "MiniRocket", "cnn": "1D-CNN"}


def is_raw(name: str) -> bool:
    return name in RAW_MODELS


def make_model(name: str, cfg: dict, seed: int = 0, sensors_channels: list[str] | None = None):
    m = cfg.get("models", {})
    n_jobs = cfg.get("n_jobs", -1)
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "dt":
        return DecisionTreeClassifier(min_samples_leaf=2, class_weight="balanced", random_state=seed)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=m.get("knn", {}).get("n_neighbors", 5),
                                                                    weights="distance"))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=m.get("svm", {}).get("C", 10.0), gamma="scale",
                                                   probability=True, class_weight="balanced", random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=m.get("rf", {}).get("n_estimators", 500), max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    if name == "minirocket":
        return MiniRocketClassifier(n_kernels=m.get("minirocket", {}).get("n_kernels", 10000), seed=seed,
                                    n_jobs=1 if n_jobs in (None, 1) else -1)
    if name == "cnn":
        return CNNClassifier(m.get("cnn", {}), seed=seed, channels=sensors_channels)
    raise ValueError(f"Unbekanntes Modell: {name}")


class ChannelScaler:
    """z-Standardisierung je Kanal für Rohfenster (n, L, C) – Statistik nur aus Trainingsdaten."""

    def fit(self, X):
        self.mu = X.mean(axis=(0, 1), keepdims=True)
        self.sd = X.std(axis=(0, 1), keepdims=True) + 1e-6
        return self

    def transform(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MiniRocketClassifier:
    """MiniRocket (Dempster et al., KDD 2021; Implementierung: aeon) + RidgeClassifierCV.
    ``predict_proba`` = Softmax der Ridge-Scores (für Aggregation über Fenster ausreichend)."""

    def __init__(self, n_kernels=10000, seed=0, n_jobs=1):
        self.n_kernels, self.seed, self.n_jobs = n_kernels, seed, n_jobs

    def fit(self, X, y):
        from aeon.transformations.collection.convolution_based import MiniRocket
        self.cs = ChannelScaler().fit(X)
        Xt = self.cs.transform(X).transpose(0, 2, 1)
        self.mr = MiniRocket(n_kernels=self.n_kernels, random_state=self.seed, n_jobs=self.n_jobs)
        Z = self.mr.fit_transform(Xt)
        self.zs = StandardScaler().fit(Z)
        self.clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(self.zs.transform(Z), y)
        self.classes_ = self.clf.classes_
        return self

    def decision_function(self, X):
        Z = self.mr.transform(self.cs.transform(X).transpose(0, 2, 1))
        d = self.clf.decision_function(self.zs.transform(Z))
        return np.column_stack([-d, d]) if d.ndim == 1 else d

    def predict_proba(self, X):
        return _softmax(self.decision_function(X))


def _import_keras():
    """Keras 3 ist Backend-agnostisch: nutzt torch, tensorflow oder jax – was installiert ist."""
    if "KERAS_BACKEND" not in os.environ:
        for backend, module in (("torch", "torch"), ("tensorflow", "tensorflow"), ("jax", "jax")):
            if importlib.util.find_spec(module):
                os.environ["KERAS_BACKEND"] = backend
                break
    import keras
    return keras


def _random_rotations(n, max_deg, rng):
    """n zufällige Rotationsmatrizen mit Winkel <= max_deg um zufällige Achsen (Rodrigues)."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = np.radians(rng.uniform(-max_deg, max_deg, n))
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = axis[:, 2], -axis[:, 1], axis[:, 0]
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


class CNNClassifier:
    """Kleines 1D-CNN (3 Faltungsblöcke + Global Average Pooling), Keras 3.

    Datenaugmentation pro Batch: zufällige 3D-Rotation je Sensor (Acc und Gyro mit DERSELBEN
    Rotation -> physikalisch konsistent; simuliert leicht anderen Sensorsitz), Amplituden-
    skalierung, Rauschen. Feste Epochenzahl (kein Early Stopping auf Testdaten).
    """

    def __init__(self, params: dict, seed: int = 0, channels: list[str] | None = None):
        self.p = {"epochs": 60, "batch_size": 64, "rot_deg": 15.0, "width": 64, "dropout": 0.3, "lr": 1e-3}
        self.p.update(params or {})
        self.seed, self.channels = seed, channels

    def _groups(self, C):
        if not self.channels:
            return []
        g = []
        for s in dict.fromkeys(c.split("_", 1)[0] for c in self.channels):
            for kind in ("acc", "gyr"):
                idx = [self.channels.index(f"{s}_{kind}_{a}") for a in "xyz" if f"{s}_{kind}_{a}" in self.channels]
                if len(idx) == 3:
                    g.append((s, kind, idx))
        return g

    def _build(self, L, C, n_classes):
        keras = self.keras
        L_ = keras.layers
        w, inp = self.p["width"], keras.Input((L, C))
        x = inp
        for filters, k, pool in ((w, 7, True), (2 * w, 5, True), (2 * w, 3, False)):
            x = L_.Conv1D(filters, k, padding="same", use_bias=False)(x)
            x = L_.BatchNormalization()(x)
            x = L_.Activation("relu")(x)
            if pool:
                x = L_.MaxPooling1D(2)(x)
        x = L_.GlobalAveragePooling1D()(x)
        x = L_.Dropout(self.p["dropout"])(x)
        out = L_.Dense(n_classes, activation="softmax")(x)
        return keras.Model(inp, out)

    def fit(self, X, y):
        self.keras = keras = _import_keras()
        keras.utils.set_random_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.cs = ChannelScaler().fit(X)
        groups = self._groups(X.shape[2])
        n, L, C = X.shape
        bs, rot = int(self.p["batch_size"]), float(self.p["rot_deg"])
        sensors = list(dict.fromkeys(s for s, _, _ in groups))
        cw = len(yi) / (len(self.classes_) * np.bincount(yi, minlength=len(self.classes_)))
        cs = self.cs

        class Aug(keras.utils.PyDataset):
            def __len__(self_):
                return int(np.ceil(n / bs))

            def __getitem__(self_, b):
                idx = order[b * bs:(b + 1) * bs]
                xb = X[idx].copy()
                if rot > 0 and groups:
                    for s in sensors:
                        R = _random_rotations(len(idx), rot, rng).astype(np.float32)
                        for s2, _, ch in groups:
                            if s2 == s:
                                xb[:, :, ch] = np.einsum("nij,ntj->nti", R, xb[:, :, ch])
                xb = cs.transform(xb)
                xb *= rng.normal(1.0, 0.1, (len(idx), 1, C)).astype(np.float32)
                xb += rng.normal(0, 0.05, xb.shape).astype(np.float32)
                return xb, yi[idx], cw[yi[idx]].astype(np.float32)

            def on_epoch_end(self_):
                rng.shuffle(order)

        order = rng.permutation(n)
        self.model = self._build(L, C, len(self.classes_))
        steps = int(np.ceil(n / bs)) * int(self.p["epochs"])
        sched = keras.optimizers.schedules.CosineDecay(float(self.p["lr"]), decay_steps=max(1, steps))
        self.model.compile(optimizer=keras.optimizers.AdamW(learning_rate=sched, weight_decay=1e-3),
                           loss="sparse_categorical_crossentropy")
        self.model.fit(Aug(), epochs=int(self.p["epochs"]), verbose=0)
        return self

    def predict_proba(self, X):
        return np.asarray(self.model.predict(self.cs.transform(X), batch_size=256, verbose=0))
