"""Detector training, calibration, fusion, and synthetic negative generation."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from . import attacks, config, features

HARSH = 1
EASY = 2

DETECTOR_FEATURES = {
    "anchor": "transition",
    "knn": "transition",
    "svm": "extended",
    "rf": "aggregate",
}

# Floors keep a perfectly consistent typist from producing a zero-width
# distribution that every perturbation would then fall outside of.
_MIN_HOLD = 0.005
_MIN_GAP = 0.005
_MIN_UD = -0.5
_STD_FLOOR = 0.008


def normalize_choice(choice, default=HARSH):
    """Coerce a model choice to an int."""
    try:
        value = int(str(choice).strip())
    except (TypeError, ValueError):
        return default
    return value if value in (HARSH, EASY) else default


def _calibration_folds(y):
    """Deterministic stratified folds for per-detector calibration."""
    min_class = int(np.min(np.bincount(np.asarray(y, dtype=int))))
    folds = max(2, min(3, min_class))
    return StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED
    )


@dataclass
class IdentityScaler:
    """Compatibility shim for callers that still expect ``scaler.transform``."""

    def transform(self, X):
        return np.asarray(X, dtype=float)


@dataclass
class RobustScaler:
    """Feature-wise normalization with optional winsorization and clipping."""

    mode: str = config.DEFAULT_SCALER
    clip: float = config.SCALER_CLIP
    winsor_limit: float = config.WINSOR_LIMIT
    center_: np.ndarray = None
    scale_: np.ndarray = None
    lower_: np.ndarray = None
    upper_: np.ndarray = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        base = X

        if self.mode == "winsorized_mad":
            lower = np.quantile(X, self.winsor_limit, axis=0)
            upper = np.quantile(X, 1.0 - self.winsor_limit, axis=0)
            self.lower_ = lower
            self.upper_ = upper
            base = np.clip(X, lower, upper)

        if self.mode == "standard":
            center = base.mean(axis=0)
            scale = base.std(axis=0)
        elif self.mode in ("mad", "winsorized_mad"):
            center = np.median(base, axis=0)
            scale = np.median(np.abs(base - center), axis=0)
        else:
            raise ValueError(f"unknown scaler mode: {self.mode!r}")

        self.center_ = np.asarray(center, dtype=float)
        self.scale_ = np.maximum(np.asarray(scale, dtype=float), _STD_FLOOR)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.lower_ is not None and self.upper_ is not None:
            X = np.clip(X, self.lower_, self.upper_)
        Z = (X - self.center_) / self.scale_
        if self.clip is not None:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)


@dataclass
class ScaledManhattanDetector:
    """Robust scaled-Manhattan anchor detector."""

    feature_set: str = DETECTOR_FEATURES["anchor"]
    scale_floor: float = config.MANHATTAN_SCALE_FLOOR
    scaler_mode: str = config.DEFAULT_SCALER
    scaler: RobustScaler = None
    median_: np.ndarray = None
    mad_: np.ndarray = None
    reliability_: np.ndarray = None
    calibrator_: object = None

    def fit(self, X_pos, X_neg, n_chars, extended):
        X_pos = _project_features(X_pos, n_chars, extended, self.feature_set)
        X_neg = _project_features(X_neg, n_chars, extended, self.feature_set)

        self.scaler = RobustScaler(mode=self.scaler_mode)
        pos_scaled = self.scaler.fit_transform(X_pos)
        neg_scaled = self.scaler.transform(X_neg)

        self.median_ = np.median(pos_scaled, axis=0)
        mad = np.median(np.abs(pos_scaled - self.median_), axis=0)
        self.mad_ = np.maximum(mad, self.scale_floor)

        variability = np.maximum(self.mad_, 1e-6)
        self.reliability_ = 1.0 / variability
        self.reliability_ *= len(self.reliability_) / self.reliability_.sum()

        pos_distance = self.distance_from_scaled(pos_scaled)
        neg_distance = self.distance_from_scaled(neg_scaled)
        raw = np.concatenate([-pos_distance, -neg_distance]).reshape(-1, 1)
        y = np.concatenate(
            [np.ones(len(pos_distance), dtype=int), np.zeros(len(neg_distance), dtype=int)]
        )
        self.calibrator_ = LogisticRegression(
            random_state=config.RANDOM_SEED, max_iter=500
        )
        self.calibrator_.fit(raw, y)
        return self

    def distance_from_scaled(self, X_scaled):
        delta = np.abs(np.asarray(X_scaled, dtype=float) - self.median_)
        scaled = self.reliability_ * delta / np.maximum(self.mad_, self.scale_floor)
        return scaled.mean(axis=1)

    def analyse(self, X, n_chars, extended):
        X_proj = _project_features(X, n_chars, extended, self.feature_set)
        X_scaled = self.scaler.transform(X_proj)
        distance = self.distance_from_scaled(X_scaled)
        score = self.calibrator_.predict_proba((-distance).reshape(-1, 1))[:, 1]
        return {"score": score, "distance": distance}


@dataclass
class CalibratedEstimator:
    """One detector with its own feature view and robust normalization."""

    name: str
    estimator: object
    feature_set: str
    scaler_mode: str = config.DEFAULT_SCALER
    scaler: RobustScaler = None
    model: object = None

    def fit(self, X_pos, X_neg, n_chars, extended):
        X_pos = _project_features(X_pos, n_chars, extended, self.feature_set)
        X_neg = _project_features(X_neg, n_chars, extended, self.feature_set)
        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

        self.scaler = RobustScaler(mode=self.scaler_mode)
        X_scaled = self.scaler.fit_transform(X)
        self.model = CalibratedClassifierCV(
            estimator=deepcopy(self.estimator),
            method="sigmoid",
            cv=_calibration_folds(y),
            ensemble=False,
        )
        self.model.fit(X_scaled, y)
        return self

    def analyse(self, X, n_chars, extended):
        X_proj = _project_features(X, n_chars, extended, self.feature_set)
        X_scaled = self.scaler.transform(X_proj)
        score = self.model.predict_proba(X_scaled)[:, 1]
        return {"score": score}


@dataclass
class FusionModel:
    """Self-contained multi-detector biometric scorer."""

    n_chars: int
    extended: bool
    detectors: dict
    weights: dict
    self_contained: bool = True
    detector_order: tuple = ("anchor", "svm", "knn", "rf")
    thresholds: dict = field(default_factory=dict)

    def analyse(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        detector_scores = {}
        anchor_distance = np.zeros(len(X))

        for name in self.detector_order:
            details = self.detectors[name].analyse(X, self.n_chars, self.extended)
            detector_scores[name] = details["score"]
            if name == "anchor":
                anchor_distance = details["distance"]

        score_matrix = np.column_stack(
            [detector_scores[name] for name in self.detector_order]
        )
        weights = np.array([self.weights[name] for name in self.detector_order], dtype=float)
        fused = np.sum(score_matrix * weights, axis=1)
        disagreement = np.sqrt(
            np.sum(weights * np.square(score_matrix - fused[:, None]), axis=1)
        )
        return {
            "fused": fused,
            "disagreement": disagreement,
            "scores": detector_scores,
            "anchor_distance": anchor_distance,
        }

    def predict_proba(self, X):
        fused = self.analyse(X)["fused"]
        return np.column_stack([1.0 - fused, fused])


def _build_estimators(choice_train=HARSH, n_samples=None):
    choice = normalize_choice(choice_train)

    if choice == HARSH:
        svm = SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
        neighbors = 3
        forest = RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_SEED,
            class_weight="balanced",
        )
    else:
        svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
        neighbors = 5
        forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=config.RANDOM_SEED,
            class_weight="balanced",
        )

    if n_samples:
        neighbors = max(1, min(neighbors, n_samples))
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    return svm, knn, forest


def replicate_by_recency(X, timestamps, now=None, half_life_days=None, max_replication=None):
    """Duplicate recent rows so newer typing dominates the fit."""
    X = np.asarray(X, dtype=float)
    if timestamps is None or len(timestamps) != len(X):
        return X, np.ones(len(X), dtype=int)

    import time as _time

    now = now if now is not None else _time.time()
    half_life = half_life_days or config.RECENCY_HALF_LIFE_DAYS
    max_rep = max_replication or config.MAX_REPLICATION

    counts = []
    for stamp in timestamps:
        age_days = max(0.0, (now - float(stamp)) / 86400.0)
        weight = 0.5 ** (age_days / half_life) if half_life > 0 else 1.0
        counts.append(max(1, int(round(weight * max_rep))))

    counts = np.asarray(counts, dtype=int)
    return np.repeat(X, counts, axis=0), counts


def _column_std(matrix):
    if matrix.size == 0:
        return np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0)
    std = matrix.std(axis=0)
    return np.maximum(std, _STD_FLOOR)


def generate_negatives(authentic, n_chars, extended=True, rng=None, count=None, return_metadata=False):
    """Synthesise impostor samples from a user's authentic samples."""
    records = attacks.generate_negative_records(
        authentic,
        n_chars,
        extended=extended,
        rng=rng or np.random.default_rng(config.RANDOM_SEED),
        count=count,
    )
    negatives = attacks.records_to_matrix(records)
    if return_metadata:
        return negatives, [record.metadata for record in records]
    return negatives


def _normalize_weights(raw, names):
    raw = np.asarray(raw, dtype=float)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 1e-12:
        raw = np.ones(len(raw), dtype=float)
    raw = raw / raw.sum()
    return {name: float(weight) for name, weight in zip(names, raw)}


def _infer_layout(vector_dim):
    for n_chars in range(1, 256):
        if features.feature_dim(n_chars, True) == vector_dim:
            return n_chars, True
        if features.feature_dim(n_chars, False) == vector_dim:
            return n_chars, False
    return None, False


def _project_features(X, n_chars, extended, feature_set):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if n_chars is None:
        return X
    return features.select_set(X, n_chars, extended, feature_set)


def _learn_fusion_weights(model, X_pos, X_neg):
    details_pos = model.analyse(X_pos)
    details_neg = model.analyse(X_neg)
    names = list(model.detector_order)

    learned = []
    for name in names:
        pos = details_pos["scores"][name]
        neg = details_neg["scores"][name]
        separation = max(0.0, float(np.mean(pos) - np.mean(neg)))
        stability = max(0.0, 1.0 - float(np.std(pos)))
        learned.append(separation * stability)

    prior = np.array(
        [config.GLOBAL_FUSION_WEIGHTS.get(name, 1.0) for name in names], dtype=float
    )
    prior = prior / prior.sum()
    shrink = min(1.0, len(X_pos) / float(config.FUSION_SHRINK_SAMPLES))
    combined = shrink * np.asarray(learned, dtype=float) + (1.0 - shrink) * prior
    weights = _normalize_weights(combined, names)
    model.weights = weights
    return weights


def train(authentic, negatives, choice_train=HARSH, timestamps=None, now=None):
    """Fit the calibrated detector ensemble."""
    authentic = np.atleast_2d(np.asarray(authentic, dtype=float))
    negatives = np.atleast_2d(np.asarray(negatives, dtype=float))
    weighted, counts = replicate_by_recency(authentic, timestamps, now=now)

    svm, knn, forest = _build_estimators(choice_train, n_samples=len(weighted) + len(negatives))
    detectors = {
        "anchor": ScaledManhattanDetector(),
        "svm": CalibratedEstimator("svm", svm, DETECTOR_FEATURES["svm"]),
        "knn": CalibratedEstimator("knn", knn, DETECTOR_FEATURES["knn"]),
        "rf": CalibratedEstimator("rf", forest, DETECTOR_FEATURES["rf"]),
    }

    n_chars, extended = _infer_layout(authentic.shape[1])

    for detector in detectors.values():
        detector.fit(weighted, negatives, n_chars, extended)

    model = FusionModel(
        n_chars=n_chars,
        extended=extended,
        detectors=detectors,
        weights=_normalize_weights(
            [config.GLOBAL_FUSION_WEIGHTS[k] for k in ("anchor", "svm", "knn", "rf")],
            ("anchor", "svm", "knn", "rf"),
        ),
    )
    weights = _learn_fusion_weights(model, authentic, negatives)

    info = {
        "authentic_samples": int(len(authentic)),
        "effective_positives": int(len(weighted)),
        "negatives": int(len(negatives)),
        "replication": counts.tolist(),
        "model_choice": normalize_choice(choice_train),
        "fusion_weights": weights,
    }
    return model, IdentityScaler(), info


def analyse(model, vector):
    """Detailed per-detector analysis for a sample or matrix."""
    return model.analyse(np.asarray(vector, dtype=float))


def score(model, scaler, vector):
    """Probability that ``vector`` came from the enrolled user."""
    scaled = scaler.transform(np.asarray(vector, dtype=float).reshape(1, -1))
    return float(model.predict_proba(scaled)[0][1])
