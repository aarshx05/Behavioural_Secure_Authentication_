"""Model training, recency weighting, and synthetic negative generation."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config, features

HARSH = 1
EASY = 2

# Floors keep a perfectly consistent typist from producing a zero-width
# distribution that every perturbation would then fall outside of.
_MIN_HOLD = 0.005
_MIN_GAP = 0.005
_MIN_UD = -0.5
_STD_FLOOR = 0.008


def normalize_choice(choice, default=HARSH):
    """Coerce a model choice to an int.

    The CLI reads this with input(), so it arrives as a string. The original
    code compared that string against the integer 1, which was never true --
    meaning the 'Harsh' preset was unreachable and every model ever trained
    silently used the 'Easy' one.
    """
    try:
        value = int(str(choice).strip())
    except (TypeError, ValueError):
        return default
    return value if value in (HARSH, EASY) else default


def _calibrated_svm(**kwargs):
    """RBF SVM exposing predict_proba, without SVC(probability=True).

    Soft voting needs probabilities, but SVC's own ``probability=True`` is
    deprecated as of scikit-learn 1.9 and removed in 1.11. Wrapping in
    CalibratedClassifierCV is the supported replacement and measured slightly
    more stable here (higher worst-case genuine score) on top of being
    forward-compatible.
    """
    return CalibratedClassifierCV(
        SVC(kernel="rbf", class_weight="balanced", **kwargs),
        method="sigmoid",
        cv=3,
        ensemble=False,
    )


def build_estimators(choice_train=HARSH, n_samples=None):
    """Build the three base estimators for a preset.

    Both presets use an RBF kernel. The original 'Harsh' preset used
    ``kernel='linear'``, which cannot work here: the synthetic negatives
    surround the authentic cluster in every direction, and no hyperplane
    separates a blob from a shell enclosing it. Measured on simulated typists,
    the linear kernel scored genuine samples at ~0.53 against RBF's ~0.73,
    dragging the whole ensemble down. That preset was unreachable in practice
    because of the string/int comparison bug in normalize_choice, so the flaw
    never surfaced.

    The presets now differ in strictness rather than in kernel.
    """
    choice = normalize_choice(choice_train)

    if choice == HARSH:
        svm = _calibrated_svm(C=5.0, gamma="scale")
        neighbors = 3
        forest = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
    else:
        svm = _calibrated_svm(C=1.0, gamma="scale")
        neighbors = 5
        forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            class_weight="balanced",
        )

    # KNN raises if it is asked for more neighbours than there are samples.
    if n_samples:
        neighbors = max(1, min(neighbors, n_samples))
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")

    return svm, knn, forest


def replicate_by_recency(X, timestamps, now=None, half_life_days=None, max_replication=None):
    """Duplicate recent rows so newer typing dominates the fit.

    Recency is expressed by replication rather than ``sample_weight`` because
    VotingClassifier only forwards sample weights when *every* estimator accepts
    them, and KNeighborsClassifier does not -- passing weights would raise.

    Returns ``(X_replicated, counts)``.
    """
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


def generate_negatives(authentic, n_chars, extended=True, rng=None, count=None):
    """Synthesise impostor samples from a user's authentic samples.

    The original generator added a fixed 0.1s Gaussian to the assembled vector.
    That is a large perturbation next to a ~0.09s dwell time but a negligible
    one next to total_time, and it left the aggregate features inconsistent with
    the per-key values.

    This builds negatives in raw timing space and re-derives the vector, using
    several impostor archetypes:

    ``jitter``   someone typing the password almost right (the hard negative)
    ``slow``     hunt-and-peck typist
    ``fast``     faster typist with a different rhythm
    ``flat``     uniform intervals -- scripted or replayed input
    ``shuffle``  the user's own intervals in the wrong order: same overall
                 speed, wrong rhythm, which forces the model to learn rhythm
                 rather than just how fast the password gets typed
    ``random``   broad draws across a plausible human range
    """
    rng = rng or np.random.default_rng(42)
    authentic = np.atleast_2d(np.asarray(authentic, dtype=float))

    decomposed = [features.decompose(row, n_chars, extended) for row in authentic]
    holds = np.array([d[0] for d in decomposed])
    dds = np.array([d[1] for d in decomposed])
    uds = np.array([d[2] for d in decomposed])
    uus = np.array([d[3] for d in decomposed])

    std_hold = _column_std(holds)
    std_dd = _column_std(dds)
    std_ud = _column_std(uds)
    std_uu = _column_std(uus)

    if count is None:
        count = int(
            np.clip(
                len(authentic) * config.NEGATIVE_RATIO,
                config.MIN_NEGATIVES,
                config.MAX_NEGATIVES,
            )
        )

    archetypes = ("jitter", "slow", "fast", "flat", "shuffle", "random")
    negatives = []

    for i in range(count):
        source = i % len(authentic)
        hold = holds[source].copy()
        dd = dds[source].copy()
        ud = uds[source].copy()
        uu = uus[source].copy()
        kind = archetypes[i % len(archetypes)]

        if kind == "jitter":
            scale = 3.0
            hold += rng.normal(0, np.maximum(std_hold * scale, hold * 0.35 + 0.01))
            dd += rng.normal(0, np.maximum(std_dd * scale, dd * 0.35 + 0.01))
            ud += rng.normal(0, np.maximum(std_ud * scale, np.abs(ud) * 0.35 + 0.01))
            uu += rng.normal(0, np.maximum(std_uu * scale, uu * 0.35 + 0.01))

        elif kind in ("slow", "fast"):
            factor = rng.uniform(1.4, 2.5) if kind == "slow" else rng.uniform(0.35, 0.7)
            hold = hold * factor + rng.normal(0, std_hold, hold.shape)
            dd = dd * factor + rng.normal(0, std_dd, dd.shape)
            ud = ud * factor + rng.normal(0, std_ud, ud.shape)
            uu = uu * factor + rng.normal(0, std_uu, uu.shape)

        elif kind == "flat":
            hold = np.full_like(hold, hold.mean() if hold.size else 0.1)
            hold += rng.normal(0, 0.004, hold.shape)
            if dd.size:
                dd = np.full_like(dd, dd.mean()) + rng.normal(0, 0.004, dd.shape)
                ud = np.full_like(ud, ud.mean()) + rng.normal(0, 0.004, ud.shape)
                uu = np.full_like(uu, uu.mean()) + rng.normal(0, 0.004, uu.shape)

        elif kind == "shuffle":
            hold = rng.permutation(hold)
            if dd.size:
                order = rng.permutation(dd.size)
                dd, ud, uu = dd[order], ud[order], uu[order]
            hold += rng.normal(0, std_hold * 0.5, hold.shape)

        else:  # random
            hold = rng.uniform(0.03, 0.30, hold.shape)
            dd = rng.uniform(0.05, 0.60, dd.shape)
            ud = rng.uniform(-0.05, 0.50, ud.shape)
            uu = rng.uniform(0.05, 0.60, uu.shape)

        hold = np.maximum(hold, _MIN_HOLD)
        dd = np.maximum(dd, _MIN_GAP)
        uu = np.maximum(uu, _MIN_GAP)
        ud = np.maximum(ud, _MIN_UD)

        negatives.append(features.assemble(hold, dd, ud, uu, extended=extended))

    return np.array(negatives)


def train(authentic, negatives, choice_train=HARSH, timestamps=None, now=None):
    """Fit the soft-voting ensemble.

    Returns ``(model, scaler, info)``.
    """
    authentic = np.atleast_2d(np.asarray(authentic, dtype=float))
    negatives = np.atleast_2d(np.asarray(negatives, dtype=float))

    weighted, counts = replicate_by_recency(authentic, timestamps, now=now)

    X = np.vstack([weighted, negatives])
    y = np.hstack([np.ones(len(weighted)), np.zeros(len(negatives))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm, knn, forest = build_estimators(choice_train, n_samples=len(X))
    model = VotingClassifier(
        estimators=[("svm", svm), ("knn", knn), ("rf", forest)], voting="soft"
    )
    model.fit(X_scaled, y)

    info = {
        "authentic_samples": int(len(authentic)),
        "effective_positives": int(len(weighted)),
        "negatives": int(len(negatives)),
        "replication": counts.tolist(),
        "model_choice": normalize_choice(choice_train),
    }
    return model, scaler, info


def score(model, scaler, vector):
    """Probability that ``vector`` came from the enrolled user."""
    scaled = scaler.transform(np.asarray(vector, dtype=float).reshape(1, -1))
    return float(model.predict_proba(scaled)[0][1])
