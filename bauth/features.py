"""Feature vector assembly.

Layout (n = password length)::

    index                 contents
    -----                 --------
    0                     total_time = sum(hold) + sum(dd)
    1        .. n          hold times            (n values)
    n+1      .. 2n-1       down-down latencies   (n-1 values)
    --- v1 profiles stop here (dim = 2n) -------------------------------
    2n       .. 3n-2       up-down latencies     (n-1 values)
    3n-1     .. 4n-3       up-up latencies       (n-1 values)
    4n-2     .. 4n+11      aggregates            (14 values)

The first 2n entries are byte-for-byte what the original implementation
produced, so a profile enrolled under v1 keeps working: it simply assembles
with ``extended=False`` and gets the identical vector its model was fit on.
"""

import numpy as np

AGGREGATE_NAMES = [
    "mean_hold",
    "std_hold",
    "min_hold",
    "max_hold",
    "mean_dd",
    "std_dd",
    "mean_ud",
    "std_ud",
    "mean_uu",
    "std_uu",
    "cv_hold",
    "cv_dd",
    "typing_speed",
    "overlap_ratio",
]

N_AGGREGATES = len(AGGREGATE_NAMES)

FEATURE_SETS = {
    # Stable raw timings shared by the anomaly-style detectors.
    "core": ("hold", "dd"),
    # Adds true flight time, which captures overlap and key transitions.
    "transition": ("hold", "dd", "ud"),
    # Richer rhythm signal for discriminative models.
    "extended": (
        "hold",
        "dd",
        "ud",
        "mean_hold",
        "std_hold",
        "mean_dd",
        "std_dd",
        "mean_ud",
        "std_ud",
        "typing_speed",
        "overlap_ratio",
    ),
    # Forests benefit more from coarser timing summaries than from every column.
    "aggregate": (
        "hold",
        "dd",
        "mean_hold",
        "std_hold",
        "min_hold",
        "max_hold",
        "mean_dd",
        "std_dd",
        "mean_ud",
        "std_ud",
        "mean_uu",
        "std_uu",
        "typing_speed",
        "overlap_ratio",
    ),
}


def feature_dim(n_chars, extended=True):
    """Length of the feature vector for an ``n_chars``-long password."""
    if n_chars < 1:
        raise ValueError("password must have at least one character")
    if not extended:
        return 2 * n_chars
    return 4 * n_chars + N_AGGREGATES - 2


def _stats(values):
    """(mean, std, min, max), all zero for an empty input."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(array.mean()),
        float(array.std()),
        float(array.min()),
        float(array.max()),
    )


def _cv(mean, std):
    """Coefficient of variation -- rhythm consistency, independent of speed."""
    return float(std / mean) if abs(mean) > 1e-9 else 0.0


def _aggregates(hold, dd, ud, uu):
    mean_hold, std_hold, min_hold, max_hold = _stats(hold)
    mean_dd, std_dd, _, _ = _stats(dd)
    mean_ud, std_ud, _, _ = _stats(ud)
    mean_uu, std_uu, _, _ = _stats(uu)

    total = float(np.sum(hold) + np.sum(dd))
    n_chars = len(hold)
    typing_speed = float(n_chars / total) if total > 1e-9 else 0.0

    ud_array = np.asarray(ud, dtype=float)
    overlap_ratio = (
        float(np.mean(ud_array < 0.0)) if ud_array.size else 0.0
    )

    return [
        mean_hold,
        std_hold,
        min_hold,
        max_hold,
        mean_dd,
        std_dd,
        mean_ud,
        std_ud,
        mean_uu,
        std_uu,
        _cv(mean_hold, std_hold),
        _cv(mean_dd, std_dd),
        typing_speed,
        overlap_ratio,
    ]


def assemble(hold, dd, ud, uu, extended=True):
    """Build one feature vector from the four raw timing sequences.

    Synthetic negatives are produced by perturbing the raw timings and calling
    this, so aggregates always stay consistent with the per-key values they
    summarise. Perturbing the assembled vector directly would leave the two out
    of sync and hand the classifier a shortcut that has nothing to do with
    typing behaviour.
    """
    hold = np.asarray(hold, dtype=float)
    dd = np.asarray(dd, dtype=float)
    ud = np.asarray(ud, dtype=float)
    uu = np.asarray(uu, dtype=float)

    total = float(hold.sum() + dd.sum())
    parts = [np.array([total]), hold, dd]

    if extended:
        parts.extend([ud, uu, np.asarray(_aggregates(hold, dd, ud, uu), dtype=float)])

    return np.concatenate(parts)


def from_capture(capture, extended=True):
    """Feature vector for a completed :class:`~bauth.capture.KeystrokeCapture`."""
    hold, dd, ud, uu = capture.timings()
    return assemble(hold, dd, ud, uu, extended=extended)


def decompose(vector, n_chars, extended=True):
    """Inverse of :func:`assemble` -- recover ``(hold, dd, ud, uu)``.

    For v1 vectors the release-derived timings were never recorded, so ``ud``
    and ``uu`` come back as zeros.
    """
    vector = np.asarray(vector, dtype=float)
    expected = feature_dim(n_chars, extended)
    if vector.shape[0] != expected:
        raise ValueError(
            f"expected {expected} features for a {n_chars}-char password, "
            f"got {vector.shape[0]}"
        )

    hold = vector[1 : 1 + n_chars]
    dd = vector[1 + n_chars : 2 * n_chars]

    if not extended:
        zeros = np.zeros(max(n_chars - 1, 0))
        return hold, dd, zeros, zeros.copy()

    ud_start = 2 * n_chars
    uu_start = ud_start + (n_chars - 1)
    ud = vector[ud_start:uu_start]
    uu = vector[uu_start : uu_start + (n_chars - 1)]
    return hold, dd, ud, uu


def describe(n_chars, extended=True):
    """Human-readable name for every position in the vector."""
    names = ["total_time"]
    names += [f"hold[{i}]" for i in range(n_chars)]
    names += [f"dd[{i}->{i + 1}]" for i in range(n_chars - 1)]
    if extended:
        names += [f"ud[{i}->{i + 1}]" for i in range(n_chars - 1)]
        names += [f"uu[{i}->{i + 1}]" for i in range(n_chars - 1)]
        names += list(AGGREGATE_NAMES)
    return names


def feature_indices(n_chars, extended=True):
    """Map every feature name returned by :func:`describe` to its column index."""
    return {name: idx for idx, name in enumerate(describe(n_chars, extended))}


def select_set(X, n_chars, extended=True, set_name="extended"):
    """Project ``X`` onto one of the named feature configurations."""
    matrix = np.atleast_2d(np.asarray(X, dtype=float))
    names = describe(n_chars, extended)
    index = feature_indices(n_chars, extended)
    selected = []

    for token in FEATURE_SETS[set_name]:
        if token in AGGREGATE_NAMES:
            selected.append(index[token])
            continue
        if token == "hold":
            selected.extend(index[f"hold[{i}]"] for i in range(n_chars))
            continue
        if token == "dd":
            selected.extend(index[f"dd[{i}->{i + 1}]"] for i in range(n_chars - 1))
            continue
        if token == "ud" and extended:
            selected.extend(index[f"ud[{i}->{i + 1}]"] for i in range(n_chars - 1))
            continue
        if token == "uu" and extended:
            selected.extend(index[f"uu[{i}->{i + 1}]"] for i in range(n_chars - 1))
            continue
        raise KeyError(f"unknown feature token: {token!r}")

    if not selected:
        raise ValueError(f"feature set {set_name!r} produced no columns from {names!r}")
    return matrix[:, selected]
