"""Biometric performance metrics.

Every scorer in this harness returns a *genuineness* score where higher means
more likely genuine. Anomaly detectors, whose natural output is a distance,
negate it so the convention holds everywhere.
"""

import numpy as np


def far_frr_curve(genuine, impostor):
    """False-accept and false-reject rates across all decision thresholds.

    A sample is accepted when ``score >= threshold``, so FAR falls and FRR
    rises as the threshold increases.
    """
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)

    points = np.unique(np.concatenate([genuine, impostor]))
    span = (points.max() - points.min()) or 1.0
    thresholds = np.concatenate([[points.min() - span], points, [points.max() + span]])

    frr = np.array([(genuine < t).mean() for t in thresholds])
    far = np.array([(impostor >= t).mean() for t in thresholds])
    return thresholds, far, frr


def eer(genuine, impostor):
    """Equal error rate: the point where FAR and FRR coincide.

    The two curves are step functions, so they rarely meet exactly. The
    crossing is interpolated linearly between the bracketing thresholds, which
    is the usual convention and avoids the quantisation you get from simply
    taking the closest sampled point.
    """
    thresholds, far, frr = far_frr_curve(genuine, impostor)
    diff = far - frr

    crossings = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(crossings) == 0:
        i = int(np.argmin(np.abs(diff)))
        return float((far[i] + frr[i]) / 2.0), float(thresholds[i])

    i = int(crossings[0])
    d0, d1 = diff[i], diff[i + 1]
    w = 0.0 if d0 == d1 else d0 / (d0 - d1)
    rate = (far[i] + w * (far[i + 1] - far[i]) + frr[i] + w * (frr[i + 1] - frr[i])) / 2.0
    threshold = thresholds[i] + w * (thresholds[i + 1] - thresholds[i])
    return float(rate), float(threshold)


def auc(genuine, impostor):
    """Area under the ROC curve, via the rank-sum identity.

    Equivalent to the probability that a random genuine sample outranks a
    random impostor one, with ties counted as half.
    """
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    combined = np.concatenate([genuine, impostor])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)

    # Average ranks within ties so the result is tie-aware.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    n_g, n_i = len(genuine), len(impostor)
    rank_sum = ranks[:n_g].sum()
    return float((rank_sum - n_g * (n_g + 1) / 2.0) / (n_g * n_i))


def zero_miss_far(genuine, impostor):
    """FAR at the threshold that rejects no genuine sample.

    Reported by Killourhy & Maxion alongside EER; it is the operating point of
    a system tuned never to inconvenience the real user.
    """
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    if len(genuine) == 0:
        return 1.0
    return float((impostor >= genuine.min()).mean())


def summarise(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "n": int(len(values)),
    }
