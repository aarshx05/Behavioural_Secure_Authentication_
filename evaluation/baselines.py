"""Baseline detectors from Killourhy & Maxion (DSN 2009).

These are *anomaly* detectors: they see only the genuine user's training data
and score how unlike it a new sample is. That is a different setting from this
project's classifier, which manufactures synthetic negatives and solves a
two-class problem -- which is exactly why the comparison is worth making.

Published mean EER on the CMU dataset, for reference:

    Manhattan (scaled)          ~0.096
    Nearest neighbour (Mahal.)  ~0.100
    Manhattan (plain)           ~0.153
    Euclidean                   ~0.171

Reproducing the scaled-Manhattan figure is the check that this harness
implements the same protocol the paper did.

Every ``score`` here returns *genuineness* (higher = more genuine), so the
distances are negated.
"""

import numpy as np


class Detector:
    name = "detector"

    def fit(self, X):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError


class ScaledManhattan(Detector):
    """Manhattan distance, each dimension divided by its mean absolute deviation.

    The best-performing detector in the paper. Scaling by MAD rather than
    standard deviation is what makes it robust: keystroke timings are heavily
    right-skewed by occasional long pauses, which inflate a variance estimate
    far more than a MAD one.
    """

    name = "manhattan-scaled"

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        mad = np.abs(X - self.mean_).mean(axis=0)
        # A dimension with no observed variation would otherwise divide by zero.
        self.mad_ = np.where(mad > 1e-12, mad, 1e-12)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        return -np.abs(X - self.mean_).__truediv__(self.mad_).sum(axis=1)


class Manhattan(Detector):
    name = "manhattan"

    def fit(self, X):
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def score(self, X):
        return -np.abs(np.asarray(X, dtype=float) - self.mean_).sum(axis=1)


class Euclidean(Detector):
    name = "euclidean"

    def fit(self, X):
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def score(self, X):
        delta = np.asarray(X, dtype=float) - self.mean_
        return -np.sqrt((delta ** 2).sum(axis=1))


class MahalanobisNN(Detector):
    """Nearest neighbour under a Mahalanobis metric.

    Distance to the closest training sample rather than to the centroid, which
    lets a user with two distinct typing modes be modelled properly instead of
    being averaged into a rhythm they never actually use.

    The covariance is estimated from few samples relative to its dimension, so
    it is regularised and inverted with a pseudo-inverse.
    """

    name = "mahalanobis-nn"

    def __init__(self, ridge=1e-6):
        self.ridge = ridge

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.train_ = X
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0]) * self.ridge * max(np.trace(cov) / cov.shape[0], 1e-12)
        self.inv_cov_ = np.linalg.pinv(cov)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        for i, row in enumerate(X):
            delta = self.train_ - row
            d = np.einsum("ij,jk,ik->i", delta, self.inv_cov_, delta)
            out[i] = d.min()
        return -np.sqrt(np.maximum(out, 0.0))


ALL = {
    d.name: d
    for d in (ScaledManhattan(), Manhattan(), Euclidean(), MahalanobisNN())
}


def build(name):
    """Fresh instance of a named detector."""
    mapping = {
        "manhattan-scaled": ScaledManhattan,
        "manhattan": Manhattan,
        "euclidean": Euclidean,
        "mahalanobis-nn": MahalanobisNN,
    }
    return mapping[name]()
