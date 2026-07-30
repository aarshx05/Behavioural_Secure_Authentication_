"""The Killourhy & Maxion (DSN 2009) evaluation protocol.

For each of the 51 subjects in turn, treated as the genuine user:

* **train** on their first 200 repetitions (sessions 1-4);
* **genuine test** on their remaining 200 repetitions (sessions 5-8);
* **impostor test** on the first 5 repetitions of each of the other 50
  subjects, giving 250 impostor samples.

An EER is computed per subject and the mean across subjects is reported. The
train/test split is by session, so genuine test scores come from sessions the
model never saw -- which also means the reported numbers include whatever
within-subject drift occurred between sessions.

Note the asymmetry this protocol has for a two-class system: the baselines see
only genuine data, whereas this project's classifier manufactures its own
negatives. Neither sees real impostor data during training.
"""

import sys
import time

import numpy as np

from bauth import config, features, models

from . import baselines, cmu, metrics

TRAIN_REPS = 200
IMPOSTOR_REPS = 5


def _ordered_index(data, subject):
    """Row indices for one subject, in (session, rep) order."""
    idx = np.where(data.mask(subject))[0]
    return idx[np.lexsort((data.reps[idx], data.sessions[idx]))]


def split(data, subject):
    """(train, genuine_test, impostor_test) row indices for one genuine user."""
    own = _ordered_index(data, subject)
    train, genuine_test = own[:TRAIN_REPS], own[TRAIN_REPS:]

    impostor = []
    for other in data.subject_ids:
        if other == subject:
            continue
        impostor.extend(_ordered_index(data, other)[:IMPOSTOR_REPS])

    return train, genuine_test, np.asarray(impostor, dtype=int)


def project_raw31(X_extended, n_chars=cmu.N_KEYS):
    """Extended feature matrix -> the published 31-column representation.

    ``decompose`` is the exact inverse of ``assemble``, so this loses nothing;
    it simply presents the same timings in the layout the paper used.
    """
    rows = []
    for row in np.atleast_2d(X_extended):
        hold, dd, ud, _ = features.decompose(row, n_chars, True)
        rows.append(np.concatenate([hold, dd, ud]))
    return np.array(rows)


def project_raw21(X_extended, n_chars=cmu.N_KEYS):
    """Extended feature matrix -> hold and down-down only (the free parameters)."""
    rows = []
    for row in np.atleast_2d(X_extended):
        hold, dd, _, _ = features.decompose(row, n_chars, True)
        rows.append(np.concatenate([hold, dd]))
    return np.array(rows)


def project_identity(X_extended, n_chars=cmu.N_KEYS):
    return np.atleast_2d(np.asarray(X_extended, dtype=float))


class BauthEnsemble(baselines.Detector):
    """This project's voting ensemble, trained against synthetic negatives.

    Always receives extended vectors, because that is the layout the negative
    generator understands; ``project`` then maps both genuine samples and
    generated negatives into whichever representation is being evaluated.
    """

    def __init__(self, n_chars=cmu.N_KEYS, choice=models.HARSH, project=None, label=None):
        self.n_chars = n_chars
        self.choice = choice
        self.project = project or project_identity
        self.name = label or f"bauth-ensemble-{'harsh' if choice == models.HARSH else 'easy'}"

    def fit(self, X_extended):
        X_extended = np.atleast_2d(np.asarray(X_extended, dtype=float))
        negatives = models.generate_negatives(
            X_extended, self.n_chars, extended=True,
            rng=np.random.default_rng(config.RANDOM_SEED),
        )
        positives_p = self.project(X_extended, self.n_chars)
        negatives_p = self.project(negatives, self.n_chars)
        # timestamps=None: no recency weighting, so the fit is deterministic.
        self.model_, self.scaler_, self.info_ = models.train(
            positives_p, negatives_p, choice_train=self.choice, timestamps=None,
        )
        return self

    def score(self, X_extended):
        Xp = self.project(np.atleast_2d(np.asarray(X_extended, dtype=float)), self.n_chars)
        return self.model_.predict_proba(self.scaler_.transform(Xp))[:, 1]


def evaluate_subject(data, subject, detector, representation):
    """EER, AUC and zero-miss FAR for one genuine user."""
    train_idx, genuine_idx, impostor_idx = split(data, subject)

    X_train = representation(data, train_idx)
    X_genuine = representation(data, genuine_idx)
    X_impostor = representation(data, impostor_idx)

    detector.fit(X_train)
    genuine_scores = detector.score(X_genuine)
    impostor_scores = detector.score(X_impostor)

    rate, threshold = metrics.eer(genuine_scores, impostor_scores)
    return {
        "subject": subject,
        "eer": rate,
        "threshold": threshold,
        "auc": metrics.auc(genuine_scores, impostor_scores),
        "zero_miss_far": metrics.zero_miss_far(genuine_scores, impostor_scores),
        "n_train": len(train_idx),
        "n_genuine": len(genuine_idx),
        "n_impostor": len(impostor_idx),
    }


def raw31_representation(data, index):
    return data.raw31(index)


def raw21_representation(data, index):
    return data.raw21(index)


def extended_representation(data, index):
    return data.extended(index)


def run(data, systems, subjects=None, progress=True):
    """Evaluate every system over every subject.

    ``systems`` maps a label to ``(detector_factory, representation)``.
    """
    subjects = subjects or data.subject_ids
    results = {}

    for label, (factory, representation) in systems.items():
        started = time.time()
        per_subject = []
        # Carriage-return progress is unreadable when stdout is redirected to a
        # file or captured, so fall back to periodic lines there.
        live = progress and sys.stdout.isatty()

        for i, subject in enumerate(subjects, 1):
            per_subject.append(
                evaluate_subject(data, subject, factory(), representation)
            )
            if live:
                running = np.mean([r["eer"] for r in per_subject])
                print(
                    f"\r  {label:<28} {i:>3}/{len(subjects)}  "
                    f"running mean EER {running:.4f}",
                    end="", flush=True,
                )
            elif progress and (i % 10 == 0 or i == len(subjects)):
                running = np.mean([r["eer"] for r in per_subject])
                print(
                    f"  {label:<28} {i:>3}/{len(subjects)}  "
                    f"running mean EER {running:.4f}",
                    flush=True,
                )
        if progress:
            mean = np.mean([r["eer"] for r in per_subject])
            print(f"  {label:<28} done: EER {mean:.4f}  "
                  f"[{time.time() - started:.1f}s]", flush=True)

        results[label] = {
            "per_subject": per_subject,
            "eer": metrics.summarise([r["eer"] for r in per_subject]),
            "auc": metrics.summarise([r["auc"] for r in per_subject]),
            "zero_miss_far": metrics.summarise(
                [r["zero_miss_far"] for r in per_subject]
            ),
            "seconds": time.time() - started,
        }
    return results


def default_systems():
    """Baselines on the published features, plus this project's ensemble.

    The ensemble is run on both representations so the effect of the extended
    feature set can be separated from the effect of the classifier.
    """
    return {
        "manhattan-scaled (raw31)": (
            lambda: baselines.ScaledManhattan(), raw31_representation,
        ),
        "mahalanobis-nn (raw31)": (
            lambda: baselines.MahalanobisNN(), raw31_representation,
        ),
        "manhattan (raw31)": (
            lambda: baselines.Manhattan(), raw31_representation,
        ),
        "euclidean (raw31)": (
            lambda: baselines.Euclidean(), raw31_representation,
        ),
        # Same information as raw31, 10 fewer columns -- isolates the cost of
        # carrying redundant encodings.
        "manhattan-scaled (raw21)": (
            lambda: baselines.ScaledManhattan(), raw21_representation,
        ),
        "bauth harsh (raw21)": (
            lambda: BauthEnsemble(choice=models.HARSH, project=project_raw21),
            extended_representation,
        ),
        "bauth harsh (raw31)": (
            lambda: BauthEnsemble(choice=models.HARSH, project=project_raw31),
            extended_representation,
        ),
        "bauth harsh (extended56)": (
            lambda: BauthEnsemble(choice=models.HARSH, project=project_identity),
            extended_representation,
        ),
        "bauth easy (extended56)": (
            lambda: BauthEnsemble(choice=models.EASY, project=project_identity),
            extended_representation,
        ),
    }


# Published means from Killourhy & Maxion (DSN 2009), for validating this
# harness rather than for citation.
PUBLISHED_EER = {
    "manhattan-scaled (raw31)": 0.096,
    "mahalanobis-nn (raw31)": 0.100,
    "manhattan (raw31)": 0.153,
    "euclidean (raw31)": 0.171,
}
