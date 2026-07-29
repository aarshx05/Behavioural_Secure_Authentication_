"""Profile persistence.

On-disk layout (``user_data/<user_id>/``)::

    metadata.pkl             schema version, password, feature spec, counters
    model.pkl                trained VotingClassifier
    scaler.pkl               fitted StandardScaler
    authentic_data.npy       (N, D) authentic samples, oldest row first
    synthetic_data.npy       most recent generated negatives
    sample_meta.pkl          per-sample timestamp / source / context
    context_history.pkl      contexts seen for this user
    match_probabilities.pkl  recent genuine match scores
    history.pkl              enrollment / retrain / drift event log

Profiles written by the original single-file version carry only
``{'password': ...}`` in metadata.pkl. Those load as schema version 1 with the
legacy 2n feature layout, so they keep verifying against the model they were
actually fit on.
"""

import os
import pickle
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import config, features

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # older scikit-learn
    InconsistentVersionWarning = None


def _path(user_id, *parts):
    return os.path.join(config.USER_DATA_PATH, str(user_id), *parts)


def _read_pickle(path, default=None):
    """Load a pickle, or return ``default`` if it is missing or unreadable.

    A model pickled by a different scikit-learn version emits one
    InconsistentVersionWarning per estimator -- seven lines every time a
    profile is opened, which buries the program's own output. The mismatch is
    already handled: profiles record their schema version, and a stale model is
    replaced on the next retrain. So the warning is suppressed here and
    reported once, per profile, through ``sklearn_version_mismatch``.
    """
    global _saw_version_mismatch
    if not os.path.exists(path):
        return default
    try:
        with open(path, "rb") as handle, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = pickle.load(handle)
        if InconsistentVersionWarning is not None and any(
            issubclass(w.category, InconsistentVersionWarning) for w in caught
        ):
            _saw_version_mismatch = True
        return value
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError):
        return default


_saw_version_mismatch = False


def _write_pickle(path, value):
    with open(path, "wb") as handle:
        pickle.dump(value, handle)


@dataclass
class Profile:
    user_id: str
    password: str
    schema_version: int = config.SCHEMA_VERSION
    extended: bool = config.EXTENDED_FEATURES
    model_choice: int = 1

    model: object = None
    scaler: object = None

    authentic: np.ndarray = None
    synthetic: np.ndarray = None
    sample_meta: list = field(default_factory=list)

    match_probabilities: list = field(default_factory=list)
    context_history: list = field(default_factory=list)
    history: list = field(default_factory=list)
    # Rejected attempts that had the correct password. Used only to report
    # suspected drift; never fed to the model.
    recent_failures: list = field(default_factory=list)

    created_at: float = 0.0
    updated_at: float = 0.0
    samples_since_retrain: int = 0

    # Snapshot of the template at the last password-verified enroll/retrain.
    # Auto-adoption is bounded relative to this so the profile cannot be
    # walked away from verified-genuine typing (see adaptive.template_drift).
    anchor_centroid: np.ndarray = None
    anchor_spread: np.ndarray = None
    anchor_time: float = 0.0

    # True when the stored model was pickled by a different scikit-learn
    # version. Set on load; not persisted.
    sklearn_version_mismatch: bool = False

    # -- derived -------------------------------------------------------------
    @property
    def char_count(self):
        return len(self.password)

    @property
    def feature_dim(self):
        return features.feature_dim(self.char_count, self.extended)

    @property
    def is_legacy(self):
        return self.schema_version < config.SCHEMA_VERSION

    @property
    def sample_count(self):
        return 0 if self.authentic is None else len(self.authentic)

    def timestamps(self):
        """Per-sample capture times, or None when they are unknown (v1)."""
        if not self.sample_meta or len(self.sample_meta) != self.sample_count:
            return None
        return [entry.get("timestamp", 0.0) for entry in self.sample_meta]

    def auto_fraction(self):
        """Share of the profile that came from unsupervised auto-adoption.

        Reported in the status view. This is informational only -- the actual
        anti-poisoning bound is the template drift check in adaptive.py.
        """
        if not self.sample_meta:
            return 0.0
        auto = sum(1 for entry in self.sample_meta if entry.get("source") == "auto")
        return auto / len(self.sample_meta)

    def set_anchor(self):
        """Record the current template as the trusted reference point."""
        if self.authentic is None or len(self.authentic) == 0:
            return
        self.anchor_centroid = self.authentic.mean(axis=0)
        self.anchor_spread = np.maximum(self.authentic.std(axis=0), 1e-6)
        self.anchor_time = time.time()

    # -- mutation ------------------------------------------------------------
    def add_sample(self, vector, context=None, source="enroll", timestamp=None):
        """Append a sample, evicting the oldest once the window is full."""
        vector = np.asarray(vector, dtype=float).reshape(1, -1)
        if self.authentic is None or len(self.authentic) == 0:
            self.authentic = vector
        else:
            self.authentic = np.vstack([self.authentic, vector])

        self.sample_meta.append(
            {
                "timestamp": timestamp if timestamp is not None else time.time(),
                "source": source,
                "context": context.to_dict() if context is not None else None,
            }
        )
        if context is not None:
            self.context_history.append(context.to_dict())

        overflow = len(self.authentic) - config.MAX_AUTHENTIC_SAMPLES
        if overflow > 0:
            self.authentic = self.authentic[overflow:]
            self.sample_meta = self.sample_meta[overflow:]

    def log(self, event, **details):
        self.history.append({"event": event, "timestamp": time.time(), **details})

    def record_failure(self, vector, probability):
        """Remember a rejected attempt that supplied the correct password."""
        self.recent_failures.append(
            {
                "timestamp": time.time(),
                "probability": float(probability),
                "vector": np.asarray(vector, dtype=float),
            }
        )
        if len(self.recent_failures) > config.MAX_FAILURE_HISTORY:
            self.recent_failures = self.recent_failures[-config.MAX_FAILURE_HISTORY :]

    def record_probability(self, probability):
        self.match_probabilities.append(float(probability))
        if len(self.match_probabilities) > config.PROB_HISTORY_SIZE:
            self.match_probabilities = self.match_probabilities[
                -config.PROB_HISTORY_SIZE :
            ]


def exists(user_id):
    return os.path.isdir(_path(user_id))


def list_users():
    root = config.USER_DATA_PATH
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, "metadata.pkl"))
    )


def load(user_id):
    """Load a profile, or None when the user does not exist."""
    user_id = str(user_id)
    if not exists(user_id):
        return None

    global _saw_version_mismatch
    _saw_version_mismatch = False

    metadata = _read_pickle(_path(user_id, "metadata.pkl"))
    if not metadata or "password" not in metadata:
        return None

    version = metadata.get("schema_version", config.LEGACY_SCHEMA_VERSION)
    # v1 profiles predate the extended feature set; their model was fit on the
    # 2n-length vector, so they must keep assembling features that way.
    extended = metadata.get("extended", version >= config.SCHEMA_VERSION)

    authentic_path = _path(user_id, "authentic_data.npy")
    synthetic_path = _path(user_id, "synthetic_data.npy")

    profile = Profile(
        user_id=user_id,
        password=metadata["password"],
        schema_version=version,
        extended=bool(extended),
        model_choice=metadata.get("model_choice", 1),
        model=_read_pickle(_path(user_id, "model.pkl")),
        scaler=_read_pickle(_path(user_id, "scaler.pkl")),
        authentic=np.load(authentic_path) if os.path.exists(authentic_path) else None,
        synthetic=np.load(synthetic_path) if os.path.exists(synthetic_path) else None,
        sample_meta=_read_pickle(_path(user_id, "sample_meta.pkl"), []) or [],
        match_probabilities=_read_pickle(_path(user_id, "match_probabilities.pkl"), [])
        or [],
        context_history=_read_pickle(_path(user_id, "context_history.pkl"), []) or [],
        history=_read_pickle(_path(user_id, "history.pkl"), []) or [],
        recent_failures=_read_pickle(_path(user_id, "recent_failures.pkl"), []) or [],
        created_at=metadata.get("created_at", 0.0),
        updated_at=metadata.get("updated_at", 0.0),
        samples_since_retrain=metadata.get("samples_since_retrain", 0),
        anchor_centroid=metadata.get("anchor_centroid"),
        anchor_spread=metadata.get("anchor_spread"),
        anchor_time=metadata.get("anchor_time", 0.0),
    )

    if profile.model is None or profile.scaler is None:
        return None

    profile.sklearn_version_mismatch = _saw_version_mismatch
    return profile


def save(profile):
    """Write a profile to disk."""
    directory = _path(profile.user_id)
    os.makedirs(directory, exist_ok=True)

    now = time.time()
    if not profile.created_at:
        profile.created_at = now
    profile.updated_at = now

    _write_pickle(
        _path(profile.user_id, "metadata.pkl"),
        {
            "schema_version": profile.schema_version,
            "password": profile.password,
            "extended": profile.extended,
            "char_count": profile.char_count,
            "feature_dim": profile.feature_dim,
            "model_choice": profile.model_choice,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "samples_since_retrain": profile.samples_since_retrain,
            "anchor_centroid": profile.anchor_centroid,
            "anchor_spread": profile.anchor_spread,
            "anchor_time": profile.anchor_time,
        },
    )
    _write_pickle(_path(profile.user_id, "model.pkl"), profile.model)
    _write_pickle(_path(profile.user_id, "scaler.pkl"), profile.scaler)
    _write_pickle(_path(profile.user_id, "sample_meta.pkl"), profile.sample_meta)
    _write_pickle(
        _path(profile.user_id, "match_probabilities.pkl"), profile.match_probabilities
    )
    _write_pickle(_path(profile.user_id, "context_history.pkl"), profile.context_history)
    _write_pickle(_path(profile.user_id, "history.pkl"), profile.history)
    _write_pickle(_path(profile.user_id, "recent_failures.pkl"), profile.recent_failures)

    if profile.authentic is not None:
        np.save(_path(profile.user_id, "authentic_data.npy"), profile.authentic)
    if profile.synthetic is not None:
        np.save(_path(profile.user_id, "synthetic_data.npy"), profile.synthetic)
