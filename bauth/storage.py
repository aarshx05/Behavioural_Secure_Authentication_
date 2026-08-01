"""Profile persistence."""

from copy import deepcopy
import hmac
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import config, passwords

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # older scikit-learn
    InconsistentVersionWarning = None


def _path(user_id, *parts):
    return os.path.join(config.USER_DATA_PATH, str(user_id), *parts)


def _read_pickle(path, default=None):
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


def _write_pickle(path, value):
    with open(path, "wb") as handle:
        pickle.dump(value, handle)


_saw_version_mismatch = False


def _robust_center_scale(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    center = np.median(X, axis=0)
    scale = np.median(np.abs(X - center), axis=0)
    return center, np.maximum(scale, 1e-6)


@dataclass
class Profile:
    user_id: str
    password_hash: dict = None
    password_length: int = 0
    schema_version: int = config.SCHEMA_VERSION
    extended: bool = config.EXTENDED_FEATURES
    model_choice: int = 1

    model: object = None
    scaler: object = None

    active_samples: np.ndarray = None
    anchor_samples: np.ndarray = None
    synthetic: np.ndarray = None
    synthetic_meta: list = field(default_factory=list)
    sample_meta: list = field(default_factory=list)

    match_probabilities: list = field(default_factory=list)
    context_history: list = field(default_factory=list)
    history: list = field(default_factory=list)
    recent_failures: list = field(default_factory=list)
    quarantine: list = field(default_factory=list)
    versions: list = field(default_factory=list)
    recent_sample_fingerprints: list = field(default_factory=list)

    created_at: float = 0.0
    updated_at: float = 0.0
    samples_since_retrain: int = 0
    profile_version: int = 1

    anchor_centroid: np.ndarray = None
    anchor_spread: np.ndarray = None
    anchor_time: float = 0.0
    active_centroid: np.ndarray = None
    active_scale: np.ndarray = None
    thresholds: dict = field(default_factory=dict)
    fusion_weights: dict = field(default_factory=dict)
    adaptation_policy: str = config.DEFAULT_ADAPTATION_POLICY
    policy_state: dict = field(default_factory=dict)

    sklearn_version_mismatch: bool = False
    legacy_plaintext: str = None

    def set_password(self, password):
        self.password_hash = passwords.hash_password(password)
        self.password_length = len(password)
        self.legacy_plaintext = None

    def check_password(self, candidate):
        if self.password_hash:
            return passwords.verify_password(candidate, self.password_hash)

        if self.legacy_plaintext is not None:
            if not hmac.compare_digest(candidate, self.legacy_plaintext):
                return False
            self.set_password(candidate)
            self.log("password_hashed", scheme=passwords.describe(self.password_hash))
            return True

        return False

    @property
    def password_is_plaintext(self):
        return self.password_hash is None and self.legacy_plaintext is not None

    @property
    def char_count(self):
        return self.password_length

    @property
    def feature_dim(self):
        from . import features

        return features.feature_dim(self.char_count, self.extended)

    @property
    def is_legacy(self):
        return self.schema_version < config.SCHEMA_VERSION

    @property
    def authentic(self):
        return self.active_samples

    @authentic.setter
    def authentic(self, value):
        self.active_samples = value
        self.refresh_active_stats()

    @property
    def sample_count(self):
        return 0 if self.active_samples is None else len(self.active_samples)

    @property
    def anchor_scale(self):
        return self.anchor_spread

    def timestamps(self):
        if not self.sample_meta or len(self.sample_meta) != self.sample_count:
            return None
        return [entry.get("timestamp", 0.0) for entry in self.sample_meta]

    def auto_fraction(self):
        if not self.sample_meta:
            return 0.0
        auto = sum(1 for entry in self.sample_meta if entry.get("source") == "auto")
        return auto / len(self.sample_meta)

    def refresh_active_stats(self):
        if self.active_samples is None or len(self.active_samples) == 0:
            self.active_centroid = None
            self.active_scale = None
            return
        self.active_centroid, self.active_scale = _robust_center_scale(self.active_samples)

    def set_anchor(self, samples=None, touch_time=True):
        source = self.active_samples if samples is None else np.asarray(samples, dtype=float)
        if source is None or len(source) == 0:
            return
        self.anchor_samples = np.array(source, dtype=float, copy=True)
        self.anchor_centroid, self.anchor_spread = _robust_center_scale(self.anchor_samples)
        if touch_time:
            self.anchor_time = time.time()

    def add_sample(self, vector, context=None, source="enroll", timestamp=None, metadata=None):
        vector = np.asarray(vector, dtype=float).reshape(1, -1)
        if self.active_samples is None or len(self.active_samples) == 0:
            self.active_samples = vector
        else:
            self.active_samples = np.vstack([self.active_samples, vector])

        context_dict = None
        if context is not None:
            if isinstance(context, dict):
                context_dict = context
            else:
                context_dict = context.to_dict()

        entry = {
            "timestamp": timestamp if timestamp is not None else time.time(),
            "source": source,
            "context": context_dict,
        }
        if metadata:
            entry.update(deepcopy(metadata))
        self.sample_meta.append(entry)
        if context_dict is not None:
            self.context_history.append(context_dict)

        overflow = len(self.active_samples) - config.MAX_AUTHENTIC_SAMPLES
        if overflow > 0:
            self.active_samples = self.active_samples[overflow:]
            self.sample_meta = self.sample_meta[overflow:]

        self.refresh_active_stats()

    def log(self, event, **details):
        self.history.append({"event": event, "timestamp": time.time(), **details})

    def record_failure(self, vector, probability, disagreement=0.0):
        self.recent_failures.append(
            {
                "timestamp": time.time(),
                "probability": float(probability),
                "disagreement": float(disagreement),
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

    def snapshot_version(self, reason, shift=None, promoted=None):
        self.profile_version += 1
        self.versions.append(
            {
                "version": self.profile_version,
                "timestamp": time.time(),
                "reason": reason,
                "shift": float(shift or 0.0),
                "promoted": int(promoted or 0),
                "active_centroid": None
                if self.active_centroid is None
                else np.array(self.active_centroid, copy=True),
                "active_samples": None
                if self.active_samples is None
                else np.array(self.active_samples, copy=True),
                "sample_meta": deepcopy(self.sample_meta),
                "context_history": deepcopy(self.context_history),
                "match_probabilities": list(self.match_probabilities),
                "thresholds": deepcopy(self.thresholds),
                "fusion_weights": deepcopy(self.fusion_weights),
                "adaptation_policy": self.adaptation_policy,
                "policy_state": deepcopy(self.policy_state),
                "synthetic": None if self.synthetic is None else np.array(self.synthetic, copy=True),
                "synthetic_meta": deepcopy(self.synthetic_meta),
                "model": deepcopy(self.model),
                "scaler": deepcopy(self.scaler),
            }
        )
        if len(self.versions) > config.MAX_PROFILE_VERSIONS:
            self.versions = self.versions[-config.MAX_PROFILE_VERSIONS :]

    def has_sample_fingerprint(self, fingerprint):
        return any(entry.get("fingerprint") == fingerprint for entry in self.recent_sample_fingerprints)

    def record_sample_fingerprint(self, fingerprint, timestamp=None, source="verify"):
        if not fingerprint:
            return
        self.recent_sample_fingerprints.append(
            {
                "fingerprint": str(fingerprint),
                "timestamp": time.time() if timestamp is None else float(timestamp),
                "source": source,
            }
        )
        if len(self.recent_sample_fingerprints) > config.MAX_FINGERPRINT_HISTORY:
            self.recent_sample_fingerprints = self.recent_sample_fingerprints[
                -config.MAX_FINGERPRINT_HISTORY :
            ]

    def rollback(self, version_id):
        for entry in reversed(self.versions):
            if entry.get("version") != int(version_id):
                continue
            self.active_samples = None if entry.get("active_samples") is None else np.array(
                entry["active_samples"], copy=True
            )
            self.sample_meta = deepcopy(entry.get("sample_meta", []))
            self.context_history = deepcopy(entry.get("context_history", []))
            self.match_probabilities = list(entry.get("match_probabilities", []))
            self.thresholds = deepcopy(entry.get("thresholds", {}))
            self.fusion_weights = deepcopy(entry.get("fusion_weights", {}))
            self.adaptation_policy = entry.get("adaptation_policy", config.DEFAULT_ADAPTATION_POLICY)
            self.policy_state = deepcopy(entry.get("policy_state", {}))
            self.synthetic = None if entry.get("synthetic") is None else np.array(
                entry["synthetic"], copy=True
            )
            self.synthetic_meta = deepcopy(entry.get("synthetic_meta", []))
            self.model = deepcopy(entry.get("model"))
            self.scaler = deepcopy(entry.get("scaler"))
            self.quarantine = []
            self.recent_failures = []
            self.refresh_active_stats()
            self.log("rollback", restored_version=int(version_id))
            return True
        return False


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
    user_id = str(user_id)
    if not exists(user_id):
        return None

    global _saw_version_mismatch
    _saw_version_mismatch = False

    metadata = _read_pickle(_path(user_id, "metadata.pkl"))
    if not metadata:
        return None

    password_hash = metadata.get("password_hash")
    legacy_plaintext = None if password_hash else metadata.get("password")
    if password_hash is None and legacy_plaintext is None:
        return None

    password_length = metadata.get("password_length", len(legacy_plaintext or ""))
    version = metadata.get("schema_version", config.LEGACY_SCHEMA_VERSION)
    extended = metadata.get("extended", version >= config.SCHEMA_VERSION)

    active_path = _path(user_id, "authentic_data.npy")
    anchor_path = _path(user_id, "anchor_data.npy")
    synthetic_path = _path(user_id, "synthetic_data.npy")

    profile = Profile(
        user_id=user_id,
        password_hash=password_hash,
        password_length=password_length,
        legacy_plaintext=legacy_plaintext,
        schema_version=version,
        extended=bool(extended),
        model_choice=metadata.get("model_choice", 1),
        model=_read_pickle(_path(user_id, "model.pkl")),
        scaler=_read_pickle(_path(user_id, "scaler.pkl")),
        active_samples=np.load(active_path) if os.path.exists(active_path) else None,
        anchor_samples=np.load(anchor_path) if os.path.exists(anchor_path) else None,
        synthetic=np.load(synthetic_path) if os.path.exists(synthetic_path) else None,
        synthetic_meta=_read_pickle(_path(user_id, "synthetic_meta.pkl"), []) or [],
        sample_meta=_read_pickle(_path(user_id, "sample_meta.pkl"), []) or [],
        match_probabilities=_read_pickle(_path(user_id, "match_probabilities.pkl"), [])
        or [],
        context_history=_read_pickle(_path(user_id, "context_history.pkl"), []) or [],
        history=_read_pickle(_path(user_id, "history.pkl"), []) or [],
        recent_failures=_read_pickle(_path(user_id, "recent_failures.pkl"), []) or [],
        quarantine=_read_pickle(_path(user_id, "quarantine.pkl"), []) or [],
        versions=_read_pickle(_path(user_id, "versions.pkl"), []) or [],
        recent_sample_fingerprints=_read_pickle(
            _path(user_id, "recent_sample_fingerprints.pkl"), []
        )
        or [],
        created_at=metadata.get("created_at", 0.0),
        updated_at=metadata.get("updated_at", 0.0),
        samples_since_retrain=metadata.get("samples_since_retrain", 0),
        profile_version=metadata.get("profile_version", 1),
        anchor_centroid=metadata.get("anchor_centroid"),
        anchor_spread=metadata.get("anchor_spread"),
        anchor_time=metadata.get("anchor_time", 0.0),
        active_centroid=metadata.get("active_centroid"),
        active_scale=metadata.get("active_scale"),
        thresholds=metadata.get("thresholds", {}) or {},
        fusion_weights=metadata.get("fusion_weights", {}) or {},
        adaptation_policy=metadata.get("adaptation_policy", config.DEFAULT_ADAPTATION_POLICY),
        policy_state=metadata.get("policy_state", {}) or {},
    )

    if profile.model is None:
        return None
    if profile.scaler is None and not getattr(profile.model, "self_contained", False):
        return None

    if profile.active_samples is not None:
        profile.refresh_active_stats()
    if profile.anchor_samples is None and profile.active_samples is not None:
        profile.set_anchor(samples=profile.active_samples, touch_time=False)
    elif profile.anchor_centroid is None and profile.anchor_samples is not None:
        profile.set_anchor(samples=profile.anchor_samples, touch_time=False)

    profile.sklearn_version_mismatch = _saw_version_mismatch
    return profile


def save(profile):
    directory = _path(profile.user_id)
    os.makedirs(directory, exist_ok=True)

    now = time.time()
    if not profile.created_at:
        profile.created_at = now
    profile.updated_at = now
    profile.refresh_active_stats()
    if profile.anchor_samples is not None and profile.anchor_centroid is None:
        profile.set_anchor(samples=profile.anchor_samples, touch_time=False)

    _write_pickle(
        _path(profile.user_id, "metadata.pkl"),
        {
            "schema_version": profile.schema_version,
            "password_hash": profile.password_hash,
            "password_length": profile.password_length,
            **(
                {"password": profile.legacy_plaintext}
                if profile.password_hash is None and profile.legacy_plaintext is not None
                else {}
            ),
            "extended": profile.extended,
            "char_count": profile.char_count,
            "feature_dim": profile.feature_dim,
            "model_choice": profile.model_choice,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "samples_since_retrain": profile.samples_since_retrain,
            "profile_version": profile.profile_version,
            "anchor_centroid": profile.anchor_centroid,
            "anchor_spread": profile.anchor_spread,
            "anchor_time": profile.anchor_time,
            "active_centroid": profile.active_centroid,
            "active_scale": profile.active_scale,
            "thresholds": profile.thresholds,
            "fusion_weights": profile.fusion_weights,
            "adaptation_policy": profile.adaptation_policy,
            "policy_state": profile.policy_state,
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
    _write_pickle(_path(profile.user_id, "quarantine.pkl"), profile.quarantine)
    _write_pickle(_path(profile.user_id, "versions.pkl"), profile.versions)
    _write_pickle(_path(profile.user_id, "synthetic_meta.pkl"), profile.synthetic_meta)
    _write_pickle(
        _path(profile.user_id, "recent_sample_fingerprints.pkl"),
        profile.recent_sample_fingerprints,
    )

    if profile.active_samples is not None:
        np.save(_path(profile.user_id, "authentic_data.npy"), profile.active_samples)
    if profile.anchor_samples is not None:
        np.save(_path(profile.user_id, "anchor_data.npy"), profile.anchor_samples)
    if profile.synthetic is not None:
        np.save(_path(profile.user_id, "synthetic_data.npy"), profile.synthetic)
