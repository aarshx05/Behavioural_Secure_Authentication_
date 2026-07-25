"""Adaptive enrollment, verification, drift detection, and retraining.

Typing changes: a password typed for the first time this week is slow and
deliberate, and the same password six months later is muscle memory. A profile
frozen at enrollment steadily drifts away from its owner, so the false rejection
rate climbs until the user gives up. This module keeps the profile tracking the
user through three mechanisms:

1. A sliding window with recency weighting, so recent typing dominates the fit
   and samples older than the window fall out entirely.
2. Auto-adoption: verifications the model is very confident about are folded
   back into the profile, so ordinary day-to-day logins are the training data.
3. Drift detection, which compares the oldest and newest samples in the window
   and reports when the user's rhythm has moved far enough to justify an
   explicit retrain.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from . import config, features, models, risk, storage


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
def dynamic_threshold(profile):
    """Acceptance threshold derived from this user's own score history.

    The original computed ``mean + std`` over *successful* logins. Only scores
    above the threshold are ever recorded, so that mean is high by construction
    and adding a standard deviation pushed the bar above almost every future
    genuine attempt -- the threshold ratcheted upward until the real user was
    locked out.

    The bar belongs at the *lower* edge of the genuine distribution: accept
    anything within k standard deviations below the user's typical score, and
    clamp so it can never drift somewhere absurd.
    """
    probabilities = profile.match_probabilities
    if len(probabilities) < config.MIN_SCORES_FOR_DYNAMIC:
        return config.STATIC_THRESHOLD

    mean = float(np.mean(probabilities))
    std = float(np.std(probabilities))

    # Widen the interval when the history is short, so a handful of unusually
    # consistent logins cannot set a bar the user then struggles to clear.
    multiplier = config.THRESHOLD_STD_MULTIPLIER
    if len(probabilities) < config.PROB_HISTORY_SIZE:
        multiplier *= config.PROB_HISTORY_SIZE / len(probabilities)

    threshold = mean - multiplier * std
    return float(np.clip(threshold, config.MIN_THRESHOLD, config.MAX_THRESHOLD))


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------
@dataclass
class DriftReport:
    detected: bool = False
    magnitude: float = 0.0
    speed_change: float = 0.0
    message: str = ""
    top_features: list = field(default_factory=list)

    def describe(self):
        return self.message


def detect_drift(profile):
    """Compare the oldest and newest samples in the window.

    The shift is measured in standard deviations of the profile's own timing
    spread, so a user with naturally variable typing is not flagged for
    variation that is normal for them.
    """
    samples = profile.authentic
    if samples is None or len(samples) < config.DRIFT_MIN_SAMPLES:
        have = 0 if samples is None else len(samples)
        return DriftReport(
            message=f"Not enough history to assess drift ({have}/"
            f"{config.DRIFT_MIN_SAMPLES} samples)."
        )

    window = min(config.DRIFT_WINDOW, len(samples) // 2)
    oldest = samples[:window]
    newest = samples[-window:]

    core = slice(0, 2 * profile.char_count)  # total_time + hold + dd
    spread = np.maximum(samples[:, core].std(axis=0), 1e-6)
    shift = (newest[:, core].mean(axis=0) - oldest[:, core].mean(axis=0)) / spread

    magnitude = float(np.mean(np.abs(shift)))
    detected = magnitude > config.DRIFT_Z_THRESHOLD

    old_total = float(oldest[:, 0].mean())
    new_total = float(newest[:, 0].mean())
    speed_change = (
        (old_total - new_total) / old_total * 100.0 if old_total > 1e-9 else 0.0
    )

    names = features.describe(profile.char_count, profile.extended)[: 2 * profile.char_count]
    ranked = sorted(zip(names, shift), key=lambda pair: abs(pair[1]), reverse=True)[:3]

    direction = "faster" if speed_change > 0 else "slower"
    if detected:
        message = (
            f"Typing has drifted (mean shift {magnitude:.2f} sd); "
            f"you now type this password {abs(speed_change):.1f}% {direction} "
            f"than when the profile was built. A retrain is recommended."
        )
    else:
        message = (
            f"Typing is stable (mean shift {magnitude:.2f} sd, "
            f"{abs(speed_change):.1f}% {direction})."
        )

    return DriftReport(
        detected=detected,
        magnitude=magnitude,
        speed_change=speed_change,
        message=message,
        top_features=[(name, float(value)) for name, value in ranked],
    )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class FailureAnalysis:
    """What repeated rejections of the correct password suggest."""

    verdict: str = "none"  # none | drift | attack | inconclusive
    count: int = 0
    magnitude: float = 0.0
    cohesion: float = 0.0
    speed_change: float = 0.0
    message: str = ""

    @property
    def suggests_drift(self):
        return self.verdict == "drift"


def analyse_failures(profile):
    """Diagnose a run of rejected attempts that carried the correct password.

    Drift measured over stored samples cannot see this situation at all: once a
    user's typing moves far enough to be rejected, nothing new is adopted and
    that measure keeps reporting whatever the stale window says. The rejected
    attempts are the only remaining evidence, so they are read here.

    Two cases have to be told apart:

    * one person typing consistently differently -- the attempts cluster
      tightly around a new rhythm, which is genuine drift;
    * several different people guessing at a known password -- the attempts are
      scattered, which is an attack and must *not* be described to the user as
      "your typing changed".

    Nothing here feeds the model. Retraining still requires the password and
    freshly captured typing.
    """
    failures = profile.recent_failures
    if len(failures) < config.CONSECUTIVE_FAILURE_HINT:
        return FailureAnalysis(count=len(failures))

    if profile.authentic is None or len(profile.authentic) == 0:
        return FailureAnalysis(count=len(failures), verdict="inconclusive")

    vectors = np.array([f["vector"] for f in failures])
    if vectors.shape[1] != profile.authentic.shape[1]:
        return FailureAnalysis(count=len(failures), verdict="inconclusive")

    core = slice(0, 2 * profile.char_count)
    spread = np.maximum(profile.authentic[:, core].std(axis=0), 1e-6)

    shift = (vectors[:, core].mean(axis=0) - profile.authentic[:, core].mean(axis=0)) / spread
    magnitude = float(np.mean(np.abs(shift)))
    # How tightly the rejected attempts agree with each other.
    cohesion = float(np.mean(vectors[:, core].std(axis=0) / spread))

    template_total = float(profile.authentic[:, 0].mean())
    failure_total = float(vectors[:, 0].mean())
    speed_change = (
        (template_total - failure_total) / template_total * 100.0
        if template_total > 1e-9
        else 0.0
    )
    direction = "faster" if speed_change > 0 else "slower"

    result = FailureAnalysis(
        count=len(failures),
        magnitude=magnitude,
        cohesion=cohesion,
        speed_change=speed_change,
    )

    if magnitude <= config.DRIFT_Z_THRESHOLD:
        result.verdict = "inconclusive"
        result.message = (
            f"{len(failures)} recent rejections, but they sit close to your "
            f"stored profile. Try typing at your usual pace."
        )
    elif cohesion <= config.FAILURE_COHESION_MAX:
        result.verdict = "drift"
        result.message = (
            f"{len(failures)} rejected attempts used the correct password and "
            f"share a consistent rhythm {magnitude:.2f} sd from your profile "
            f"({abs(speed_change):.0f}% {direction}). Your typing has most "
            f"likely changed -- retrain (option 2) to catch the profile up."
        )
    else:
        result.verdict = "attack"
        result.message = (
            f"{len(failures)} rejected attempts used the correct password but "
            f"their rhythms are inconsistent (spread {cohesion:.2f} sd). That "
            f"looks like different people rather than your own typing "
            f"changing -- consider changing the password."
        )
    return result


def template_drift(profile):
    """How far the window centroid has moved from its trusted anchor.

    Measured in standard deviations of the anchor's own spread, so it is
    comparable across users and passwords. Returns 0.0 when no anchor exists.
    """
    if profile.anchor_centroid is None or profile.authentic is None:
        return 0.0
    if len(profile.authentic) == 0:
        return 0.0
    if profile.anchor_centroid.shape[0] != profile.authentic.shape[1]:
        return 0.0  # Layout changed (legacy upgrade); anchor no longer applies.

    core = slice(0, 2 * profile.char_count)
    shift = (
        profile.authentic.mean(axis=0)[core] - profile.anchor_centroid[core]
    ) / profile.anchor_spread[core]
    return float(np.mean(np.abs(shift)))


def fit_profile(profile, choice_train=None, now=None):
    """Regenerate negatives and refit the ensemble from the current window.

    Negatives are regenerated from scratch every time rather than accumulated.
    The original stacked freshly generated negatives onto the stored ones at
    every retrain, so the set grew without bound (10 samples became 2500 after
    two retrains) while most of it described typing the user had already moved
    away from.
    """
    choice = models.normalize_choice(
        profile.model_choice if choice_train is None else choice_train
    )
    negatives = models.generate_negatives(
        profile.authentic, profile.char_count, extended=profile.extended
    )
    model, scaler, info = models.train(
        profile.authentic,
        negatives,
        choice_train=choice,
        timestamps=profile.timestamps(),
        now=now,
    )

    profile.model = model
    profile.scaler = scaler
    profile.synthetic = negatives
    profile.model_choice = choice
    profile.samples_since_retrain = 0
    return info


def retrain(profile, samples, choice_train=None, source="retrain"):
    """Add ``samples`` to the profile and refit.

    ``samples`` is a sequence of ``(vector, context)`` pairs.
    """
    for vector, context in samples:
        profile.add_sample(vector, context=context, source=source)

    drift_before = detect_drift(profile)
    info = fit_profile(profile, choice_train=choice_train)

    # Past scores were produced by the previous model, so they no longer
    # describe this one; keeping them would skew the dynamic threshold.
    profile.match_probabilities = []

    # This retrain was gated on the password, so the resulting template is
    # trusted: re-anchor, which also releases any auto-adoption lockout.
    profile.set_anchor()
    # The profile now describes current typing, so the old rejections no longer
    # say anything about it.
    profile.recent_failures = []

    profile.log(
        source,
        added=len(samples),
        window=profile.sample_count,
        drift=drift_before.magnitude,
        **info,
    )
    return info, drift_before


# --------------------------------------------------------------------------
# Enrollment
# --------------------------------------------------------------------------
def enroll(user_id, password, samples, choice_train=1):
    """Build a fresh profile from enrollment samples."""
    profile = storage.Profile(
        user_id=str(user_id),
        password=password,
        schema_version=config.SCHEMA_VERSION,
        extended=config.EXTENDED_FEATURES,
        model_choice=models.normalize_choice(choice_train),
    )
    for vector, context in samples:
        profile.add_sample(vector, context=context, source="enroll")

    info = fit_profile(profile, choice_train=choice_train)
    profile.set_anchor()
    profile.log("enroll", samples=profile.sample_count, **info)
    return profile, info


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
@dataclass
class VerificationResult:
    authenticated: bool = False
    probability: float = 0.0
    base_threshold: float = 0.0
    required: float = 0.0
    assessment: object = None
    adopted: bool = False
    retrained: bool = False
    reason: str = ""
    adopt_bar: float = 0.0
    template_drift: float = 0.0
    lockout: str = ""
    failure_analysis: object = None


def verify(profile, vector, context):
    """Score a sample and decide, combining biometric and contextual evidence.

    Returns a :class:`VerificationResult`. The profile is mutated in place when
    the sample is adopted or triggers a retrain; the caller is responsible for
    persisting it.
    """
    probability = models.score(profile.model, profile.scaler, vector)
    base = dynamic_threshold(profile)
    assessment = risk.assess(context, profile.context_history)
    required = risk.required_probability(base, assessment)

    result = VerificationResult(
        probability=probability,
        base_threshold=base,
        required=required,
        assessment=assessment,
    )

    if config.RISK_BLOCK_ENABLED and assessment.is_high:
        result.reason = "blocked: contextual risk too high"
        return result

    if probability < required:
        result.reason = (
            f"typing pattern scored {probability:.3f}, below the required "
            f"{required:.3f}"
        )
        # The password was already checked by the caller, so this is a rejected
        # attempt by someone who knows it -- the only evidence available if the
        # user has drifted out of their own profile.
        profile.record_failure(vector, probability)
        result.failure_analysis = analyse_failures(profile)
        return result

    result.authenticated = True
    result.reason = "typing pattern matched"
    profile.record_probability(probability)
    # Getting in clears the run; "consecutive" is what carries the signal.
    profile.recent_failures = []

    _maybe_adapt(profile, vector, context, result)
    return result


def _maybe_adapt(profile, vector, context, result):
    """Fold a confident, low-risk verification back into the profile.

    Three independent guards keep this from becoming a poisoning channel:

    * the sample must clear an absolute confidence floor *and* beat the bar it
      was actually judged against by a margin, so merely squeaking past
      verification is not enough to become training data;
    * the context must look ordinary, so a login from an unrecognised device
      never teaches the model anything;
    * the template as a whole must still sit close to its last
      password-verified anchor, which bounds the slow walk attack that the
      per-sample checks alone cannot stop.
    """
    result.adopt_bar = max(
        config.AUTO_ADOPT_MIN_PROB, result.required + config.AUTO_ADOPT_MARGIN
    )
    result.template_drift = template_drift(profile)

    if not config.ADAPTIVE_LEARNING:
        return
    if profile.is_legacy:
        return  # v1 profiles have no per-sample metadata to extend.
    if result.probability < result.adopt_bar:
        return
    if result.assessment is not None and result.assessment.is_elevated:
        return
    if result.template_drift > config.MAX_TEMPLATE_DRIFT:
        result.lockout = (
            f"template has moved {result.template_drift:.2f} sd from its last "
            f"verified anchor; run a retrain to re-anchor before adaptive "
            f"learning resumes"
        )
        return

    profile.add_sample(vector, context=context, source="auto")
    profile.samples_since_retrain += 1
    result.adopted = True

    if profile.samples_since_retrain >= config.AUTO_RETRAIN_AFTER:
        info = fit_profile(profile)
        # Score history is deliberately *not* cleared here. An auto-retrain
        # refits on nearly the same window plus a few samples, so previous
        # scores still describe the model closely. Clearing would reset the
        # count on every fifth login, so the history could never reach the
        # length the dynamic threshold needs and the bar would stay pinned at
        # the static value forever. A manual retrain still clears, because
        # there the user has deliberately supplied new typing.
        profile.log("auto_retrain", window=profile.sample_count, **info)
        result.retrained = True


def status(profile):
    """Summary lines describing a profile's current state."""
    drift = detect_drift(profile)
    failures = analyse_failures(profile)
    sources = {}
    for entry in profile.sample_meta:
        sources[entry.get("source", "unknown")] = (
            sources.get(entry.get("source", "unknown"), 0) + 1
        )

    timestamps = profile.timestamps()
    if timestamps:
        oldest = time.strftime("%Y-%m-%d", time.localtime(min(timestamps)))
        newest = time.strftime("%Y-%m-%d", time.localtime(max(timestamps)))
        span = f"{oldest} to {newest}"
    else:
        span = "unknown (profile predates sample timestamps)"

    devices = {
        c.get("hostname", "unknown") for c in profile.context_history if c
    }

    return {
        "schema": f"v{profile.schema_version}"
        + (" (legacy)" if profile.is_legacy else ""),
        "features": f"{profile.feature_dim} "
        + ("extended" if profile.extended else "legacy v1"),
        "samples": f"{profile.sample_count}/{config.MAX_AUTHENTIC_SAMPLES}",
        "sample_span": span,
        "sources": ", ".join(f"{k}={v}" for k, v in sorted(sources.items())) or "n/a",
        "devices_seen": ", ".join(sorted(devices)) or "none recorded",
        "threshold": f"{dynamic_threshold(profile):.3f}",
        "recent_scores": len(profile.match_probabilities),
        "pending_retrain": profile.samples_since_retrain,
        "auto_adopted": f"{profile.auto_fraction() * 100:.0f}% of window",
        "template_drift": f"{template_drift(profile):.2f} sd from anchor "
        f"(limit {config.MAX_TEMPLATE_DRIFT})",
        "drift": drift.message,
        "rejections": failures.message
        or f"{failures.count} recent rejection(s) with the correct password",
    }
