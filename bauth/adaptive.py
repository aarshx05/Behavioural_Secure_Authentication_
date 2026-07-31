"""Adaptive enrollment, verification, quarantine, and anchor-bounded updates."""

import time
from dataclasses import dataclass, field

import numpy as np

from . import config, features, models, policies, quality, risk, storage


def dynamic_threshold(profile):
    """Authentication threshold selected during the last profile fit."""
    return float(profile.thresholds.get("auth", config.AUTH_THRESHOLD_FLOOR))


@dataclass
class DriftReport:
    detected: bool = False
    magnitude: float = 0.0
    speed_change: float = 0.0
    state: str = "stable"
    message: str = ""
    recommendation: str = ""
    top_features: list = field(default_factory=list)

    def describe(self):
        return self.message


@dataclass
class FailureAnalysis:
    verdict: str = "none"
    count: int = 0
    magnitude: float = 0.0
    cohesion: float = 0.0
    speed_change: float = 0.0
    disagreement: float = 0.0
    message: str = ""

    @property
    def suggests_drift(self):
        return self.verdict == "drift"


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
    lockout: str = ""
    failure_analysis: object = None
    disagreement: float = 0.0
    anchor_distance: float = 0.0
    detector_scores: dict = field(default_factory=dict)
    quarantined: bool = False
    quality_score: float = 1.0
    quality_flags: list = field(default_factory=list)
    sample_fingerprint: str = ""
    trust_score: float = 0.0


@dataclass
class AdaptationRuntime:
    profile: object
    vector: np.ndarray
    context: object
    result: VerificationResult
    timestamp: float

    @property
    def thresholds(self):
        return self.profile.thresholds or {}

    @property
    def update_threshold(self):
        return float(self.thresholds.get("update", config.UPDATE_THRESHOLD_FLOOR))

    @property
    def disagreement_limit(self):
        return float(self.thresholds.get("disagreement", config.DISAGREEMENT_LIMIT))

    @property
    def risk_limit(self):
        return float(self.thresholds.get("risk", config.UPDATE_RISK_LIMIT))

    @property
    def anchor_limit(self):
        return float(self.thresholds.get("anchor_candidate", config.ANCHOR_CANDIDATE_LIMIT))

    @property
    def risk_score(self):
        if self.result.assessment is None:
            return 0.0
        return float(self.result.assessment.score)

    def common_gate(
        self,
        *,
        require_update_threshold=False,
        require_disagreement=False,
        require_context=False,
        require_anchor=False,
        require_quality=False,
        require_replay_guard=False,
    ):
        if require_update_threshold and self.result.probability < self.update_threshold:
            self.result.lockout = "accepted for authentication, but below the adaptation threshold"
            return False
        if require_disagreement and self.result.disagreement > self.disagreement_limit:
            self.result.lockout = "classifier disagreement too high for adaptation"
            return False
        if require_context and self.risk_score > self.risk_limit:
            self.result.lockout = "context risk too high for adaptation"
            return False
        if require_anchor and self.result.anchor_distance > self.anchor_limit:
            self.result.lockout = "sample is too far from the trusted anchor"
            return False
        if require_quality and self.result.quality_score < config.QUALITY_UPDATE_FLOOR:
            self.result.lockout = "sample quality too low for adaptation"
            return False
        if require_replay_guard and "replayed-or-duplicate-sample" in self.result.quality_flags:
            self.result.lockout = "replayed samples can authenticate but are never adapted"
            return False
        return True

    def _entry(self):
        self.result.trust_score = _trust_score(self.result)
        return {
            "features": np.asarray(self.vector, dtype=float),
            "score": float(self.result.probability),
            "disagreement": float(self.result.disagreement),
            "quality_score": float(self.result.quality_score),
            "quality_flags": list(self.result.quality_flags),
            "context_risk": float(self.risk_score),
            "anchor_distance": float(self.result.anchor_distance),
            "trust_score": float(self.result.trust_score),
            "timestamp": float(self.timestamp),
            "context": None if self.context is None else self.context.to_dict(),
            "sample_fingerprint": self.result.sample_fingerprint,
        }

    def queue_quarantine(self):
        self.profile.quarantine.append(self._entry())
        if len(self.profile.quarantine) > config.QUARANTINE_MAX_SAMPLES:
            self.profile.quarantine = self.profile.quarantine[-config.QUARANTINE_MAX_SAMPLES :]
        self.result.quarantined = True

    def maybe_promote_quarantine(self):
        _maybe_promote_quarantine(self.profile, self.result)

    def promote_immediately(self, reason, bounded=False, immediate_refit=True):
        entry = self._entry()
        sample = entry["features"]
        if bounded:
            sample = _bounded_vector(self.profile, sample, entry["trust_score"])
        allowed, message = _promotion_allowed(self.profile, [entry["features"]], [sample])
        if not allowed:
            self.result.lockout = message
            return False
        promoted = _promote_samples(
            self.profile,
            [
                {
                    **entry,
                    "features": sample,
                }
            ],
            reason=reason,
            refit=immediate_refit,
        )
        self.result.adopted = promoted
        self.result.retrained = promoted and immediate_refit
        return promoted


def _core_slice(profile):
    return slice(0, 2 * profile.char_count)


def _anchor_scale(profile):
    if profile.anchor_scale is None:
        return None
    return np.maximum(np.asarray(profile.anchor_scale, dtype=float), 1e-6)


def _anchor_shift(center, profile):
    if center is None or profile.anchor_centroid is None or profile.anchor_scale is None:
        return 0.0
    core = _core_slice(profile)
    scale = _anchor_scale(profile)[core]
    delta = np.abs(np.asarray(center, dtype=float)[core] - profile.anchor_centroid[core])
    return float(np.mean(delta / scale))


def template_drift(profile):
    return _anchor_shift(profile.active_centroid, profile)


def _context_continuity(profile, window):
    recent = [c for c in profile.context_history[-window:] if c]
    if len(recent) < 2:
        return 1.0
    fingerprints = [risk.from_dict(entry).device_fingerprint for entry in recent]
    dominant = max(fingerprints.count(value) for value in set(fingerprints))
    return float(dominant / len(fingerprints))


def _classify_drift_state(profile, magnitude, anchor_move, score_delta, consistency, failures):
    if failures.verdict == "attack" and (
        anchor_move >= config.MAX_TEMPLATE_DRIFT * config.LIKELY_POISONING_ANCHOR_FRACTION
        or failures.disagreement > config.DISAGREEMENT_LIMIT * 1.5
    ):
        return (
            "likely_poisoning",
            "Attack-like failures near the anchor budget; freeze adaptation and roll back if this persists.",
        )
    if failures.verdict == "attack" or consistency > config.HETEROGENEOUS_CLUSTER_THRESHOLD:
        return (
            "suspicious_heterogeneous_shift",
            "Recent samples are inconsistent with one stable user change; keep adaptation quarantined.",
        )
    if magnitude >= config.STRONG_DRIFT_Z_THRESHOLD and score_delta < -0.05:
        return (
            "strong_consistent_drift",
            "Typing appears to have changed consistently; gather verified samples before re-anchoring.",
        )
    if magnitude >= config.DRIFT_Z_THRESHOLD:
        return (
            "mild_drift",
            "Typing is moving, but not yet far enough to justify a full re-anchor.",
        )
    return ("stable", "No meaningful drift signal is present.")


def detect_drift(profile):
    samples = profile.active_samples
    if samples is None or len(samples) < config.DRIFT_MIN_SAMPLES:
        have = 0 if samples is None else len(samples)
        return DriftReport(
            state="insufficient_data",
            message=f"Not enough history to assess drift ({have}/{config.DRIFT_MIN_SAMPLES} samples)."
        )

    window = min(config.DRIFT_WINDOW, len(samples) // 2)
    oldest = samples[:window]
    newest = samples[-window:]
    core = _core_slice(profile)
    spread = np.maximum(profile.active_scale[core] if profile.active_scale is not None else samples[:, core].std(axis=0), 1e-6)
    oldest_center = np.median(oldest[:, core], axis=0)
    newest_center = np.median(newest[:, core], axis=0)
    shift = (newest_center - oldest_center) / spread

    magnitude = float(np.mean(np.abs(shift)))
    detected = magnitude > config.DRIFT_Z_THRESHOLD

    old_total = float(oldest[:, 0].mean())
    new_total = float(newest[:, 0].mean())
    speed_change = (
        (old_total - new_total) / old_total * 100.0 if old_total > 1e-9 else 0.0
    )
    direction = "faster" if speed_change > 0 else "slower"
    names = features.describe(profile.char_count, profile.extended)[: 2 * profile.char_count]
    ranked = sorted(zip(names, shift), key=lambda pair: abs(pair[1]), reverse=True)[:3]

    anchor_move = template_drift(profile)
    consistency = _candidate_consistency(profile, newest)
    score_delta = 0.0
    if len(profile.match_probabilities) >= 4:
        scores = np.asarray(profile.match_probabilities, dtype=float)
        chunk = max(1, len(scores) // 3)
        score_delta = float(np.mean(scores[-chunk:]) - np.mean(scores[:chunk]))
    failures = analyse_failures(profile)
    state, recommendation = _classify_drift_state(
        profile,
        magnitude,
        anchor_move,
        score_delta,
        consistency,
        failures,
    )
    continuity = _context_continuity(profile, max(window, 3))
    if detected:
        message = (
            f"Typing has drifted ({state.replace('_', ' ')}; mean shift {magnitude:.2f} sd, "
            f"anchor shift {anchor_move:.2f} sd, context continuity {continuity:.2f}); "
            f"you now type this password {abs(speed_change):.1f}% {direction} than when the profile was built."
        )
    else:
        message = (
            f"Typing is stable (mean shift {magnitude:.2f} sd, anchor shift {anchor_move:.2f} sd, "
            f"{abs(speed_change):.1f}% {direction}, context continuity {continuity:.2f})."
        )

    return DriftReport(
        detected=detected,
        magnitude=magnitude,
        speed_change=speed_change,
        state=state,
        message=message,
        recommendation=recommendation,
        top_features=[(name, float(value)) for name, value in ranked],
    )


def analyse_failures(profile):
    failures = profile.recent_failures
    if len(failures) < config.CONSECUTIVE_FAILURE_HINT:
        return FailureAnalysis(count=len(failures))
    if profile.active_samples is None or len(profile.active_samples) == 0:
        return FailureAnalysis(count=len(failures), verdict="inconclusive")

    vectors = np.array([f["vector"] for f in failures])
    if vectors.shape[1] != profile.active_samples.shape[1]:
        return FailureAnalysis(count=len(failures), verdict="inconclusive")

    core = _core_slice(profile)
    spread = np.maximum(profile.active_samples[:, core].std(axis=0), 1e-6)
    shift = (vectors[:, core].mean(axis=0) - profile.active_samples[:, core].mean(axis=0)) / spread
    magnitude = float(np.mean(np.abs(shift)))
    cohesion = float(np.mean(vectors[:, core].std(axis=0) / spread))
    disagreement = float(np.mean([f.get("disagreement", 0.0) for f in failures]))

    template_total = float(profile.active_samples[:, 0].mean())
    failure_total = float(vectors[:, 0].mean())
    speed_change = (
        (template_total - failure_total) / template_total * 100.0 if template_total > 1e-9 else 0.0
    )
    direction = "faster" if speed_change > 0 else "slower"

    result = FailureAnalysis(
        count=len(failures),
        magnitude=magnitude,
        cohesion=cohesion,
        speed_change=speed_change,
        disagreement=disagreement,
    )

    if magnitude <= config.DRIFT_Z_THRESHOLD:
        result.verdict = "inconclusive"
        result.message = (
            f"{len(failures)} recent rejections, but they sit close to your active profile. "
            f"Try typing at your usual pace."
        )
    elif cohesion <= config.FAILURE_COHESION_MAX and disagreement <= config.DISAGREEMENT_LIMIT:
        result.verdict = "drift"
        result.message = (
            f"{len(failures)} rejected attempts used the correct password and share a consistent rhythm "
            f"{magnitude:.2f} sd from your profile ({abs(speed_change):.0f}% {direction}). "
            f"Your typing has most likely changed."
        )
    else:
        result.verdict = "attack"
        result.message = (
            f"{len(failures)} rejected attempts used the correct password but their rhythms are inconsistent "
            f"(spread {cohesion:.2f} sd, disagreement {disagreement:.2f}). That looks more like an attack "
            f"than a stable change in your typing."
        )
    return result


def _best_threshold(genuine, impostor):
    values = np.unique(np.concatenate([genuine, impostor]))
    best_threshold = config.AUTH_THRESHOLD_FLOOR
    best_score = -1.0
    for threshold in values:
        tpr = float(np.mean(genuine >= threshold))
        fpr = float(np.mean(impostor >= threshold))
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _calibrate_thresholds(model, positives, negatives):
    genuine = model.analyse(positives)
    impostor = model.analyse(negatives)
    count = max(len(positives), 1)

    def shrink(empirical, prior, scale=20):
        weight = min(1.0, count / float(scale))
        return float(weight * empirical + (1.0 - weight) * prior)

    auth = _best_threshold(genuine["fused"], impostor["fused"])
    auth = shrink(float(np.clip(auth, config.AUTH_THRESHOLD_FLOOR, 0.98)), config.AUTH_THRESHOLD_FLOOR)
    update = max(
        auth + config.UPDATE_THRESHOLD_MARGIN,
        float(np.quantile(genuine["fused"], 0.50)),
        config.UPDATE_THRESHOLD_FLOOR,
    )
    update = shrink(float(np.clip(update, auth, 0.995)), config.UPDATE_THRESHOLD_FLOOR)
    disagreement = shrink(
        float(np.clip(np.quantile(genuine["disagreement"], 0.85), 0.03, 0.25)),
        config.DISAGREEMENT_LIMIT,
    )
    anchor_candidate = shrink(
        float(max(config.ANCHOR_CANDIDATE_LIMIT, np.quantile(genuine["anchor_distance"], 0.90))),
        config.ANCHOR_CANDIDATE_LIMIT,
    )

    return {
        "auth": auth,
        "update": update,
        "disagreement": disagreement,
        "risk": config.UPDATE_RISK_LIMIT,
        "anchor_candidate": anchor_candidate,
        "anchor_total": config.MAX_TEMPLATE_DRIFT,
        "promotion_consistency": config.QUARANTINE_CONSISTENCY_LIMIT,
        "drift_warning": config.DRIFT_Z_THRESHOLD,
        "attack_warning": config.FAILURE_COHESION_MAX,
    }


def fit_profile(profile, choice_train=None, now=None):
    choice = models.normalize_choice(
        profile.model_choice if choice_train is None else choice_train
    )
    negatives, negative_meta = models.generate_negatives(
        profile.active_samples,
        profile.char_count,
        extended=profile.extended,
        return_metadata=True,
    )
    model, scaler, info = models.train(
        profile.active_samples,
        negatives,
        choice_train=choice,
        timestamps=profile.timestamps(),
        now=now,
    )

    profile.model = model
    profile.scaler = scaler
    profile.synthetic = negatives
    profile.synthetic_meta = negative_meta
    profile.model_choice = choice
    profile.samples_since_retrain = 0
    profile.refresh_active_stats()
    if profile.anchor_samples is None:
        profile.set_anchor(samples=profile.active_samples)

    profile.thresholds = _calibrate_thresholds(model, profile.active_samples, negatives)
    profile.fusion_weights = info.get("fusion_weights", {})
    generator_counts = {}
    for entry in negative_meta:
        name = entry.get("generator", "unknown")
        generator_counts[name] = generator_counts.get(name, 0) + 1
    info["negative_generators"] = generator_counts
    return info


def _candidate_consistency(profile, candidates):
    matrix = np.atleast_2d(np.asarray(candidates, dtype=float))
    center = np.median(matrix, axis=0)
    scale = profile.active_scale
    if scale is None:
        scale = np.ones(matrix.shape[1], dtype=float)
    scale = np.maximum(scale, 1e-6)
    return float(np.mean(np.abs(matrix - center) / scale))


def _bounded_vector(profile, vector, trust_score):
    vector = np.asarray(vector, dtype=float)
    if (
        profile.active_centroid is None
        or profile.anchor_scale is None
        or trust_score <= 0.0
    ):
        return vector
    if profile.extended:
        current = features.decompose(profile.active_centroid, profile.char_count, True)
        target = features.decompose(vector, profile.char_count, True)
        scale = features.decompose(profile.anchor_scale, profile.char_count, True)
        bounded = []
        alpha = config.PROMOTION_ALPHA * float(np.clip(trust_score, 0.0, 1.0))
        for current_part, target_part, scale_part in zip(current, target, scale):
            if len(target_part) == 0:
                bounded.append(target_part)
                continue
            step = np.maximum(np.asarray(scale_part, dtype=float), 1e-6)
            limit = step * config.MAX_PROMOTION_FEATURE_STEP
            delta = np.asarray(target_part, dtype=float) - np.asarray(current_part, dtype=float)
            bounded.append(np.asarray(current_part, dtype=float) + alpha * np.clip(delta, -limit, limit))
        return features.assemble(*bounded, extended=True)
    delta = vector - profile.active_centroid
    scale = np.maximum(np.asarray(profile.anchor_scale, dtype=float), 1e-6)
    clipped = np.clip(delta, -scale * config.MAX_PROMOTION_FEATURE_STEP, scale * config.MAX_PROMOTION_FEATURE_STEP)
    return profile.active_centroid + config.PROMOTION_ALPHA * float(np.clip(trust_score, 0.0, 1.0)) * clipped


def _promotion_allowed(profile, raw_candidates, promoted_candidates):
    raw_matrix = np.atleast_2d(np.asarray(raw_candidates, dtype=float))
    matrix = np.atleast_2d(np.asarray(promoted_candidates, dtype=float))
    if len(matrix) == 0:
        return False, "no candidates to promote"

    consistency = _candidate_consistency(profile, raw_matrix)
    consistency_limit = float(
        profile.thresholds.get("promotion_consistency", config.QUARANTINE_CONSISTENCY_LIMIT)
    )
    if consistency > consistency_limit:
        return False, f"quarantine cluster is too loose ({consistency:.2f} sd)"

    proposed = np.vstack([profile.active_samples, matrix])
    proposed_center = np.median(proposed, axis=0)
    total_shift = _anchor_shift(proposed_center, profile)
    if total_shift > profile.thresholds.get("anchor_total", config.MAX_TEMPLATE_DRIFT):
        return False, f"promotion would move the active profile {total_shift:.2f} sd from the anchor"

    scale = _anchor_scale(profile)
    if scale is not None and profile.active_centroid is not None:
        core = _core_slice(profile)
        step = np.abs(proposed_center[core] - profile.active_centroid[core]) / scale[core]
        if float(np.mean(step)) > config.MAX_PROMOTION_SHIFT:
            return False, "promotion step is too large"
        feature_shift = np.abs(proposed_center[core] - profile.anchor_centroid[core]) / scale[core]
        if float(np.max(feature_shift)) > config.PER_FEATURE_DRIFT_LIMIT:
            return False, "one or more timing features moved too far from the anchor"

    return True, ""


def _trust_score(result):
    disagreement_factor = max(
        0.0,
        1.0 - result.disagreement / max(config.TRUST_DISAGREEMENT_SCALE, 1e-6),
    )
    risk_factor = 1.0
    if result.assessment is not None:
        risk_factor = max(0.0, 1.0 - float(result.assessment.score))
    quality_factor = float(np.clip(result.quality_score, 0.0, 1.0))
    return float(np.clip(result.probability * disagreement_factor * risk_factor * quality_factor, 0.0, 1.0))


def _promote_samples(profile, entries, reason, refit=True):
    if not entries:
        return False
    before = template_drift(profile)
    for entry in entries:
        profile.add_sample(
            entry["features"],
            context=entry.get("context"),
            source="auto",
            timestamp=entry.get("timestamp"),
        )
        profile.record_sample_fingerprint(
            entry.get("sample_fingerprint"),
            timestamp=entry.get("timestamp"),
            source="promoted",
        )
    profile.quarantine = []
    profile.samples_since_retrain = 0 if refit else profile.samples_since_retrain + len(entries)
    if refit:
        fit_profile(profile)
    after = template_drift(profile)
    profile.snapshot_version(reason, shift=after - before, promoted=len(entries))
    profile.log(
        reason,
        promoted=len(entries),
        anchor_shift=after,
        trust=float(np.mean([entry.get("trust_score", 0.0) for entry in entries])),
    )
    return True


def _maybe_promote_quarantine(profile, result):
    if len(profile.quarantine) < config.QUARANTINE_MIN_SAMPLES:
        return

    ordered = sorted(profile.quarantine, key=lambda entry: entry["timestamp"])
    span = ordered[-1]["timestamp"] - ordered[0]["timestamp"]
    if span < config.QUARANTINE_MIN_SPAN_SECONDS:
        return

    raw_candidates = np.array([entry["features"] for entry in ordered])
    promoted_candidates = np.array(
        [
            _bounded_vector(profile, entry["features"], entry.get("trust_score", 0.0))
            for entry in ordered
        ]
    )
    allowed, reason = _promotion_allowed(profile, raw_candidates, promoted_candidates)
    if not allowed:
        result.lockout = reason
        return

    entries = []
    for entry, promoted in zip(ordered, promoted_candidates):
        entries.append({**entry, "features": promoted})
    promoted = _promote_samples(profile, entries, reason="quarantine_promotion", refit=True)
    result.adopted = promoted
    result.retrained = promoted


def retrain(profile, samples, choice_train=None, source="retrain"):
    for vector, context in samples:
        profile.add_sample(vector, context=context, source=source)

    drift_before = detect_drift(profile)
    profile.policy_state = {}
    info = fit_profile(profile, choice_train=choice_train)
    profile.match_probabilities = []
    profile.quarantine = []
    profile.recent_failures = []
    profile.set_anchor(samples=profile.active_samples)
    profile.snapshot_version(source, shift=template_drift(profile), promoted=len(samples))
    profile.log(
        source,
        added=len(samples),
        window=profile.sample_count,
        drift=drift_before.magnitude,
        **info,
    )
    return info, drift_before


def enroll(user_id, password, samples, choice_train=1):
    profile = storage.Profile(
        user_id=str(user_id),
        schema_version=config.SCHEMA_VERSION,
        extended=config.EXTENDED_FEATURES,
        model_choice=models.normalize_choice(choice_train),
        adaptation_policy=config.DEFAULT_ADAPTATION_POLICY,
    )
    profile.set_password(password)
    for vector, context in samples:
        profile.add_sample(vector, context=context, source="enroll")

    profile.set_anchor(samples=profile.active_samples)
    info = fit_profile(profile, choice_train=choice_train)
    profile.log("enroll", samples=profile.sample_count, **info)
    return profile, info


def _maybe_queue_adaptation(profile, vector, context, result, timestamp=None):
    if not config.ADAPTIVE_LEARNING or profile.is_legacy:
        return
    runtime = AdaptationRuntime(
        profile=profile,
        vector=np.asarray(vector, dtype=float),
        context=context,
        result=result,
        timestamp=time.time() if timestamp is None else float(timestamp),
    )
    policy = policies.get_policy(profile.adaptation_policy)
    policy.on_authentication(runtime)


def verify(profile, vector, context, quality_report=None, timestamp=None):
    analysis = models.analyse(profile.model, vector)
    probability = float(analysis["fused"][0])
    disagreement = float(analysis["disagreement"][0])
    anchor_distance = float(analysis["anchor_distance"][0])
    detector_scores = {
        name: float(values[0]) for name, values in analysis["scores"].items()
    }

    base = dynamic_threshold(profile)
    assessment = risk.assess(context, profile.context_history)
    required = risk.required_probability(base, assessment)

    result = VerificationResult(
        probability=probability,
        base_threshold=base,
        required=required,
        assessment=assessment,
        disagreement=disagreement,
        anchor_distance=anchor_distance,
        detector_scores=detector_scores,
    )
    if quality_report is not None:
        result.quality_score = float(quality_report.score)
        result.quality_flags = list(quality_report.flags)
        result.sample_fingerprint = quality_report.fingerprint
    if quality_report is not None and not quality_report.acceptable:
        result.reason = "sample quality was too low to score safely"
        profile.record_sample_fingerprint(result.sample_fingerprint, source="rejected_quality")
        return result

    if config.RISK_BLOCK_ENABLED and assessment.is_high:
        result.reason = "blocked: contextual risk too high"
        profile.record_sample_fingerprint(result.sample_fingerprint, source="blocked_risk")
        return result

    if probability < required:
        result.reason = (
            f"fused score {probability:.3f} was below the required {required:.3f}"
        )
        profile.record_failure(vector, probability, disagreement)
        profile.record_sample_fingerprint(result.sample_fingerprint, source="rejected")
        result.failure_analysis = analyse_failures(profile)
        return result

    result.authenticated = True
    result.reason = "typing pattern matched"
    profile.record_probability(probability)
    profile.record_sample_fingerprint(result.sample_fingerprint, source="accepted")
    profile.recent_failures = []

    _maybe_queue_adaptation(profile, vector, context, result, timestamp=timestamp)
    return result


def rollback(profile, version_id):
    """Restore a previously snapshotted active profile version."""
    return profile.rollback(version_id)


def status(profile):
    drift = detect_drift(profile)
    failures = analyse_failures(profile)
    sources = {}
    for entry in profile.sample_meta:
        key = entry.get("source", "unknown")
        sources[key] = sources.get(key, 0) + 1
    generator_counts = {}
    for entry in profile.synthetic_meta:
        key = entry.get("generator", "unknown")
        generator_counts[key] = generator_counts.get(key, 0) + 1

    timestamps = profile.timestamps()
    if timestamps:
        oldest = time.strftime("%Y-%m-%d", time.localtime(min(timestamps)))
        newest = time.strftime("%Y-%m-%d", time.localtime(max(timestamps)))
        span = f"{oldest} to {newest}"
    else:
        span = "unknown (profile predates sample timestamps)"

    devices = {c.get("hostname", "unknown") for c in profile.context_history if c}
    schema = f"v{profile.schema_version}" + (" (legacy)" if profile.is_legacy else "")
    if profile.sklearn_version_mismatch:
        schema += " - model built by a different scikit-learn version; retrain to rebuild"

    return {
        "schema": schema,
        "features": f"{profile.feature_dim} " + ("extended" if profile.extended else "legacy v1"),
        "samples": f"{profile.sample_count}/{config.MAX_AUTHENTIC_SAMPLES}",
        "sample_span": span,
        "sources": ", ".join(f"{k}={v}" for k, v in sorted(sources.items())) or "n/a",
        "devices_seen": ", ".join(sorted(devices)) or "none recorded",
        "threshold": f"{dynamic_threshold(profile):.3f}",
        "update_threshold": f"{profile.thresholds.get('update', config.UPDATE_THRESHOLD_FLOOR):.3f}",
        "disagreement_limit": f"{profile.thresholds.get('disagreement', config.DISAGREEMENT_LIMIT):.3f}",
        "adaptation_policy": profile.adaptation_policy,
        "quarantine": f"{len(profile.quarantine)} pending sample(s)",
        "versions": len(profile.versions),
        "negative_generators": ", ".join(
            f"{name}={count}" for name, count in sorted(generator_counts.items())
        )
        or "n/a",
        "fusion_weights": ", ".join(
            f"{name}={weight:.2f}" for name, weight in sorted(profile.fusion_weights.items())
        )
        or "n/a",
        "anchor_shift": f"{template_drift(profile):.2f} sd from anchor",
        "drift_state": drift.state.replace("_", " "),
        "recent_scores": len(profile.match_probabilities),
        "auto_adopted": f"{profile.auto_fraction() * 100:.0f}% of window",
        "drift": drift.message,
        "drift_action": drift.recommendation or "none",
        "rejections": failures.message
        or f"{failures.count} recent rejection(s) with the correct password",
    }
