"""Sample-quality and replay/fabrication heuristics."""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import config, features


@dataclass
class QualityReport:
    score: float = 1.0
    flags: list = field(default_factory=list)
    fingerprint: str = ""
    replay_like: bool = False
    severe: bool = False
    metrics: dict = field(default_factory=dict)

    @property
    def acceptable(self):
        return not self.severe and self.score >= config.QUALITY_REJECT_FLOOR

    @property
    def update_ready(self):
        return self.acceptable and self.score >= config.QUALITY_UPDATE_FLOOR


def _fingerprint(vector, step_ms):
    millis = np.asarray(vector, dtype=float) * 1000.0
    quantized = np.rint(millis / step_ms).astype(int)
    digest = hashlib.sha256(quantized.tobytes()).hexdigest()
    return digest[:24], quantized


def _resolution_ratio(values, step_ms):
    if values.size == 0:
        return 0.0
    scaled = values * 1000.0 / step_ms
    return float(np.mean(np.isclose(scaled, np.rint(scaled), atol=0.03)))


def _repeat_ratio(values, step_ms):
    if values.size == 0:
        return 0.0
    quantized = np.rint(values * 1000.0 / step_ms).astype(int)
    _, counts = np.unique(quantized, return_counts=True)
    return float(counts.max() / len(quantized))


def _subvector_repeat_ratio(values, step_ms, window=3):
    values = np.asarray(values, dtype=float)
    if values.size < window:
        return 0.0
    quantized = np.rint(values * 1000.0 / step_ms).astype(int)
    windows = [tuple(quantized[i : i + window]) for i in range(len(quantized) - window + 1)]
    _, counts = np.unique(np.asarray(windows, dtype=object), return_counts=True)
    return float(counts.max() / len(windows)) if len(windows) else 0.0


def _variance_ratio(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 1.0
    mean_abs = float(np.mean(np.abs(values)))
    if mean_abs <= 1e-9:
        return 0.0
    return float(np.std(values) / mean_abs)


def _assess_timings(report, hold, dd, ud, uu, vector, profile=None):
    intervals = np.concatenate([hold, dd, ud, uu]) if vector.size else np.array([], dtype=float)
    fingerprint, quantized = _fingerprint(vector, config.REPLAY_QUANTIZATION_MS)
    report.fingerprint = fingerprint
    report.metrics["quantized_length"] = int(len(quantized))

    hold = np.asarray(hold, dtype=float)
    dd = np.asarray(dd, dtype=float)
    ud = np.asarray(ud, dtype=float)
    uu = np.asarray(uu, dtype=float)

    if np.any(hold < 0.0) or np.any(dd < 0.0):
        report.flags.append("impossible-latencies")
        report.score -= 0.80
        report.severe = True

    if dd.size and ud.size:
        relation_error = np.abs(dd - (hold[:-1] + ud))
        report.metrics["relation_error_ms"] = float(np.max(relation_error, initial=0.0) * 1000.0)
        if np.any(relation_error > 0.020):
            report.flags.append("inconsistent-timing-relations")
            report.score -= 0.35

    if intervals.size:
        resolution_ratio = _resolution_ratio(intervals, config.REPLAY_QUANTIZATION_MS)
        repeat_ratio = _repeat_ratio(intervals, config.REPLAY_QUANTIZATION_MS)
        subvector_ratio = _subvector_repeat_ratio(intervals, config.REPLAY_QUANTIZATION_MS)
        variance_ratio = _variance_ratio(intervals)
        report.metrics["resolution_ratio"] = resolution_ratio
        report.metrics["repeat_ratio"] = repeat_ratio
        report.metrics["subvector_repeat_ratio"] = subvector_ratio
        report.metrics["variance_ratio"] = variance_ratio

        if resolution_ratio > config.QUALITY_MAX_RESOLUTION_RATIO:
            report.flags.append("resolution-limited-sample")
            report.score -= 0.12
        if repeat_ratio > config.QUALITY_MAX_REPEAT_RATIO:
            report.flags.append("uniform-interval-pattern")
            report.score -= 0.18
        if subvector_ratio > 0.66:
            report.flags.append("repeated-timing-subsequence")
            report.score -= 0.20
        if variance_ratio < 0.08:
            report.flags.append("variance-collapse")
            report.score -= 0.18

    if profile is not None and profile.active_samples is not None and len(profile.active_samples) >= 5:
        baseline = np.asarray(profile.active_samples[:, 0], dtype=float)
        center = float(np.median(baseline))
        scale = max(float(np.median(np.abs(baseline - center))), 1e-6)
        total_z = abs(float(vector[0]) - center) / scale
        report.metrics["total_time_z"] = total_z
        if total_z > config.QUALITY_MAX_TOTAL_Z:
            report.flags.append("abnormal-total-duration")
            report.score -= 0.20

    if profile is not None and profile.has_sample_fingerprint(fingerprint):
        report.flags.append("replayed-or-duplicate-sample")
        report.score -= 0.25
        report.replay_like = True

    report.score = float(np.clip(report.score, 0.0, 1.0))
    return report


def assess_capture(capture, vector, profile=None):
    """Assess a captured sample before authentication and adaptation."""
    report = QualityReport()
    vector = np.asarray(vector, dtype=float)

    if not capture.complete:
        report.flags.append("incomplete-sequence")
        report.score -= 0.80
        report.severe = True

    if len(capture.press_times) != len(capture.release_times) or not capture.press_times:
        report.flags.append("missing-key-events")
        report.score -= 0.70
        report.severe = True
    else:
        press = np.asarray(capture.press_times, dtype=float)
        release = np.asarray(capture.release_times, dtype=float)
        if np.any(np.diff(press) < 0.0):
            report.flags.append("non-monotonic-press-times")
            report.score -= 0.70
            report.severe = True
        if np.any(release < press):
            report.flags.append("key-up-before-key-down")
            report.score -= 0.70
            report.severe = True

    hold, dd, ud, uu = capture.timings()
    return _assess_timings(report, hold, dd, ud, uu, vector, profile=profile)


def assess_vector(vector, profile=None, n_chars=None, extended=True):
    """Assess a feature vector without raw key-event timestamps.

    Used by synthetic attacks and sequential simulations, where only the timing
    vector exists.
    """
    vector = np.asarray(vector, dtype=float)
    if n_chars is None:
        if profile is not None:
            n_chars = profile.char_count
            extended = profile.extended
        else:
            raise ValueError("n_chars is required when profile is not provided")
    hold, dd, ud, uu = features.decompose(vector, n_chars, extended)
    return _assess_timings(QualityReport(), hold, dd, ud, uu, vector, profile=profile)
