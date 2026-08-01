"""Sequential authentication and poisoning simulator."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from . import adaptive, attacks, context, quality


@dataclass
class SimulationStep:
    index: int
    source: str
    attack: str
    authenticated: bool
    adaptation_eligible: bool
    quarantined: bool
    promoted: bool
    profile_shift: float
    anchor_shift: float
    probability: float
    genuine_score: float | None
    attacker_score: float | None
    anchor_distance: float
    disagreement: float
    context_risk: float
    quality_score: float
    quality_flags: list = field(default_factory=list)
    lockout: str = ""
    risk_level: str = ""
    promoted_count: int = 0
    promoted_genuine_count: int = 0
    promoted_attacker_count: int = 0
    profile_size: int = 0
    profile_genuine_fraction: float = 0.0
    profile_attacker_fraction: float = 0.0
    metadata: dict = field(default_factory=dict)


def _default_context(profile):
    if profile.context_history:
        return context.from_dict(profile.context_history[-1])
    return context.CaptureContext(
        hostname="simulated-host",
        local_ip="192.168.0.50",
        timezone_name="UTC",
        os_name="Linux",
        machine="x86_64",
    )


class PoisoningSimulator:
    def __init__(
        self,
        profile,
        adaptation_policy=None,
        rng=None,
        default_context=None,
        start_time=0.0,
        step_seconds=60.0,
    ):
        self.profile = deepcopy(profile)
        if adaptation_policy:
            self.profile.adaptation_policy = adaptation_policy
        self.rng = rng or np.random.default_rng(20260731)
        self.default_context = default_context or _default_context(self.profile)
        self.history = []
        self.last_result = None
        self.last_quality = None
        self.current_time = float(start_time)
        self.step_seconds = float(step_seconds)

    def _profile_truth_counts(self):
        counts = {}
        for entry in self.profile.sample_meta:
            truth_source = entry.get("truth_source")
            if truth_source is None:
                truth_source = "genuine" if entry.get("source") != "auto" else "unknown"
            counts[str(truth_source)] = counts.get(str(truth_source), 0) + 1
        return counts

    def _coerce_context(self, value):
        if value is None:
            return self.default_context
        if isinstance(value, dict):
            return context.from_dict(value)
        return value

    def step(self, sample, source="genuine", context_override=None, metadata=None):
        record = sample if isinstance(sample, attacks.AttackSample) else None
        vector = np.asarray(record.vector if record is not None else sample, dtype=float)
        attack_name = record.generator if record is not None else ("genuine" if source == "genuine" else "unknown")
        sample_metadata = {}
        if record is not None:
            sample_metadata.update(record.metadata)
        if metadata:
            sample_metadata.update(metadata)

        ctx = self._coerce_context(context_override)
        report = quality.assess_vector(
            vector,
            profile=self.profile,
            n_chars=self.profile.char_count,
            extended=self.profile.extended,
        )
        before_shift = adaptive.template_drift(self.profile)
        step_time = self.current_time
        result = adaptive.verify(
            self.profile,
            vector,
            ctx,
            quality_report=report,
            timestamp=step_time,
            sample_source=source,
            sample_metadata=sample_metadata,
        )
        after_shift = adaptive.template_drift(self.profile)
        self.last_result = result
        self.last_quality = report
        self.current_time += self.step_seconds
        truth_counts = self._profile_truth_counts()
        profile_size = max(sum(truth_counts.values()), 1)
        promoted_truth_counts = dict(result.promoted_truth_counts or {})

        entry = SimulationStep(
            index=len(self.history),
            source=source,
            attack=attack_name,
            authenticated=result.authenticated,
            adaptation_eligible=bool(
                result.authenticated and (result.quarantined or result.adopted or not result.lockout)
            ),
            quarantined=result.quarantined,
            promoted=result.adopted,
            profile_shift=float(after_shift - before_shift),
            anchor_shift=float(after_shift),
            probability=float(result.probability),
            genuine_score=float(result.probability) if source == "genuine" else None,
            attacker_score=float(result.probability) if source != "genuine" else None,
            anchor_distance=float(result.anchor_distance),
            disagreement=float(result.disagreement),
            context_risk=0.0 if result.assessment is None else float(result.assessment.score),
            quality_score=float(result.quality_score),
            quality_flags=list(result.quality_flags),
            lockout=result.lockout,
            risk_level="" if result.assessment is None else result.assessment.level,
            promoted_count=int(result.promoted_count),
            promoted_genuine_count=int(promoted_truth_counts.get("genuine", 0)),
            promoted_attacker_count=int(promoted_truth_counts.get("attacker", 0)),
            profile_size=int(sum(truth_counts.values())),
            profile_genuine_fraction=float(truth_counts.get("genuine", 0) / profile_size),
            profile_attacker_fraction=float(truth_counts.get("attacker", 0) / profile_size),
            metadata=sample_metadata,
        )
        self.history.append(entry)
        return entry

    def run_sequence(self, samples, source="genuine", context_override=None):
        return [
            self.step(sample, source=source, context_override=context_override)
            for sample in samples
        ]

    def run_strategy(self, strategy, steps, source="attacker", context_override=None):
        strategy.reset()
        output = []
        for _ in range(int(steps)):
            sample = strategy.sample(self.profile, self.history, self.rng)
            entry = self.step(sample, source=source, context_override=context_override)
            strategy.feedback(self.last_result, self.history)
            output.append(entry)
        return output

    def summary(self):
        if not self.history:
            return {
                "steps": 0,
                "accept_rate": 0.0,
                "promotion_rate": 0.0,
                "max_anchor_shift": adaptive.template_drift(self.profile),
            }
        accepted = sum(1 for entry in self.history if entry.authenticated)
        promoted = sum(1 for entry in self.history if entry.promoted)
        return {
            "steps": len(self.history),
            "accept_rate": accepted / len(self.history),
            "promotion_rate": promoted / len(self.history),
            "max_anchor_shift": max(entry.anchor_shift for entry in self.history),
            "policy": self.profile.adaptation_policy,
        }
