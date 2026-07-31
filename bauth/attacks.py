"""Synthetic attack generators and sequential attacker strategies."""

from dataclasses import dataclass, field

import numpy as np

from . import config, features

_MIN_HOLD = 0.005
_MIN_GAP = 0.005
_MIN_UD = -0.5


@dataclass
class AttackSample:
    vector: np.ndarray
    generator: str
    strength: float
    source_index: int = 0
    metadata: dict = field(default_factory=dict)


def _safe_stack(rows):
    rows = [np.asarray(row, dtype=float) for row in rows]
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(rows)


def _column_scale(matrix, floor=0.01):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0, dtype=float)
    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0)
    return np.maximum(scale, floor)


def _clip_timings(hold, dd, ud, uu):
    return (
        np.maximum(np.asarray(hold, dtype=float), _MIN_HOLD),
        np.maximum(np.asarray(dd, dtype=float), _MIN_GAP),
        np.maximum(np.asarray(ud, dtype=float), _MIN_UD),
        np.maximum(np.asarray(uu, dtype=float), _MIN_GAP),
    )


def _assemble(hold, dd, ud, uu, extended):
    hold, dd, ud, uu = _clip_timings(hold, dd, ud, uu)
    return features.assemble(hold, dd, ud, uu, extended=extended)


def _decompose(vector, n_chars, extended):
    return features.decompose(vector, n_chars, extended)


def _timing_context(authentic, n_chars, extended):
    authentic = np.atleast_2d(np.asarray(authentic, dtype=float))
    parts = [_decompose(row, n_chars, extended) for row in authentic]
    holds = _safe_stack([p[0] for p in parts])
    dds = _safe_stack([p[1] for p in parts])
    uds = _safe_stack([p[2] for p in parts])
    uus = _safe_stack([p[3] for p in parts])
    center = np.median(authentic, axis=0)
    scale = _column_scale(authentic, floor=0.01)
    return {
        "n_chars": n_chars,
        "extended": extended,
        "authentic": authentic,
        "holds": holds,
        "dds": dds,
        "uds": uds,
        "uus": uus,
        "center": center,
        "scale": scale,
        "hold_center": np.median(holds, axis=0) if holds.size else np.zeros(0),
        "dd_center": np.median(dds, axis=0) if dds.size else np.zeros(0),
        "ud_center": np.median(uds, axis=0) if uds.size else np.zeros(0),
        "uu_center": np.median(uus, axis=0) if uus.size else np.zeros(0),
        "hold_scale": _column_scale(holds, floor=0.01),
        "dd_scale": _column_scale(dds, floor=0.01),
        "ud_scale": _column_scale(uds, floor=0.01),
        "uu_scale": _column_scale(uus, floor=0.01),
    }


class NegativeGenerator:
    name = "generator"

    def generate(self, genuine_sample, strength, rng, context):
        raise NotImplementedError


class GlobalSpeedShiftAttack(NegativeGenerator):
    name = "global_speed_shift"

    def generate(self, genuine_sample, strength, rng, context):
        hold, dd, ud, uu = _decompose(genuine_sample, context["n_chars"], context["extended"])
        slower = rng.random() < 0.5
        factor = rng.uniform(1.15, 1.15 + 1.2 * strength) if slower else rng.uniform(
            max(0.35, 1.0 - 0.9 * strength), 0.95
        )
        noise = max(0.002, 0.015 * strength)
        hold = hold * factor + rng.normal(0, noise, hold.shape)
        dd = dd * factor + rng.normal(0, noise, dd.shape)
        ud = ud * factor + rng.normal(0, noise, ud.shape)
        uu = uu * factor + rng.normal(0, noise, uu.shape)
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"factor": float(factor), "direction": "slower" if slower else "faster"},
        )


class PerKeyJitterAttack(NegativeGenerator):
    name = "per_key_jitter"

    def generate(self, genuine_sample, strength, rng, context):
        hold, dd, ud, uu = _decompose(genuine_sample, context["n_chars"], context["extended"])
        hold = hold + rng.normal(0, np.maximum(context["hold_scale"] * (1.2 + strength), 0.01))
        dd = dd + rng.normal(0, np.maximum(context["dd_scale"] * (1.2 + strength), 0.01))
        ud = ud + rng.normal(0, np.maximum(context["ud_scale"] * (1.0 + strength), 0.01))
        uu = uu + rng.normal(0, np.maximum(context["uu_scale"] * (1.0 + strength), 0.01))
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"jitter_scale": float(1.2 + strength)},
        )


class RhythmPermutationAttack(NegativeGenerator):
    name = "rhythm_permutation"

    def generate(self, genuine_sample, strength, rng, context):
        hold, dd, ud, uu = _decompose(genuine_sample, context["n_chars"], context["extended"])
        if hold.size:
            hold = hold[rng.permutation(len(hold))]
        if dd.size:
            order = rng.permutation(len(dd))
            dd = dd[order]
            ud = ud[order]
            uu = uu[order]
        hold = hold + rng.normal(0, np.maximum(context["hold_scale"] * 0.35 * strength, 0.003))
        dd = dd + rng.normal(0, np.maximum(context["dd_scale"] * 0.35 * strength, 0.003))
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"permuted": True},
        )


class PartialMimicryAttack(NegativeGenerator):
    name = "partial_mimicry"

    def generate(self, genuine_sample, strength, rng, context):
        hold, dd, ud, uu = _decompose(genuine_sample, context["n_chars"], context["extended"])
        blend = np.clip(0.35 + 0.45 * strength, 0.2, 0.9)
        target_index = int(rng.integers(len(context["authentic"])))
        target = context["authentic"][target_index]
        t_hold, t_dd, t_ud, t_uu = _decompose(target, context["n_chars"], context["extended"])
        hold = blend * hold + (1.0 - blend) * t_hold + rng.normal(0, 0.005, hold.shape)
        dd = blend * dd + (1.0 - blend) * t_dd + rng.normal(0, 0.005, dd.shape)
        ud = blend * ud + (1.0 - blend) * t_ud + rng.normal(0, 0.004, ud.shape)
        uu = blend * uu + (1.0 - blend) * t_uu + rng.normal(0, 0.004, uu.shape)
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"target_index": target_index, "blend": float(blend)},
        )


class TimeScaledReplayAttack(NegativeGenerator):
    name = "time_scaled_replay"

    def generate(self, genuine_sample, strength, rng, context):
        hold, dd, ud, uu = _decompose(genuine_sample, context["n_chars"], context["extended"])
        factor = rng.uniform(max(0.55, 1.0 - 0.35 * strength), 1.0 + 0.45 * strength)
        jitter = rng.normal(0, 0.0025 * max(strength, 0.2), hold.shape)
        hold = hold * factor + jitter
        dd = dd * factor + rng.normal(0, 0.0025 * max(strength, 0.2), dd.shape)
        ud = ud * factor + rng.normal(0, 0.0020 * max(strength, 0.2), ud.shape)
        uu = uu * factor + rng.normal(0, 0.0020 * max(strength, 0.2), uu.shape)
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"factor": float(factor)},
        )


class AnchorWalkAttack(NegativeGenerator):
    name = "anchor_walk_adversary"

    def generate(self, genuine_sample, strength, rng, context):
        hold, dd, ud, uu = _decompose(genuine_sample, context["n_chars"], context["extended"])
        center_hold = context["hold_center"]
        center_dd = context["dd_center"]
        center_ud = context["ud_center"]
        center_uu = context["uu_center"]
        delta_hold = hold - center_hold
        delta_dd = dd - center_dd
        delta_ud = ud - center_ud
        delta_uu = uu - center_uu
        if delta_hold.size and np.allclose(delta_hold, 0.0):
            delta_hold = rng.normal(0, context["hold_scale"])
        if delta_dd.size and np.allclose(delta_dd, 0.0):
            delta_dd = rng.normal(0, context["dd_scale"])
        if delta_ud.size and np.allclose(delta_ud, 0.0):
            delta_ud = rng.normal(0, context["ud_scale"])
        if delta_uu.size and np.allclose(delta_uu, 0.0):
            delta_uu = rng.normal(0, context["uu_scale"])
        step = 1.0 + 0.75 * strength
        hold = center_hold + delta_hold * step
        dd = center_dd + delta_dd * step
        ud = center_ud + delta_ud * step
        uu = center_uu + delta_uu * step
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"step": float(step)},
        )


class BoundarySeekingAttack(NegativeGenerator):
    name = "boundary_seeking_negative"

    def generate(self, genuine_sample, strength, rng, context):
        vector = np.asarray(genuine_sample, dtype=float)
        direction = np.sign(vector - context["center"])
        direction = np.where(direction == 0.0, rng.choice([-1.0, 1.0], size=vector.shape), direction)
        boundary = vector + direction * context["scale"] * (0.6 + 0.8 * strength)
        hold, dd, ud, uu = _decompose(boundary, context["n_chars"], context["extended"])
        hold = hold + rng.normal(0, 0.003, hold.shape)
        dd = dd + rng.normal(0, 0.003, dd.shape)
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"scale_multiple": float(0.6 + 0.8 * strength)},
        )


class PopulationDerivedImpostorAttack(NegativeGenerator):
    name = "population_derived_impostor"

    def generate(self, genuine_sample, strength, rng, context):
        def draw(matrix, scale):
            if matrix.size == 0:
                return np.zeros(0, dtype=float)
            values = []
            for idx in range(matrix.shape[1]):
                column = matrix[:, idx]
                base = float(column[int(rng.integers(len(column)))])
                values.append(base + rng.normal(0, max(scale[idx] * (0.5 + 0.5 * strength), 0.004)))
            return np.asarray(values, dtype=float)

        hold = draw(context["holds"], context["hold_scale"])
        dd = draw(context["dds"], context["dd_scale"])
        ud = draw(context["uds"], context["ud_scale"])
        uu = draw(context["uus"], context["uu_scale"])
        return AttackSample(
            vector=_assemble(hold, dd, ud, uu, context["extended"]),
            generator=self.name,
            strength=float(strength),
            metadata={"sampling": "column_marginals"},
        )


DEFAULT_NEGATIVE_GENERATORS = (
    GlobalSpeedShiftAttack,
    PerKeyJitterAttack,
    RhythmPermutationAttack,
    PartialMimicryAttack,
    TimeScaledReplayAttack,
    AnchorWalkAttack,
    BoundarySeekingAttack,
    PopulationDerivedImpostorAttack,
)


def default_negative_generators():
    return [factory() for factory in DEFAULT_NEGATIVE_GENERATORS]


def generate_negative_records(authentic, n_chars, extended=True, rng=None, count=None, generators=None):
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    authentic = np.atleast_2d(np.asarray(authentic, dtype=float))
    if count is None:
        count = int(
            np.clip(
                len(authentic) * config.NEGATIVE_RATIO,
                config.MIN_NEGATIVES,
                config.MAX_NEGATIVES,
            )
        )
    generators = generators or default_negative_generators()
    context = _timing_context(authentic, n_chars, extended)

    records = []
    for i in range(int(count)):
        generator = generators[i % len(generators)]
        source_index = i % len(authentic)
        strength = float(rng.uniform(0.35, 1.0))
        record = generator.generate(authentic[source_index], strength, rng, context)
        record.source_index = source_index
        record.metadata = {
            "source_index": source_index,
            "generator": record.generator,
            **record.metadata,
        }
        records.append(record)
    return records


def records_to_matrix(records):
    if not records:
        return np.zeros((0, 0), dtype=float)
    return np.vstack([np.asarray(record.vector, dtype=float) for record in records])


def generator_counts(records):
    counts = {}
    for record in records:
        counts[record.generator] = counts.get(record.generator, 0) + 1
    return counts


class AttackStrategy:
    name = "attacker"

    def reset(self):
        """Reset any internal state between simulation runs."""

    def sample(self, profile, history, rng):
        raise NotImplementedError

    def feedback(self, result, history):
        """Observe the previous outcome, for adaptive attackers."""


class RandomAttacker(AttackStrategy):
    name = "random_attacker"

    def __init__(self):
        self.generator = PopulationDerivedImpostorAttack()

    def sample(self, profile, history, rng):
        source = profile.anchor_samples if profile.anchor_samples is not None else profile.active_samples
        context = _timing_context(source, profile.char_count, profile.extended)
        index = int(rng.integers(len(source)))
        return self.generator.generate(source[index], rng.uniform(0.5, 1.0), rng, context)


class ReplayAttacker(AttackStrategy):
    name = "replay_attacker"

    def __init__(self, captured_samples=None):
        self.captured_samples = captured_samples
        self.index = 0

    def reset(self):
        self.index = 0

    def sample(self, profile, history, rng):
        bank = self.captured_samples
        if bank is None:
            bank = profile.anchor_samples if profile.anchor_samples is not None else profile.active_samples
        bank = np.atleast_2d(np.asarray(bank, dtype=float))
        vector = np.array(bank[self.index % len(bank)], copy=True)
        self.index += 1
        return AttackSample(vector=vector, generator=self.name, strength=1.0, metadata={"replay": True})


class TimeScaledReplayAttacker(AttackStrategy):
    name = "time_scaled_replay_attacker"

    def __init__(self, captured_samples=None, factor=1.18):
        self.replay = ReplayAttacker(captured_samples)
        self.factor = float(factor)

    def reset(self):
        self.replay.reset()

    def sample(self, profile, history, rng):
        replayed = self.replay.sample(profile, history, rng)
        context = _timing_context(
            profile.anchor_samples if profile.anchor_samples is not None else profile.active_samples,
            profile.char_count,
            profile.extended,
        )
        generator = TimeScaledReplayAttack()
        return generator.generate(replayed.vector, abs(self.factor - 1.0), rng, context)


class MimicryAttacker(AttackStrategy):
    name = "mimicry_attacker"

    def __init__(self, attacker_samples=None):
        self.attacker_samples = attacker_samples

    def sample(self, profile, history, rng):
        bank = self.attacker_samples
        if bank is None:
            bank = profile.active_samples[::-1]
        bank = np.atleast_2d(np.asarray(bank, dtype=float))
        source = np.array(bank[int(rng.integers(len(bank)))], copy=True)
        context = _timing_context(
            profile.anchor_samples if profile.anchor_samples is not None else profile.active_samples,
            profile.char_count,
            profile.extended,
        )
        generator = PartialMimicryAttack()
        record = generator.generate(source, rng.uniform(0.4, 0.9), rng, context)
        record.generator = self.name
        return record


class GradualWalkAttacker(AttackStrategy):
    name = "gradual_walk_attacker"

    def __init__(self, target=None):
        self.target = target
        self.current = None

    def reset(self):
        self.current = None

    def sample(self, profile, history, rng):
        base = profile.anchor_samples if profile.anchor_samples is not None else profile.active_samples
        base = np.atleast_2d(np.asarray(base, dtype=float))
        if self.current is None:
            self.current = np.array(base[int(rng.integers(len(base)))], copy=True)
        target = self.target
        if target is None:
            idx = int(rng.integers(len(base)))
            target = np.array(base[idx], copy=True) + rng.normal(0, np.maximum(base.std(axis=0), 0.01))
        direction = np.asarray(target, dtype=float) - self.current
        step = 0.10 + 0.10 * min(len(history), 10) / 10.0
        self.current = self.current + step * direction
        return AttackSample(
            vector=np.array(self.current, copy=True),
            generator=self.name,
            strength=float(step),
            metadata={"step": float(step)},
        )

    def feedback(self, result, history):
        if self.current is None:
            return
        if result.authenticated:
            self.current = np.array(self.current, copy=True)


class FeedbackAwareAttacker(AttackStrategy):
    name = "feedback_aware_attacker"

    def __init__(self, target=None):
        self.target = target
        self.current = None

    def reset(self):
        self.current = None

    def sample(self, profile, history, rng):
        source = profile.anchor_samples if profile.anchor_samples is not None else profile.active_samples
        source = np.atleast_2d(np.asarray(source, dtype=float))
        anchor = np.median(source, axis=0)
        if self.current is None:
            self.current = np.array(anchor, copy=True)
        target = np.asarray(self.target, dtype=float) if self.target is not None else (
            anchor + np.median(np.abs(source - anchor), axis=0) * 2.0
        )
        self.current = self.current + 0.15 * (target - self.current)
        self.current = self.current + rng.normal(0, np.maximum(source.std(axis=0), 0.01) * 0.08)
        return AttackSample(
            vector=np.array(self.current, copy=True),
            generator=self.name,
            strength=0.15,
            metadata={"history_len": len(history)},
        )

    def feedback(self, result, history):
        if self.current is None:
            return
        anchor_distance = getattr(result, "anchor_distance", 0.0)
        probability = getattr(result, "probability", 0.0)
        if result.authenticated:
            self.current = self.current + 0.05 * probability
        else:
            self.current = self.current * max(0.85, 1.0 - 0.03 * max(anchor_distance, 1.0))


ATTACK_STRATEGIES = {
    "random_attacker": RandomAttacker,
    "replay_attacker": ReplayAttacker,
    "time_scaled_replay_attacker": TimeScaledReplayAttacker,
    "mimicry_attacker": MimicryAttacker,
    "gradual_walk_attacker": GradualWalkAttacker,
    "feedback_aware_attacker": FeedbackAwareAttacker,
}


def build_strategy(name):
    factory = ATTACK_STRATEGIES.get(name, RandomAttacker)
    return factory()
