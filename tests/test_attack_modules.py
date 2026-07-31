import unittest

import numpy as np

from bauth import adaptive, attacks, context, features, policies, quality, simulator


def _vector(offset=0.0):
    hold = np.array([0.090, 0.082, 0.079, 0.088]) + offset
    dd = np.array([0.115, 0.102, 0.111]) + offset * 0.5
    ud = np.array([-0.012, -0.009, -0.011]) + offset * 0.2
    uu = np.array([0.101, 0.094, 0.099]) + offset * 0.4
    return features.assemble(hold, dd, ud, uu, extended=True)


class DummyProfile:
    def __init__(self):
        self.char_count = 4
        self.extended = True
        self.active_samples = np.vstack([_vector(i * 0.002) for i in range(6)])
        self._fingerprints = set()

    def has_sample_fingerprint(self, fingerprint):
        return fingerprint in self._fingerprints

    def record_sample_fingerprint(self, fingerprint):
        self._fingerprints.add(fingerprint)


class AttackGeneratorTests(unittest.TestCase):
    def test_negative_records_cover_named_generators(self):
        authentic = np.vstack([_vector(i * 0.002) for i in range(10)])
        records = attacks.generate_negative_records(
            authentic,
            n_chars=4,
            extended=True,
            rng=np.random.default_rng(7),
            count=16,
        )

        self.assertEqual(len(records), 16)
        matrix = attacks.records_to_matrix(records)
        self.assertEqual(matrix.shape[0], 16)
        counts = attacks.generator_counts(records)
        self.assertGreaterEqual(len(counts), 8)
        self.assertIn("anchor_walk_adversary", counts)
        self.assertIn("population_derived_impostor", counts)


class QualityVectorTests(unittest.TestCase):
    def test_assess_vector_flags_duplicate_replay(self):
        profile = DummyProfile()
        vector = _vector(0.0)

        first = quality.assess_vector(vector, profile=profile)
        profile.record_sample_fingerprint(first.fingerprint)
        second = quality.assess_vector(vector, profile=profile)

        self.assertTrue(first.acceptable)
        self.assertIn("replayed-or-duplicate-sample", second.flags)
        self.assertTrue(second.replay_like)

    def test_assess_vector_flags_uniform_sequences(self):
        hold = np.array([0.05, 0.05, 0.05, 0.05])
        dd = np.array([0.10, 0.10, 0.10])
        ud = np.array([0.05, 0.05, 0.05])
        uu = np.array([0.10, 0.10, 0.10])
        vector = features.assemble(hold, dd, ud, uu, extended=True)

        report = quality.assess_vector(vector, n_chars=4, extended=True)

        self.assertIn("resolution-limited-sample", report.flags)
        self.assertIn("repeated-timing-subsequence", report.flags)


class PolicyRegistryTests(unittest.TestCase):
    def test_policy_registry_exposes_anchor_guard_variant(self):
        policy = policies.get_policy("quarantine_consensus_anchor")
        self.assertEqual(policy.name, "quarantine_consensus_anchor")
        self.assertEqual(
            policies.get_policy("does-not-exist").name,
            "quarantine_consensus_anchor",
        )


class SimulatorTests(unittest.TestCase):
    def test_poisoning_simulator_emits_summary(self):
        ctx = context.CaptureContext(
            hostname="host-a",
            local_ip="192.168.1.20",
            timezone_name="UTC",
            os_name="Linux",
            machine="x86_64",
        )
        samples = [(_vector(i * 0.002), ctx) for i in range(10)]
        profile, _ = adaptive.enroll("sim-user", "abcd", samples)
        profile.thresholds["update"] = 0.60
        profile.thresholds["disagreement"] = 0.30

        sim = simulator.PoisoningSimulator(profile)
        steps = sim.run_strategy(attacks.build_strategy("random_attacker"), steps=3)
        summary = sim.summary()

        self.assertEqual(len(steps), 3)
        self.assertEqual(summary["steps"], 3)
        self.assertIn("accept_rate", summary)
        self.assertIn("max_anchor_shift", summary)


if __name__ == "__main__":
    unittest.main()
