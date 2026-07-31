import unittest
from unittest import mock

import numpy as np

from bauth import adaptive, config, context, features, models, quality


def _vector(offset=0.0):
    hold = np.array([0.090, 0.082, 0.079, 0.088]) + offset
    dd = np.array([0.115, 0.102, 0.111]) + offset * 0.5
    ud = np.array([-0.012, -0.009, -0.011]) + offset * 0.2
    uu = np.array([0.101, 0.094, 0.099]) + offset * 0.4
    return features.assemble(hold, dd, ud, uu, extended=True)


def _samples(count=10):
    ctx = context.CaptureContext(
        hostname="host-a",
        local_ip="192.168.1.20",
        timezone_name="UTC",
        os_name="Linux",
        machine="x86_64",
    )
    return [(_vector(i * 0.002), ctx) for i in range(count)]


class FeatureRegistryTests(unittest.TestCase):
    def test_named_feature_sets_expand_in_expected_order(self):
        matrix = np.vstack([_vector(0.0), _vector(0.01)])
        core = features.select_set(matrix, 4, True, "core")
        transition = features.select_set(matrix, 4, True, "transition")
        extended = features.select_set(matrix, 4, True, "extended")
        aggregate = features.select_set(matrix, 4, True, "aggregate")

        self.assertEqual(core.shape[1], 7)
        self.assertEqual(transition.shape[1], 10)
        self.assertGreater(extended.shape[1], transition.shape[1])
        self.assertGreaterEqual(aggregate.shape[1], core.shape[1])


class FusionModelTests(unittest.TestCase):
    def test_fusion_model_emits_weights_scores_and_disagreement(self):
        authentic = np.vstack([_vector(i * 0.002) for i in range(10)])
        negatives = models.generate_negatives(authentic, 4, extended=True)
        model, scaler, info = models.train(authentic, negatives)
        analysis = models.analyse(model, authentic[:3])

        self.assertAlmostEqual(sum(info["fusion_weights"].values()), 1.0, places=6)
        self.assertEqual(analysis["fused"].shape[0], 3)
        self.assertEqual(analysis["disagreement"].shape[0], 3)
        self.assertTrue(np.all(analysis["fused"] >= 0.0))
        self.assertTrue(np.all(analysis["fused"] <= 1.0))
        self.assertTrue(np.all(analysis["disagreement"] >= 0.0))
        self.assertEqual(scaler.transform(authentic[:1]).shape, (1, authentic.shape[1]))


class QuarantinePolicyTests(unittest.TestCase):
    def test_samples_promote_only_after_quarantine_consensus(self):
        profile, _ = adaptive.enroll("user-1", "abcd", _samples())
        sample = _vector(0.0)
        ctx = _samples(1)[0][1]
        before = profile.sample_count
        profile.thresholds["update"] = 0.60
        profile.thresholds["disagreement"] = 0.25
        profile.thresholds["anchor_candidate"] = 2.0

        with mock.patch.object(config, "QUARANTINE_MIN_SPAN_SECONDS", 0.0):
            first = adaptive.verify(profile, sample, ctx)
            second = adaptive.verify(profile, sample, ctx)
            third = adaptive.verify(profile, sample, ctx)

        self.assertTrue(first.authenticated)
        self.assertTrue(first.quarantined)
        self.assertFalse(first.adopted)
        self.assertEqual(len(profile.quarantine), 0)
        self.assertTrue(third.adopted)
        self.assertTrue(third.retrained)
        self.assertEqual(len(profile.anchor_samples), before)
        self.assertGreater(profile.sample_count, before)


class QualityAndRollbackTests(unittest.TestCase):
    def test_duplicate_fingerprint_is_flagged_for_adaptation(self):
        profile, _ = adaptive.enroll("user-2", "abcd", _samples())

        class DummyCapture:
            complete = True
            press_times = [0.0, 0.1, 0.2, 0.3]
            release_times = [0.05, 0.15, 0.25, 0.35]

            def timings(self):
                return (
                    [0.05, 0.05, 0.05, 0.05],
                    [0.10, 0.10, 0.10],
                    [0.05, 0.05, 0.05],
                    [0.10, 0.10, 0.10],
                )

        vector = _vector(0.0)
        first = quality.assess_capture(DummyCapture(), vector, profile=profile)
        profile.record_sample_fingerprint(first.fingerprint)
        second = quality.assess_capture(DummyCapture(), vector, profile=profile)

        self.assertTrue(first.acceptable)
        self.assertIn("replayed-or-duplicate-sample", second.flags)
        self.assertTrue(second.replay_like)

    def test_profile_can_roll_back_to_saved_version(self):
        profile, _ = adaptive.enroll("user-3", "abcd", _samples())
        original = np.array(profile.active_samples, copy=True)
        profile.snapshot_version("baseline")
        profile.add_sample(_vector(0.2), context=_samples(1)[0][1], source="auto")
        self.assertNotEqual(profile.active_samples.shape, original.shape)

        version_id = profile.versions[-1]["version"]
        self.assertTrue(adaptive.rollback(profile, version_id))
        self.assertTrue(np.allclose(profile.active_samples, original))


if __name__ == "__main__":
    unittest.main()
