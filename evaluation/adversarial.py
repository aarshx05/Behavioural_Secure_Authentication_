"""Adversarial evaluation using the sequential poisoning simulator.

Builds per-subject profiles from the CMU benchmark, runs genuine and attacker
trajectories once to the maximum requested horizon, then derives prefix-based
metrics, confidence intervals, and paired tests from those runs.
"""

from copy import deepcopy

import numpy as np
from scipy import stats

from bauth import adaptive, attacks, config, context, simulator, storage

from . import cmu, protocol

DEFAULT_HORIZONS = (25, 50, 100, 250)
DEFAULT_POLICIES = (
    "frozen",
    "high_confidence",
    "anchor_bounded",
    "consensus_anchor",
    "quarantine_anchor",
    "quarantine_consensus_no_anchor",
    "quarantine_consensus_anchor",
)
DEFAULT_ATTACKERS = (
    "random_attacker",
    "replay_attacker",
    "time_scaled_replay_attacker",
    "mimicry_attacker",
    "gradual_walk_attacker",
    "feedback_aware_attacker",
)

FULL_POLICY = "quarantine_consensus_anchor"
POLICY_ABLATIONS = {
    "remove_quarantine": "consensus_anchor",
    "remove_consensus": "quarantine_anchor",
    "remove_anchor_bounds": "quarantine_consensus_no_anchor",
}
PAIRWISE_METRICS = (
    "genuine_acceptance_rate",
    "authentication_acceptance_rate",
    "attacker_promotion_rate",
    "takeover_success",
    "final_anchor_displacement",
    "final_attacker_profile_percentage",
    "adaptation_precision",
    "adaptation_recall",
)


def _dummy_context(hostname="cmu-lab"):
    return context.CaptureContext(
        hostname=hostname,
        local_ip="10.0.0.25",
        timezone_name="UTC",
        os_name="Linux",
        machine="x86_64",
        username="cmu-user",
    )


def _calibrate_for_benchmark(profile, dev_vectors):
    if dev_vectors is None or len(dev_vectors) == 0:
        return
    genuine = profile.model.analyse(dev_vectors)
    impostor = profile.model.analyse(profile.synthetic)
    auth = adaptive._best_threshold(genuine["fused"], impostor["fused"])
    profile.thresholds["auth"] = float(auth)
    profile.thresholds["update"] = float(
        max(
            auth + config.UPDATE_THRESHOLD_MARGIN,
            np.quantile(genuine["fused"], 0.25),
        )
    )
    profile.thresholds["disagreement"] = float(
        np.quantile(genuine["disagreement"], 0.85)
    )
    profile.thresholds["anchor_candidate"] = float(
        max(
            config.ANCHOR_CANDIDATE_LIMIT,
            np.quantile(genuine["anchor_distance"], 0.90),
        )
    )


def build_profile(data, subject, max_train_samples=None, dev_samples=40, now=None):
    train_idx, genuine_idx, impostor_idx = protocol.split(data, subject)
    vectors = data.extended(train_idx)
    if max_train_samples is not None:
        fit_vectors = vectors[:max_train_samples]
    else:
        fit_vectors = vectors
    dev_vectors = vectors[len(fit_vectors) : len(fit_vectors) + dev_samples]
    adaptation_vectors = vectors[len(fit_vectors) + len(dev_vectors) :]

    profile = storage.Profile(
        user_id=str(subject),
        password_length=cmu.N_KEYS,
        schema_version=config.SCHEMA_VERSION,
        extended=True,
        model_choice=1,
        adaptation_policy=config.DEFAULT_ADAPTATION_POLICY,
        legacy_plaintext="x" * cmu.N_KEYS,
    )
    ctx = _dummy_context(hostname=f"{subject}-host")
    for i, vector in enumerate(fit_vectors, 1):
        profile.add_sample(
            vector,
            context=ctx,
            source="enroll",
            timestamp=float(i),
            metadata={"truth_source": "genuine"},
        )

    adaptive.fit_profile(
        profile,
        now=float(now if now is not None else len(fit_vectors) + 1),
    )
    _calibrate_for_benchmark(profile, dev_vectors)
    profile.log(
        "adversarial_eval_profile_built",
        subject=subject,
        samples=len(fit_vectors),
    )
    return (
        profile,
        adaptation_vectors,
        data.extended(genuine_idx),
        data.extended(impostor_idx),
    )


def _safe_mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _first_true(values):
    for index, value in enumerate(values, 1):
        if value:
            return index
    return None


def _series_from_steps(steps, attr, cast=float):
    return [cast(getattr(step, attr)) for step in steps]


def _distribution(values):
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return {
            "values": [],
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(cleaned, dtype=float)
    return {
        "values": [float(value) for value in cleaned],
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _bootstrap_mean_ci(values, rng, resamples=1000, confidence=0.95):
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    arr = np.asarray(cleaned, dtype=float)
    if len(arr) == 1:
        point = float(arr[0])
        return [point, point]
    samples = rng.choice(arr, size=(resamples, len(arr)), replace=True)
    means = samples.mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(low), float(high)]


def _paired_wilcoxon(left, right):
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if not pairs:
        return {
            "n": 0,
            "statistic": None,
            "p_value": None,
            "median_difference": None,
        }
    arr_left = np.asarray([pair[0] for pair in pairs], dtype=float)
    arr_right = np.asarray([pair[1] for pair in pairs], dtype=float)
    diff = arr_left - arr_right
    if np.allclose(diff, 0.0):
        return {
            "n": int(len(diff)),
            "statistic": 0.0,
            "p_value": 1.0,
            "median_difference": 0.0,
        }
    test = stats.wilcoxon(arr_left, arr_right, zero_method="wilcox", alternative="two-sided")
    return {
        "n": int(len(diff)),
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "median_difference": float(np.median(diff)),
    }


def _summarise_metric(values, rng):
    distribution = _distribution(values)
    return {
        **distribution,
        "ci95": _bootstrap_mean_ci(distribution["values"], rng),
    }


def _summarise_genuine_prefix(steps):
    if not steps:
        return {
            "attempts": 0,
            "genuine_acceptance_rate": 0.0,
            "promotion_event_rate": 0.0,
            "promoted_genuine_samples": 0,
            "max_anchor_displacement": 0.0,
            "final_anchor_displacement": 0.0,
            "mean_genuine_score": 0.0,
        }
    accepted = [bool(step.authenticated) for step in steps]
    promoted_genuine = [int(step.promoted_genuine_count) for step in steps]
    anchor_shift = [float(step.anchor_shift) for step in steps]
    scores = [float(step.probability) for step in steps]
    return {
        "attempts": int(len(steps)),
        "genuine_acceptance_rate": float(np.mean(accepted)),
        "promotion_event_rate": float(np.mean([bool(step.promoted_count) for step in steps])),
        "promoted_genuine_samples": int(sum(promoted_genuine)),
        "max_anchor_displacement": float(np.max(anchor_shift)),
        "final_anchor_displacement": float(anchor_shift[-1]),
        "mean_genuine_score": float(np.mean(scores)),
    }


def _summarise_attacker_prefix(steps):
    if not steps:
        return {
            "attempts": 0,
            "authentication_acceptance_rate": 0.0,
            "attacker_promotion_rate": 0.0,
            "promoted_attacker_samples": 0,
            "takeover_success": 0.0,
            "attempts_to_takeover": None,
            "max_anchor_displacement": 0.0,
            "final_anchor_displacement": 0.0,
            "max_attacker_profile_percentage": 0.0,
            "final_attacker_profile_percentage": 0.0,
            "mean_attacker_score": 0.0,
            "lockouts": 0,
        }
    accepted = [bool(step.authenticated) for step in steps]
    promoted = [bool(step.promoted_count) for step in steps]
    promoted_attacker = [int(step.promoted_attacker_count) for step in steps]
    anchor_shift = [float(step.anchor_shift) for step in steps]
    attacker_fraction = [100.0 * float(step.profile_attacker_fraction) for step in steps]
    scores = [float(step.probability) for step in steps]
    takeovers = [fraction >= 50.0 for fraction in attacker_fraction]
    return {
        "attempts": int(len(steps)),
        "authentication_acceptance_rate": float(np.mean(accepted)),
        "attacker_promotion_rate": float(np.mean(promoted)),
        "promoted_attacker_samples": int(sum(promoted_attacker)),
        "takeover_success": 1.0 if any(takeovers) else 0.0,
        "attempts_to_takeover": _first_true(takeovers),
        "max_anchor_displacement": float(np.max(anchor_shift)),
        "final_anchor_displacement": float(anchor_shift[-1]),
        "max_attacker_profile_percentage": float(np.max(attacker_fraction)),
        "final_attacker_profile_percentage": float(attacker_fraction[-1]),
        "mean_attacker_score": float(np.mean(scores)),
        "lockouts": int(sum(1 for step in steps if step.lockout)),
    }


def _combine_precision_recall(genuine_metrics, attacker_metrics):
    promoted_genuine = int(genuine_metrics["promoted_genuine_samples"])
    promoted_attacker = int(attacker_metrics["promoted_attacker_samples"])
    attempts = max(int(genuine_metrics["attempts"]), 1)
    denominator = promoted_genuine + promoted_attacker
    precision = None if denominator == 0 else float(promoted_genuine / denominator)
    recall = float(promoted_genuine / attempts)
    return precision, recall


def _attacker_time_series(attacker_steps):
    return {
        "attacker_score_over_time": _series_from_steps(attacker_steps, "probability"),
        "attacker_anchor_displacement_over_time": _series_from_steps(
            attacker_steps,
            "anchor_shift",
        ),
        "attacker_profile_percentage_over_time": [
            100.0 * float(step.profile_attacker_fraction) for step in attacker_steps
        ],
    }


def evaluate_subject(
    data,
    subject,
    *,
    policies=DEFAULT_POLICIES,
    attackers=DEFAULT_ATTACKERS,
    horizons=DEFAULT_HORIZONS,
    max_train_samples=60,
):
    max_horizon = int(max(horizons))
    profile, adaptation_pool, future_genuine_pool, _ = build_profile(
        data,
        subject,
        max_train_samples=max_train_samples,
        now=max_train_samples + 1,
    )
    genuine_pool = np.vstack(
        [segment for segment in (adaptation_pool, future_genuine_pool) if len(segment)]
    )
    if len(genuine_pool) < max_horizon:
        raise ValueError(
            f"subject {subject} only has {len(genuine_pool)} genuine samples after calibration"
        )

    out = {"subject": subject, "policies": {}}
    baseline_context = _dummy_context(hostname=f"{subject}-host")

    for policy_name in policies:
        genuine_sim = simulator.PoisoningSimulator(
            deepcopy(profile),
            adaptation_policy=policy_name,
            default_context=baseline_context,
            start_time=0.0,
            step_seconds=120.0,
        )
        genuine_steps = genuine_sim.run_sequence(
            genuine_pool[:max_horizon],
            source="genuine",
        )

        attacker_runs = {}
        for attacker_name in attackers:
            atk_sim = simulator.PoisoningSimulator(
                deepcopy(profile),
                adaptation_policy=policy_name,
                default_context=baseline_context,
                start_time=0.0,
                step_seconds=120.0,
            )
            strategy = attacks.build_strategy(attacker_name)
            steps_out = atk_sim.run_strategy(
                strategy,
                steps=max_horizon,
                source="attacker",
            )

            horizons_out = {}
            time_series = _attacker_time_series(steps_out)
            for horizon in horizons:
                attacker_metrics = _summarise_attacker_prefix(steps_out[:horizon])
                genuine_metrics = _summarise_genuine_prefix(genuine_steps[:horizon])
                precision, recall = _combine_precision_recall(
                    genuine_metrics,
                    attacker_metrics,
                )
                horizons_out[str(horizon)] = {
                    **attacker_metrics,
                    "adaptation_precision": precision,
                    "adaptation_recall": recall,
                }
            attacker_runs[attacker_name] = {
                "time_series": time_series,
                "horizons": horizons_out,
            }

        genuine_horizons = {
            str(horizon): _summarise_genuine_prefix(genuine_steps[:horizon])
            for horizon in horizons
        }
        out["policies"][policy_name] = {
            "genuine": {
                "time_series": {
                    "genuine_acceptance_over_time": _series_from_steps(
                        genuine_steps,
                        "authenticated",
                        cast=lambda value: 1.0 if value else 0.0,
                    ),
                    "genuine_score_over_time": _series_from_steps(
                        genuine_steps,
                        "probability",
                    ),
                    "genuine_anchor_displacement_over_time": _series_from_steps(
                        genuine_steps,
                        "anchor_shift",
                    ),
                },
                "horizons": genuine_horizons,
            },
            "attackers": attacker_runs,
        }
    return out


def _subject_metric(per_subject, policy_name, attacker_name, horizon, metric):
    horizon_key = str(horizon)
    if attacker_name is None:
        return per_subject["policies"][policy_name]["genuine"]["horizons"][horizon_key][metric]
    return per_subject["policies"][policy_name]["attackers"][attacker_name]["horizons"][horizon_key][metric]


def _mean_time_series(arrays):
    if not arrays:
        return []
    return np.mean(np.asarray(arrays, dtype=float), axis=0).tolist()


def _summarise_genuine_across_subjects(per_subject, policy_name, horizons, rng):
    summary = {
        "time_series": {
            "genuine_acceptance_over_time": _mean_time_series(
                [
                    subject["policies"][policy_name]["genuine"]["time_series"]["genuine_acceptance_over_time"]
                    for subject in per_subject
                ]
            ),
            "genuine_score_over_time": _mean_time_series(
                [
                    subject["policies"][policy_name]["genuine"]["time_series"]["genuine_score_over_time"]
                    for subject in per_subject
                ]
            ),
            "genuine_anchor_displacement_over_time": _mean_time_series(
                [
                    subject["policies"][policy_name]["genuine"]["time_series"]["genuine_anchor_displacement_over_time"]
                    for subject in per_subject
                ]
            ),
        },
        "horizons": {},
    }
    metric_names = (
        "genuine_acceptance_rate",
        "promotion_event_rate",
        "promoted_genuine_samples",
        "max_anchor_displacement",
        "final_anchor_displacement",
        "mean_genuine_score",
    )
    for horizon in horizons:
        horizon_key = str(horizon)
        summary["horizons"][horizon_key] = {
            metric: _summarise_metric(
                [
                    _subject_metric(subject, policy_name, None, horizon, metric)
                    for subject in per_subject
                ],
                rng,
            )
            for metric in metric_names
        }
    return summary


def _summarise_attacker_across_subjects(per_subject, policy_name, attacker_name, horizons, rng):
    summary = {
        "time_series": {
            "attacker_score_over_time": _mean_time_series(
                [
                    subject["policies"][policy_name]["attackers"][attacker_name]["time_series"]["attacker_score_over_time"]
                    for subject in per_subject
                ]
            ),
            "attacker_anchor_displacement_over_time": _mean_time_series(
                [
                    subject["policies"][policy_name]["attackers"][attacker_name]["time_series"]["attacker_anchor_displacement_over_time"]
                    for subject in per_subject
                ]
            ),
            "attacker_profile_percentage_over_time": _mean_time_series(
                [
                    subject["policies"][policy_name]["attackers"][attacker_name]["time_series"]["attacker_profile_percentage_over_time"]
                    for subject in per_subject
                ]
            ),
        },
        "horizons": {},
    }
    metric_names = (
        "authentication_acceptance_rate",
        "attacker_promotion_rate",
        "promoted_attacker_samples",
        "takeover_success",
        "attempts_to_takeover",
        "max_anchor_displacement",
        "final_anchor_displacement",
        "max_attacker_profile_percentage",
        "final_attacker_profile_percentage",
        "mean_attacker_score",
        "lockouts",
        "adaptation_precision",
        "adaptation_recall",
    )
    for horizon in horizons:
        horizon_key = str(horizon)
        summary["horizons"][horizon_key] = {
            metric: _summarise_metric(
                [
                    _subject_metric(subject, policy_name, attacker_name, horizon, metric)
                    for subject in per_subject
                ],
                rng,
            )
            for metric in metric_names
        }
    return summary


def _paired_tests(per_subject, policies, attackers, horizons):
    if FULL_POLICY not in policies:
        return {}
    output = {
        "reference_policy": FULL_POLICY,
        "ablation_mapping": POLICY_ABLATIONS,
        "results": {},
    }
    for horizon in horizons:
        horizon_key = str(horizon)
        output["results"][horizon_key] = {
            "genuine": {},
            "attackers": {},
        }
        for policy_name in policies:
            if policy_name == FULL_POLICY:
                continue
            output["results"][horizon_key]["genuine"][policy_name] = {}
            for metric in ("genuine_acceptance_rate", "final_anchor_displacement"):
                baseline = [
                    _subject_metric(subject, FULL_POLICY, None, horizon, metric)
                    for subject in per_subject
                ]
                comparison = [
                    _subject_metric(subject, policy_name, None, horizon, metric)
                    for subject in per_subject
                ]
                output["results"][horizon_key]["genuine"][policy_name][metric] = _paired_wilcoxon(
                    baseline,
                    comparison,
                )

        for attacker_name in attackers:
            output["results"][horizon_key]["attackers"][attacker_name] = {}
            for policy_name in policies:
                if policy_name == FULL_POLICY:
                    continue
                output["results"][horizon_key]["attackers"][attacker_name][policy_name] = {}
                for metric in PAIRWISE_METRICS[1:]:
                    baseline = [
                        _subject_metric(subject, FULL_POLICY, attacker_name, horizon, metric)
                        for subject in per_subject
                    ]
                    comparison = [
                        _subject_metric(subject, policy_name, attacker_name, horizon, metric)
                        for subject in per_subject
                    ]
                    output["results"][horizon_key]["attackers"][attacker_name][policy_name][metric] = _paired_wilcoxon(
                        baseline,
                        comparison,
                    )
    return output


def evaluate(
    data,
    *,
    subjects=None,
    policies=DEFAULT_POLICIES,
    attackers=DEFAULT_ATTACKERS,
    horizons=DEFAULT_HORIZONS,
    max_train_samples=60,
    progress=True,
):
    subjects = subjects or data.subject_ids
    horizons = tuple(sorted(int(horizon) for horizon in horizons))
    per_subject = []
    for index, subject in enumerate(subjects, 1):
        if progress:
            print(f"  adversarial {index:>3}/{len(subjects)}  {subject}", flush=True)
        per_subject.append(
            evaluate_subject(
                data,
                subject,
                policies=policies,
                attackers=attackers,
                horizons=horizons,
                max_train_samples=max_train_samples,
            )
        )

    rng = np.random.default_rng(config.RANDOM_SEED)
    summary = {}
    for policy_name in policies:
        summary[policy_name] = {
            "genuine": _summarise_genuine_across_subjects(
                per_subject,
                policy_name,
                horizons,
                rng,
            ),
            "attackers": {
                attacker_name: _summarise_attacker_across_subjects(
                    per_subject,
                    policy_name,
                    attacker_name,
                    horizons,
                    rng,
                )
                for attacker_name in attackers
            },
        }

    return {
        "subjects": list(subjects),
        "policies": list(policies),
        "attackers": list(attackers),
        "horizons": [int(horizon) for horizon in horizons],
        "max_train_samples": int(max_train_samples),
        "full_policy": FULL_POLICY,
        "ablations": POLICY_ABLATIONS,
        "per_subject": per_subject,
        "summary": summary,
        "paired_tests": _paired_tests(per_subject, policies, attackers, horizons),
    }
