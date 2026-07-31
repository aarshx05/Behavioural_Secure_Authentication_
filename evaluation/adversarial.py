"""Adversarial evaluation using the sequential poisoning simulator.

Builds per-subject profiles from the CMU benchmark, then measures how
different adaptation policies behave under repeated attacker strategies.
"""

from copy import deepcopy

import numpy as np

from bauth import adaptive, attacks, config, context, simulator, storage

from . import cmu, protocol

DEFAULT_POLICIES = (
    "frozen",
    "high_confidence",
    "anchor_bounded",
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
        profile.add_sample(vector, context=ctx, source="enroll", timestamp=float(i))

    adaptive.fit_profile(profile, now=float(now if now is not None else len(fit_vectors) + 1))
    _calibrate_for_benchmark(profile, dev_vectors)
    profile.log("adversarial_eval_profile_built", subject=subject, samples=len(fit_vectors))
    return profile, adaptation_vectors, data.extended(genuine_idx), data.extended(impostor_idx)


def _summarise_steps(steps):
    if not steps:
        return {
            "attempts": 0,
            "accept_rate": 0.0,
            "promotion_rate": 0.0,
            "mean_anchor_shift": 0.0,
            "max_anchor_shift": 0.0,
            "mean_probability": 0.0,
        }
    return {
        "attempts": len(steps),
        "accept_rate": float(np.mean([entry.authenticated for entry in steps])),
        "promotion_rate": float(np.mean([entry.promoted for entry in steps])),
        "mean_anchor_shift": float(np.mean([entry.anchor_shift for entry in steps])),
        "max_anchor_shift": float(np.max([entry.anchor_shift for entry in steps])),
        "mean_probability": float(np.mean([entry.probability for entry in steps])),
    }


def _average_dicts(items):
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def evaluate_subject(
    data,
    subject,
    *,
    policies=DEFAULT_POLICIES,
    attackers=DEFAULT_ATTACKERS,
    genuine_steps=20,
    attacker_steps=20,
    max_train_samples=60,
):
    profile, adaptation_pool, future_genuine_pool, _ = build_profile(
        data,
        subject,
        max_train_samples=max_train_samples,
        now=max_train_samples + 1,
    )

    out = {"subject": subject, "policies": {}}
    genuine_pool = (adaptation_pool if len(adaptation_pool) else future_genuine_pool)[:genuine_steps]
    future_genuine_pool = future_genuine_pool[:genuine_steps]
    baseline_context = _dummy_context(hostname=f"{subject}-host")

    for policy_name in policies:
        sim = simulator.PoisoningSimulator(
            deepcopy(profile),
            adaptation_policy=policy_name,
            default_context=baseline_context,
            start_time=0.0,
            step_seconds=120.0,
        )
        genuine_steps_out = sim.run_sequence(genuine_pool, source="genuine")
        future_sim = simulator.PoisoningSimulator(
            deepcopy(profile),
            adaptation_policy=policy_name,
            default_context=baseline_context,
            start_time=0.0,
            step_seconds=120.0,
        )
        future_steps_out = future_sim.run_sequence(future_genuine_pool, source="genuine_future")
        attacker_results = {}
        for attacker_name in attackers:
            atk_sim = simulator.PoisoningSimulator(
                deepcopy(profile),
                adaptation_policy=policy_name,
                default_context=baseline_context,
                start_time=0.0,
                step_seconds=120.0,
            )
            strategy = attacks.build_strategy(attacker_name)
            steps_out = atk_sim.run_strategy(strategy, steps=attacker_steps, source="attacker")
            attacker_results[attacker_name] = {
                **_summarise_steps(steps_out),
                "final_anchor_shift": float(steps_out[-1].anchor_shift if steps_out else 0.0),
                "lockouts": int(sum(1 for entry in steps_out if entry.lockout)),
            }
        out["policies"][policy_name] = {
            "genuine": _summarise_steps(genuine_steps_out),
            "future_genuine": _summarise_steps(future_steps_out),
            "attackers": attacker_results,
        }
    return out


def evaluate(
    data,
    *,
    subjects=None,
    policies=DEFAULT_POLICIES,
    attackers=DEFAULT_ATTACKERS,
    genuine_steps=20,
    attacker_steps=20,
    max_train_samples=60,
    progress=True,
):
    subjects = subjects or data.subject_ids
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
                genuine_steps=genuine_steps,
                attacker_steps=attacker_steps,
                max_train_samples=max_train_samples,
            )
        )

    summary = {}
    for policy_name in policies:
        genuine = [
            subject["policies"][policy_name]["genuine"]
            for subject in per_subject
        ]
        future_genuine = [
            subject["policies"][policy_name]["future_genuine"]
            for subject in per_subject
        ]
        attack_summary = {}
        for attacker_name in attackers:
            attack_summary[attacker_name] = _average_dicts(
                [subject["policies"][policy_name]["attackers"][attacker_name] for subject in per_subject]
            )
        summary[policy_name] = {
            "genuine": _average_dicts(genuine),
            "future_genuine": _average_dicts(future_genuine),
            "attackers": attack_summary,
        }

    return {
        "subjects": list(subjects),
        "policies": list(policies),
        "attackers": list(attackers),
        "genuine_steps": int(genuine_steps),
        "attacker_steps": int(attacker_steps),
        "max_train_samples": int(max_train_samples),
        "per_subject": per_subject,
        "summary": summary,
    }
