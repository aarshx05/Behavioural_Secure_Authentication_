"""Adaptation-policy baselines for sequential profile updates."""

from . import config


class AdaptationPolicy:
    name = "policy"

    def on_authentication(self, runtime):
        raise NotImplementedError


class FrozenPolicy(AdaptationPolicy):
    name = "frozen"

    def on_authentication(self, runtime):
        runtime.result.lockout = "adaptation policy is frozen"


class NaiveAcceptedUpdatePolicy(AdaptationPolicy):
    name = "naive_accepted_update"

    def on_authentication(self, runtime):
        runtime.promote_immediately(
            "naive_accepted_update",
            bounded=False,
            enforce_bounds=False,
        )


class HighConfidencePolicy(AdaptationPolicy):
    name = "high_confidence"

    def on_authentication(self, runtime):
        if not runtime.common_gate(
            require_update_threshold=True,
            require_quality=True,
            require_replay_guard=True,
        ):
            return
        runtime.promote_immediately(
            "high_confidence",
            bounded=False,
            enforce_bounds=False,
        )


class SlidingWindowPolicy(AdaptationPolicy):
    name = "sliding_window"

    def on_authentication(self, runtime):
        if not runtime.common_gate(require_quality=True, require_replay_guard=True):
            return
        runtime.promote_immediately(
            "sliding_window",
            bounded=False,
            enforce_bounds=False,
        )


class ConfidenceAndContextPolicy(AdaptationPolicy):
    name = "confidence_and_context"

    def on_authentication(self, runtime):
        if not runtime.common_gate(
            require_update_threshold=True,
            require_context=True,
            require_quality=True,
            require_replay_guard=True,
        ):
            return
        runtime.promote_immediately(
            "confidence_and_context",
            bounded=False,
            enforce_bounds=False,
        )


class AnchorBoundedPolicy(AdaptationPolicy):
    name = "anchor_bounded"

    def on_authentication(self, runtime):
        if not runtime.common_gate(
            require_update_threshold=True,
            require_context=True,
            require_anchor=True,
            require_quality=True,
            require_replay_guard=True,
        ):
            return
        runtime.promote_immediately("anchor_bounded", bounded=True)


class ConsensusAnchorPolicy(AdaptationPolicy):
    name = "consensus_anchor"

    def on_authentication(self, runtime):
        if not runtime.common_gate(
            require_update_threshold=True,
            require_disagreement=True,
            require_context=True,
            require_anchor=True,
            require_quality=True,
            require_replay_guard=True,
        ):
            return
        runtime.promote_immediately("consensus_anchor", bounded=True)


class QuarantineAnchorPolicy(AdaptationPolicy):
    name = "quarantine_anchor"

    def on_authentication(self, runtime):
        if not runtime.common_gate(
            require_update_threshold=True,
            require_context=True,
            require_anchor=True,
            require_quality=True,
            require_replay_guard=True,
        ):
            return
        runtime.queue_quarantine()
        runtime.maybe_promote_quarantine(bounded=True, enforce_bounds=True)


class QuarantineConsensusNoAnchorPolicy(AdaptationPolicy):
    name = "quarantine_consensus_no_anchor"

    def on_authentication(self, runtime):
        if not runtime.common_gate(
            require_update_threshold=True,
            require_disagreement=True,
            require_context=True,
            require_quality=True,
            require_replay_guard=True,
        ):
            return
        runtime.queue_quarantine()
        runtime.maybe_promote_quarantine(bounded=False, enforce_bounds=False)


class QuarantineConsensusAnchorPolicy(AdaptationPolicy):
    name = "quarantine_consensus_anchor"

    def on_authentication(self, runtime):
        if not runtime.common_gate(
            require_update_threshold=True,
            require_disagreement=True,
            require_context=True,
            require_anchor=True,
            require_quality=True,
            require_replay_guard=True,
        ):
            return
        runtime.queue_quarantine()
        runtime.maybe_promote_quarantine()


class SupervisedPeriodicUpdatePolicy(AdaptationPolicy):
    name = "supervised_periodic_update"

    def on_authentication(self, runtime):
        if not runtime.common_gate(require_quality=True, require_replay_guard=True):
            return
        state = runtime.profile.policy_state.setdefault(self.name, {})
        state["accepted_since_supervision"] = int(state.get("accepted_since_supervision", 0)) + 1
        if state["accepted_since_supervision"] >= config.SUPERVISED_RETRAIN_INTERVAL:
            runtime.profile.log(
                "supervised_update_due",
                count=state["accepted_since_supervision"],
            )
            runtime.result.lockout = "manual supervised retrain recommended before further adaptation"


POLICIES = {
    policy.name: policy
    for policy in (
        FrozenPolicy(),
        NaiveAcceptedUpdatePolicy(),
        HighConfidencePolicy(),
        SlidingWindowPolicy(),
        ConfidenceAndContextPolicy(),
        AnchorBoundedPolicy(),
        ConsensusAnchorPolicy(),
        QuarantineAnchorPolicy(),
        QuarantineConsensusNoAnchorPolicy(),
        QuarantineConsensusAnchorPolicy(),
        SupervisedPeriodicUpdatePolicy(),
    )
}


def get_policy(name):
    return POLICIES.get(name, POLICIES[config.DEFAULT_ADAPTATION_POLICY])
