# Adversarial Evaluation Report

Run date: Saturday, August 1, 2026

Artifact: [adversarial-full-results.json](/C:/Users/12345/OneDrive/Documents/Aarsh%20All%20Data/Personal%20Projects(AntiGravity)/Behavioural_Secure_Authentication_/docs/adversarial-full-results.json)

## Setup

- Subjects: all 51 CMU users.
- Attacker horizons: 25, 50, 100, 250 attempts.
- Policies: `frozen`, `high_confidence`, `anchor_bounded`, `consensus_anchor`, `quarantine_anchor`, `quarantine_consensus_no_anchor`, `quarantine_consensus_anchor`.
- Attackers: `random_attacker`, `replay_attacker`, `time_scaled_replay_attacker`, `mimicry_attacker`, `gradual_walk_attacker`, `feedback_aware_attacker`.
- Takeover definition: attacker-origin samples occupy at least 50% of the 60-sample active window.
- Adaptation precision: genuine promoted samples / (genuine promoted samples + attacker promoted samples) at the same horizon.
- Adaptation recall: genuine promoted samples / genuine attempts at the same horizon.

The raw JSON contains:

- 95% bootstrap confidence intervals for every reported mean.
- Paired Wilcoxon signed-rank tests across users against the full method `quarantine_consensus_anchor`.
- Per-user distributions in each metric's `values` array.
- Mean time series under `summary.*.genuine.time_series.*` and `summary.*.attackers.*.time_series.*`.

## Main Findings

- No evaluated policy reached the takeover threshold for any attacker up to 250 attempts. `takeover_success_rate = 0.0` everywhere, so `attempts_to_takeover` is undefined throughout this run.
- The full method `quarantine_consensus_anchor` keeps attacker-induced profile contamination modest even for the strongest replay-family attacks, but its genuine acceptance falls steadily with longer horizons.
- Removing quarantine sharply improves attacker rejection, but it also destroys genuine acceptance and adaptation recall.
- Removing anchor bounds raises replay-family acceptance and attacker share in the active profile.
- Removing consensus is closest to the full method overall, but it is materially weaker against mimicry at 250 attempts.

## Full Method

Policy: `quarantine_consensus_anchor`

### Genuine User Retention

| Horizon | Genuine acceptance rate | 95% CI | Max anchor displacement | Final anchor displacement |
| --- | ---: | --- | ---: | ---: |
| 25 | 0.4353 | [0.3694, 0.5012] | 0.0429 | 0.0429 |
| 50 | 0.3569 | [0.2765, 0.4373] | 0.0574 | 0.0574 |
| 100 | 0.2606 | [0.1937, 0.3375] | 0.0767 | 0.0767 |
| 250 | 0.1565 | [0.1118, 0.2144] | 0.0948 | 0.0948 |

### Attacker Results

| Horizon | Attacker | Authentication acceptance | Promotion rate | Takeover success | Max anchor displacement | Final anchor displacement | Final attacker profile % | Adaptation precision | Adaptation recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | `replay_attacker` | 0.5741 | 0.0267 | 0.0000 | 0.0136 | 0.0136 | 4.8039 | 0.4152 | 0.0706 |
| 25 | `time_scaled_replay_attacker` | 0.6902 | 0.0173 | 0.0000 | 0.0177 | 0.0177 | 2.8105 | 0.5686 | 0.0706 |
| 25 | `mimicry_attacker` | 0.5953 | 0.0243 | 0.0000 | 0.0340 | 0.0340 | 3.1046 | 0.4679 | 0.0706 |
| 25 | `gradual_walk_attacker` | 0.8447 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0706 |
| 25 | `feedback_aware_attacker` | 0.4494 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0706 |
| 25 | `random_attacker` | 0.4753 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0706 |
| 50 | `replay_attacker` | 0.4384 | 0.0157 | 0.0000 | 0.0173 | 0.0173 | 7.1569 | 0.3800 | 0.0471 |
| 50 | `time_scaled_replay_attacker` | 0.5639 | 0.0133 | 0.0000 | 0.0370 | 0.0370 | 7.5163 | 0.4119 | 0.0471 |
| 50 | `mimicry_attacker` | 0.4706 | 0.0129 | 0.0000 | 0.0389 | 0.0389 | 3.8889 | 0.4866 | 0.0471 |
| 50 | `gradual_walk_attacker` | 0.8522 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0471 |
| 50 | `feedback_aware_attacker` | 0.4439 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0471 |
| 50 | `random_attacker` | 0.4663 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0471 |
| 100 | `replay_attacker` | 0.3496 | 0.0078 | 0.0000 | 0.0173 | 0.0173 | 7.1569 | 0.4338 | 0.0312 |
| 100 | `time_scaled_replay_attacker` | 0.4451 | 0.0071 | 0.0000 | 0.0407 | 0.0407 | 8.3007 | 0.4163 | 0.0312 |
| 100 | `mimicry_attacker` | 0.3775 | 0.0071 | 0.0000 | 0.0446 | 0.0446 | 4.7712 | 0.5128 | 0.0312 |
| 100 | `gradual_walk_attacker` | 0.8220 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0312 |
| 100 | `feedback_aware_attacker` | 0.4414 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0312 |
| 100 | `random_attacker` | 0.4455 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0312 |
| 250 | `replay_attacker` | 0.2969 | 0.0031 | 0.0000 | 0.0173 | 0.0173 | 7.1569 | 0.4961 | 0.0158 |
| 250 | `time_scaled_replay_attacker` | 0.3464 | 0.0030 | 0.0000 | 0.0433 | 0.0433 | 9.0850 | 0.4533 | 0.0158 |
| 250 | `mimicry_attacker` | 0.2871 | 0.0031 | 0.0000 | 0.0549 | 0.0549 | 6.3399 | 0.5096 | 0.0158 |
| 250 | `gradual_walk_attacker` | 0.8106 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0158 |
| 250 | `feedback_aware_attacker` | 0.4419 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0158 |
| 250 | `random_attacker` | 0.4297 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0158 |

## Ablation Snapshot At 250 Attempts

The three requested single-removal ablations are:

- Remove quarantine: `consensus_anchor`
- Remove consensus: `quarantine_anchor`
- Remove anchor bounds: `quarantine_consensus_no_anchor`

### Means

| Policy | Genuine acceptance | Final anchor displacement | Replay accept | Replay final attacker % | Time-scaled accept | Time-scaled final attacker % | Mimicry accept | Mimicry final attacker % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `quarantine_consensus_anchor` | 0.1565 | 0.0948 | 0.2969 | 7.1569 | 0.3464 | 9.0850 | 0.2871 | 6.3399 |
| `consensus_anchor` | 0.0193 | 0.0326 | 0.1343 | 1.6667 | 0.0233 | 1.6667 | 0.0176 | 1.6667 |
| `quarantine_anchor` | 0.1266 | 0.1015 | 0.2969 | 7.1569 | 0.3425 | 8.6275 | 0.1620 | 7.3203 |
| `quarantine_consensus_no_anchor` | 0.1824 | 0.0917 | 0.4281 | 11.7647 | 0.4135 | 10.6863 | 0.3079 | 6.9281 |

### Paired Tests Against The Full Method At 250 Attempts

- Remove quarantine (`consensus_anchor`)
  - Genuine acceptance: `p = 7.31e-10`
  - Final anchor displacement: `p = 4.64e-06`
  - Replay acceptance: `p = 4.00e-04`
  - Replay attacker profile share: `p = 9.49e-09`
  - Adaptation recall on replay: `p = 4.03e-09`
  - Time-scaled replay acceptance: `p = 1.55e-09`
  - Mimicry acceptance: `p = 7.49e-10`
- Remove consensus (`quarantine_anchor`)
  - Genuine acceptance: `p = 8.83e-04`
  - Replay acceptance: `p = 1.00`
  - Time-scaled replay acceptance: `p = 0.203`
  - Mimicry acceptance: `p = 8.23e-06`
- Remove anchor bounds (`quarantine_consensus_no_anchor`)
  - Genuine acceptance: `p = 0.173`
  - Replay acceptance: `p = 2.62e-03`
  - Replay attacker profile share: `p = 1.76e-04`
  - Adaptation precision on replay: `p = 4.90e-03`
  - Time-scaled replay acceptance: `p = 8.25e-03`
  - Mimicry acceptance: `p = 5.85e-02`

## Where To Find The Requested Outputs

- Authentication acceptance rate:
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.authentication_acceptance_rate`
- Attacker promotion rate:
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.attacker_promotion_rate`
- Takeover success rate and attempts to takeover:
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.takeover_success`
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.attempts_to_takeover`
- Maximum and final anchor displacement:
  - `summary.<policy>.genuine.horizons.<horizon>.max_anchor_displacement`
  - `summary.<policy>.genuine.horizons.<horizon>.final_anchor_displacement`
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.max_anchor_displacement`
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.final_anchor_displacement`
- Genuine acceptance over time:
  - `summary.<policy>.genuine.time_series.genuine_acceptance_over_time`
- Attacker score over time:
  - `summary.<policy>.attackers.<attacker>.time_series.attacker_score_over_time`
- Percentage of attacker samples in the profile:
  - `summary.<policy>.attackers.<attacker>.time_series.attacker_profile_percentage_over_time`
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.final_attacker_profile_percentage`
- Adaptation precision and recall:
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.adaptation_precision`
  - `summary.<policy>.attackers.<attacker>.horizons.<horizon>.adaptation_recall`

