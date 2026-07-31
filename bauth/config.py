"""Tunable constants for the keystroke authentication system.

Everything the system's behaviour depends on lives here so it can be tuned
without hunting through the code.
"""

# --- Reproducibility ---------------------------------------------------------
# One seed drives every stochastic step: synthetic negative generation, the
# SVM's probability calibration folds, and the random forest. Given the same
# samples and the same seed, training is bit-for-bit repeatable, which is what
# lets an experiment be re-run and checked.
#
# Two sources of non-determinism remain by design and must be pinned explicitly
# when running experiments:
#   * recency weighting depends on wall-clock age, so models.train() and
#     adaptive.fit_profile() take a `now` argument -- pass a fixed timestamp;
#   * context capture reads the live clock and network.
RANDOM_SEED = 20260726

# --- Storage -----------------------------------------------------------------
USER_DATA_PATH = "user_data"

# Bumped when the on-disk profile layout changes. Profiles written by the
# original single-file version have no version field and are treated as v1.
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1

# --- Feature extraction ------------------------------------------------------
# When True, samples carry release-derived timings (UD/UU) and statistical
# aggregates on top of the v1 [total, hold..., dd...] prefix. Turning this off
# reproduces the original v1 feature vector exactly, which is useful for A/B
# comparison. A profile records the setting it was enrolled with, so flipping
# this does not break existing users.
EXTENDED_FEATURES = True

# --- Enrollment / retraining -------------------------------------------------
ENROLL_SAMPLES = 10
RETRAIN_SAMPLES = 5

# Sliding window: only the newest N authentic samples are kept. Older ones are
# dropped so the profile tracks how the user types *now*, not a year ago.
MAX_AUTHENTIC_SAMPLES = 60

# Recency weighting. A sample this old counts half as much as a fresh one.
# Implemented by replicating recent rows (see models.replicate_by_recency)
# because KNeighborsClassifier does not accept sample_weight, so a real
# sample_weight passthrough would break the VotingClassifier.
RECENCY_HALF_LIFE_DAYS = 30.0
MAX_REPLICATION = 4

# --- Detector fusion --------------------------------------------------------
DEFAULT_SCALER = "mad"       # standard | mad | winsorized_mad
SCALER_CLIP = 5.0
WINSOR_LIMIT = 0.05
MANHATTAN_SCALE_FLOOR = 0.20

GLOBAL_FUSION_WEIGHTS = {
    "anchor": 0.35,
    "svm": 0.30,
    "knn": 0.20,
    "rf": 0.15,
}
FUSION_SHRINK_SAMPLES = 25
DEFAULT_ADAPTATION_POLICY = "quarantine_consensus_anchor"

# --- Separate thresholds ----------------------------------------------------
AUTH_THRESHOLD_FLOOR = 0.55
UPDATE_THRESHOLD_MARGIN = 0.10
UPDATE_THRESHOLD_FLOOR = 0.80
DISAGREEMENT_LIMIT = 0.12
UPDATE_RISK_LIMIT = 0.25
ANCHOR_CANDIDATE_LIMIT = 2.00

# --- Adaptive learning -------------------------------------------------------
# A successful verification enters quarantine only when it is well clear of the
# authentication bar, the detector ensemble agrees, and the sample still sits
# close to the trusted anchor profile.
ADAPTIVE_LEARNING = True
QUARANTINE_MIN_SAMPLES = 3
QUARANTINE_MAX_SAMPLES = 12
QUARANTINE_CONSISTENCY_LIMIT = 1.25
QUARANTINE_MIN_SPAN_SECONDS = 30.0
PROMOTION_ALPHA = 0.05
TRUST_DISAGREEMENT_SCALE = 0.25
MAX_PROMOTION_FEATURE_STEP = 0.75
SUPERVISED_RETRAIN_INTERVAL = 5

# Anti-poisoning bound. Every adopted sample must individually beat the bar
# above, which limits how far any single login can move the template -- but an
# attacker with sustained access could still walk it across in small steps. So
# the window centroid is also kept within this many standard deviations of the
# anchor recorded at the last password-verified enroll/retrain. Past that,
# auto-adoption stops and an explicit retrain is required to re-anchor.
#
# A share-of-profile cap cannot do this job: the window slides, so enrollment
# samples age out and the profile legitimately becomes mostly auto-adopted,
# which would disable adaptation permanently.
MAX_TEMPLATE_DRIFT = 2.5
MAX_PROMOTION_SHIFT = 0.35
PER_FEATURE_DRIFT_LIMIT = 3.0

# --- Sample quality and replay detection -----------------------------------
QUALITY_REJECT_FLOOR = 0.25
QUALITY_UPDATE_FLOOR = 0.70
QUALITY_MAX_RESOLUTION_RATIO = 0.60
QUALITY_MAX_REPEAT_RATIO = 0.80
QUALITY_MAX_TOTAL_Z = 4.0
REPLAY_QUANTIZATION_MS = 2.0
MAX_FINGERPRINT_HISTORY = 64

# --- Profile versioning -----------------------------------------------------
MAX_PROFILE_VERSIONS = 12

# --- Drift detection ---------------------------------------------------------
DRIFT_MIN_SAMPLES = 10
DRIFT_WINDOW = 5          # samples compared at each end of the history
DRIFT_Z_THRESHOLD = 1.0   # mean |z| shift above this is reported as drift
STRONG_DRIFT_Z_THRESHOLD = 1.75
LIKELY_POISONING_ANCHOR_FRACTION = 0.80
HETEROGENEOUS_CLUSTER_THRESHOLD = 1.75

# Drift measured over stored samples can only see logins that were accepted.
# Once a user drifts far enough to start failing, nothing new enters the
# profile and that measure goes blind -- reporting "stable" while the user is
# locked out. Rejected attempts that supplied the correct password are
# therefore kept (for reporting only, never for training) so the system can
# still say "your typing seems to have changed".
MAX_FAILURE_HISTORY = 10
CONSECUTIVE_FAILURE_HINT = 3
# Rejected attempts clustered more tightly than this look like one person
# typing consistently differently (drift). Scattered ones look like several
# different people (attacks).
FAILURE_COHESION_MAX = 1.5

# --- Thresholds --------------------------------------------------------------
STATIC_THRESHOLD = 0.4
MIN_THRESHOLD = 0.30
MAX_THRESHOLD = 0.70
PROB_HISTORY_SIZE = 10
# Scores needed before the threshold is derived from the user's own history.
# Waiting for a full history left new profiles running on the static global
# guess for a long stretch; five scores already beat a one-size-fits-all
# constant, and the multiplier below is widened for small samples to stay
# conservative.
MIN_SCORES_FOR_DYNAMIC = 5
# Number of standard deviations below the genuine mean the threshold sits at.
THRESHOLD_STD_MULTIPLIER = 1.5

# --- Contextual risk ---------------------------------------------------------
# Public IP lookup contacts a third-party service, so it is opt-in and off by
# default. Everything else in context.py is collected purely locally.
ENABLE_PUBLIC_IP_LOOKUP = False
PUBLIC_IP_URL = "https://api.ipify.org"
PUBLIC_IP_TIMEOUT = 2.0

RISK_ELEVATED = 0.40
RISK_HIGH = 0.75
# How much harder the biometric bar gets when context looks unusual.
ELEVATED_PROB_BONUS = 0.15
HIGH_RISK_MIN_PROB = 0.85
# When False, a high-risk context tightens the bar but never hard-blocks.
RISK_BLOCK_ENABLED = False

# --- Synthetic negatives -----------------------------------------------------
# Total negatives are scaled to the positive count rather than being a fixed
# multiplier per sample, which is what made the original data set explode on
# every retrain (10 samples became 2500 negatives after two retrains).
#
# 8:1 measured best: impostor scores stay at ~0.01 while genuine samples keep a
# healthy margin. The original's effective 100:1 pushed genuine scores down
# without buying any additional separation.
NEGATIVE_RATIO = 8
MIN_NEGATIVES = 60
MAX_NEGATIVES = 1500
