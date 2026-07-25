"""Contextual risk scoring.

Why the context attributes are scored here instead of being appended to the
feature vector in features.py:

* They are constant throughout enrollment, so a classifier trained on them
  would learn "local_ip == 192.168.1.7 implies authentic". That is trivially
  spoofable and it collapses the moment the user changes network.
* They are categorical. Standard-scaling an encoded IP is meaningless, and the
  synthetic negative generator perturbs timings -- adding Gaussian noise to an
  encoded hostname produces nothing an impostor would ever look like.
* The user's network legitimately changes far more often than their typing
  rhythm does, so context belongs on a slower, separate axis.

So context forms a second layer: the biometric decides *how* the password was
typed, and this decides whether *where and when* looks normal. The two combine
into the final decision in adaptive.verify.
"""

from dataclasses import dataclass, field

from . import config
from .context import from_dict

# Weight each signal contributes to the total risk score.
_WEIGHTS = {
    "device": 0.45,
    "username": 0.20,
    "timezone": 0.20,
    "os": 0.15,
    "subnet": 0.15,
    "public_ip": 0.10,
    "hour": 0.10,
    "layout": 0.10,
}

LOW = "low"
ELEVATED = "elevated"
HIGH = "high"
BASELINE = "baseline"


@dataclass
class RiskAssessment:
    score: float = 0.0
    level: str = BASELINE
    factors: list = field(default_factory=list)

    @property
    def is_elevated(self):
        return self.level in (ELEVATED, HIGH)

    @property
    def is_high(self):
        return self.level == HIGH

    def describe(self):
        if not self.factors:
            return f"{self.level} (score {self.score:.2f}) - context matches history"
        return f"{self.level} (score {self.score:.2f}) - " + "; ".join(self.factors)


def _seen(history, attribute):
    values = set()
    for entry in history:
        value = getattr(from_dict(entry), attribute, None)
        if value:
            values.add(value)
    return values


def _hour_is_unusual(hour, history, tolerance=3):
    hours = [from_dict(e).hour_of_day for e in history]
    if not hours:
        return False
    # Circular distance so 23:00 and 01:00 count as two hours apart.
    return all(min(abs(hour - h), 24 - abs(hour - h)) > tolerance for h in hours)


def assess(context, history):
    """Compare ``context`` against previously seen contexts for this user."""
    if not history:
        return RiskAssessment(0.0, BASELINE, [])

    factors = []
    score = 0.0

    def flag(key, message):
        nonlocal score
        score += _WEIGHTS[key]
        factors.append(message)

    fingerprints = {from_dict(e).device_fingerprint for e in history}
    if context.device_fingerprint not in fingerprints:
        flag("device", f"unrecognised device ({context.hostname})")

    if context.username and context.username not in _seen(history, "username"):
        flag("username", f"new OS account ({context.username})")

    if context.timezone_name not in _seen(history, "timezone_name"):
        flag("timezone", f"new timezone ({context.timezone_name})")

    if context.os_name and context.os_name not in _seen(history, "os_name"):
        flag("os", f"new operating system ({context.os_name})")

    subnets = {from_dict(e).subnet for e in history}
    if context.subnet not in subnets:
        flag("subnet", f"new network ({context.subnet})")

    if context.public_ip:
        known_public = _seen(history, "public_ip")
        if known_public and context.public_ip not in known_public:
            flag("public_ip", f"new public IP ({context.public_ip})")

    if _hour_is_unusual(context.hour_of_day, history):
        flag("hour", f"unusual hour ({context.hour_of_day:02d}:00)")

    if context.keyboard_layout:
        layouts = _seen(history, "keyboard_layout")
        if layouts and context.keyboard_layout not in layouts:
            flag("layout", "different keyboard layout")

    score = min(score, 1.0)
    if score >= config.RISK_HIGH:
        level = HIGH
    elif score >= config.RISK_ELEVATED:
        level = ELEVATED
    else:
        level = LOW

    return RiskAssessment(score, level, factors)


def required_probability(base_threshold, assessment):
    """Raise the biometric bar when the context looks unusual.

    An unfamiliar device does not prove an impostor, so the default is to demand
    stronger biometric evidence rather than to hard-block. Set
    ``RISK_BLOCK_ENABLED`` to refuse high-risk contexts outright.
    """
    if assessment.is_high:
        return max(base_threshold + config.ELEVATED_PROB_BONUS, config.HIGH_RISK_MIN_PROB)
    if assessment.is_elevated:
        return base_threshold + config.ELEVATED_PROB_BONUS
    return base_threshold
