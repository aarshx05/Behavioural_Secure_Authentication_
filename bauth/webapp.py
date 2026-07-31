"""Flask dashboard for the keystroke authentication system.

Capturing keystrokes in the browser sidesteps every problem the terminal
version has: no OS keyboard hook to install (so no accessibility permissions
and no race against the hook going live), no console echo of the password, and
`keydown`/`keyup` give press *and* release timing directly.

The whole model, risk and storage stack is shared with the CLI -- this module
only handles transport and presentation.

Run with::

    python webapp.py
"""

import os
import platform
import socket

from flask import Flask, jsonify, render_template, request

from . import adaptive, attacks, capture, config, context, features, models, policies, quality, simulator, storage

app = Flask(__name__)


def _client_context():
    """Context snapshot describing the *client*, not just this server.

    ``context.capture_context`` reads the local machine, which is right for the
    CLI. Over HTTP the browser may be somewhere else, so the request's own
    address and user agent take precedence when they are available.
    """
    ctx = context.capture_context()

    remote = request.remote_addr
    if remote and remote not in ("127.0.0.1", "::1", "localhost"):
        ctx.local_ip = remote
        try:
            ctx.hostname = socket.gethostbyaddr(remote)[0]
        except (socket.herror, socket.gaierror, OSError):
            ctx.hostname = f"client@{remote}"
        # A remote browser's MAC is not visible to us; claiming the server's
        # would make every client look like the same device.
        ctx.mac_address = None

    agent = request.headers.get("User-Agent", "")
    for token, name in (
        ("Windows", "Windows"),
        ("Android", "Android"),
        ("iPhone", "iOS"),
        ("iPad", "iOS"),
        ("Mac OS", "macOS"),
        ("Linux", "Linux"),
    ):
        if token in agent:
            ctx.os_name = name
            break
    ctx.keyboard_layout = None  # not knowable from the browser
    return ctx


def _capture_from_request(payload, password):
    """Build an EventCapture from posted keystroke events."""
    return capture.EventCapture(
        payload.get("events", []),
        password,
        corrections=payload.get("corrections", 0),
    )


def _context_view(ctx):
    """Context rendered for display, grouped the way the UI shows it."""
    return {
        "network": {
            "Local IP": ctx.local_ip,
            "Subnet": ctx.subnet,
            "Public IP": ctx.public_ip or "not collected (opt-in)",
            "Hostname": ctx.hostname,
            "MAC address": ctx.mac_address or "not visible",
        },
        "device": {
            "Operating system": f"{ctx.os_name} {ctx.os_release}".strip(),
            "Machine": ctx.machine or "unknown",
            "Processor": ctx.processor or "unknown",
            "OS account": ctx.username,
            "Keyboard layout": ctx.keyboard_layout or "not visible",
            "Python": ctx.python_version,
        },
        "clock": {
            "Captured at": ctx.iso_time,
            "Hour of day": f"{ctx.hour_of_day:02d}:00",
            "Weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][ctx.weekday],
            "Timezone": ctx.timezone_name,
            "UTC offset": f"{ctx.utc_offset_minutes // 60:+d}:{abs(ctx.utc_offset_minutes) % 60:02d}",
        },
        "fingerprint": ctx.device_fingerprint,
    }


def _profile_summary(profile):
    return {
        "user_id": profile.user_id,
        "schema": profile.schema_version,
        "legacy": profile.is_legacy,
        "samples": profile.sample_count,
        "max_samples": config.MAX_AUTHENTIC_SAMPLES,
        "features": profile.feature_dim,
        "threshold": round(adaptive.dynamic_threshold(profile), 3),
        "adaptation_policy": profile.adaptation_policy,
        "status": adaptive.status(profile),
    }


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", enroll_samples=config.ENROLL_SAMPLES)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.route("/api/users")
def api_users():
    users = []
    for user_id in storage.list_users():
        profile = storage.load(user_id)
        if profile is None:
            continue
        users.append(
            {
                "user_id": user_id,
                "samples": profile.sample_count,
                "legacy": profile.is_legacy,
                "password_length": profile.password_length,
            }
        )
    return jsonify({"users": users})


@app.route("/api/register", methods=["POST"])
def api_register():
    payload = request.get_json(silent=True) or {}
    user_id = (payload.get("user_id") or "").strip()
    password = payload.get("password") or ""
    choice = models.normalize_choice(payload.get("model_choice", 1))
    policy_name = payload.get("adaptation_policy") or config.DEFAULT_ADAPTATION_POLICY
    samples = payload.get("samples", [])

    if not user_id or not password:
        return jsonify({"ok": False, "error": "User ID and password are required."}), 400
    if storage.exists(user_id):
        return jsonify({"ok": False, "error": f"User '{user_id}' already exists."}), 409

    collected = []
    for entry in samples:
        rec = _capture_from_request(entry, password)
        if not rec.complete:
            continue
        vector = features.from_capture(rec, extended=config.EXTENDED_FEATURES)
        report = quality.assess_capture(rec, vector)
        if report.acceptable:
            collected.append((vector, _client_context()))

    if len(collected) < config.ENROLL_SAMPLES:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"Only {len(collected)} of {config.ENROLL_SAMPLES} "
                    f"samples were usable. Please try again.",
                }
            ),
            400,
        )

    profile, info = adaptive.enroll(user_id, password, collected, choice_train=choice)
    profile.adaptation_policy = policies.get_policy(policy_name).name
    storage.save(profile)
    return jsonify({"ok": True, "info": info, "profile": _profile_summary(profile)})


@app.route("/api/verify", methods=["POST"])
def api_verify():
    payload = request.get_json(silent=True) or {}
    user_id = (payload.get("user_id") or "").strip()
    password = payload.get("password") or ""

    profile = storage.load(user_id)
    if profile is None:
        return jsonify({"ok": False, "error": f"User '{user_id}' does not exist."}), 404
    if not profile.check_password(password):
        return jsonify({"ok": False, "authenticated": False, "reason": "wrong_password",
                        "error": "Wrong password."}), 200

    rec = _capture_from_request(payload, password)
    if not rec.complete:
        return jsonify({"ok": False, "error": "The password was not typed correctly."}), 400

    vector = features.from_capture(rec, extended=profile.extended)
    report = quality.assess_capture(rec, vector, profile=profile)
    ctx = _client_context()
    result = adaptive.verify(profile, vector, ctx, quality_report=report)
    storage.save(profile)

    analysis = result.failure_analysis
    return jsonify(
        {
            "ok": True,
            "authenticated": result.authenticated,
            "probability": round(result.probability, 4),
            "disagreement": round(result.disagreement, 4),
            "anchor_distance": round(result.anchor_distance, 4),
            "base_threshold": round(result.base_threshold, 4),
            "required": round(result.required, 4),
            "reason": result.reason,
            "adopted": result.adopted,
            "retrained": result.retrained,
            "quarantined": result.quarantined,
            "lockout": result.lockout,
            "risk": {
                "score": round(result.assessment.score, 3),
                "level": result.assessment.level,
                "factors": result.assessment.factors,
            },
            "detectors": {
                key: round(value, 4) for key, value in result.detector_scores.items()
            },
            "quality": {
                "score": round(result.quality_score, 3),
                "flags": result.quality_flags,
                "fingerprint": result.sample_fingerprint,
            },
            "context": _context_view(ctx),
            "timing": {
                "characters": len(rec.typed),
                "total_ms": round(
                    (rec.press_times[-1] - rec.press_times[0]) * 1000, 1
                ) if len(rec.press_times) > 1 else 0.0,
                "mean_hold_ms": round(
                    sum(rec.hold_times()) / len(rec.hold_times()) * 1000, 1
                ) if rec.hold_times() else 0.0,
                "corrections": rec.corrections,
            },
            "analysis": (
                {"verdict": analysis.verdict, "message": analysis.message}
                if analysis is not None and analysis.message
                else None
            ),
        }
    )


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    payload = request.get_json(silent=True) or {}
    user_id = (payload.get("user_id") or "").strip()
    password = payload.get("password") or ""
    policy_name = payload.get("adaptation_policy")

    profile = storage.load(user_id)
    if profile is None:
        return jsonify({"ok": False, "error": f"User '{user_id}' does not exist."}), 404
    if not profile.check_password(password):
        return jsonify({"ok": False, "error": "Password mismatch - cannot retrain."}), 403

    if profile.is_legacy:
        # v1 samples have no release timings, so the window must be rebuilt.
        profile.authentic = None
        profile.sample_meta = []
        profile.extended = config.EXTENDED_FEATURES
        profile.schema_version = config.SCHEMA_VERSION

    collected = []
    for entry in payload.get("samples", []):
        rec = _capture_from_request(entry, password)
        if not rec.complete:
            continue
        vector = features.from_capture(rec, extended=profile.extended)
        report = quality.assess_capture(rec, vector, profile=profile)
        if report.acceptable:
            collected.append((vector, _client_context()))

    if not collected:
        return jsonify({"ok": False, "error": "No usable samples were captured."}), 400

    if policy_name:
        profile.adaptation_policy = policies.get_policy(policy_name).name
    info, drift = adaptive.retrain(profile, collected, choice_train=payload.get("model_choice"))
    storage.save(profile)
    return jsonify(
        {
            "ok": True,
            "info": info,
            "drift_before": round(drift.magnitude, 3),
            "profile": _profile_summary(profile),
        }
    )


@app.route("/api/status/<user_id>")
def api_status(user_id):
    profile = storage.load(user_id)
    if profile is None:
        return jsonify({"ok": False, "error": f"User '{user_id}' does not exist."}), 404

    drift = adaptive.detect_drift(profile)
    failures = adaptive.analyse_failures(profile)
    return jsonify(
        {
            "ok": True,
            "profile": _profile_summary(profile),
            "drift": {
                "detected": drift.detected,
                "state": drift.state,
                "magnitude": round(drift.magnitude, 3),
                "speed_change": round(drift.speed_change, 1),
                "message": drift.message,
                "recommendation": drift.recommendation,
            },
            "failures": {"verdict": failures.verdict, "count": failures.count,
                         "message": failures.message},
            "events": profile.history[-8:],
            "versions": profile.versions[-5:],
            "contexts": profile.context_history[-5:],
        }
    )


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    payload = request.get_json(silent=True) or {}
    user_id = (payload.get("user_id") or "").strip()
    strategy_name = payload.get("attacker") or "random_attacker"
    steps = int(payload.get("steps") or 10)
    policy_name = payload.get("adaptation_policy")
    persist = bool(payload.get("persist"))

    if not user_id:
        return jsonify({"ok": False, "error": "User ID is required."}), 400
    if steps < 1 or steps > 200:
        return jsonify({"ok": False, "error": "Steps must be between 1 and 200."}), 400

    profile = storage.load(user_id)
    if profile is None:
        return jsonify({"ok": False, "error": f"User '{user_id}' does not exist."}), 404

    active_policy = profile.adaptation_policy if not policy_name else policies.get_policy(policy_name).name
    sim = simulator.PoisoningSimulator(profile, adaptation_policy=active_policy)
    strategy = attacks.build_strategy(strategy_name)
    steps_out = sim.run_strategy(strategy, steps=steps)
    summary = sim.summary()

    if persist:
        storage.save(sim.profile)

    return jsonify(
        {
            "ok": True,
            "policy": active_policy,
            "attacker": strategy_name,
            "persisted": persist,
            "summary": summary,
            "steps": [
                {
                    "index": entry.index,
                    "source": entry.source,
                    "attack": entry.attack,
                    "authenticated": entry.authenticated,
                    "adaptation_eligible": entry.adaptation_eligible,
                    "quarantined": entry.quarantined,
                    "promoted": entry.promoted,
                    "profile_shift": round(entry.profile_shift, 4),
                    "anchor_shift": round(entry.anchor_shift, 4),
                    "probability": round(entry.probability, 4),
                    "anchor_distance": round(entry.anchor_distance, 4),
                    "disagreement": round(entry.disagreement, 4),
                    "context_risk": round(entry.context_risk, 4),
                    "quality_score": round(entry.quality_score, 4),
                    "quality_flags": entry.quality_flags,
                    "lockout": entry.lockout,
                    "risk_level": entry.risk_level,
                }
                for entry in steps_out
            ],
        }
    )


@app.route("/api/context")
def api_context():
    """Everything the risk layer can see about the current client."""
    ctx = _client_context()
    return jsonify({"ok": True, "context": _context_view(ctx)})


@app.route("/api/config")
def api_config():
    return jsonify(
        {
            "enroll_samples": config.ENROLL_SAMPLES,
            "retrain_samples": config.RETRAIN_SAMPLES,
            "static_threshold": config.STATIC_THRESHOLD,
            "auth_threshold_floor": config.AUTH_THRESHOLD_FLOOR,
            "update_threshold_floor": config.UPDATE_THRESHOLD_FLOOR,
            "default_adaptation_policy": config.DEFAULT_ADAPTATION_POLICY,
            "adaptation_policies": sorted(policies.POLICIES),
            "attack_strategies": sorted(attacks.ATTACK_STRATEGIES),
            "server": {"host": socket.gethostname(), "os": platform.system()},
        }
    )


def main(host="127.0.0.1", port=5000, debug=False):
    os.makedirs(config.USER_DATA_PATH, exist_ok=True)
    print("Keystroke Authentication dashboard")
    print(f"  http://{host}:{port}")
    print("  Press Ctrl+C to stop.\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
