"""Command line interface for the keystroke authentication system."""

import os
import sys

from . import adaptive, capture, config, features, models, storage


def _collect_samples(password, count, extended, purpose="enrollment"):
    """Gather ``count`` clean typing samples of ``password``.

    Returns a list of ``(feature_vector, context)`` pairs.
    """
    collected = []
    attempts = 0
    limit = count * 3 + 5  # generous: a mistyped sample costs nothing but time

    print(f"\nType the password {count} times for {purpose}.")
    print("Type it the way you naturally would - the rhythm is what is learned.")
    print("Backspace works; press Enter after each attempt.\n")

    while len(collected) < count:
        print(f"[{len(collected) + 1}/{count}]")
        recorder, typed, context = capture.collect_keystroke_data(password)
        attempts += 1

        if recorder is not None:
            collected.append((features.from_capture(recorder, extended=extended), context))
            note = f"  captured{f' ({recorder.corrections} correction(s))' if recorder.corrections else ''}"
            print(note)

        if attempts >= limit:
            print(f"\nStopping after {attempts} attempts with {len(collected)} good samples.")
            break

    return collected


def _prompt_model_choice():
    raw = input("Enter choice of model (1 - Harsh / 2 - Easy): ")
    return models.normalize_choice(raw)


def register_user():
    user_id = input("Enter user ID: ").strip()
    if not user_id:
        print("\nUser ID cannot be empty.")
        return
    if storage.exists(user_id):
        print(f"\nUser '{user_id}' already exists. Use the retrain option instead.")
        return

    password = input("Enter password: ")
    if not password:
        print("\nPassword cannot be empty.")
        return
    choice = _prompt_model_choice()

    samples = _collect_samples(
        password, config.ENROLL_SAMPLES, config.EXTENDED_FEATURES, "enrollment"
    )
    if len(samples) < config.ENROLL_SAMPLES:
        print("\nRegistration failed due to insufficient data.")
        return

    profile, info = adaptive.enroll(user_id, password, samples, choice_train=choice)
    storage.save(profile)

    context = samples[-1][1]
    print(f"\nUser '{user_id}' registered successfully.")
    print(f"  Features per sample : {profile.feature_dim}")
    print(f"  Samples             : {info['authentic_samples']}")
    print(f"  Synthetic negatives : {info['negatives']}")
    print(f"  Context recorded    : {context.summary()}")


def retrain_user():
    user_id = input("Enter user ID: ").strip()
    profile = storage.load(user_id)
    if profile is None:
        print(f"\nUser '{user_id}' does not exist. Please register first.")
        return

    password = input("Enter password: ")
    if not profile.check_password(password):
        print("\nPassword mismatch! Cannot retrain model.")
        return

    before = adaptive.detect_drift(profile)
    print(f"\nCurrent profile: {profile.sample_count} samples.")
    print(f"Drift check: {before.describe()}")

    failures = adaptive.analyse_failures(profile)
    if failures.message:
        print(f"Recent rejections: {failures.message}")

    if profile.is_legacy:
        print(
            "\nThis profile was created by an older version and uses the legacy "
            "feature set. Retraining will upgrade it to the extended features "
            "(richer timings plus captured context)."
        )
        if input("Continue? (y/n): ").strip().lower() != "y":
            return
        # Old samples lack the release timings the new layout needs, so the
        # window has to be rebuilt from freshly captured typing.
        profile.authentic = None
        profile.sample_meta = []
        profile.extended = config.EXTENDED_FEATURES
        profile.schema_version = config.SCHEMA_VERSION
        needed = config.ENROLL_SAMPLES
    else:
        needed = config.RETRAIN_SAMPLES

    choice = _prompt_model_choice()
    samples = _collect_samples(password, needed, profile.extended, "retraining")
    if not samples:
        print("\nRetraining failed due to insufficient data.")
        return

    info, drift = adaptive.retrain(profile, samples, choice_train=choice)
    storage.save(profile)

    print(f"\nUser '{user_id}' model retrained successfully.")
    print(f"  Window size         : {profile.sample_count} samples")
    print(f"  Effective positives : {info['effective_positives']} (recency-weighted)")
    print(f"  Synthetic negatives : {info['negatives']}")
    print(f"  Drift before retrain: {drift.magnitude:.2f} sd")


def verify_user():
    user_id = input("Enter user ID: ").strip()
    profile = storage.load(user_id)
    if profile is None:
        print(f"\nUser '{user_id}' does not exist. Please register first.")
        return

    password = input("Enter password: ")
    if not profile.check_password(password):
        print("\nWrong initial password.")
        return

    print("\nNow type the password the way you normally do.")
    recorder, typed, context = capture.collect_keystroke_data(password)
    if recorder is None:
        print("\nVerification cancelled - the password was not typed correctly.")
        return

    vector = features.from_capture(recorder, extended=profile.extended)
    result = adaptive.verify(profile, vector, context)

    print(f"\nBiometric score : {result.probability:.3f}")
    print(f"Required score  : {result.required:.3f} (base {result.base_threshold:.3f})")
    print(f"Context risk    : {result.assessment.describe()}")

    if result.authenticated:
        print("\nUser verified successfully!")
        if result.adopted:
            print("  This sample was added to your profile (adaptive learning).")
        if result.retrained:
            print("  Profile automatically retrained on your recent typing.")
        if result.lockout:
            print(f"  Adaptive learning paused: {result.lockout}")
    else:
        print(f"\nUser verification failed - {result.reason}")
        analysis = result.failure_analysis
        if analysis is not None and analysis.message:
            label = {
                "drift": "Your typing appears to have changed",
                "attack": "Warning",
            }.get(analysis.verdict, "Note")
            print(f"\n  {label}: {analysis.message}")

    storage.save(profile)


def show_status():
    users = storage.list_users()
    if not users:
        print("\nNo registered users.")
        return
    print(f"\nRegistered users: {', '.join(users)}")

    user_id = input("Enter user ID for details (blank to skip): ").strip()
    if not user_id:
        return

    profile = storage.load(user_id)
    if profile is None:
        print(f"\nUser '{user_id}' does not exist.")
        return

    print(f"\nProfile: {user_id}")
    for key, value in adaptive.status(profile).items():
        print(f"  {key.replace('_', ' '):<16}: {value}")

    if profile.history:
        print("\n  Recent events:")
        for entry in profile.history[-5:]:
            import time as _time

            stamp = _time.strftime(
                "%Y-%m-%d %H:%M", _time.localtime(entry.get("timestamp", 0))
            )
            print(f"    {stamp}  {entry.get('event')}")


def cli():
    os.makedirs(config.USER_DATA_PATH, exist_ok=True)
    print("Welcome to the Keystroke Authentication System!")

    actions = {
        "1": register_user,
        "2": retrain_user,
        "3": verify_user,
        "4": show_status,
    }

    while True:
        print("\nSelect an option:")
        print("1. Register a new user")
        print("2. Retrain an existing user model")
        print("3. Verify an existing user")
        print("4. Profile status and drift report")
        print("5. Exit")

        try:
            choice = input("Enter choice (1-5): ").strip()
        except EOFError:
            # stdin closed (piped input exhausted, or Ctrl+D).
            print("\nExiting Keystroke Authentication System.")
            return

        if choice == "5":
            print("\nExiting Keystroke Authentication System.")
            sys.exit()

        action = actions.get(choice)
        if action is None:
            print("\nInvalid choice. Please try again.")
            continue

        try:
            action()
        except KeyboardInterrupt:
            print("\nCancelled.")
        except EOFError:
            print("\nInput ended; returning to the menu.")
