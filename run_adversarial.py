"""Run adversarial evaluation with the sequential poisoning simulator.

    python run_adversarial.py --download
    python run_adversarial.py --limit 10
    python run_adversarial.py --json docs/adversarial-results.json
"""

import argparse
import json
import time

from bauth import config
from evaluation import adversarial, cmu


def _print_summary(results):
    print("\nAdversarial summary")
    for policy_name, policy in results["summary"].items():
        print(f"\n{policy_name}")
        for horizon in results["horizons"]:
            horizon_key = str(horizon)
            genuine = policy["genuine"]["horizons"][horizon_key]
            print(
                f"  horizon {horizon:>3}  "
                f"genuine accept {genuine['genuine_acceptance_rate']['mean']:.3f}  "
                f"final shift {genuine['final_anchor_displacement']['mean']:.3f}"
            )
            for attacker_name, attack in policy["attackers"].items():
                attack_h = attack["horizons"][horizon_key]
                print(
                    f"    {attacker_name:<26} "
                    f"accept {attack_h['authentication_acceptance_rate']['mean']:.3f}  "
                    f"promote {attack_h['attacker_promotion_rate']['mean']:.3f}  "
                    f"takeover {attack_h['takeover_success']['mean']:.3f}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=cmu.DEFAULT_PATH)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, help="evaluate the first N subjects only")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=list(adversarial.DEFAULT_HORIZONS),
    )
    parser.add_argument("--max-train-samples", type=int, default=60)
    parser.add_argument("--policies", nargs="*")
    parser.add_argument("--attackers", nargs="*")
    parser.add_argument("--json", help="write results here")
    args = parser.parse_args()

    if args.download:
        print(f"Downloading {cmu.URL}")
        size = cmu.download(args.data)
        print(f"  wrote {args.data} ({size:,} bytes)\n")

    print(f"Loading {args.data} ...")
    data = cmu.load(args.data)
    subjects = data.subject_ids[: args.limit] if args.limit else data.subject_ids
    policies = args.policies or adversarial.DEFAULT_POLICIES
    attackers = args.attackers or adversarial.DEFAULT_ATTACKERS

    started = time.time()
    results = adversarial.evaluate(
        data,
        subjects=subjects,
        policies=policies,
        attackers=attackers,
        horizons=args.horizons,
        max_train_samples=args.max_train_samples,
        progress=True,
    )
    seconds = time.time() - started
    results["seed"] = config.RANDOM_SEED
    results["seconds"] = seconds
    results["dataset"] = cmu.describe(data)
    _print_summary(results)
    print(f"\nCompleted in {seconds:.1f}s")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
