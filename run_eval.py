"""Run the CMU keystroke dynamics benchmark.

    python run_eval.py --download          # fetch the dataset (~4.7 MB)
    python run_eval.py --check             # verify the loader, run nothing
    python run_eval.py --limit 5           # quick pass over 5 subjects
    python run_eval.py                     # full run, all 51 subjects
    python run_eval.py --json results.json # also write machine-readable output
"""

import argparse
import json
import sys
import time

import numpy as np

from bauth import config
from evaluation import cmu, protocol


def print_dataset(data):
    info = cmu.describe(data)
    print("Dataset")
    for key, value in info.items():
        print(f"  {key.replace('_', ' '):<20} {value}")

    ok, worst = cmu.check_consistency(data)
    status = "OK" if ok else "FAILED"
    print(f"  {'UD = DD - H check':<20} {status} (max error {worst:.2e})")
    if not ok:
        print("\n  Loader disagrees with the published columns; aborting.")
        sys.exit(1)


def print_results(results):
    width = max(len(k) for k in results) + 2
    print(f"\n{'system':<{width}} {'EER':>8} {'sd':>7} {'median':>8} "
          f"{'AUC':>7} {'0-miss FAR':>11} {'published':>10}")
    print("-" * (width + 56))

    for label, res in sorted(results.items(), key=lambda kv: kv[1]["eer"]["mean"]):
        published = protocol.PUBLISHED_EER.get(label)
        pub = f"{published:.3f}" if published else "-"
        e, a, z = res["eer"], res["auc"], res["zero_miss_far"]
        print(
            f"{label:<{width}} {e['mean']:>8.4f} {e['std']:>7.4f} "
            f"{e['median']:>8.4f} {a['mean']:>7.4f} {z['mean']:>11.4f} {pub:>10}"
        )

    print("\nEER: equal error rate, lower is better. Mean over subjects, sd across them.")
    print("0-miss FAR: impostor accept rate at the threshold that rejects no genuine sample.")

    deltas = [
        (label, res["eer"]["mean"] - protocol.PUBLISHED_EER[label])
        for label, res in results.items()
        if label in protocol.PUBLISHED_EER
    ]
    if deltas:
        worst = max(abs(d) for _, d in deltas)
        print(f"\nHarness validation: baselines differ from the published means by at "
              f"most {worst:.4f}.")
        for label, d in sorted(deltas, key=lambda kv: abs(kv[1]), reverse=True):
            print(f"  {label:<28} {d:+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=cmu.DEFAULT_PATH)
    parser.add_argument("--download", action="store_true", help="fetch the dataset first")
    parser.add_argument("--check", action="store_true", help="validate the loader only")
    parser.add_argument("--limit", type=int, help="evaluate only the first N subjects")
    parser.add_argument("--only", nargs="*", help="restrict to named systems")
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    if args.download:
        print(f"Downloading {cmu.URL}")
        size = cmu.download(args.data)
        print(f"  wrote {args.data} ({size:,} bytes)\n")

    print(f"Loading {args.data} ...")
    started = time.time()
    data = cmu.load(args.data)
    print(f"  {len(data):,} rows in {time.time() - started:.1f}s\n")

    print_dataset(data)
    if args.check:
        return

    subjects = data.subject_ids
    if args.limit:
        subjects = subjects[: args.limit]

    systems = protocol.default_systems()
    if args.only:
        systems = {k: v for k, v in systems.items() if k in args.only}
        if not systems:
            print(f"No systems matched. Available: {list(protocol.default_systems())}")
            sys.exit(1)

    print(f"\nProtocol: train on first {protocol.TRAIN_REPS} reps, test on the rest, "
          f"impostors = first {protocol.IMPOSTOR_REPS} reps of each other subject")
    print(f"Subjects: {len(subjects)}   seed: {config.RANDOM_SEED}\n")

    results = protocol.run(data, systems, subjects=subjects)
    print_results(results)

    if args.json:
        payload = {
            "seed": config.RANDOM_SEED,
            "subjects": subjects,
            "dataset": cmu.describe(data),
            "protocol": {
                "train_reps": protocol.TRAIN_REPS,
                "impostor_reps": protocol.IMPOSTOR_REPS,
            },
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=float)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
