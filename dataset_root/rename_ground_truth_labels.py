#!/usr/bin/env python3
"""Rename specific labels in turbo_result ground_truth.json files."""

import argparse
import glob
import json
import os
from collections import Counter

LABEL_RENAMES = {
    "ball_out_of_play_clear": "Ball out of play",
    "ball_out_of_play_distant": "Ball out of play",
    "shot": "Shots on target",
    "free_kick": "Direct free-kick",
    "throw_in": "Throw-in",
    "foul": "Foul",
    "corner": "Corner",
    "goal": "Goal",
    "substitution": "Substitution",
}


def main():
    parser = argparse.ArgumentParser(description="Rename labels in ground_truth.json files.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "turbo_result"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    renamed = Counter()
    final_labels = Counter()
    files_changed = 0

    paths = sorted(glob.glob(os.path.join(args.dataset_root, "**", "ground_truth.json"), recursive=True))
    if not paths:
        raise SystemExit(f"No ground_truth.json under {args.dataset_root}")

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        changed = False
        for ann in data.get("annotations", []):
            old = ann.get("label")
            if old in LABEL_RENAMES:
                new = LABEL_RENAMES[old]
                ann["label"] = new
                renamed[f"{old} -> {new}"] += 1
                changed = True
            final_labels[ann.get("label")] += 1

        if changed:
            files_changed += 1
            if not args.dry_run:
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")

    print(f"Files scanned: {len(paths)}")
    print(f"Files changed: {files_changed}")
    print("\nRenames applied:")
    for mapping, cnt in renamed.most_common():
        print(f"  {mapping:55s} {cnt}")
    print("\nFinal label distribution:")
    for label, cnt in final_labels.most_common():
        print(f"  {label:30s} {cnt}")


if __name__ == "__main__":
    main()
