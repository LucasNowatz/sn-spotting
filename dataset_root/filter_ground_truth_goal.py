#!/usr/bin/env python3
"""Keep only Goal labels in ground_truth.json (strip other event types)."""

import argparse
import glob
import json
import os
from collections import Counter

GOAL_LABEL = "Goal"


def main():
    parser = argparse.ArgumentParser(description="Filter ground_truth.json to Goal labels only.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "turbo_result"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    removed_labels = Counter()
    files_changed = 0
    goal_count = 0
    total_before = 0
    total_after = 0

    paths = sorted(glob.glob(os.path.join(args.dataset_root, "**", "ground_truth.json"), recursive=True))
    if not paths:
        raise SystemExit(f"No ground_truth.json under {args.dataset_root}")

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        total_before += len(anns)
        kept = []
        for ann in anns:
            label = ann.get("label")
            if label == GOAL_LABEL:
                kept.append(ann)
                goal_count += 1
            else:
                removed_labels[label] += 1

        total_after += len(kept)
        if len(kept) != len(anns):
            files_changed += 1
            if not args.dry_run:
                data["annotations"] = kept
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                    f.write("\n")

    print(f"Files scanned: {len(paths)}")
    print(f"Files changed: {files_changed}")
    print(f"Goal annotations kept: {goal_count}")
    print(f"Annotations before: {total_before}")
    print(f"Annotations after:  {total_after}")
    print(f"Removed: {total_before - total_after}")
    if removed_labels:
        print("\nRemoved labels (top 15):")
        for label, cnt in removed_labels.most_common(15):
            print(f"  {label:30s} {cnt}")


if __name__ == "__main__":
    main()
