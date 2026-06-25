#!/usr/bin/env python3
"""Create train/valid/test splits for turbo clip dataset."""

import argparse
import glob
import importlib.util
import json
import os
import random
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURE_FILE = "1_ResNET_TF2_PCA512.npy"
GROUND_TRUTH = "ground_truth.json"

CLASS_SET_MODULES = {
    "goal_1": os.path.join(REPO_ROOT, "config", "goal_classes.py"),
    "foul_1": os.path.join(REPO_ROOT, "config", "foul_classes.py"),
    "foul_ball_2": os.path.join(REPO_ROOT, "config", "foul_ball_classes.py"),
    "turbo_8": os.path.join(REPO_ROOT, "config", "action_classes.py"),
    "v2_17": os.path.join(REPO_ROOT, "Benchmarks", "CALF", "src", "config", "classes.py"),
}


def load_event_dictionary(class_set: str) -> dict:
    module_path = CLASS_SET_MODULES.get(class_set)
    if module_path is None:
        raise ValueError(f"Unknown class_set={class_set!r}")
    spec = importlib.util.spec_from_file_location(f"split_{class_set}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EVENT_DICTIONARY_V2


def clip_has_valid_annotations(gt_path: str, event_dictionary: dict) -> bool:
    with open(gt_path) as fobj:
        data = json.load(fobj)
    for ann in data.get("annotations", []):
        if ann.get("label") in event_dictionary:
            return True
    return False


def count_exact_target_labels(gt_path: str, event_dictionary: dict) -> tuple[int, int, bool]:
    """Count exact Foul / Ball out labels (foul subtypes excluded)."""
    with open(gt_path) as fobj:
        data = json.load(fobj)
    foul = 0
    ball = 0
    for ann in data.get("annotations", []):
        label = ann.get("label")
        if label not in event_dictionary:
            continue
        if label == "Foul":
            foul += 1
        elif label == "Ball out of play":
            ball += 1
    return foul, ball, (foul + ball) > 0


def feat_frame_count(clip_dir: str) -> int | None:
    feat_path = os.path.join(clip_dir, FEATURE_FILE)
    if not os.path.isfile(feat_path):
        return None
    return int(np.load(feat_path, mmap_mode="r").shape[0])


def is_30s_clip(feat_frames: int, min_frames: int, max_frames: int) -> bool:
    return min_frames <= feat_frames <= max_frames


def collect_clip_dirs(
    dataset_root: str,
    require_features: bool,
    require_annotations: bool,
    event_dictionary: dict,
) -> list[str]:
    clip_dirs = []
    for gt_path in glob.glob(os.path.join(dataset_root, "**", GROUND_TRUTH), recursive=True):
        clip_dir = os.path.dirname(gt_path)
        rel = os.path.relpath(clip_dir, dataset_root)
        if require_annotations and not clip_has_valid_annotations(gt_path, event_dictionary):
            continue
        if require_features and not os.path.isfile(os.path.join(clip_dir, FEATURE_FILE)):
            continue
        clip_dirs.append(rel)
    clip_dirs.sort()
    return clip_dirs


def build_30s_eval_splits(
    dataset_root: str,
    all_clips: list[str],
    event_dictionary: dict,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    min_feat_frames: int,
    max_feat_frames: int,
) -> tuple[dict[str, list[str]], dict]:
    """Train = full feature pool minus valid/test; valid/test = 30s clips with exact Foul/Ball out."""
    eval_candidates: list[str] = []
    inventory: list[dict] = []

    for rel in all_clips:
        clip_dir = os.path.join(dataset_root, rel)
        gt_path = os.path.join(clip_dir, GROUND_TRUTH)
        feat_frames = feat_frame_count(clip_dir)
        if feat_frames is None:
            continue
        foul, ball, has_target = count_exact_target_labels(gt_path, event_dictionary)
        row = {
            "clip": rel,
            "feat_frames": feat_frames,
            "foul": foul,
            "ball_out": ball,
            "is_30s": is_30s_clip(feat_frames, min_feat_frames, max_feat_frames),
            "eval_eligible": False,
        }
        if has_target and row["is_30s"]:
            eval_candidates.append(rel)
            row["eval_eligible"] = True
        inventory.append(row)

    if not eval_candidates:
        raise SystemExit(
            f"No ~30s clips ({min_feat_frames}-{max_feat_frames} feat frames) with exact "
            "Foul / Ball out of play labels."
        )

    rng = random.Random(seed)
    rng.shuffle(eval_candidates)

    n = len(eval_candidates)
    n_train_eval = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train_eval - n_valid

    # 15% valid + 15% test from 30s pool; remaining 70% of 30s pool stays in train.
    valid = eval_candidates[n_train_eval : n_train_eval + n_valid]
    test = eval_candidates[n_train_eval + n_valid :]
    held_out = set(valid) | set(test)
    train = [c for c in all_clips if c not in held_out]

    def split_stats(clips: list[str]) -> dict:
        foul_total = 0
        ball_total = 0
        count_30s = 0
        for rel in clips:
            clip_dir = os.path.join(dataset_root, rel)
            ff = feat_frame_count(clip_dir)
            if ff is not None and is_30s_clip(ff, min_feat_frames, max_feat_frames):
                count_30s += 1
            f, b, _ = count_exact_target_labels(
                os.path.join(clip_dir, GROUND_TRUTH), event_dictionary
            )
            foul_total += f
            ball_total += b
        return {
            "clips": len(clips),
            "clips_30s": count_30s,
            "foul_labels": foul_total,
            "ball_out_labels": ball_total,
        }

    meta = {
        "strategy": "train_full_dataset_valid_test_30s_only",
        "seed": seed,
        "train_ratio_on_30s_pool": train_ratio,
        "valid_ratio_on_30s_pool": valid_ratio,
        "feat_frames_30s_range": [min_feat_frames, max_feat_frames],
        "foul_subtypes_excluded": True,
        "eval_pool_size": n,
        "eval_pool_unused_in_train": n_train_eval,
        "inventory": inventory,
        "split_stats": {
            "train": split_stats(train),
            "valid": split_stats(valid),
            "test": split_stats(test),
        },
    }
    splits = {"train": train, "valid": valid, "test": test}
    return splits, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "turbo_result"),
    )
    parser.add_argument("--output", default="splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument(
        "--class_set",
        default="turbo_8",
        choices=tuple(CLASS_SET_MODULES),
        help="Which label vocabulary to use when filtering annotated clips",
    )
    parser.add_argument(
        "--require-features",
        action="store_true",
        help="Only include clips that already have feature files",
    )
    parser.add_argument(
        "--require-annotations",
        action="store_true",
        default=True,
        help="Only include clips with at least one label from class_set (default: on)",
    )
    parser.add_argument(
        "--no-require-annotations",
        action="store_false",
        dest="require_annotations",
        help="Include all clips (with features); unlabeled frames become background",
    )
    parser.add_argument(
        "--eval-30s-valid-test",
        action="store_true",
        help=(
            "Put only ~30s clips (see --min-feat-frames/--max-feat-frames) with exact "
            "Foul/Ball out labels into valid/test; train uses the full feature pool minus those."
        ),
    )
    parser.add_argument(
        "--min-feat-frames",
        type=int,
        default=55,
        help="Minimum feature frames for ~30s eval clips (2 fps, default 55)",
    )
    parser.add_argument(
        "--max-feat-frames",
        type=int,
        default=65,
        help="Maximum feature frames for ~30s eval clips (2 fps, default 65)",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON path for split metadata (default: <output>_summary.json)",
    )
    args = parser.parse_args()

    event_dictionary = load_event_dictionary(args.class_set)
    clips = collect_clip_dirs(
        args.dataset_root,
        require_features=args.require_features,
        require_annotations=args.require_annotations,
        event_dictionary=event_dictionary,
    )
    if not clips:
        raise SystemExit(f"No clips found under {args.dataset_root}")

    meta = None
    if args.eval_30s_valid_test:
        splits, meta = build_30s_eval_splits(
            args.dataset_root,
            clips,
            event_dictionary,
            args.seed,
            args.train_ratio,
            args.valid_ratio,
            args.min_feat_frames,
            args.max_feat_frames,
        )
    else:
        rng = random.Random(args.seed)
        rng.shuffle(clips)

        n = len(clips)
        n_train = int(n * args.train_ratio)
        n_valid = int(n * args.valid_ratio)
        splits = {
            "train": clips[:n_train],
            "valid": clips[n_train : n_train + n_valid],
            "test": clips[n_train + n_valid :],
        }

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.dataset_root, output_path)

    with open(output_path, "w") as fobj:
        json.dump(splits, fobj, indent=2)
        fobj.write("\n")

    print(f"Wrote {output_path} (class_set={args.class_set})")
    for split_name, split_clips in splits.items():
        print(f"  {split_name}: {len(split_clips)}")

    if meta is not None:
        if args.summary_output is None:
            base, ext = os.path.splitext(output_path)
            summary_path = f"{base}_summary{ext or '.json'}"
        else:
            summary_path = args.summary_output
            if not os.path.isabs(summary_path) and os.path.dirname(summary_path) == "":
                summary_path = os.path.join(args.dataset_root, summary_path)
            elif not os.path.isabs(summary_path):
                summary_path = os.path.normpath(
                    os.path.join(os.path.dirname(output_path) or args.dataset_root, summary_path)
                )
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w") as fobj:
            json.dump(meta, fobj, indent=2)
            fobj.write("\n")
        print(f"Wrote {summary_path}")
        for split_name, stats in meta["split_stats"].items():
            print(
                f"  {split_name} stats: clips={stats['clips']} "
                f"(30s={stats['clips_30s']}), foul={stats['foul_labels']}, "
                f"ball_out={stats['ball_out_labels']}"
            )


if __name__ == "__main__":
    main()
