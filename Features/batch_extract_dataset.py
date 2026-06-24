#!/usr/bin/env python3
"""Batch-extract ResNET TF2 PCA512 features for clip directories (e.g. turbo_result)."""

import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from VideoFeatureExtractor import PCAReducer, VideoFeatureExtractor  # noqa: E402


FEATURE_NAME = "1_ResNET_TF2_PCA512.npy"


def find_mp4(clip_dir):
    matches = glob.glob(os.path.join(clip_dir, "*.mp4"))
    if not matches:
        return None
    matches.sort()
    return matches[0]


def needs_extraction(output_path):
    if not os.path.exists(output_path):
        return True
    try:
        feat = np.load(output_path, mmap_mode="r")
        return feat.ndim != 2 or feat.shape[1] != 512
    except Exception:
        return True


def collect_clip_dirs(dataset_root):
    clip_dirs = set()
    for mp4_path in glob.glob(os.path.join(dataset_root, "**", "*.mp4"), recursive=True):
        clip_dirs.add(os.path.dirname(mp4_path))
    for gt_path in glob.glob(os.path.join(dataset_root, "**", "ground_truth.json"), recursive=True):
        clip_dirs.add(os.path.dirname(gt_path))
    return sorted(clip_dirs)


def main():
    parser = argparse.ArgumentParser(description="Batch extract PCA512 features for clip dataset.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing clip folders with mp4 files",
    )
    parser.add_argument("--pca", type=str, default=None, help="PCA pickle (default: Features/pca_512_TF2.pkl)")
    parser.add_argument(
        "--pca_scaler",
        type=str,
        default=None,
        help="PCA scaler pickle (default: Features/average_512_TF2.pkl)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id; use -1 for CPU")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pca = args.pca or os.path.join(repo_root, "Features", "pca_512_TF2.pkl")
    pca_scaler = args.pca_scaler or os.path.join(repo_root, "Features", "average_512_TF2.pkl")

    if args.gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    clip_dirs = collect_clip_dirs(args.dataset_root)
    if not clip_dirs:
        raise SystemExit(f"No mp4 files found under {args.dataset_root}")

    todo = []
    for clip_dir in clip_dirs:
        video_path = find_mp4(clip_dir)
        if video_path is None:
            continue
        output_path = os.path.join(clip_dir, FEATURE_NAME)
        if args.overwrite or needs_extraction(output_path):
            todo.append((video_path, output_path))

    print(f"Found {len(clip_dirs)} clips; {len(todo)} need extraction.")

    if not todo:
        return

    extractor = VideoFeatureExtractor(FPS=args.fps)
    reducer = PCAReducer(pca_file=pca, scaler_file=pca_scaler)

    failed = []
    for video_path, output_path in tqdm(todo, desc="Extracting"):
        tmp_path = output_path + ".raw.tmp.npy"
        try:
            extractor.extractFeatures(
                path_video_input=video_path,
                path_features_output=tmp_path,
                overwrite=True,
            )
            reducer.reduceFeatures(
                input_features=tmp_path,
                output_features=output_path,
                overwrite=True,
            )
            os.remove(tmp_path)
        except Exception as exc:
            failed.append((video_path, str(exc)))
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    print(f"Done. Success: {len(todo) - len(failed)} / {len(todo)}")
    if failed:
        print("Failures:")
        for video_path, err in failed[:20]:
            print(f"  {video_path}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
