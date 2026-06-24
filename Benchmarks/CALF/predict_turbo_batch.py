#!/usr/bin/env python3
"""Batch 17-class CALF inference for turbo_result clip directories."""

import argparse
import glob
import logging
import os
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

CALF_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CALF_DIR, "..", ".."))
INFERENCE_DIR = os.path.join(CALF_DIR, "inference")

sys.path.insert(0, INFERENCE_DIR)

from model import ContextAwareModel  # noqa: E402
from preprocessing import NMS, timestamps2long  # noqa: E402
from json_io import predictions2json  # noqa: E402

FEATURE_NAME = "1_ResNET_TF2_PCA512.npy"
PREDICTIONS_NAME = "Predictions-v2.json"
DEFAULT_EXTRACT_PYTHON = os.path.expanduser(
    "~/miniconda3/envs/SoccerNet-FeatureExtraction/bin/python"
)


def resolve_extract_python(explicit_path):
    if explicit_path:
        return explicit_path
    if os.path.isfile(DEFAULT_EXTRACT_PYTHON):
        return DEFAULT_EXTRACT_PYTHON
    return sys.executable


def collect_clip_dirs(dataset_root):
    clip_dirs = set()
    for mp4_path in glob.glob(os.path.join(dataset_root, "**", "*.mp4"), recursive=True):
        clip_dirs.add(os.path.dirname(mp4_path))
    return sorted(clip_dirs)


def run_feature_extraction(dataset_root, extract_gpu, overwrite, extract_python):
    script = os.path.join(REPO_ROOT, "Features", "batch_extract_dataset.py")
    python_bin = resolve_extract_python(extract_python)
    if python_bin == sys.executable:
        logging.warning(
            "SoccerNet-FeatureExtraction env not found; using %s. "
            "Feature extraction needs opencv + tensorflow 2.3.",
            python_bin,
        )
    cmd = [
        python_bin,
        script,
        "--dataset_root",
        dataset_root,
        "--gpu",
        str(extract_gpu),
    ]
    if overwrite:
        cmd.append("--overwrite")
    logging.info("Feature extraction command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def feats2clip(feats, stride, clip_length):
    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = []
    for i in torch.arange(0, clip_length):
        idxs.append(idx + i)
    idx = torch.stack(idxs, dim=1)
    idx = idx.clamp(0, feats.shape[0] - 1)
    idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length
    return feats[idx, :]


def predict_clip(feat_path, model, chunk_size, receptive_field, framerate, device):
    feat = np.load(feat_path)
    original_size = feat.shape[0]
    if original_size < chunk_size:
        pad = chunk_size - original_size
        feat = np.pad(feat, ((0, pad), (0, 0)), mode="edge")

    inference_size = feat.shape[0]
    stride = chunk_size - receptive_field
    feat_clip = feats2clip(torch.from_numpy(feat), stride=stride, clip_length=chunk_size)
    feat_clip = feat_clip.unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        _, output_spotting = model(feat_clip)

    timestamp_long = timestamps2long(
        output_spotting.cpu().detach(), inference_size, chunk_size, receptive_field
    )
    detections = NMS(timestamp_long.numpy(), 20 * framerate)
    return detections[:original_size]


def load_class_config(class_set):
    src_dir = os.path.join(CALF_DIR, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if class_set == "goal_1":
        from config import goal_classes as class_cfg
    elif class_set == "foul_1":
        from config import foul_classes as class_cfg
    elif class_set == "foul_ball_2":
        from config import foul_ball_classes as class_cfg
    elif class_set == "turbo_8":
        from config import turbo_classes as class_cfg
    elif class_set == "v2_17":
        from config import classes as class_cfg
    else:
        raise ValueError(f"Unknown class_set={class_set!r}")
    return class_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run CALF on all clips under dataset_root (turbo_result layout)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.path.join(REPO_ROOT, "dataset_root", "turbo_result"),
        help="Root folder containing clip subdirectories with mp4 files",
    )
    parser.add_argument("--model_name", type=str, default="CALF_benchmark")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model.pth.tar (overrides --model_name)",
    )
    parser.add_argument(
        "--class_set",
        type=str,
        default="v2_17",
        choices=("goal_1", "foul_1", "foul_ball_2", "turbo_8", "v2_17"),
        help="Label vocabulary for prediction JSON",
    )
    parser.add_argument("--num_classes", type=int, default=17)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=120,
        help="Chunk size in seconds (CALF_benchmark uses 120)",
    )
    parser.add_argument(
        "--receptive_field",
        type=int,
        default=40,
        help="Receptive field in seconds (CALF_benchmark uses 40)",
    )
    parser.add_argument("--framerate", type=int, default=2)
    parser.add_argument("--num_features", type=int, default=512)
    parser.add_argument("--dim_capsule", type=int, default=16)
    parser.add_argument("--GPU", type=int, default=0, help="GPU for CALF inference (-1 for CPU)")
    parser.add_argument(
        "--extract_gpu",
        type=int,
        default=-1,
        help="GPU for feature extraction (-1 for CPU; recommended on RTX 4090)",
    )
    parser.add_argument(
        "--extract_python",
        type=str,
        default=None,
        help="Python for feature extraction (default: SoccerNet-FeatureExtraction env)",
    )
    parser.add_argument(
        "--extract_features",
        action="store_true",
        help="Extract missing 1_ResNET_TF2_PCA512.npy before inference",
    )
    parser.add_argument(
        "--features_only",
        action="store_true",
        help="Only run feature extraction, skip CALF inference",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--loglevel", type=str, default="INFO")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"dataset_root not found: {dataset_root}")

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    clip_dirs = collect_clip_dirs(dataset_root)
    if not clip_dirs:
        raise SystemExit(f"No mp4 files found under {dataset_root}")
    logging.info("Found %d clips under %s", len(clip_dirs), dataset_root)

    if args.extract_features or args.features_only:
        logging.info("Running feature extraction (extract_gpu=%d)", args.extract_gpu)
        run_feature_extraction(
            dataset_root, args.extract_gpu, args.overwrite, args.extract_python
        )

    if args.features_only:
        return

    class_cfg = load_class_config(args.class_set)
    inverse_event_dictionary = class_cfg.INVERSE_EVENT_DICTIONARY_V2

    model_path = args.weights
    if model_path is None:
        model_path = os.path.join(CALF_DIR, "models", args.model_name, "model.pth.tar")
    if not os.path.isfile(model_path):
        raise SystemExit(
            f"Missing model weights: {model_path}\n"
            "Train with train_turbo_8class.sh or copy CALF_benchmark weights."
        )

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    chunk_frames = args.chunk_size * args.framerate
    receptive_frames = args.receptive_field * args.framerate

    model = ContextAwareModel(
        input_size=args.num_features,
        num_classes=args.num_classes,
        chunk_size=chunk_frames,
        dim_capsule=args.dim_capsule,
        receptive_field=receptive_frames,
        num_detections=15,
        framerate=args.framerate,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    logging.info("Loaded %s", model_path)

    todo = []
    skipped = 0
    missing_feat = 0
    for clip_dir in clip_dirs:
        feat_path = os.path.join(clip_dir, FEATURE_NAME)
        pred_path = os.path.join(clip_dir, PREDICTIONS_NAME)
        if not os.path.isfile(feat_path):
            missing_feat += 1
            logging.warning("Missing features, skipping: %s", feat_path)
            continue
        if os.path.isfile(pred_path) and not args.overwrite:
            skipped += 1
            continue
        todo.append((clip_dir, feat_path, pred_path))

    logging.info(
        "Predicting %d clips (%d skipped existing, %d missing features)",
        len(todo),
        skipped,
        missing_feat,
    )

    if missing_feat > 0 and not todo:
        raise SystemExit(
            f"{missing_feat} clips have no {FEATURE_NAME}. "
            "Run without --no-extract to extract features first:\n"
            "  ./predict_turbo.sh"
        )

    if missing_feat > 0:
        logging.warning(
            "%d clips still missing features and will be skipped", missing_feat
        )

    failed = []
    start = time.time()
    for clip_dir, feat_path, pred_path in tqdm(todo, desc="CALF inference"):
        try:
            detections = predict_clip(
                feat_path, model, chunk_frames, receptive_frames, args.framerate, device
            )
            predictions2json(
                detections,
                clip_dir + os.sep,
                args.framerate,
                inverse_event_dictionary=inverse_event_dictionary,
            )
            if not os.path.isfile(pred_path):
                raise RuntimeError(f"predictions not written to {pred_path}")
        except Exception as exc:
            failed.append((clip_dir, str(exc)))

    elapsed = time.time() - start
    ok = len(todo) - len(failed)
    logging.info("Done in %.1fs. Success: %d / %d", elapsed, ok, len(todo))
    if failed:
        logging.error("Failures (%d):", len(failed))
        for clip_dir, err in failed[:10]:
            logging.error("  %s: %s", clip_dir, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
