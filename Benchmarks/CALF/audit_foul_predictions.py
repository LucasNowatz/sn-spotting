#!/usr/bin/env python3
"""Run CALF inference on valid/test and audit foul predictions (no eval-tolerance changes)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

CALF_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CALF_DIR, "..", ".."))
INFERENCE_DIR = os.path.join(CALF_DIR, "inference")
SRC_DIR = os.path.join(CALF_DIR, "src")

for path in (REPO_ROOT, SRC_DIR, INFERENCE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from model import ContextAwareModel  # noqa: E402
from preprocessing import timestamps2long  # noqa: E402
from metrics_visibility_fast import NMS  # noqa: E402
from json_io import predictions2json  # noqa: E402
from config import foul_ball_classes as class_cfg  # noqa: E402

FEATURE_NAME = "1_ResNET_TF2_PCA512.npy"
PREDICTIONS_NAME = "Predictions-v2.json"
PREFIX = "5HGR34ZxUMn8RvMoiD5Bb7XWbqn8RfpquFMbwa6GBqExa48w"

# CALF defaults (same as train.py / metrics_visibility_fast delta sweep minimum).
NMS_DELTA_FRAMES = 20 * 2  # 20 s window at 2 fps
AUDIT_MATCH_DELTA_FRAMES = 5 * 2  # 5 s full delta -> ±2.5 s match window


def is_foul_gt_label(label: str) -> bool:
    return "foul" in label.lower()


@dataclass
class EventRow:
    split: str
    clip_id: str
    label: str
    trained_foul: bool
    game_time: str
    position_ms: int
    frame_index_25fps: int | None
    feat_frame_2fps: int
    visibility: str
    team: str
    body_part: str


@dataclass
class PredRow:
    label: str
    class_idx: int
    game_time: str
    position_ms: int
    feat_frame_2fps: int
    confidence: float


def clip_id_from_rel(rel_path: str) -> str:
    return rel_path.split("/")[-1]


def load_splits(path: str) -> dict[str, list[str]]:
    with open(path) as fobj:
        return json.load(fobj)


def load_ground_truth(clip_dir: str) -> tuple[list[dict], float]:
    gt_path = os.path.join(clip_dir, class_cfg.GROUND_TRUTH_JSON)
    with open(gt_path) as fobj:
        data = json.load(fobj)
    return data.get("annotations", []), float(data.get("fps_assumed", 25.0))


def annotation_to_feat_frame(annotation: dict, framerate: int, fps_assumed: float) -> int | None:
    parsed = class_cfg.annotation_half_and_frame(annotation, framerate, fps_assumed)
    if parsed is None:
        return None
    return parsed[1]


def load_predictions_json(pred_path: str, framerate: int) -> list[PredRow]:
    if not os.path.isfile(pred_path):
        return []
    with open(pred_path) as fobj:
        data = json.load(fobj)
    rows: list[PredRow] = []
    for pred in data.get("predictions", []):
        label = pred.get("label", "")
        class_idx = class_cfg.event_index(label)
        if class_idx is None:
            continue
        ann = {"gameTime": pred.get("gameTime"), "position": pred.get("position")}
        feat = class_cfg.annotation_half_and_frame(ann, framerate)
        if feat is None:
            continue
        rows.append(
            PredRow(
                label=label,
                class_idx=class_idx,
                game_time=str(pred.get("gameTime", "")),
                position_ms=int(pred.get("position", 0)),
                feat_frame_2fps=feat[1],
                confidence=float(pred.get("confidence", 0.0)),
            )
        )
    return rows


def feats2clip(feats: torch.Tensor, stride: int, clip_length: int) -> torch.Tensor:
    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = [idx + i for i in torch.arange(0, clip_length)]
    idx = torch.stack(idxs, dim=1)
    idx = idx.clamp(0, feats.shape[0] - 1)
    idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length
    return feats[idx, :]


def predict_clip(
    feat_path: str,
    model: ContextAwareModel,
    chunk_size: int,
    receptive_field: int,
    framerate: int,
    device: torch.device,
) -> np.ndarray:
    feat = np.load(feat_path)
    original_size = feat.shape[0]
    if original_size < chunk_size:
        feat = np.pad(feat, ((0, chunk_size - original_size), (0, 0)), mode="edge")

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
    return NMS(timestamp_long.numpy(), NMS_DELTA_FRAMES)[:original_size]


def run_inference_for_clips(
    dataset_root: str,
    clip_rels: list[str],
    output_root: str,
    weights: str,
    chunk_size_s: int,
    receptive_field_s: int,
    framerate: int,
    device: torch.device,
    overwrite: bool,
) -> None:
    chunk_frames = chunk_size_s * framerate
    receptive_frames = receptive_field_s * framerate

    model = ContextAwareModel(
        input_size=512,
        num_classes=class_cfg.NUM_ACTION_CLASSES,
        chunk_size=chunk_frames,
        dim_capsule=16,
        receptive_field=receptive_frames,
        num_detections=15,
        framerate=framerate,
    ).to(device)

    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    logging.info("Loaded weights from %s", weights)

    for rel in tqdm(clip_rels, desc="Inference"):
        clip_dir = os.path.join(dataset_root, rel)
        feat_path = os.path.join(clip_dir, FEATURE_NAME)
        out_dir = os.path.join(output_root, rel)
        pred_path = os.path.join(out_dir, PREDICTIONS_NAME)

        if not os.path.isfile(feat_path):
            logging.warning("Missing features: %s", feat_path)
            continue
        if os.path.isfile(pred_path) and not overwrite:
            continue

        detections = predict_clip(
            feat_path, model, chunk_frames, receptive_frames, framerate, device
        )
        os.makedirs(out_dir, exist_ok=True)
        predictions2json(
            detections,
            out_dir + os.sep,
            framerate,
            inverse_event_dictionary=class_cfg.INVERSE_EVENT_DICTIONARY_V2,
        )


def match_half_width() -> int:
    return AUDIT_MATCH_DELTA_FRAMES // 2


def nearest_pred(
    preds: list[PredRow], feat_frame: int, class_idx: int | None = None
) -> PredRow | None:
    candidates = preds if class_idx is None else [p for p in preds if p.class_idx == class_idx]
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p.feat_frame_2fps - feat_frame))


def match_within(preds: list[PredRow], feat_frame: int, class_idx: int) -> PredRow | None:
    half = match_half_width()
    in_window = [
        p for p in preds if p.class_idx == class_idx and abs(p.feat_frame_2fps - feat_frame) <= half
    ]
    if not in_window:
        return None
    return max(in_window, key=lambda p: p.confidence)


def nearest_gt_label(
    all_annotations: list[dict], feat_frame: int, framerate: int, fps_assumed: float
) -> dict | None:
    best = None
    best_dist = None
    for ann in all_annotations:
        ff = annotation_to_feat_frame(ann, framerate, fps_assumed)
        if ff is None:
            continue
        dist = abs(ff - feat_frame)
        if best is None or dist < best_dist:
            best = ann
            best_dist = dist
    return best


def collect_foul_gt(split: str, clip_rel: str, dataset_root: str, framerate: int) -> list[EventRow]:
    clip_dir = os.path.join(dataset_root, clip_rel)
    annotations, fps_assumed = load_ground_truth(clip_dir)
    rows: list[EventRow] = []
    for ann in annotations:
        label = ann.get("label", "")
        if not is_foul_gt_label(label):
            continue
        feat = annotation_to_feat_frame(ann, framerate, fps_assumed)
        if feat is None:
            continue
        rows.append(
            EventRow(
                split=split,
                clip_id=clip_id_from_rel(clip_rel),
                label=label,
                trained_foul=(label == "Foul"),
                game_time=str(ann.get("gameTime", "")),
                position_ms=int(ann.get("position", 0)),
                frame_index_25fps=ann.get("frame_index"),
                feat_frame_2fps=feat,
                visibility=str(ann.get("visibility", "")),
                team=str(ann.get("team", "")),
                body_part=str(ann.get("body_part", "")),
            )
        )
    return rows


def audit_foul_events(
    split: str,
    clip_rels: list[str],
    dataset_root: str,
    pred_root: str,
    framerate: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for rel in clip_rels:
        pred_path = os.path.join(pred_root, rel, PREDICTIONS_NAME)
        preds = load_predictions_json(pred_path, framerate)
        foul_events = collect_foul_gt(split, rel, dataset_root, framerate)

        for ev in foul_events:
            foul_match = match_within(preds, ev.feat_frame_2fps, 0)
            ball_near = match_within(preds, ev.feat_frame_2fps, 1)
            nearest_foul = nearest_pred(preds, ev.feat_frame_2fps, 0)
            nearest_ball = nearest_pred(preds, ev.feat_frame_2fps, 1)

            if foul_match is not None:
                outcome = "TP"
                weakness = ""
                if foul_match.confidence < 0.1:
                    weakness = "low_conf"
                err_s = (foul_match.feat_frame_2fps - ev.feat_frame_2fps) / framerate
                if err_s < -0.5:
                    weakness = (weakness + ";early").strip(";")
                elif err_s > 0.5:
                    weakness = (weakness + ";late").strip(";")
            else:
                outcome = "FN"
                weakness = "confused_ball_out" if ball_near is not None else "no_detection"

            records.append(
                {
                    "split": split,
                    "clip_id": ev.clip_id,
                    "outcome": outcome,
                    "gt_label": ev.label,
                    "trained_foul": ev.trained_foul,
                    "gameTime": ev.game_time,
                    "position_ms": ev.position_ms,
                    "frame_index_25fps": ev.frame_index_25fps,
                    "feat_frame_2fps": ev.feat_frame_2fps,
                    "visibility": ev.visibility,
                    "team": ev.team,
                    "body_part": ev.body_part,
                    "match_foul_gameTime": foul_match.game_time if foul_match else "",
                    "match_foul_position_ms": foul_match.position_ms if foul_match else "",
                    "match_foul_confidence": round(foul_match.confidence, 6) if foul_match else "",
                    "match_foul_time_error_s": round(
                        (foul_match.feat_frame_2fps - ev.feat_frame_2fps) / framerate, 3
                    )
                    if foul_match
                    else "",
                    "nearest_foul_confidence": round(nearest_foul.confidence, 6) if nearest_foul else "",
                    "nearest_foul_offset_s": round(
                        (nearest_foul.feat_frame_2fps - ev.feat_frame_2fps) / framerate, 3
                    )
                    if nearest_foul
                    else "",
                    "nearest_ball_confidence": round(nearest_ball.confidence, 6) if nearest_ball else "",
                    "nearest_ball_offset_s": round(
                        (nearest_ball.feat_frame_2fps - ev.feat_frame_2fps) / framerate, 3
                    )
                    if nearest_ball
                    else "",
                    "weakness_tag": weakness,
                }
            )
    return records


def audit_clip_whole(
    clip_rel: str,
    dataset_root: str,
    pred_root: str,
    framerate: int,
) -> tuple[list[dict], list[dict], dict]:
    clip_dir = os.path.join(dataset_root, clip_rel)
    clip_id = clip_id_from_rel(clip_rel)
    annotations, fps_assumed = load_ground_truth(clip_dir)
    pred_path = os.path.join(pred_root, clip_rel, PREDICTIONS_NAME)
    preds = load_predictions_json(pred_path, framerate)
    half = match_half_width()

    pred_records = [
        {
            "clip_id": clip_id,
            "pred_label": p.label,
            "pred_class_idx": p.class_idx,
            "gameTime": p.game_time,
            "position_ms": p.position_ms,
            "feat_frame_2fps": p.feat_frame_2fps,
            "confidence": round(p.confidence, 6),
        }
        for p in preds
    ]

    confusion_rows: list[dict] = []
    foul_preds = [p for p in preds if p.class_idx == 0]

    for p in foul_preds:
        gt_foul = None
        for ann in annotations:
            if not is_foul_gt_label(ann.get("label", "")):
                continue
            ff = annotation_to_feat_frame(ann, framerate, fps_assumed)
            if ff is not None and abs(ff - p.feat_frame_2fps) <= half:
                gt_foul = ann
                break

        nearest = nearest_gt_label(annotations, p.feat_frame_2fps, framerate, fps_assumed)
        if gt_foul is not None:
            outcome = "TP_foul_pred"
            nearest_label = gt_foul.get("label", "")
            nearest_offset_s = round(
                (annotation_to_feat_frame(gt_foul, framerate, fps_assumed) - p.feat_frame_2fps)
                / framerate,
                3,
            )
        else:
            outcome = "FP_foul_pred"
            nearest_label = nearest.get("label", "") if nearest else ""
            nearest_offset_s = round(
                (annotation_to_feat_frame(nearest, framerate, fps_assumed) - p.feat_frame_2fps)
                / framerate,
                3,
            ) if nearest else ""

        confusion_rows.append(
            {
                "clip_id": clip_id,
                "outcome": outcome,
                "pred_gameTime": p.game_time,
                "pred_position_ms": p.position_ms,
                "pred_feat_frame_2fps": p.feat_frame_2fps,
                "pred_confidence": round(p.confidence, 6),
                "nearest_gt_label": nearest_label,
                "nearest_gt_offset_s": nearest_offset_s,
            }
        )

    foul_gts = [a for a in annotations if is_foul_gt_label(a.get("label", ""))]
    for ann in foul_gts:
        ff = annotation_to_feat_frame(ann, framerate, fps_assumed)
        if ff is None:
            continue
        if match_within(preds, ff, 0) is None:
            ball_near = match_within(preds, ff, 1)
            nearest_foul = nearest_pred(preds, ff, 0)
            confusion_rows.append(
                {
                    "clip_id": clip_id,
                    "outcome": "FN_foul_gt",
                    "gt_label": ann.get("label", ""),
                    "gameTime": ann.get("gameTime", ""),
                    "position_ms": ann.get("position", ""),
                    "feat_frame_2fps": ff,
                    "nearest_foul_conf": round(nearest_foul.confidence, 6) if nearest_foul else "",
                    "nearest_foul_offset_s": round(
                        (nearest_foul.feat_frame_2fps - ff) / framerate, 3
                    )
                    if nearest_foul
                    else "",
                    "ball_in_window": bool(ball_near),
                    "ball_conf": round(ball_near.confidence, 6) if ball_near else "",
                }
            )

    fp_confusion = Counter(
        r["nearest_gt_label"] for r in confusion_rows if r["outcome"] == "FP_foul_pred"
    )
    fn_rows = [r for r in confusion_rows if r["outcome"] == "FN_foul_gt"]
    fn_by_label = Counter(r["gt_label"] for r in fn_rows)

    summary = {
        "clip_id": clip_id,
        "num_foul_predictions": len(foul_preds),
        "num_foul_gt": len(foul_gts),
        "foul_tp": sum(1 for r in confusion_rows if r["outcome"] == "TP_foul_pred"),
        "foul_fp": sum(1 for r in confusion_rows if r["outcome"] == "FP_foul_pred"),
        "foul_fn": len(fn_rows),
        "fp_confused_with_gt_label": dict(fp_confusion.most_common()),
        "fn_by_gt_label": dict(fn_by_label),
        "audit_match_half_width_s": match_half_width() / framerate,
        "nms_window_s": NMS_DELTA_FRAMES / framerate,
    }
    return pred_records, confusion_rows, summary


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        logging.warning("No rows for %s", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Wrote %s (%d rows)", path, len(rows))


def summarize_foul_audit(all_rows: list[dict], framerate: int) -> dict:
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_split[row["split"]].append(row)

    summary: dict[str, Any] = {
        "audit_match_half_width_s": match_half_width() / framerate,
        "nms_window_s": NMS_DELTA_FRAMES / framerate,
        "splits": {},
    }

    for split, rows in by_split.items():
        tp = [r for r in rows if r["outcome"] == "TP"]
        fn = [r for r in rows if r["outcome"] == "FN"]
        confs = [float(r["match_foul_confidence"]) for r in tp if r["match_foul_confidence"] != ""]

        summary["splits"][split] = {
            "total_foul_gt": len(rows),
            "tp": len(tp),
            "fn": len(fn),
            "recall_at_match": round(len(tp) / len(rows), 4) if rows else 0.0,
            "gt_by_label": dict(Counter(r["gt_label"] for r in rows)),
            "fn_by_label": dict(Counter(r["gt_label"] for r in fn)),
            "fn_weakness_tags": dict(Counter(r["weakness_tag"] for r in fn if r["weakness_tag"])),
            "tp_confidence_mean": round(float(np.mean(confs)), 4) if confs else None,
            "tp_confidence_median": round(float(np.median(confs)), 4) if confs else None,
            "worst_fn_clips": Counter(r["clip_id"] for r in fn).most_common(10),
        }
    return summary


def write_006_summary(path: str, summary: dict, confusion_rows: list[dict]) -> None:
    half_s = summary["audit_match_half_width_s"]
    lines = [
        "# Clip 006 foul analysis",
        "",
        f"- Match window: ±{half_s}s (CALF default minimum delta, unchanged)",
        f"- Foul GT events: **{summary['num_foul_gt']}**",
        f"- Foul predictions: **{summary['num_foul_predictions']}**",
        f"- TP: **{summary['foul_tp']}** | FP: **{summary['foul_fp']}** | FN: **{summary['foul_fn']}**",
        "",
        "## False foul predictions — nearest GT label",
        "",
    ]
    for label, count in summary.get("fp_confused_with_gt_label", {}).items():
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Missed fouls by GT label", ""])
    for label, count in summary.get("fn_by_gt_label", {}).items():
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Example FP foul preds (first 15)", ""])
    for r in [x for x in confusion_rows if x.get("outcome") == "FP_foul_pred"][:15]:
        lines.append(
            f"- {r['pred_gameTime']} conf={r['pred_confidence']} → nearest GT `{r['nearest_gt_label']}` "
            f"(offset {r['nearest_gt_offset_s']}s)"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fobj:
        fobj.write("\n".join(lines) + "\n")
    logging.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit foul predictions on valid/test splits.")
    parser.add_argument(
        "--dataset_root",
        default=os.path.join(REPO_ROOT, "dataset_root", "turbo_result"),
    )
    parser.add_argument(
        "--splits_path",
        default=os.path.join(REPO_ROOT, "dataset_root", "turbo_result", "splits_foul_ball.json"),
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(CALF_DIR, "models", "CALF_turbo_foul_ball", "model.pth.tar"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(CALF_DIR, "outputs", "audit"),
    )
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--receptive_field", type=int, default=6)
    parser.add_argument("--framerate", type=int, default=2)
    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    splits = load_splits(args.splits_path)
    valid_clips = splits["valid"]
    test_clips = splits["test"]
    all_clips = valid_clips + test_clips

    if args.GPU >= 0 and torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    pred_root = os.path.join(args.output_dir, "predictions")
    if not args.skip_inference:
        run_inference_for_clips(
            args.dataset_root,
            all_clips,
            pred_root,
            args.weights,
            args.chunk_size,
            args.receptive_field,
            args.framerate,
            device,
            args.overwrite,
        )

    valid_rows = audit_foul_events("valid", valid_clips, args.dataset_root, pred_root, args.framerate)
    test_rows = audit_foul_events("test", test_clips, args.dataset_root, pred_root, args.framerate)
    all_foul_rows = valid_rows + test_rows

    audit_dir = args.output_dir
    write_csv(os.path.join(audit_dir, "foul_audit_valid.csv"), valid_rows)
    write_csv(os.path.join(audit_dir, "foul_audit_test.csv"), test_rows)

    summary = summarize_foul_audit(all_foul_rows, args.framerate)
    summary_path = os.path.join(audit_dir, "foul_audit_summary.json")
    with open(summary_path, "w") as fobj:
        json.dump(summary, fobj, indent=2)
    logging.info("Wrote %s", summary_path)

    clip_006_rel = f"{PREFIX}/006"
    pred_records, confusion_rows, summary_006 = audit_clip_whole(
        clip_006_rel, args.dataset_root, pred_root, args.framerate
    )
    write_csv(os.path.join(audit_dir, "006_whole_clip_preds.csv"), pred_records)
    write_csv(os.path.join(audit_dir, "006_foul_confusion.csv"), confusion_rows)
    with open(os.path.join(audit_dir, "006_summary.json"), "w") as fobj:
        json.dump(summary_006, fobj, indent=2)
    write_006_summary(os.path.join(audit_dir, "006_summary.md"), summary_006, confusion_rows)


if __name__ == "__main__":
    main()
