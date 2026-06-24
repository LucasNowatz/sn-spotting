"""Partial loading of CALF checkpoints (backbone / mapped segmentation heads)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

BACKBONE_KEYS = ("conv_1.weight", "conv_1.bias", "conv_2.weight", "conv_2.bias")


def _checkpoint_state_dict(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


def _remap_conv_seg(
    src_weight: torch.Tensor,
    src_bias: torch.Tensor,
    class_map: dict[int, int],
    dim_capsule: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tgt_num_classes = len(class_map)
    out_channels = dim_capsule * tgt_num_classes
    tgt_weight = torch.zeros(
        out_channels,
        *src_weight.shape[1:],
        dtype=src_weight.dtype,
        device=src_weight.device,
    )
    tgt_bias = torch.zeros(out_channels, dtype=src_bias.dtype, device=src_bias.device)
    for tgt_idx in range(tgt_num_classes):
        src_idx = class_map[tgt_idx]
        t0, t1 = tgt_idx * dim_capsule, (tgt_idx + 1) * dim_capsule
        s0, s1 = src_idx * dim_capsule, (src_idx + 1) * dim_capsule
        tgt_weight[t0:t1] = src_weight[s0:s1]
        tgt_bias[t0:t1] = src_bias[s0:s1]
    return tgt_weight, tgt_bias


def load_pretrained_calf(
    model: nn.Module,
    checkpoint_path: str,
    *,
    mode: str = "full",
    class_map: dict[int, int] | None = None,
    source_num_classes: int = 17,
    dim_capsule: int = 16,
) -> None:
    """Load CALF weights into *model* (ContextAwareModel).

    Modes:
      - full: strict full state dict (architecture must match).
      - backbone: conv_1 + conv_2 only (512-d feature trunk).
      - backbone_seg: backbone + conv_seg with target->source class_map.
    """
    src_state = _checkpoint_state_dict(checkpoint_path)
    mode = mode.lower()

    if mode == "full":
        model.load_state_dict(src_state)
        logging.info("Loaded full CALF checkpoint from %s", checkpoint_path)
        return

    if mode not in ("backbone", "backbone_seg"):
        raise ValueError(f"Unknown load_mode={mode!r}; use full, backbone, or backbone_seg")

    dst_state = model.state_dict()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    for key in BACKBONE_KEYS:
        if key in src_state and key in dst_state and src_state[key].shape == dst_state[key].shape:
            dst_state[key] = src_state[key]
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    if mode == "backbone_seg":
        if class_map is None:
            raise ValueError("backbone_seg requires class_map (target_idx -> source_class_idx)")
        seg_w_key, seg_b_key = "conv_seg.weight", "conv_seg.bias"
        if seg_w_key not in src_state or seg_b_key not in src_state:
            raise KeyError(f"Missing {seg_w_key} in checkpoint {checkpoint_path}")
        src_w = src_state[seg_w_key]
        src_b = src_state[seg_b_key]
        expected_src = source_num_classes * dim_capsule
        if src_w.shape[0] != expected_src:
            logging.warning(
                "conv_seg source channels %d != expected %d (source_num_classes=%d, dim_capsule=%d)",
                src_w.shape[0],
                expected_src,
                source_num_classes,
                dim_capsule,
            )
        tgt_w, tgt_b = _remap_conv_seg(src_w, src_b, class_map, dim_capsule)
        if seg_w_key in dst_state and tgt_w.shape == dst_state[seg_w_key].shape:
            dst_state[seg_w_key] = tgt_w
            dst_state[seg_b_key] = tgt_b
            loaded_keys.extend([seg_w_key, seg_b_key])
        else:
            skipped_keys.extend([seg_w_key, seg_b_key])

    model.load_state_dict(dst_state)
    logging.info(
        "Partial CALF load (%s) from %s: loaded %s",
        mode,
        checkpoint_path,
        loaded_keys,
    )
    if skipped_keys:
        logging.info("Skipped (shape mismatch or missing): %s", skipped_keys)
    logging.info(
        "Not transferred: temporal pyramid (RF may differ), batch_seg (chunk size), spotting head"
    )
