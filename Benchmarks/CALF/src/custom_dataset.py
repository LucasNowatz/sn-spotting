"""Dataset loaders for turbo clip directories (single-half, ground_truth.json)."""

import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def _load_class_config(class_set: str):
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
        raise ValueError(
            f"Unknown class_set={class_set!r}; use 'goal_1', 'foul_1', 'foul_ball_2', 'turbo_8', or 'v2_17'"
        )
    return class_cfg


def _class_symbols(class_cfg):
    return (
        class_cfg.EVENT_DICTIONARY_V2,
        class_cfg.K_V2,
        class_cfg.NUM_ACTION_CLASSES,
        class_cfg.annotation_half_and_frame,
        class_cfg.event_index,
        class_cfg.find_labels_path,
        getattr(class_cfg, "INVERSE_EVENT_DICTIONARY_V2", None),
    )


def _parse_event_class_weights(raw, num_classes: int):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        weights = [float(w) for w in raw]
    else:
        weights = [float(w.strip()) for w in str(raw).split(",") if w.strip()]
    if len(weights) != num_classes:
        raise ValueError(
            f"event_class_weights length {len(weights)} != num_classes {num_classes}"
        )
    return [max(float(w), 0.0) for w in weights]


from preprocessing import getChunks_anchors, getTimestampTargets, oneHotToShifts


FEATURE_FILE = "1_ResNET_TF2_PCA512.npy"


def load_splits(splits_path: str) -> dict[str, list[str]]:
    with open(splits_path) as fobj:
        return json.load(fobj)


def _load_labels_for_clip(clip_dir: str, find_labels_path_fn):
    labels_path = find_labels_path_fn(clip_dir)
    fps_assumed = 25.0
    if labels_path is not None:
        with open(labels_path) as fobj:
            labels = json.load(fobj)
        fps_assumed = float(labels.get("fps_assumed", 25.0))
    else:
        labels = {"annotations": []}
    return labels, fps_assumed


def _labels_to_onehot(
    labels,
    feat_len: int,
    framerate: int,
    num_classes: int,
    fps_assumed: float,
    annotation_half_and_frame_fn,
    event_index_fn,
):
    label_arr = np.zeros((feat_len, num_classes))
    for annotation in labels.get("annotations", []):
        parsed = annotation_half_and_frame_fn(
            annotation, framerate, fps_assumed=fps_assumed, default_half=1
        )
        if parsed is None:
            continue
        half, frame = parsed
        if half != 1:
            continue
        label = event_index_fn(annotation["label"])
        if label is None:
            continue
        frame = min(max(frame, 0), feat_len - 1)
        label_arr[frame][label] = 1
    return label_arr


def _pad_feat_and_labels(feat, labels, chunk_size: int):
    if feat.shape[0] >= chunk_size:
        return feat, labels
    pad = chunk_size - feat.shape[0]
    feat = np.pad(feat, ((0, pad), (0, 0)), mode="edge")
    labels = np.pad(labels, ((0, pad), (0, 0)), mode="constant", constant_values=-1)
    return feat, labels


class TurboClips(Dataset):
    """Training dataset for single-half turbo clips."""

    single_half = True

    def __init__(
        self,
        path,
        splits_path,
        split="train",
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        chunks_per_epoch=6000,
        class_set="turbo_8",
        background_weight=1.0,
        event_class_weights=None,
    ):
        self.path = path
        self.split = split
        self.framerate = framerate
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.class_set = class_set
        self.background_weight = max(float(background_weight), 0.0)

        class_cfg = _load_class_config(class_set)
        (
            _event_dict,
            k_v2,
            num_action_classes,
            self._annotation_half_and_frame,
            self._event_index,
            self._find_labels_path,
            self.inverse_event_dictionary,
        ) = _class_symbols(class_cfg)

        self.num_classes = num_action_classes
        self._event_class_weights = _parse_event_class_weights(
            event_class_weights, self.num_classes
        )
        self._event_shift_min_before_s = getattr(
            class_cfg, "EVENT_SHIFT_MIN_BEFORE_S", None
        )
        self._event_shift_min_after_s = getattr(
            class_cfg, "EVENT_SHIFT_MIN_AFTER_S", None
        )
        self.K_parameters = k_v2 * framerate
        if torch.cuda.is_available():
            self.K_parameters = self.K_parameters.cuda()
        self.num_detections = 15

        splits = load_splits(splits_path)
        self.listGames = splits[split]

        self.game_feats = []
        self.game_labels = []
        self.game_anchors = [[] for _ in range(self.num_classes + 1)]

        logging.info("Pre-compute turbo clips for split=%s (%d clips)", split, len(self.listGames))
        game_counter = 0
        for rel_clip in tqdm(self.listGames):
            clip_dir = os.path.join(self.path, rel_clip)
            feat_path = os.path.join(clip_dir, FEATURE_FILE)
            if not os.path.isfile(feat_path):
                logging.warning("Missing features: %s", feat_path)
                continue

            feat = np.load(feat_path)
            labels, fps_assumed = _load_labels_for_clip(
                clip_dir, self._find_labels_path
            )
            label_arr = _labels_to_onehot(
                labels,
                feat.shape[0],
                framerate,
                self.num_classes,
                fps_assumed,
                self._annotation_half_and_frame,
                self._event_index,
            )
            shift = oneHotToShifts(label_arr, self.K_parameters.cpu().numpy())
            feat, shift = _pad_feat_and_labels(feat, shift, self.chunk_size)
            anchors = getChunks_anchors(
                shift,
                game_counter,
                self.K_parameters.cpu().numpy(),
                self.chunk_size,
                self.receptive_field,
            )
            if not any(a[2] == self.num_classes for a in anchors):
                anchors.append([game_counter, [0, self.chunk_size - 1], self.num_classes])

            self.game_feats.append(feat)
            self.game_labels.append(shift)
            for anchor in anchors:
                self.game_anchors[anchor[2]].append(anchor)
            game_counter += 1

        if not self.game_feats:
            raise RuntimeError(f"No usable clips found for split={split}")

        self.nonempty_classes = [
            i for i, anchors in enumerate(self.game_anchors) if anchors
        ]
        if not self.nonempty_classes:
            raise RuntimeError(f"No chunk anchors found for split={split}")
        logging.info(
            "Anchor counts per class (0-%d=events, %d=background): %s",
            self.num_classes - 1,
            self.num_classes,
            [len(a) for a in self.game_anchors],
        )
        if self.background_weight == 1.0:
            logging.info("Chunk sampling: uniform over nonempty classes (legacy 1:1 for goal_1)")
        else:
            p_event = 1.0 / (1.0 + self.background_weight)
            logging.info(
                "Chunk sampling: event:bg = 1:%.1f (P(event)=%.3f, P(background)=%.3f)",
                self.background_weight,
                p_event,
                1.0 - p_event,
            )
        if self._event_class_weights is not None:
            for idx, weight in enumerate(self._event_class_weights):
                name = (
                    self.inverse_event_dictionary.get(idx, str(idx))
                    if self.inverse_event_dictionary
                    else str(idx)
                )
                logging.info("Event class sampling weight: %s (idx=%d) = %.4f", name, idx, weight)
        if (
            self._event_shift_min_before_s is not None
            and self._event_shift_min_after_s is not None
        ):
            shift_lo, shift_hi = self._event_shift_bounds()
            logging.info(
                "Event shift bounds: [%d, %d] (min_before=%.1fs, min_after=%.1fs)",
                shift_lo,
                shift_hi,
                self._event_shift_min_before_s,
                self._event_shift_min_after_s,
            )

    def _event_shift_bounds(self):
        """Return inclusive [shift_lo, shift_hi] so the keyframe stays in unmasked context."""
        half_rf = int(np.ceil(self.receptive_field / 2))
        min_before = int(np.ceil(self._event_shift_min_before_s * self.framerate))
        min_after = int(np.ceil(self._event_shift_min_after_s * self.framerate))
        last_unmasked = self.chunk_size - half_rf - 1
        k_min = half_rf + min_before
        k_max = last_unmasked - min_after
        if k_min > k_max:
            logging.warning(
                "Event shift bounds infeasible (k_min=%d > k_max=%d); using legacy range",
                k_min,
                k_max,
            )
            return (
                -self.chunk_size + self.receptive_field,
                -self.receptive_field - 1,
            )
        return (-k_max, -k_min)

    def _pick_event_class(self, event_classes):
        if self._event_class_weights is None:
            return random.choice(event_classes)
        weights = [self._event_class_weights[c] for c in event_classes]
        if sum(weights) <= 0:
            return random.choice(event_classes)
        return random.choices(event_classes, weights=weights, k=1)[0]

    def _pick_class_selection(self):
        event_classes = [c for c in self.nonempty_classes if c < self.num_classes]
        bg_class = self.num_classes
        has_bg = bg_class in self.nonempty_classes

        if not event_classes:
            return random.choice(self.nonempty_classes)
        if not has_bg:
            return self._pick_event_class(event_classes)

        if self.background_weight == 1.0 and self._event_class_weights is None:
            return random.choice(self.nonempty_classes)

        p_event = 1.0 / (1.0 + self.background_weight)
        if random.random() < p_event:
            return self._pick_event_class(event_classes)
        return bg_class

    def __getitem__(self, index):
        class_selection = self._pick_class_selection()
        event_selection = random.randint(0, len(self.game_anchors[class_selection]) - 1)
        game_index = self.game_anchors[class_selection][event_selection][0]
        anchor = self.game_anchors[class_selection][event_selection][1]

        if class_selection < self.num_classes:
            if (
                self._event_shift_min_before_s is not None
                and self._event_shift_min_after_s is not None
            ):
                shift_lo, shift_hi = self._event_shift_bounds()
                shift = np.random.randint(shift_lo, shift_hi + 1)
            else:
                shift = np.random.randint(
                    -self.chunk_size + self.receptive_field, -self.receptive_field
                )
            start = anchor + shift
        else:
            hi = max(anchor[0], anchor[1] - self.chunk_size)
            start = random.randint(anchor[0], hi)

        start = max(0, start)
        if start + self.chunk_size > self.game_feats[game_index].shape[0]:
            start = max(0, self.game_feats[game_index].shape[0] - self.chunk_size)

        clip_feat = self.game_feats[game_index][start:start + self.chunk_size]
        clip_labels = self.game_labels[game_index][start:start + self.chunk_size]

        if clip_feat.shape[0] < self.chunk_size:
            pad = self.chunk_size - clip_feat.shape[0]
            clip_feat = np.pad(clip_feat, ((0, pad), (0, 0)), mode="edge")
            clip_labels = np.pad(clip_labels, ((0, pad), (0, 0)), mode="constant", constant_values=-1)

        clip_labels[0:int(np.ceil(self.receptive_field / 2)), :] = -1
        clip_labels[-int(np.ceil(self.receptive_field / 2)):, :] = -1

        clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]
        return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets)

    def __len__(self):
        return self.chunks_per_epoch


class TurboClipsTesting(Dataset):
    """Evaluation dataset for single-half turbo clips."""

    single_half = True

    def __init__(
        self,
        path,
        splits_path,
        split="test",
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        class_set="turbo_8",
    ):
        self.path = path
        self.split = split
        self.framerate = framerate
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.class_set = class_set

        class_cfg = _load_class_config(class_set)
        (
            _event_dict,
            k_v2,
            num_action_classes,
            self._annotation_half_and_frame,
            self._event_index,
            self._find_labels_path,
            self.inverse_event_dictionary,
        ) = _class_symbols(class_cfg)

        self.num_classes = num_action_classes
        self.K_parameters = k_v2 * framerate
        if torch.cuda.is_available():
            self.K_parameters = self.K_parameters.cuda()
        self.num_detections = 15

        splits = load_splits(splits_path)
        self.listGames = [c for c in splits[split] if os.path.isfile(
            os.path.join(path, c, FEATURE_FILE)
        )]
        self.original_sizes = {}
        for rel_clip in self.listGames:
            feat_path = os.path.join(path, rel_clip, FEATURE_FILE)
            self.original_sizes[rel_clip] = int(np.load(feat_path, mmap_mode="r").shape[0])

    def __getitem__(self, index):
        rel_clip = self.listGames[index]
        clip_dir = os.path.join(self.path, rel_clip)
        feat = np.load(os.path.join(clip_dir, FEATURE_FILE))
        labels, fps_assumed = _load_labels_for_clip(clip_dir, self._find_labels_path)
        label_arr = np.zeros((feat.shape[0], self.num_classes))

        for annotation in labels.get("annotations", []):
            parsed = self._annotation_half_and_frame(
                annotation, self.framerate, fps_assumed=fps_assumed, default_half=1
            )
            if parsed is None:
                continue
            half, frame = parsed
            if half != 1:
                continue
            label = self._event_index(annotation["label"])
            if label is None:
                continue
            value = 1
            if annotation.get("visibility") == "not shown":
                value = -1
            frame = min(max(frame, 0), feat.shape[0] - 1)
            label_arr[frame][label] = value

        if feat.shape[0] < self.chunk_size:
            pad = self.chunk_size - feat.shape[0]
            feat = np.pad(feat, ((0, pad), (0, 0)), mode="edge")
            label_arr = np.pad(label_arr, ((0, pad), (0, 0)), constant_values=0)

        feat_half1 = self._feats2clip(torch.from_numpy(feat))
        label_half1 = torch.from_numpy(label_arr)

        feat_half2 = feat_half1[:1]
        label_half2 = label_half1[:1] * 0

        return feat_half1, feat_half2, label_half1, label_half2, rel_clip

    def _feats2clip(self, feats):
        stride = self.chunk_size - self.receptive_field
        clip_length = self.chunk_size
        idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
        idxs = []
        for i in torch.arange(0, clip_length):
            idxs.append(idx + i)
        idx = torch.stack(idxs, dim=1)
        idx = idx.clamp(0, feats.shape[0] - 1)
        idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length
        return feats[idx, :]

    def __len__(self):
        return len(self.listGames)
