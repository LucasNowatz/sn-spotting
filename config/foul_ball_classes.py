"""Foul + Ball out of play vocabulary: 2 event classes + implicit background during training."""

from __future__ import annotations

import os

import torch

NUM_ACTION_CLASSES = 2

EVENT_DICTIONARY: dict[str, int] = {
    "Foul": 0,
    "Ball out of play": 1,
}

INVERSE_EVENT_DICTIONARY: dict[int, str] = {
    idx: name for name, idx in EVENT_DICTIONARY.items()
}

EVENT_DICTIONARY_V2 = EVENT_DICTIONARY
INVERSE_EVENT_DICTIONARY_V2 = INVERSE_EVENT_DICTIONARY

GROUND_TRUTH_JSON = "ground_truth.json"
LABELS_V2_JSON = "Labels-v2.json"
LABEL_FILE_CANDIDATES = (GROUND_TRUTH_JSON, LABELS_V2_JSON)

# SoccerNet-V2 class indices for partial CALF checkpoint transfer (target idx -> v2 idx).
PRETRAINED_V2_CLASS_MAP: dict[int, int] = {
    0: 10,  # Foul
    1: 8,   # Ball out of play
}
PRETRAINED_SOURCE_NUM_CLASSES = 17

# Temporal context windows (seconds): [K0 outer-before, K1 inner-before, K2 inner-after, K3 outer-after].
# Foul: K2=+16 s (whistle / immediate aftermath), K3=+24 s (tail to background). Ball out: +10 / +20 s.
K_V2 = torch.FloatTensor(
    [
        [-2, -5],
        [-0.5, -1],
        [16, 10],
        [24, 20],
    ]
)

# Minimum unmasked context around the event keyframe when sampling training windows (seconds).
# With RF=6 s and chunk=20 s, shift in [-17, -10]; see train script docs for unmasked scope.
EVENT_SHIFT_MIN_BEFORE_S = 2.0
EVENT_SHIFT_MIN_AFTER_S = 8.0


def event_index(label: str) -> int | None:
    return EVENT_DICTIONARY.get(label)


def find_labels_path(game_dir: str) -> str | None:
    for name in LABEL_FILE_CANDIDATES:
        path = os.path.join(game_dir, name)
        if os.path.isfile(path):
            return path
    return None


def annotation_half_and_frame(
    annotation: dict,
    framerate: int,
    fps_assumed: float = 25.0,
    default_half: int = 1,
) -> tuple[int, int] | None:
    """Return (half, frame_index) from a label row, or None if unusable."""
    half: int | None = None
    frame: int | None = None

    position = annotation.get("position")
    if position is not None:
        try:
            frame = int(framerate * (int(position) / 1000))
        except (TypeError, ValueError):
            frame = None

    game_time = annotation.get("gameTime")
    if game_time is not None:
        try:
            half = int(str(game_time)[0])
            minutes = int(game_time[-5:-3])
            seconds = int(game_time[-2:])
            frame = framerate * (seconds + 60 * minutes)
        except (TypeError, ValueError):
            return None

    if frame is None and annotation.get("frame_index") is not None:
        try:
            frame = int(framerate * (float(annotation["frame_index"]) / fps_assumed))
        except (TypeError, ValueError):
            frame = None

    if half is None:
        half = default_half

    if frame is None:
        return None
    return half, frame
