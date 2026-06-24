"""8-class turbo action vocabulary (matches ground_truth.json label names)."""

from __future__ import annotations

import os

import torch

NUM_ACTION_CLASSES = 8

# Label strings as stored in turbo_result ground_truth.json
EVENT_DICTIONARY: dict[str, int] = {
    "Goal": 0,
    "Foul": 1,
    "Ball out of play": 2,
    "Substitution": 3,
    "Shots on target": 4,
    "Direct free-kick": 5,
    "Throw-in": 6,
    "Corner": 7,
}

INVERSE_EVENT_DICTIONARY: dict[int, str] = {
    idx: name for name, idx in EVENT_DICTIONARY.items()
}

EVENT_DICTIONARY_V2 = EVENT_DICTIONARY
INVERSE_EVENT_DICTIONARY_V2 = INVERSE_EVENT_DICTIONARY

GROUND_TRUTH_JSON = "ground_truth.json"
LABELS_V2_JSON = "Labels-v2.json"
LABEL_FILE_CANDIDATES = (GROUND_TRUTH_JSON, LABELS_V2_JSON)

# K parameters (seconds) subset from SoccerNet-17:
# Goal, Foul, Ball out of play, Substitution, Shots on target,
# Direct free-kick, Throw-in, Corner
K_V2 = torch.FloatTensor(
    [
        [-20, -75, -99, -40, -5, -97, -31, -75],
        [-10, -37, -50, -20, -3, -49, -15, -38],
        [60, 37, 50, 10, 3, 49, 15, 38],
        [90, 75, 99, 20, 5, 97, 31, 75],
    ]
)


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
