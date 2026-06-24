"""Discover clip videos under dataset_root (turbo_result layout)."""

from __future__ import annotations

import os
from pathlib import Path

_ANNOTATION_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _ANNOTATION_ROOT.parent / "dataset_root" / "turbo_result"
_CONFIG_FILE = _ANNOTATION_ROOT / "config" / "dataset_root.txt"


def resolve_dataset_root() -> Path:
    env = os.environ.get("ANNOTATION_DATASET_ROOT")
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path.resolve()
        raise FileNotFoundError(f"ANNOTATION_DATASET_ROOT is not a directory: {path}")

    if _CONFIG_FILE.is_file():
        raw = _CONFIG_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = (_ANNOTATION_ROOT / path).resolve()
            if path.is_dir():
                return path

    if _DEFAULT_ROOT.is_dir():
        return _DEFAULT_ROOT.resolve()

    raise FileNotFoundError(
        "No dataset root found. Set ANNOTATION_DATASET_ROOT or edit config/dataset_root.txt"
    )


def _is_extracted_frames_path(path: Path) -> bool:
    return any(part.startswith(".frames_") for part in path.parts)


def discover_videos(dataset_root: Path | None = None) -> list[Path]:
    root = dataset_root or resolve_dataset_root()
    videos: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        folder = Path(dirpath)
        if _is_extracted_frames_path(folder):
            continue
        for name in sorted(filenames):
            if not name.lower().endswith(".mp4"):
                continue
            videos.append((folder / name).resolve())
    return sorted(videos, key=lambda p: str(p.relative_to(root)))


def display_name(video_path: Path, dataset_root: Path | None = None) -> str:
    root = dataset_root or resolve_dataset_root()
    try:
        return str(video_path.resolve().relative_to(root))
    except ValueError:
        return video_path.name


def resolve_label_path(video_path: str | Path) -> Path:
    import json

    video_path = Path(video_path).resolve()
    folder = video_path.parent
    for name in ("ground_truth.json", "Labels-v2.json", "Labels.json"):
        candidate = folder / name
        if candidate.is_file():
            return candidate

    created = folder / "ground_truth.json"
    payload = {"version": "1.0", "fps_assumed": 25.0, "annotations": []}
    with open(created, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return created


def infer_half(video_path: str | Path) -> int:
    stem = Path(video_path).stem
    if stem.endswith("_2"):
        return 2
    if stem.endswith("_1"):
        return 1
    name = Path(video_path).name
    if name and name[0] in ("1", "2"):
        return int(name[0])
    return 1
