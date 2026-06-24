"""Background creation of lightweight streaming proxies for remote playback."""

from __future__ import annotations

import hashlib
import subprocess
import threading
from pathlib import Path

PROXY_ROOT = Path("/tmp/annotator_proxy")
_LOCK = threading.Lock()
_JOBS: dict[str, threading.Thread] = {}
_READY: set[str] = set()


def _key(video_path: Path) -> str:
    return hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:16]


def proxy_path(video_path: Path) -> Path:
    return PROXY_ROOT / f"{_key(video_path)}.mp4"


def _validate_proxy(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024 * 100:
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "h264":
        return False
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-t", "2", "-f", "null", "-"],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return decode.returncode == 0


def is_ready(video_path: Path) -> bool:
    key = _key(video_path)
    if key in _READY:
        return True
    path = proxy_path(video_path)
    if _validate_proxy(path):
        _READY.add(key)
        return True
    if path.is_file() and not _validate_proxy(path):
        path.unlink(missing_ok=True)
        _READY.discard(key)
    return False


def playback_path(video_path: Path) -> tuple[Path, bool]:
    if is_ready(video_path):
        return proxy_path(video_path), True
    ensure_proxy(video_path)
    return video_path, False


def ensure_proxy(video_path: Path) -> None:
    key = _key(video_path)
    if is_ready(video_path):
        return
    with _LOCK:
        if key in _JOBS and _JOBS[key].is_alive():
            return

        def worker() -> None:
            PROXY_ROOT.mkdir(parents=True, exist_ok=True)
            out = proxy_path(video_path)
            tmp = out.with_suffix(".part.mp4")
            tmp.unlink(missing_ok=True)
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-an",
                "-vf",
                "scale=854:-2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "main",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-movflags",
                "+faststart",
                str(tmp),
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=7200, check=False)
            if result.returncode == 0 and _validate_proxy(tmp):
                tmp.replace(out)
                _READY.add(key)
            else:
                tmp.unlink(missing_ok=True)

        thread = threading.Thread(target=worker, daemon=True)
        _JOBS[key] = thread
        thread.start()
