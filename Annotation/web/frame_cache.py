"""Fast frame serving with persistent OpenCV decode + disk/memory cache."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

import cv2

CACHE_ROOT = Path("/tmp/annotator_frames")
MAX_MEMORY = 200
JPEG_QUALITY = 70
MAX_WIDTH = 720


class VideoSession:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.current_idx = -1
        self.lock = threading.Lock()
        key = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
        self.cache_dir = CACHE_ROOT / key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory: OrderedDict[int, bytes] = OrderedDict()

    def ms_to_idx(self, ms: int) -> int:
        if self.frame_count:
            return max(0, min(self.frame_count - 1, int(round(ms / 1000.0 * self.fps))))
        return max(0, int(round(ms / 1000.0 * self.fps)))

    def idx_to_ms(self, idx: int) -> int:
        return int(round(idx / self.fps * 1000))

    def _disk_path(self, idx: int) -> Path:
        return self.cache_dir / f"{idx:08d}.jpg"

    def _mem_get(self, idx: int) -> bytes | None:
        data = self.memory.get(idx)
        if data is not None:
            self.memory.move_to_end(idx)
        return data

    def _mem_put(self, idx: int, data: bytes) -> None:
        self.memory[idx] = data
        while len(self.memory) > MAX_MEMORY:
            self.memory.popitem(last=False)

    def _encode(self, frame) -> bytes:
        height, width = frame.shape[:2]
        if width > MAX_WIDTH:
            new_h = int(height * MAX_WIDTH / width)
            frame = cv2.resize(frame, (MAX_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    def _read_idx(self, idx: int):
        if idx == self.current_idx + 1:
            ok, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"frame {idx} unavailable")
        self.current_idx = idx
        return frame

    def _store(self, idx: int, data: bytes) -> bytes:
        self._disk_path(idx).write_bytes(data)
        self._mem_put(idx, data)
        return data

    def get_jpeg(self, ms: int) -> bytes:
        idx = self.ms_to_idx(ms)
        cached = self._mem_get(idx)
        if cached is not None:
            return cached
        disk = self._disk_path(idx)
        if disk.is_file():
            data = disk.read_bytes()
            self._mem_put(idx, data)
            self.current_idx = idx
            return data

        with self.lock:
            frame = self._read_idx(idx)
            data = self._encode(frame)
        return self._store(idx, data)

    def prefetch_from(self, count: int = 10) -> None:
        def worker() -> None:
            for _ in range(count):
                with self.lock:
                    idx = self.current_idx + 1
                    if self.frame_count and idx >= self.frame_count:
                        break
                    if idx in self.memory or self._disk_path(idx).is_file():
                        if self._disk_path(idx).is_file() and idx not in self.memory:
                            self._mem_put(idx, self._disk_path(idx).read_bytes())
                        self.current_idx = idx
                        continue
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        break
                    self.current_idx = idx
                    data = self._encode(frame)
                self._disk_path(idx).write_bytes(data)
                self._mem_put(idx, data)

        threading.Thread(target=worker, daemon=True).start()


class FrameServer:
    def __init__(self) -> None:
        self._sessions: dict[str, VideoSession] = {}
        self._lock = threading.Lock()

    def get_jpeg(self, video_path: Path, ms: int) -> bytes:
        key = str(video_path.resolve())
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = VideoSession(video_path)
                self._sessions[key] = session
        data = session.get_jpeg(ms)
        session.prefetch_from()
        return data

    def duration_ms(self, video_path: Path) -> int:
        key = str(video_path.resolve())
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = VideoSession(video_path)
                self._sessions[key] = session
        if session.frame_count:
            return session.idx_to_ms(session.frame_count - 1)
        return 0


FRAME_SERVER = FrameServer()
