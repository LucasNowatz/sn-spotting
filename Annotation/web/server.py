#!/usr/bin/env python3
"""Lightweight web annotator — native browser video, no VNC."""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ANNOTATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANNOTATION_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_cache import FRAME_SERVER  # noqa: E402
from video_proxy import ensure_proxy, is_ready, playback_path  # noqa: E402

from utils.dataset_catalog import (  # noqa: E402
    discover_videos,
    display_name,
    infer_half,
    resolve_dataset_root,
    resolve_label_path,
)
from utils.event_class import Event, ms_to_time  # noqa: E402
from utils.list_management import ListManager  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parent
_VIDEO_CACHE: list[Path] | None = None


def _load_lines(name: str) -> list[str]:
    path = ANNOTATION_ROOT / "config" / name
    return [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _video_catalog() -> list[Path]:
    global _VIDEO_CACHE
    if _VIDEO_CACHE is None:
        _VIDEO_CACHE = discover_videos()
    return _VIDEO_CACHE


def _safe_video_path(rel_path: str) -> Path:
    root = resolve_dataset_root()
    candidate = (root / rel_path).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise ValueError("path outside dataset root")
    if not candidate.is_file():
        raise FileNotFoundError(rel_path)
    return candidate


def _video_duration_ms(video_path: Path) -> int:
    try:
        return FRAME_SERVER.duration_ms(video_path)
    except Exception:
        pass
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        return 0
    try:
        return int(float(result.stdout.strip()) * 1000)
    except ValueError:
        return 0


class AnnotatorHandler(BaseHTTPRequestHandler):
    server_version = "SoccerNetWebAnnotator/1.0"

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/api/video") or self.path.startswith("/api/frame"):
            return
        super().log_message(fmt, *args)

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/video":
            return self.send_error(404)
        params = parse_qs(parsed.query)
        rel = params.get("path", [""])[0]
        if not rel:
            return self.send_error(400)
        try:
            video_path = _safe_video_path(unquote(rel))
        except (ValueError, FileNotFoundError):
            return self.send_error(404)
        size = video_path.stat().st_size
        ctype = mimetypes.guess_type(str(video_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            return self._send_static(WEB_ROOT / "index.html", "text/html; charset=utf-8")

        if parsed.path == "/api/classes":
            return self._send_json(
                {
                    "classes": _load_lines("classes.txt"),
                    "teams": _load_lines("second_classes.txt"),
                    "visibility": _load_lines("third_classes.txt"),
                }
            )

        if parsed.path == "/api/health":
            try:
                root = resolve_dataset_root()
                count = len(_video_catalog())
            except Exception as exc:
                return self._send_json({"ok": False, "error": str(exc)}, 500)
            return self._send_json({"ok": True, "dataset_root": str(root), "videos": count})

        if parsed.path == "/api/videos":
            query = (params.get("q", [""])[0] or "").lower()
            limit = min(int(params.get("limit", ["10000"])[0]), 10000)
            offset = max(int(params.get("offset", ["0"])[0]), 0)
            if params.get("refresh"):
                global _VIDEO_CACHE
                _VIDEO_CACHE = None
            root = resolve_dataset_root()
            videos = _video_catalog()
            if query:
                videos = [v for v in videos if query in display_name(v, root).lower()]
            page = videos[offset : offset + limit]
            return self._send_json(
                {
                    "total": len(videos),
                    "offset": offset,
                    "items": [
                        {
                            "path": display_name(v, root),
                            "half": infer_half(v),
                        }
                        for v in page
                    ],
                }
            )

        if parsed.path == "/api/labels":
            rel = params.get("video", [""])[0]
            if not rel:
                return self._send_json({"error": "missing video"}, 400)
            try:
                video_path = _safe_video_path(unquote(rel))
            except (ValueError, FileNotFoundError) as exc:
                return self._send_json({"error": str(exc)}, 404)
            half = int(params.get("half", [str(infer_half(video_path))])[0])
            label_path = resolve_label_path(video_path)
            manager = ListManager()
            manager.create_list_from_json(str(label_path), half)
            with open(label_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            root = resolve_dataset_root()
            try:
                label_rel = str(label_path.resolve().relative_to(root.resolve().parent))
            except ValueError:
                label_rel = str(label_path)
            fps = float(meta.get("fps_assumed", 25.0))
            events_payload = []
            for idx, e in enumerate(manager.event_list):
                frame_index = e.frame_index
                if frame_index is None and fps:
                    frame_index = int(round(int(e.position) / 1000.0 * fps))
                events_payload.append(
                    {
                        "index": idx,
                        "time": e.time,
                        "label": e.label,
                        "team": e.team,
                        "visibility": e.visibility,
                        "position": int(e.position),
                        "frame_index": frame_index,
                        "half": e.half,
                        "text": e.to_display_text(idx),
                    }
                )
            return self._send_json(
                {
                    "label_file": label_rel,
                    "label_basename": label_path.name,
                    "half": half,
                    "fps_assumed": fps,
                    "events": events_payload,
                }
            )

        if parsed.path == "/api/meta":
            rel = params.get("path", [""])[0]
            if not rel:
                return self._send_json({"error": "missing path"}, 400)
            try:
                video_path = _safe_video_path(unquote(rel))
            except (ValueError, FileNotFoundError) as exc:
                return self._send_json({"error": str(exc)}, 404)
            label_path = resolve_label_path(video_path)
            with open(label_path, encoding="utf-8") as handle:
                label_meta = json.load(handle)
            duration_ms = _video_duration_ms(video_path)
            using_proxy = is_ready(video_path)
            ensure_proxy(video_path)
            return self._send_json(
                {
                    "duration_ms": duration_ms,
                    "fps_assumed": float(label_meta.get("fps_assumed", 25.0)),
                    "label_file": label_path.name,
                    "proxy_ready": using_proxy,
                }
            )

        if parsed.path == "/api/frame":
            rel = params.get("path", [""])[0]
            if not rel:
                return self._send_json({"error": "missing path"}, 400)
            try:
                video_path = _safe_video_path(unquote(rel))
                ms = int(params.get("ms", ["0"])[0])
            except (ValueError, FileNotFoundError) as exc:
                return self._send_json({"error": str(exc)}, 404)
            try:
                frame = FRAME_SERVER.get_jpeg(video_path, ms)
            except RuntimeError as exc:
                return self._send_json({"error": str(exc)}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "public, max-age=604800, immutable")
            self.end_headers()
            self.wfile.write(frame)
            return

        if parsed.path == "/api/video":
            rel = params.get("path", [""])[0]
            if not rel:
                return self._send_json({"error": "missing path"}, 400)
            try:
                video_path = _safe_video_path(unquote(rel))
            except (ValueError, FileNotFoundError) as exc:
                return self._send_json({"error": str(exc)}, 404)
            if params.get("original"):
                return self._send_file(video_path)
            serve_path, _ = playback_path(video_path)
            return self._send_file(serve_path)

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/labels":
            return self.send_error(404)

        body = self._read_json_body()
        rel = body.get("video")
        half = int(body.get("half", 1))
        action = body.get("action", "save")

        try:
            video_path = _safe_video_path(rel)
        except (ValueError, FileNotFoundError) as exc:
            return self._send_json({"error": str(exc)}, 404)

        label_path = str(resolve_label_path(video_path))
        manager = ListManager()
        manager.create_list_from_json(label_path, half)

        if action == "add":
            position = int(body["position"])
            manager.add_event(
                Event(
                    body["label"],
                    half,
                    ms_to_time(position),
                    body["team"],
                    position,
                    body["visibility"],
                )
            )
        elif action == "update":
            idx = int(body["index"])
            position = int(body.get("position", manager.event_list[idx].position))
            manager.event_list[idx] = Event(
                body["label"],
                half,
                ms_to_time(position),
                body["team"],
                position,
                body["visibility"],
            )
            manager.sort_list()
        elif action == "delete":
            manager.delete_event(int(body["index"]))
        elif action == "save":
            pass
        else:
            return self._send_json({"error": f"unknown action: {action}"}, 400)

        manager.save_file(label_path, half)
        manager.create_list_from_json(label_path, half)
        fps = 25.0
        try:
            with open(label_path, encoding="utf-8") as handle:
                fps = float(json.load(handle).get("fps_assumed", 25.0))
        except OSError:
            pass
        events_payload = []
        for idx, e in enumerate(manager.event_list):
            frame_index = e.frame_index
            if frame_index is None and fps:
                frame_index = int(round(int(e.position) / 1000.0 * fps))
            events_payload.append(
                {
                    "index": idx,
                    "time": e.time,
                    "label": e.label,
                    "team": e.team,
                    "visibility": e.visibility,
                    "position": int(e.position),
                    "frame_index": frame_index,
                    "half": e.half,
                    "text": e.to_display_text(idx),
                }
            )
        return self._send_json(
            {
                "ok": True,
                "label_file": label_path,
                "saved_to": "ground_truth.json",
                "events": events_payload,
            }
        )

    def _send_static(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        if range_header:
            try:
                units, spec = range_header.split("=", 1)
                if units.strip() != "bytes":
                    raise ValueError
                start_s, end_s = spec.split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
            except (ValueError, IndexError):
                self.send_error(416)
                return
            if start > end or start >= size:
                self.send_error(416)
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(path, "rb") as handle:
                handle.seek(start)
                try:
                    self.wfile.write(handle.read(length))
                except (BrokenPipeError, ConnectionResetError):
                    return
            return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        try:
            with open(path, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="SoccerNet web annotator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6081)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), AnnotatorHandler)
    try:
        root = resolve_dataset_root()
        count = len(_video_catalog())
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(1) from exc
    print(f"Dataset: {root} ({count} videos)", flush=True)
    print(f"Listening on http://0.0.0.0:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
