"""Session-owned logs and Windows OpenCV camera recording, without motor access."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import threading
import time

def _inside(path: Path, boundary: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(boundary) or resolved == boundary:
        raise ValueError(f"Session path must remain inside its output directory: {resolved}")
    return resolved


class Session:
    def __init__(self, output_root: Path, script_name: str = "follow-so101"):
        if not script_name or Path(script_name).name != script_name or script_name in (".", ".."):
            raise ValueError("script_name must be one directory name")
        self.output_root = Path(output_root).resolve()
        base = _inside(self.output_root / script_name, self.output_root)
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now()
        while True:
            self.directory = _inside(base / stamp.strftime("%Y%m%d-%H%M%S-%f"), self.output_root)
            try:
                self.directory.mkdir(exist_ok=False)
                break
            except FileExistsError:
                stamp += timedelta(microseconds=1)
        self._lock = threading.RLock()
        self._owned: dict[Path, str] = {}
        self.log_path = self.allocate("session.jsonl", "log")
        self._log = self.log_path.open("a", encoding="utf-8")
        self._writers: set[object] = set()

    def allocate(self, name: str, kind: str) -> Path:
        if Path(name).name != name or kind not in ("log", "video"):
            raise ValueError("Expected a session file name and log/video kind")
        with self._lock:
            if getattr(self, "_log", None) is not None and self._log.closed:
                raise RuntimeError("Session is already closed")
            path = _inside(self.directory / name, self.directory)
            if path.parent != self.directory:
                raise ValueError("File must belong to this session directory")
            with path.open("xb"):
                pass
            self._owned[path] = kind
            return path

    def emit(self, row: dict) -> None:
        with self._lock:
            if self._log.closed:
                return
            entry = dict(row)
            entry["time"] = datetime.now().astimezone().isoformat()
            self._log.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._log.flush()

    def close_log(self) -> None:
        with self._lock:
            if not self._log.closed:
                self._log.close()

    def finish(self, save_video: bool, save_log: bool) -> list[Path]:
        with self._lock:
            if self._writers:
                raise RuntimeError("Stop and join camera workers before finish")
            self.close_log()
            kept = []
            for path, kind in self._owned.items():
                checked = _inside(path, self.directory)
                if checked.parent != self.directory or path.is_symlink():
                    raise ValueError("Refuse changed session file path")
                keep = save_video if kind == "video" else save_log
                if keep and not path.is_file():
                    raise FileNotFoundError(f"Requested session file is missing: {path}")
            for path, kind in self._owned.items():
                keep = save_video if kind == "video" else save_log
                if keep:
                    kept.append(path)
                else:
                    path.unlink(missing_ok=True)
            if not any(self.directory.iterdir()):
                self.directory.rmdir()
            return kept


class CameraWorker:
    def __init__(self, index: int, session: Session, stop_event: threading.Event,
                 fps: float = 30, width: int = 640, height: int = 480):
        if index not in (0, 1) or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Expected camera 0/1 and positive finite dimensions/fps")
        self.index, self.session, self.stop_event = index, session, stop_event
        self.fps, self.width, self.height = float(fps), int(width), int(height)
        self.error: Exception | None = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(target=self._run, name=f"camera-{index}", daemon=True)

    def start(self) -> None:
        with self.session._lock:
            if self.session._log.closed:
                raise RuntimeError("Session log is already closed")
            if self._started:
                raise RuntimeError("Camera worker already started")
            self._started = True
            self.session._writers.add(self)
        try:
            self._thread.start()
        except BaseException:
            with self.session._lock:
                self.session._writers.discard(self)
            raise

    def wait_ready(self, timeout: float = 8) -> bool:
        if not self._ready.wait(timeout):
            return False
        if self.error is not None:
            raise self.error
        return self.latest_frame() is not None

    def latest_frame(self):
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout=None) -> None:
        if self._thread.ident is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _emit(self, row: dict) -> None:
        try:
            self.session.emit(row)
        except Exception as exc:
            self.error = self.error or exc
            self.stop_event.set()

    def _run(self) -> None:
        capture = writer = timestamps = None
        received = written = 0
        started = last_report = None
        try:
            import cv2
            capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                raise RuntimeError(f"Camera {self.index} could not open")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            video = self.session.allocate(f"camera_{self.index}.avi", "video")
            times = self.session.allocate(f"camera_{self.index}_timestamps.jsonl", "video")
            timestamps = times.open("a", encoding="utf-8")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera {self.index} did not provide its first frame")
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Camera {self.index} video writer could not open")
            started = last_report = time.perf_counter()
            next_video_time = started
            self._emit({"phase": "camera_started", "camera": self.index,
                "width": width, "height": height, "writer_fps": self.fps,
                "reported_camera_fps": capture.get(cv2.CAP_PROP_FPS), "video": str(video),
                "timestamps": str(times), "timing": "fixed-fps resampling with repeated frames"})
            while not self._stop.is_set() and not self.stop_event.is_set():
                captured_at = time.perf_counter()
                received += 1
                with self._frame_lock:
                    self._frame = frame.copy()
                first_written = written
                # Preserve wall-clock duration when a camera delivers fewer frames than writer FPS.
                while next_video_time <= captured_at:
                    writer.write(frame)
                    written += 1
                    next_video_time += 1 / self.fps
                timestamps.write(json.dumps({"frame": received - 1,
                    "elapsed_s": captured_at - started,
                    "time": datetime.now().astimezone().isoformat(),
                    "video_frame_start": first_written, "video_frame_end": written}) + "\n")
                timestamps.flush()
                self._ready.set()
                if captured_at - last_report >= 1:
                    self._emit({"phase": "camera_progress", "camera": self.index,
                        "received_frames": received, "video_frames": written,
                        "elapsed_s": captured_at - started,
                        "received_fps": received / max(captured_at - started, 1e-9),
                        "video_duration_s": written / self.fps})
                    last_report = captured_at
                if self._stop.wait(max(0, next_video_time - time.perf_counter())):
                    break
                if self.stop_event.is_set():
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Camera {self.index} frame read failed")
        except Exception as exc:
            self.error = exc
            self.stop_event.set()
            self._emit({"phase": "camera_error", "camera": self.index, "error": repr(exc)})
        finally:
            for resource in (timestamps, writer, capture):
                if resource is not None:
                    try:
                        resource.close() if resource is timestamps else resource.release()
                    except Exception as exc:
                        self.error = self.error or exc
                        self.stop_event.set()
                        self._emit({"phase": "camera_cleanup_error", "camera": self.index, "error": repr(exc)})
            elapsed = 0 if started is None else time.perf_counter() - started
            self._emit({"phase": "camera_stopped", "camera": self.index,
                "received_frames": received, "video_frames": written, "elapsed_s": elapsed,
                "received_fps": received / elapsed if elapsed > 0 else 0,
                "video_duration_s": written / self.fps})
            with self.session._lock:
                self.session._writers.discard(self)
            self._ready.set()


