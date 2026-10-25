"""Single-process dataset/AVI exporter. IPC carries paths, never camera pixels."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback


class ProcessEpisodeExporter:
    """Persistent child owns one official dataset and processes requests in order."""

    def __init__(self, dataset_path, repo_id, fps, *, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        self.process = None
        self._capture_backlog = None
        try:
            self._capture_backlog = shared_memory.SharedMemory(create=True, size=1)
            self._capture_backlog.buf[0] = 0
            environment = os.environ.copy()
            environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding="utf-8", bufsize=1, env=environment,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            response = self._request("prepare", dataset_path=str(Path(dataset_path).resolve()),
                                     repo_id=repo_id, fps=fps, capture_backlog=self._capture_backlog.name)
            self.worker_pid = response["worker_pid"]
        except BaseException:
            self.close()
            raise

    def _request(self, operation, **arguments):
        with self._lock:
            process = self.process
            if self._closed or process is None or process.poll() is not None:
                code = None if process is None else process.poll()
                raise RuntimeError(f"Dataset exporter is not running (exit={code}); log: {self.log_path}")
            self._request_id += 1
            request = {"request_id": self._request_id, "operation": operation, **arguments}
            try:
                process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Dataset exporter communication failed; log: {self.log_path}") from exc
            if not line:
                raise RuntimeError(f"Dataset exporter closed its reply pipe (exit={process.poll()}); log: {self.log_path}")
            try:
                response = json.loads(line)
                if response.get("worker_pid") is not None:
                    self.worker_pid = response["worker_pid"]
                if response.get("request_id") != self._request_id:
                    raise ValueError("Unexpected reply request ID")
                if not response.get("ok"):
                    raise RuntimeError(f"Dataset exporter: {response.get('error', 'unknown failure')}; log: {self.log_path}")
                return response["result"]
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise RuntimeError(f"Invalid dataset exporter reply; log: {self.log_path}") from exc

    def save_episode(self, manifest_path, episode_index):
        return self._request("save_episode", manifest_path=str(Path(manifest_path).resolve()),
                             episode_index=episode_index)

    def set_capture_backlog(self, pending):
        # One atomic byte; never wait for an export RPC or touch disk on capture.
        if not self._closed:
            self._capture_backlog.buf[0] = min(255, max(0, pending))

    def finalize(self):
        return self._request("finalize")

    def close(self):
        if self._closed:
            return
        process = self.process
        try:
            if process is not None and process.poll() is None:
                # EOF invokes the child's best-effort finalizer. Never wait for
                # a JSON reply here: a stuck child must not make close unbounded.
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._stderr.write("Exporter close exceeded 15 seconds; terminating child, raw spool retained.\n")
                    self._stderr.flush()
                    # Windows venv python.exe may be a launcher whose PID is
                    # different from the interpreter's reported worker PID.
                    actual_pid = getattr(self, "worker_pid", None)
                    if actual_pid is not None and actual_pid != process.pid:
                        try:
                            os.kill(actual_pid, signal.SIGTERM)
                        except OSError as exc:
                            self._stderr.write(f"Worker termination {actual_pid}: {exc}\n")
                            self._stderr.flush()
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            self._closed = True
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
            self._stderr.close()
            backlog = getattr(self, "_capture_backlog", None)
            if backlog is not None:
                backlog.close()
                backlog.unlink()


def _render_video(spool, fps, cameras, joints, *, wait_for_capture=lambda: None):
    """Render the existing episode directory without touching live camera objects."""
    import cv2
    directory = spool.manifest_path.parent.parent
    writers = []
    started = time.perf_counter()
    try:
        for index, _ in enumerate(cameras):
            writer = cv2.VideoWriter(str(directory / f"camera_{index}.avi"),
                                     cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
            writers.append(writer)
            if not writer.isOpened():
                raise RuntimeError(f"AVI writer could not open camera {index}")
        with (directory / "frame_timestamps.jsonl").open("w", encoding="utf-8") as log:
            for index, (frame, timing) in enumerate(spool.iter_frames()):
                wait_for_capture()
                for name, writer in zip(cameras, writers, strict=True):
                    writer.write(cv2.cvtColor(frame[f"observation.images.{name}"], cv2.COLOR_RGB2BGR))
                row = {"frame_index": index, "logical_timestamp_s": index / fps,
                       "wall_time": datetime.now().astimezone().isoformat(),
                       "perf_counter_s": time.perf_counter(),
                       "joint_names": list(joints), **timing}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for writer in writers:
            writer.release()
    return time.perf_counter() - started


def _enable_export_cpu_priority():
    # Keep memory/I/O behavior normal: full Windows background mode severely
    # delayed this machine's exporter (observed working set: 32 MiB).
    # Live backlog separately gates PNG/AVI work; only export CPU priority lowers.
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.SetPriorityClass.restype = wintypes.BOOL
        enabled = bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000))
        print(json.dumps({"phase": "export_cpu_priority", "priority": "below_normal", "enabled": enabled,
            "winerror": None if enabled else ctypes.get_last_error(),
            "worker_pid": os.getpid()}), file=sys.stderr, flush=True)


def worker_main():
    # Parent owns user interruption; sealed episodes finish before EOF closes
    # this storage-only worker. Do not interrupt official commits with Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Keep this original pipe exclusively for protocol replies. Imports, progress
    # bars and library prints go to the parent's realtime stderr log instead.
    protocol = sys.stdout
    dataset = None
    finalized = False
    fps = None
    capture_backlog = None
    def wait_for_capture():
        # Live raw writes drain before more PNG/AVI work. No samples are dropped.
        while capture_backlog is not None and capture_backlog.buf[0] >= 8:
            time.sleep(.005)
    with redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request = {}
                quitting = False
                try:
                    request = json.loads(line)
                    operation = request["operation"]
                    if operation == "prepare":
                        if dataset is not None:
                            raise RuntimeError("Exporter already owns a dataset")
                        from record_dataset import create_local_dataset
                        capture_backlog = shared_memory.SharedMemory(name=request["capture_backlog"])
                        fps = request["fps"]
                        dataset = create_local_dataset(request["dataset_path"], request["repo_id"], fps=fps)
                        _enable_export_cpu_priority()
                        result = {"worker_pid": os.getpid()}
                    elif operation == "save_episode":
                        if dataset is None or finalized:
                            raise RuntimeError("No writable dataset in exporter")
                        from episode_spool import EpisodeSpool
                        from record_dataset import CAMERAS, JOINTS
                        spool = EpisodeSpool.from_sealed(request["manifest_path"], fps, CAMERAS)
                        seconds = spool.save(dataset, request["episode_index"], wait_for_capture=wait_for_capture, emit=lambda row: print(
                            json.dumps({**row, "wall_time": datetime.now().astimezone().isoformat()}),
                            file=sys.stderr, flush=True))
                        video_seconds = _render_video(spool, fps, CAMERAS, JOINTS, wait_for_capture=wait_for_capture)
                        result = {"dataset_commit_seconds": seconds, "video_render_seconds": video_seconds,
                                  "worker_pid": os.getpid(), "frames": len(spool.rows)}
                    elif operation in ("finalize", "quit"):
                        if dataset is not None and not finalized:
                            dataset.finalize()
                            finalized = True
                        result = {"worker_pid": os.getpid(), "finalized": finalized}
                        quitting = operation == "quit"
                    else:
                        raise ValueError("Unknown exporter operation")
                    response = {"request_id": request.get("request_id"), "ok": True, "result": result,
                                "worker_pid": os.getpid()}
                except BaseException as exc:
                    traceback.print_exc(file=sys.stderr)
                    response = {"request_id": request.get("request_id"), "ok": False,
                                "error": f"{type(exc).__name__}: {exc}", "worker_pid": os.getpid()}
                    quitting = request.get("operation") == "quit"
                protocol.write(json.dumps(response, ensure_ascii=False) + "\n")
                protocol.flush()
                if quitting:
                    break
        finally:
            if dataset is not None and not finalized:
                try:
                    dataset.finalize()
                except BaseException:
                    traceback.print_exc(file=sys.stderr)
            if capture_backlog is not None:
                capture_backlog.close()
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use ProcessEpisodeExporter; this entry is a private --worker protocol.")
    raise SystemExit(worker_main())

