"""Continuous official follow and live cameras; recording gates storage only."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training"))
from record_dataset import CAMERAS, JOINTS, close_cameras, connect_cameras, create_local_dataset, load_follow, follow_with_observations
from episode_spool import EpisodeSpool


class EpisodeVideo:
    """One AVI pair per episode, no padding across pauses, RGB input retained in PNG dataset."""
    def __init__(self, directory, fps, backend=None, *, existing_directory=False):
        if backend is None:
            import cv2
            backend = cv2
        self.cv, self.fps = backend, fps
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=existing_directory)
        self.writers = []
        self.frames = 0
        self.log = None
        try:
            self.log = (self.directory / "frame_timestamps.jsonl").open("w", encoding="utf-8")
            for index in (0, 1):
                writer = self.cv.VideoWriter(str(self.directory / f"camera_{index}.avi"),
                    self.cv.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
                self.writers.append(writer)
                if not writer.isOpened():
                    raise RuntimeError(f"AVI writer could not open camera {index}")
        except BaseException as failure:
            try:
                self.close()
            except Exception as cleanup_error:
                failure.add_note(f"AVI cleanup: {cleanup_error}")
            raise

    def write(self, frame, timing=None):
        for camera, writer in zip(CAMERAS, self.writers, strict=True):
            image = frame[f"observation.images.{camera}"]
            if image.shape != (480, 640, 3):
                raise ValueError("Episode AVI requires RGB 640x480 frames")
            writer.write(self.cv.cvtColor(image, self.cv.COLOR_RGB2BGR))
        self.log.write(json.dumps({"frame_index": self.frames, "logical_timestamp_s": self.frames / self.fps,
            "wall_time": datetime.now().astimezone().isoformat(), "perf_counter_s": time.perf_counter(),
            "joint_names": list(JOINTS), "observation.state": frame["observation.state"].tolist(),
            "action": frame["action"].tolist(), **(timing or {})}) + "\n")
        self.log.flush()
        self.frames += 1

    def write_timed(self, frame, timing):
        self.write(frame, timing)

    def close(self):
        errors = []
        for writer in self.writers:
            try:
                writer.release()
            except Exception as exc:
                errors.append(str(exc))
        self.writers = []
        if self.log is not None:
            try:
                self.log.close()
            except Exception as exc:
                errors.append(str(exc))
            self.log = None
        if errors:
            raise RuntimeError("AVI close failed: " + "; ".join(errors))


class DeferredEpisodeVideo:
    """No video encoding on the live writer; render a sealed lossless spool."""
    def __init__(self, directory, fps):
        self.directory, self.fps = Path(directory), fps
        self.directory.mkdir(parents=True, exist_ok=False)
        self.frames = 0
        self.closed = False

    def write_timed(self, frame, timing):
        if self.closed:
            raise RuntimeError("Write after recording was sealed")
        self.frames += 1

    def close(self):
        self.closed = True

    def render(self, spool):
        started = time.perf_counter()
        video = EpisodeVideo(self.directory, self.fps, existing_directory=True)
        try:
            for frame, timing in spool.iter_frames():
                video.write_timed(frame, timing)
        finally:
            video.close()
        return time.perf_counter() - started


CONTROL_DEFINITIONS = (
    ("R", "録画開始 [R]", "start_recording"),
    ("P", "一時停止／再開 [P]", "pause_recording"),
    ("S", "録画停止・保存 [S]", "save_recording"),
    ("D", "今の録画を破棄 [D]", "discard_recording"),
    ("F", "収集完全終了 [F]", "complete_recording"),
    ("Q", "プログラム終了 [Q/Esc]", "exit_program"),
)


class RecordingController:
    """One persistent control owner and one storage owner; no disk I/O in control."""
    def __init__(self, args, cameras, *, dataset_factory=create_local_dataset, arm_factory=None,
                 follow_fn=None, clip_factory=DeferredEpisodeVideo, emit=lambda row: None,
                 root_path=ROOT, output_root=None, clock=time.perf_counter, max_pending_frames=64,
                 spool_factory=EpisodeSpool, off_check=None):
        self.args, self.cameras = args, cameras
        self.dataset_factory, self.clip_factory = dataset_factory, clip_factory
        self.use_process_export = (dataset_factory is create_local_dataset
                                   and clip_factory is DeferredEpisodeVideo)
        self.spool_factory = spool_factory
        module = load_follow() if arm_factory is None or follow_fn is None else None
        self.arm_factory = arm_factory or module.build_arms
        self.follow_fn = follow_fn or module.run
        self.off_check = off_check or self._verify_recovery_off
        self.emit, self.clock = emit, clock
        self.root_path = Path(root_path)
        self.review_lock = threading.Lock()
        self.review_error = None
        self.output_root = Path(output_root) if output_root is not None else Path(root_path) / "output/record-so101"
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.exports = queue.Queue()
        self.stop_event = threading.Event()
        self.control_done, self.storage_done, self.storage_idle = threading.Event(), threading.Event(), threading.Event()
        self.storage_idle.set()
        self.worker = self.storage_worker = self.export_worker = None
        self.export_done = threading.Event()
        self.pending_saves = 0
        self.submitted_episodes = 0
        self.spool = self.collection = None
        self.last_save_seconds = None
        self.state = "preview"
        self.follow_ready = self.exit_requested = False
        self.error = self.control_error = None
        self.camera_warning = None
        self.last_discard = None
        self.latest = None
        self.last_stop = None
        self.off_confirmed = False
        self.frames = self.saved_episodes = self.total_frames = self.collections = self.attempts = 0
        self.dataset = self.log = self.clip = None
        self.recording_generation = 0
        self.clip_generation = None
        self.dataset_path = self.video_path = None
        self.report = {}
        self.pending_frames = 0
        self.max_observed_pending_frames = 0
        self.max_pending_frames = max_pending_frames
        self.record_gate = False
        self.started_at = self.paused_at = None
        self.record_accept_from = None
        self.paused_total = 0.
        self.pressed = set()
        self.recovery_ready = threading.Event()
        self.recovery_ready.set()
        self.recovery_collection = None
        self.recovery_count = 0
        self._follow_invoked = False
        self._active_follower = None
        self._off_check_required = False
        self._recovery_close_failed = False

    def _publish(self, **extra):
        self.emit({"phase": "record_state", "state": self.state, "frames": self.frames,
            "saved_episodes": self.saved_episodes, "total_frames": self.total_frames,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "video_path": str(self.video_path) if self.video_path else None,
            "pending_saves": self.pending_saves, "last_save_seconds": self.last_save_seconds,
            "error": self.error, **extra})

    def start_following(self):
        with self.lock:
            if self.worker is not None or self.exit_requested:
                return False
            self.state = "starting"
            self.storage_worker = threading.Thread(target=self._storage_loop, daemon=True)
            self.export_worker = threading.Thread(target=self._export_loop, daemon=True)
            self.worker = threading.Thread(target=self._control_loop, daemon=True)
            self.export_worker.start()
            self.storage_worker.start()
            self.worker.start()
            self._publish()
            return True

    def recovery_status(self):
        """Describe an explicit retry; never reconnect or move an arm here."""
        with self.lock:
            kind = "control" if self.control_error else "recording"
            status = {"can_retry": False, "kind": kind, "error": self.error,
                      "off_confirmed": self.off_confirmed,
                      "cleanup_errors": list((self.last_stop or {}).get("cleanup_errors", [])),
                      "recovery_count": self.recovery_count, "requires_off_check": False}
            if self.exit_requested:
                reason = "プログラム終了中です。"
            elif not self.error:
                reason = "復旧が必要なエラーはありません。"
            elif not self.recovery_ready.is_set() or self.collection is not None or self.clip is not None or self.pending_frames:
                reason = "問題の録画を除外し、保存処理を切り分けています。"
            elif any(worker is None or not worker.is_alive() for worker in (self.storage_worker, self.export_worker)):
                reason = "保存処理が終了しているため、プログラムの再起動が必要です。"
            elif self.control_error and (not self.control_done.is_set() or (self.worker and self.worker.is_alive())):
                reason = "以前の通信処理とポート切断の完了を待っています。"
            elif self.control_error and (self._recovery_close_failed or any(
                    not error.startswith("OFF ") for error in status["cleanup_errors"])):
                reason = "前の接続の終了処理に未確認の問題があります：" + "; ".join(status["cleanup_errors"])
            elif self.control_error and (self._off_check_required or (self._follow_invoked and (
                    self.last_stop is None or ((self.last_stop.get("drive_started") or self.last_stop.get("release_needed"))
                                               and not self.off_confirmed)))):
                reason = "接続をやり直し、全モーターのOFFを読み取りで確認できた場合だけ追従を再開します。"
                status.update(can_retry=True, requires_off_check=True)
            else:
                reason = "接続をやり直す操作で復旧できます。" if self.control_error else "保存済みデータを残して新しい収集を始められます。"
                status["can_retry"] = True
            return {**status, "reason": reason}

    def retry_connection(self, cameras=None):
        """Retry only after explicit UI action and completed OFF/port cleanup.

        Storage/export owners remain persistent. A stopped control owner is
        replaced with a fresh stop event and fresh arm objects; it never overlaps
        the previous owner. Failed sealed exports remain in their old collection.
        """
        with self.lock:
            status = self.recovery_status()
            if not status["can_retry"]:
                self._publish(recovery=status)
                return False
            reconnect_control = self.control_error is not None
            self._off_check_required = bool(status["requires_off_check"])
            if self.recovery_collection is not None:
                self.recovery_collection.recovery_dismissed = True
            if cameras is not None:
                self.cameras = cameras
                if not reconnect_control and self._active_follower is not None:
                    self._active_follower.cameras = cameras
            self.error = self.control_error = None
            self.camera_warning = "新しい映像と関節観測の取得を待っています。"
            self.record_gate = False
            self.latest = None
            self.recovery_count += 1
            self.recovery_ready.clear()
            if reconnect_control:
                self.stop_event = threading.Event()
                self.control_done = threading.Event()
                self.follow_ready = False
                self.last_stop = None
                self.off_confirmed = False
                self._follow_invoked = False
                self._active_follower = None
                self.worker = threading.Thread(target=self._control_loop, daemon=True)
                self.state = "starting"
                self.worker.start()
            else:
                self.state = "preview"
                # Require a fresh observation before R, especially after a
                # camera owner has supplied replacement devices.
                self.camera_warning = "新しい映像と関節観測の取得を待っています。"
            self._publish(recovery_started=True)
            return True

    def _verify_recovery_off(self, leader, follower):
        """Read-only verification before explicit recovery may start following."""
        bus = follower.bus
        try:
            bus.connect()  # Official bus handshake only; no robot.configure().
            values = {joint: bus.read("Torque_Enable", joint, normalize=False) for joint in JOINTS}
            if any(value != 0 for value in values.values()):
                raise RuntimeError(f"Motor OFF could not be confirmed by read-only recovery: {values}")
            return values
        finally:
            handler = bus.port_handler
            if bus.is_connected or getattr(getattr(handler, "ser", None), "is_open", False):
                try:
                    handler.closePort()
                except Exception:
                    self._recovery_close_failed = True
                    raise

    def _control_emit(self, row):
        with self.lock:
            if row.get("phase") in ("following", "connected"):
                self.follow_ready = True
                if self.state == "starting":
                    self.state = "preview"
            if row.get("phase") == "stopped":
                self.last_stop = row
                self.off_confirmed = bool(row.get("off_confirmed"))
            if row.get("phase") == "capture_error":
                warning = row.get("error", "Observation capture failed")
                if self.record_gate:
                    self._discard_recording_locked(warning)
                elif self.camera_warning is None:
                    # Preview/reset/pause do not write any camera frames. Keep
                    # checking for recovery instead of permanently failing F/R.
                    self.camera_warning = warning
                    self.report.setdefault("preview_observation_interruptions", []).append({
                        "started_at": datetime.now().astimezone().isoformat(), "error": warning})
        self.emit(row)

    def _control_loop(self):
        try:
            leader, follower = self.arm_factory(self.args.leader_port, self.args.follower_port)
            self._active_follower = follower
            if self._off_check_required:
                verified = self.off_check(leader, follower)
                if verified is not True and (not isinstance(verified, dict) or set(verified) != set(JOINTS)
                        or any(value != 0 for value in verified.values())):
                    raise RuntimeError("Motor OFF could not be confirmed by read-only recovery")
                self._off_check_required = False
                self.off_confirmed = True
            follower.cameras = self.cameras
            self._follow_invoked = True
            follow_with_observations(leader, follower, task=self.args.task, fps=60, record_fps=self.args.fps,
                stop_event=self.stop_event, run_follow=self.follow_fn, emit=self._control_emit,
                frame_emit=self._frame, clock=self.clock)
            if not self.exit_requested:
                raise RuntimeError("Follow ended before program exit")
            if self.last_stop and (self.last_stop.get("release_needed") or self.last_stop.get("drive_started")) and not self.off_confirmed:
                raise RuntimeError("Motor OFF could not be confirmed")
        except BaseException as exc:
            import traceback
            with self.lock:
                self.control_error = self.error = f"Control: {type(exc).__name__}: {exc}"
                self.follow_ready = False
                self.recovery_ready.clear()
                self.report.update(error=self.error, exit_code=1, control_fault={
                    "at": datetime.now().astimezone().isoformat(), "type": type(exc).__name__,
                    "message": str(exc), "traceback": traceback.format_exc(),
                    "stopped": self.last_stop})
                self.recovery_collection = self.collection
                self.record_gate = False
                if not self.exit_requested:
                    self.state = "finalizing"
                    self.storage_idle.clear()
                    self.commands.put(("fault", self.error))
        finally:
            self.control_done.set()
            self.poll()

    def _storage_fault_locked(self, error):
        if self.error is None:
            self.error = "Recording: " + str(error)
            self.report.update(error=self.error, exit_code=1)
            self.recovery_collection = self.collection
            self.recovery_ready.clear()
            self.record_gate = False
            self.state = "saving"
            self.storage_idle.clear()
            self.commands.put(("fault", self.error))
            self._publish()

    def _discard_recording_locked(self, reason):
        """Reject only the active attempt; already sealed exports keep running."""
        if not self.record_gate or self.error or self.exit_requested:
            return
        self.record_gate = False
        self.camera_warning = str(reason)
        self.state = "discarding"
        self.storage_idle.clear()
        self.commands.put(("discard", str(reason)))
        self._publish(discard_requested=True)

    def recording_error(self, error):
        with self.lock:
            self._storage_fault_locked(error)

    def _frame(self, frame, timing):
        with self.lock:
            self.latest = frame
            if self.camera_warning is not None:
                self.camera_warning = None
                events = self.report.get("preview_observation_interruptions", [])
                if events:
                    events[-1]["recovered_at"] = datetime.now().astimezone().isoformat()
            if not self.record_gate or self.exit_requested:
                return
            # A sample already in flight when R was pressed belongs to the reset interval.
            if timing["observation_start_s"] < self.record_accept_from:
                return
            if self.pending_frames >= self.max_pending_frames:
                self.report["storage_overflow"] = {"wall_time": datetime.now().astimezone().isoformat(),
                    "perf_counter_s": self.clock(), "pending_frames": self.pending_frames,
                    "capacity": self.max_pending_frames}
                self._discard_recording_locked("Storage queue full; current recording discarded without silently dropping frames")
                return
            self.pending_frames += 1
            if self.collection is not None and self.collection.exporter is not None:
                self.collection.exporter.set_capture_backlog(self.pending_frames)
            self.max_observed_pending_frames = max(self.max_observed_pending_frames, self.pending_frames)
            self.commands.put(("frame", (frame, {**timing,
                "storage_enqueued_s": self.clock(), "storage_pending_at_enqueue": self.pending_frames})))
            active = timing["action_sent_s"] - self.started_at - self.paused_total
            if active >= self.args.episode_seconds:
                self.record_gate = False
                self.state = "waiting"
                self.storage_idle.clear()
                self.commands.put(("save", None))
                self._publish()

    def start_recording(self):
        with self.lock:
            if not self.follow_ready or self.error or self.camera_warning or self.exit_requested or self.state not in ("preview", "waiting", "complete"):
                return False
            self.started_at, self.paused_total = self.clock(), 0.
            self.record_accept_from = self.started_at
            self.last_discard = None
            self.state = "recording"
            self.storage_idle.clear()
            self.recording_generation += 1
            self.commands.put(("start", self.recording_generation))
            self.record_gate = True
            self._publish()
            return True

    def pause_recording(self):
        barrier = None
        with self.lock:
            if self.state == "recording":
                self.record_gate = False
                self.paused_at = self.clock()
                self.state = "pausing"
                barrier = threading.Event()
                self.commands.put(("barrier", barrier))
            elif self.state == "paused":
                if self.camera_warning is not None:
                    return False
                self.paused_total += self.clock() - self.paused_at
                self.record_accept_from = self.clock()
                self.state = "recording"
                self.record_gate = True
                self._publish()
                return True
            else:
                return False
        # ACK is after all accepted frames, while the control worker keeps following.
        barrier.wait()
        with self.lock:
            if self.state == "pausing":
                self.state = "paused"
            self._publish()
        return True

    def save_recording(self):
        with self.lock:
            if self.state not in ("recording", "paused"):
                return False
            self.record_gate = False
            # The stop command seals this episode in FIFO order. R can enqueue the
            # next start immediately; LeRobot conversion runs on another worker.
            self.state = "waiting"
            self.storage_idle.clear()
            self.commands.put(("save", None))
            self._publish()
            return True

    def save_next(self):
        return self.save_recording()  # Compatibility alias; never starts another episode.

    def discard_recording(self, directory=None):
        """Stop and reject the current attempt without exporting it or stopping follow."""
        with self.lock:
            if self.error or self.exit_requested or self.state not in ("recording", "paused", "pausing"):
                return False
            # Review may have selected an earlier attempt: never discard a new one.
            if directory is not None:
                clip = self.clip
                if clip is None or self.clip_generation != self.recording_generation or \
                        Path(clip.directory).resolve() != Path(directory).resolve():
                    return False
            self.record_gate = False
            self.state = "discarding"
            self.storage_idle.clear()
            self.commands.put(("discard", "User discarded current recording before saving"))
            self._publish(discard_requested=True)
            return True

    def complete_recording(self):
        with self.lock:
            if self.exit_requested or self.state not in ("preview", "waiting", "recording", "paused"):
                return False
            self.record_gate = False
            self.state = "finalizing"
            self.storage_idle.clear()
            self.commands.put(("finish", None))
            self._publish()
            return True

    def exit_program(self):
        with self.lock:
            if self.exit_requested:
                return False
            self.exit_requested = True
            self.record_gate = False
            self.stop_event.set()
            self.state = "closing"
            self.storage_idle.clear()
            if self.storage_worker is None:
                self.storage_done.set()
                self.control_done.set()
                self.export_done.set()
            else:
                self.commands.put(("quit", None))
            self._publish()
        self.poll()
        return True

    def dispatch_key(self, key):
        key = str(key).lower()
        if key in ("escape", "esc"):
            return self.exit_program()
        method = next((method for letter, _, method in CONTROL_DEFINITIONS if letter.lower() == key), None)
        return bool(method and getattr(self, method)())

    def key_pressed(self, key):
        key = str(key).lower()
        with self.lock:
            if key in self.pressed:
                return False
            self.pressed.add(key)
        return self.dispatch_key(key)

    def key_released(self, key):
        with self.lock:
            self.pressed.discard(str(key).lower())

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.perf_counter() + timeout
        if not self.exit_requested:
            return self.storage_idle.wait(timeout)
        for worker in (self.worker, self.storage_worker, self.export_worker):
            if worker and worker is not threading.current_thread():
                remaining = None if deadline is None else max(0., deadline - time.perf_counter())
                worker.join(remaining)
        self.poll()
        return self.control_done.is_set() and self.storage_done.is_set() and self.export_done.is_set()

    def poll(self):
        with self.lock:
            if self.exit_requested and self.control_done.is_set() and self.storage_done.is_set() and self.export_done.is_set():
                if self.state != "closed":
                    self.state = "closed"
                    self._publish()
                return True
        return False

    def _new_collection(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.review_error = None
        self.dataset_path = Path(self.args.output) if self.collections == 0 else Path(self.args.output).with_name(
            Path(self.args.output).name + "-" + stamp)
        self.video_path = self.output_root / stamp
        self.report = {"task": self.args.task, "state": "Follower.get_observation measured positions",
            "action": "Follower.send_action actual returned targets", "fps": self.args.fps, "follow_fps": 60,
            "camera_sync": "Latest RGB frames; no hardware exposure synchronization claimed",
            "timestamp_policy": "Dataset/AVI frame_index/fps; pause/reset omitted; acquisition times in sidecar",
            "video_path": str(self.video_path), "dataset_path": str(self.dataset_path),
            "repo_id": self.args.repo_id, "episodes_recorded": [], "exit_code": 0}
        self.report["recovery_count"] = self.recovery_count
        self.report["camera_profile"] = getattr(self.args, "camera_profile", "pc")
        if self.report["camera_profile"] == "iphone":
            self.report["camera_roles"] = {"camera0": "カメラ1：iPhone", "camera1": "カメラ2：アームカメラ"}
            from iphone_camera_profile import camera_device_metadata
            devices = camera_device_metadata(self.cameras)
            self.report["camera_devices"] = devices
            self.report["requested_camera_indices"] = self.args.camera_indices
            self.report["camera_indices"] = ([devices[name]["index"] for name in CAMERAS]
                                              if devices else self.args.camera_indices)
            self.report["iphone_preprocessing"] = {"native_width": 1280, "native_height": 720,
                "output_width": 640, "output_height": 480, "resize_width": 640, "resize_height": 360,
                "padding_top": 60, "padding_bottom": 60, "color": "RGB",
                "method": "Aspect-ratio-preserving resize with black padding; no crop"}
        self.collections += 1
        self.video_path.mkdir(parents=True, exist_ok=False)
        self.log = (self.video_path / "acquisition.jsonl").open("w", encoding="utf-8")
        self.saved_episodes = self.total_frames = self.attempts = 0
        self.max_observed_pending_frames = self.pending_frames
        self.submitted_episodes = 0
        self.report["storage_pipeline"] = ("Live raw RGB+JSONL spool; isolated export process owns lossless PNG compression=1, official LeRobot episode_data commit and AVI"
            if self.use_process_export else "Live raw RGB+JSONL spool; injected in-process export for offline tests")
        self.collection = SimpleNamespace(dataset=None, report=self.report,
            dataset_path=self.dataset_path, video_path=self.video_path, failed=False, exporter=None,
            review_attempts={}, review_finalized=False)
        self._review_update(self.collection)
        self.exports.put(("prepare", self.collection))

    def _review_update(self, collection, directory=None, **changes):
        """Publish closed-attempt identity without touching live camera or motor data."""
        # This lock only serializes storage/export/review metadata writers.
        # The 60 Hz control loop never takes it or performs this disk I/O.
        with self.review_lock:
            if "finalized" in changes:
                collection.review_finalized = changes.pop("finalized")
            if directory is not None:
                directory = Path(directory)
                attempt = int(directory.name.removeprefix("episode_"))
                item = collection.review_attempts.setdefault(attempt, {
                    "attempt": attempt, "directory": str(directory), "episode_index": None,
                    "frames": 0, "status": "active"})
                item.update(changes)
            snapshot = {"schema_version": 1, "dataset_path": str(collection.dataset_path),
                "repo_id": self.args.repo_id, "fps": self.args.fps,
                "finalized": collection.review_finalized,
                "attempts": list(collection.review_attempts.values())}
            path = collection.video_path / "review-state.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def request_episode_deletion(self, directory):
        """Queue sealed-attempt removal; the dataset writer remains its sole owner."""
        from episode_review_store import request_deletion
        return request_deletion(Path(directory).parent, directory, self.root_path,
                                reason="User deleted this recording in the review screen")

    def _apply_review_deletions(self, collection):
        # The helper imports Arrow/official Dataset code only in its subprocess.
        from episode_review_store import apply_requested_deletions_subprocess
        result = apply_requested_deletions_subprocess(collection.video_path, self.root_path)
        if result.get("applied"):
            with self.lock:
                if self.report is collection.report:
                    self.saved_episodes = result.get("total_episodes", self.saved_episodes)
                    self.total_frames = result.get("total_frames", self.total_frames)
        return result

    def _save_episode(self):
        if self.clip is not None and not self.frames and not self.error:
            # R followed immediately by S/F/Q can seal before the first sample.
            # Publish a closed non-training attempt, rather than leaving review
            # metadata stuck at active after its empty spool was removed.
            self._discard_episode("No frames were recorded; not exported to dataset")
            return
        if self.clip is not None:
            clip = self.clip
            self.clip = None
            spool, self.spool = self.spool, None
            try:
                clip.close()
            finally:
                spool.close()
            if self.frames:
                index = self.submitted_episodes
                self.submitted_episodes += 1
                with self.lock:
                    self.pending_saves += 1
                self._review_update(self.collection, clip.directory, status="pending",
                                    episode_index=index, frames=self.frames)
                self.exports.put(("episode", (self.collection, spool, index, self.frames, clip)))

    def _discard_episode(self, reason):
        # No Dataset episode number is allocated and no export is submitted.
        # Retain the raw attempt outside the training dataset for diagnosis.
        clip, spool = self.clip, self.spool
        self.clip = self.spool = None
        if clip is not None:
            try:
                clip.close()
            finally:
                spool.close()
            info = {"recording_attempt": self.attempts, "frames": self.frames,
                    "directory": str(clip.directory), "reason": reason,
                    "training_allowed": False, "exported_to_dataset": False,
                    "discarded_at": datetime.now().astimezone().isoformat()}
            (clip.directory / "episode_status.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            self.report.setdefault("discarded_episodes", []).append(info)
            self.last_discard = info
            self._review_update(self.collection, clip.directory, status="discarded",
                                frames=self.frames, training_allowed=False)
        self.frames = 0
        with self.lock:
            if self.state == "discarding" and not self.exit_requested:
                self.state = "waiting"
            if not self.pending_saves:
                self.storage_idle.set()
        self._publish(discarded=True, discard=self.last_discard)

    def _finalize(self):
        errors = []
        try:
            self._save_episode()
        except Exception as exc:
            errors.append(str(exc))
        collection = self.collection
        if collection is not None and self.error:
            collection.report.update(error=self.error, exit_code=1)
            self.recovery_collection = collection
        self.dataset = self.collection = None
        if self.log is not None:
            try:
                self.log.close()
            except Exception as exc:
                errors.append(str(exc))
            self.log = None
        if errors:
            self.error = self.error or "; ".join(errors)
        if collection is not None:
            self.exports.put(("finish", collection))

    def _export_loop(self):
        """Sole owner of dataset.save_episode/finalize, never of motors/cameras."""
        process_exporters = []
        try:
            while True:
                kind, payload = self.exports.get()
                if kind == "quit":
                    break
                try:
                    if kind == "prepare":
                        collection = payload
                        if self.use_process_export:
                            from process_export import ProcessEpisodeExporter
                            collection.exporter = ProcessEpisodeExporter(collection.dataset_path,
                                self.args.repo_id, self.args.fps,
                                log_path=collection.video_path / "export-worker.log")
                            process_exporters.append(collection.exporter)
                            collection.dataset = collection.exporter
                            collection.report["export_worker_pid"] = collection.exporter.worker_pid
                        else:
                            collection.dataset = self.dataset_factory(collection.dataset_path, self.args.repo_id, fps=self.args.fps)
                        from training_selection import mark_dataset_review_state
                        mark_dataset_review_state(collection.dataset_path, ready=False,
                            reason="Collection is still recording or saving")
                        with self.lock:
                            if self.collection is collection:
                                self.dataset = collection.dataset
                        continue
                    if kind == "episode":
                        collection, spool, index, frames, clip = payload
                        if collection.failed:
                            raise RuntimeError("Previous episode commit failed; pending PNGs retained")
                        self._review_update(collection, clip.directory, status="exporting")
                        export_started_at = datetime.now().astimezone().isoformat()
                        if collection.exporter is not None:
                            saved = collection.exporter.save_episode(spool.manifest_path, index)
                            if saved["frames"] != frames or saved["worker_pid"] != collection.exporter.worker_pid:
                                raise RuntimeError("Export reply does not match the sealed recording; originals retained")
                            elapsed, video_elapsed = saved["dataset_commit_seconds"], saved["video_render_seconds"]
                        else:
                            elapsed = spool.save(collection.dataset, index)
                            video_elapsed = clip.render(spool) if hasattr(clip, "render") else 0.
                        collection.report["episodes_recorded"].append({"episode": index, "frames": frames,
                            "video_directory": str(clip.directory), "dataset_commit_seconds": elapsed,
                            "video_render_seconds": video_elapsed, "export_started_at": export_started_at,
                            "export_completed_at": datetime.now().astimezone().isoformat()})
                        spool.discard_committed_files()
                        self._review_update(collection, clip.directory, status="saved", frames=frames)
                        with self.lock:
                            self.pending_saves -= 1
                            self.last_save_seconds = elapsed + video_elapsed
                            if self.report is collection.report:
                                self.saved_episodes += 1
                                self.total_frames += frames
                            if not self.pending_saves and self.state in ("waiting", "complete", "error"):
                                self.storage_idle.set()
                        self._publish(saved=True, saved_episode=index)
                    else:
                        collection = payload
                        if collection.dataset is not None:
                            try:
                                collection.dataset.finalize()
                            finally:
                                if collection.exporter is not None:
                                    collection.exporter.close()
                        if collection.dataset_path.exists():
                            shutil.copyfile(collection.video_path / "acquisition.jsonl", collection.dataset_path / "acquisition.jsonl")
                        acquisition_error = str(collection.report.get("error") or "")
                        if not collection.failed and acquisition_error.startswith(("Control:", "Recording:", "Storage:")) \
                                and all(item.get("status") in ("saved", "discarded", "deleted", "delete_requested")
                                        for item in collection.review_attempts.values()):
                            # Only complete, successfully exported episodes enter
                            # the dataset. A rejected active attempt must not make
                            # those earlier episodes permanently ineligible.
                            collection.report["acquisition_failure"] = {
                                "error": collection.report.pop("error"),
                                "scope": "Rejected active attempt; previously sealed successful exports preserved",
                                "discarded_attempts": [item["attempt"] for item in collection.review_attempts.values()
                                                       if item.get("status") == "discarded"]}
                        collection.report["storage_queue_max_pending"] = self.max_observed_pending_frames
                        collection.report["storage_queue_capacity"] = self.max_pending_frames
                        for path in (collection.dataset_path / "collection-report.json", collection.video_path / "collection-report.json"):
                            if not path.parent.exists():
                                continue
                            path.write_text(json.dumps(collection.report, ensure_ascii=False, indent=2), encoding="utf-8")
                        self._review_update(collection, finalized=not collection.failed)
                        if not collection.failed:
                            try:
                                self._apply_review_deletions(collection)
                                from episode_review_store import complete_collection_review
                                if collection.dataset_path.exists():
                                    complete_collection_review(collection.video_path, self.root_path)
                            except Exception as exc:
                                # A review edit error does not stop following or erase recordings.
                                self.review_error = f"Recording deletion: {exc}"
                                self._publish(review_error=self.review_error)
                        with self.lock:
                            if self.state == "finalizing" and not self.exit_requested and self.report is collection.report:
                                self.state = "error" if self.error else "complete"
                            self.storage_idle.set()
                        self._publish(saved=True, collection_finalized=True)
                except BaseException as exc:
                    if kind == "prepare":
                        collection.failed = True
                    if kind == "episode":
                        collection.failed = True
                        self._review_update(collection, clip.directory, status="failed",
                                            training_allowed=False, error=str(exc))
                        with self.lock:
                            self.pending_saves -= 1
                    with self.lock:
                        export_error = f"Dataset commit: {type(exc).__name__}: {exc}"
                        # A previous failed collection may still have sealed
                        # work in this FIFO. Its failure must not poison a new
                        # collection started by an explicit recovery operation.
                        if self.report is collection.report and not getattr(collection, "recovery_dismissed", False):
                            self._storage_fault_locked(export_error)
                    collection.report.update(error=collection.report.get("error") or export_error, exit_code=1)
                    for path in (collection.dataset_path / "collection-report.json", collection.video_path / "collection-report.json"):
                        try:
                            path.write_text(json.dumps(collection.report, ensure_ascii=False, indent=2), encoding="utf-8")
                        except OSError:
                            pass
                    self._publish(saved=False)
        finally:
            for exporter in process_exporters:
                try:
                    exporter.close()
                except Exception as exc:
                    with self.lock:
                        self.error = self.error or f"Export process cleanup: {exc}"
            self.export_done.set()
            self.poll()

    def _storage_loop(self):
        try:
            while True:
                kind, payload = self.commands.get()
                try:
                    if kind == "barrier":
                        payload.set()
                        continue
                    if kind == "frame":
                        frame, timing = payload
                        try:
                            if self.clip is not None:
                                write_started = self.clock()
                                timing = {**timing, "recording_attempt": self.attempts}
                                self.spool.add_frame(frame, timing)
                                if hasattr(self.clip, "write_timed"):
                                    self.clip.write_timed(frame, timing)
                                else:
                                    self.clip.write(frame)
                                self.frames += 1
                                self.log.write(json.dumps({"episode": self.submitted_episodes, "frame_index": self.frames - 1, **timing,
                                    "storage_write_start_s": write_started,
                                    "storage_write_end_s": self.clock()}) + "\n")
                                self.log.flush()
                        finally:
                            with self.lock:
                                self.pending_frames -= 1
                                if self.collection is not None and self.collection.exporter is not None:
                                    self.collection.exporter.set_capture_backlog(self.pending_frames)
                        continue
                    if kind == "start":
                        if self.exit_requested:
                            continue
                        if self.collection is None:
                            self._new_collection()
                        self.attempts += 1
                        self.frames = 0
                        self.clip = self.clip_factory(self.video_path / f"episode_{self.attempts:06d}", self.args.fps)
                        self.spool = self.spool_factory(self.clip.directory, self.args.fps, CAMERAS)
                        self.clip_generation = payload
                        self._review_update(self.collection, self.clip.directory, status="active")
                        self.storage_idle.set()
                    elif kind == "save":
                        self._save_episode()
                        with self.lock:
                            if not self.exit_requested and self.state == "waiting":
                                self.state = "waiting"
                            if not self.pending_saves:
                                self.storage_idle.set()
                    elif kind == "discard":
                        self._discard_episode(payload)
                    elif kind in ("finish", "fault", "quit"):
                        had_collection = self.collection is not None
                        if kind == "fault" and self.clip is not None:
                            self._discard_episode(payload or self.error or "Recording failed")
                        self._finalize()
                        with self.lock:
                            if not self.exit_requested and self.error:
                                self.state = "error"
                            if not self.pending_saves and self.state == "error":
                                self.storage_idle.set()
                            if kind == "finish" and not had_collection and not self.exit_requested:
                                self.state = "error" if self.error else "complete"
                                self.storage_idle.set()
                            if kind == "fault":
                                # The active attempt and its accepted frame FIFO
                                # are closed. Older sealed exports can continue
                                # while a fresh control owner/collection starts.
                                self.recovery_ready.set()
                        if kind == "quit":
                            self.exports.put(("quit", None))
                            break
                    self._publish(queued=kind in ("save", "finish", "quit"))
                except BaseException as exc:
                    with self.lock:
                        self.error = self.error or f"Storage: {type(exc).__name__}: {exc}"
                        self.record_gate = False
                    if self.clip is not None:
                        try:
                            self._discard_episode(self.error)
                        except Exception as discard_error:
                            self.error += f"; discard cleanup: {discard_error}"
                    self._finalize()
                    with self.lock:
                        if not self.exit_requested:
                            self.state = "error"
                        self.recovery_ready.set()
                    self.storage_idle.set()
                    self._publish()
                    if kind == "quit":
                        self.exports.put(("quit", None))
                        break
        finally:
            self.storage_done.set()
            self.poll()


def run_ui(args, root_path=ROOT):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    from so101_ui_errors import SelectableErrorPanel

    window = tk.Tk()
    window.title("SO-101 常時追従・学習用データ録画")
    window.geometry("1060x840")
    messages, shutdown, signal_stop = queue.Queue(), threading.Event(), threading.Event()
    camera_latest, camera_lock = {}, threading.Lock()
    displayed_timestamps = {}
    controller = None
    error = None
    last_error = ""
    retry_connecting = False
    replacement_cameras = None
    camera_retry = threading.Event()
    exit_requested = False
    runtime_directory = Path(root_path) / "output/record-so101-runtime"
    runtime_directory.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_directory / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
    runtime_log = runtime_path.open("x", encoding="utf-8")
    def log_event(row):
        runtime_log.write(json.dumps({"wall_time": datetime.now().astimezone().isoformat(), **row},
                                     ensure_ascii=False, default=str) + "\n")
        runtime_log.flush()
    log_event({"phase": "ui_started", "task": args.task})
    previous_signals = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signum = getattr(signal, name)
                previous_signals[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: signal_stop.set())
    status = tk.StringVar(value="カメラ接続後、追従を開始します。録画開始は R です。")
    saved_path = tk.StringVar(value="保存先：録画開始後に表示")
    ttk.Label(window, textvariable=status, wraplength=1020).pack(pady=10)
    ttk.Label(window, text="直近のエラー（選択して Ctrl+C、またはボタンで全文コピー）").pack(anchor="w", padx=15)
    error_panel = SelectableErrorPanel(window, max_rows=3)
    error_panel.pack(fill=tk.X, padx=15, pady=4)
    panel = ttk.Frame(window)
    panel.pack()
    labels = []
    for name in CAMERAS:
        child = ttk.Frame(panel)
        child.pack(side=tk.LEFT, padx=6)
        title = ({"camera0": "カメラ1：iPhone", "camera1": "カメラ2：アームカメラ"}[name]
                 if getattr(args, "camera_profile", "pc") == "iphone" else name)
        ttk.Label(child, text=title).pack()
        label = ttk.Label(child, text="カメラ待機")
        label.pack()
        labels.append(label)
    table = ttk.Treeview(window, columns=("joint", "state", "action"), show="headings", height=6)
    for name, text in (("joint", "関節"), ("state", "Follower 実測"), ("action", "送信目標")):
        table.heading(name, text=text)
    for joint in JOINTS:
        table.insert("", "end", iid=joint, values=(joint, "?", "?"))
    table.pack(fill=tk.X, padx=15, pady=8)
    ttk.Label(window, textvariable=saved_path, wraplength=1020).pack(pady=8)
    controls = ttk.Frame(window)
    controls.pack(pady=12)

    def request_exit():
        nonlocal exit_requested
        exit_requested = True
        if controller is not None:
            controller.exit_program()
        else:
            shutdown.set()

    def invoke(method):
        if method == "exit_program":
            request_exit()
        elif controller is not None:
            # Pause ACK drains accepted storage frames without blocking Tk or follow.
            if method == "pause_recording":
                threading.Thread(target=controller.pause_recording, daemon=True).start()
            else:
                getattr(controller, method)()
    buttons = {}
    for key, text, method in CONTROL_DEFINITIONS:
        button = ttk.Button(controls, text=text, command=lambda method=method: invoke(method))
        button.pack(side=tk.LEFT, padx=3)
        buttons[key] = button
    def retry_connection():
        nonlocal retry_connecting
        if exit_requested or retry_connecting:
            return
        if error:
            # The sole camera owner closes the previous handles before opening
            # replacements. Control recovery runs only after its discard ACK.
            retry_connecting = True
            camera_retry.set()
            log_event({"phase": "camera_retry_requested"})
        elif controller is not None:
            log_event({"phase": "recovery_requested", "recovery": controller.recovery_status()})
            controller.retry_connection()
    retry_button = ttk.Button(controls, text="接続をやり直す", command=retry_connection, state="disabled")
    retry_button.pack(side=tk.LEFT, padx=3)
    review_window = None
    def open_review():
        nonlocal review_window
        if review_window is not None and review_window.winfo_exists():
            review_window.lift()
            return
        from so101_episode_review_ui import show_episode_review
        review_window = show_episode_review(window, root_path, active_controller=controller)
    ttk.Button(controls, text="試行の確認・削除 [V]", command=open_review).pack(side=tk.LEFT, padx=3)
    held = set()
    def key_press(event):
        if event.widget is error_panel.text:
            return  # Keep Text selection/copy separate from recording controls.
        key = event.keysym.lower()
        if key in held:
            return
        held.add(key)
        if key == "escape":
            request_exit()
        elif key == "v":
            open_review()
        else:
            method = next((m for k, _, m in CONTROL_DEFINITIONS if k.lower() == key), None)
            if method:
                invoke(method)
    window.bind("<KeyPress>", key_press)
    window.bind("<KeyRelease>", lambda event: held.discard(event.keysym.lower()))
    window.bind("<FocusOut>", lambda _: held.clear())
    window.protocol("WM_DELETE_WINDOW", request_exit)

    def camera_worker():
        while not shutdown.is_set():
            cameras = {}
            try:
                cameras = connect_cameras(SimpleNamespace(), args.camera_indices, fps=args.fps,
                                          profile=getattr(args, "camera_profile", "pc"))
                messages.put({"phase": "camera_ready", "cameras": cameras})
                while not shutdown.is_set() and not camera_retry.is_set():
                    from iphone_camera_profile import read_latest_with_metadata
                    images = {name: read_latest_with_metadata(camera, max_age_ms=float("inf"))
                              for name, camera in cameras.items()}
                    with camera_lock:
                        camera_latest.update(images)
                    shutdown.wait(1 / args.fps)
            except BaseException as exc:
                import traceback
                messages.put({"phase": "camera_error", "error": str(exc),
                              "traceback": traceback.format_exc()})
            finally:
                try:
                    close_cameras(cameras)
                except Exception as exc:
                    messages.put({"phase": "camera_error", "error": str(exc)})
            while not shutdown.is_set() and not camera_retry.wait(.1):
                pass
            camera_retry.clear()
    camera_thread = threading.Thread(target=camera_worker, daemon=True)
    camera_thread.start()

    def poll():
        nonlocal controller, error, last_error, retry_connecting, replacement_cameras
        while True:
            try:
                message = messages.get_nowait()
            except queue.Empty:
                break
            # Disk writes belong to the Tk thread, never the motor owner.
            log_event({key: value for key, value in message.items() if key != "cameras"})
            if message.get("error"):
                last_error = str(message["error"])
            if message["phase"] == "camera_ready":
                if not exit_requested:
                    if controller is None:
                        controller = RecordingController(args, message["cameras"], emit=messages.put, root_path=root_path)
                        controller.start_following()
                        retry_connecting = False
                    else:
                        replacement_cameras = message["cameras"]
                    error = None
            elif message["phase"] == "camera_error":
                error = message["error"]
                retry_connecting = False
                if controller is not None:
                    controller.recording_error(error)
        if signal_stop.is_set():
            request_exit()
        if controller:
            controller.poll()
            if replacement_cameras is not None and not exit_requested and controller.recovery_status()["can_retry"]:
                if controller.retry_connection(cameras=replacement_cameras):
                    replacement_cameras = None
                    retry_connecting = False
        state = controller.state if controller else "connecting"
        ready = bool(controller and controller.follow_ready and not controller.error and not exit_requested)
        camera_ready = bool(controller and not controller.camera_warning)
        enabled = {"R": ready and camera_ready and state in ("preview", "waiting", "complete"),
                   "P": ready and state in ("recording", "paused") and (state == "recording" or camera_ready),
                   "S": ready and state in ("recording", "paused"),
                   "D": ready and state in ("recording", "paused", "pausing"),
                   "F": ready and state in ("preview", "waiting", "recording", "paused"),
                   "Q": not exit_requested}
        for key, button in buttons.items():
            button.configure(state="normal" if enabled[key] else "disabled")
        recovery = controller.recovery_status() if controller else None
        can_retry = bool(not exit_requested and not retry_connecting and
                         ((error and controller is None) or (recovery and recovery["can_retry"])))
        retry_button.configure(state="normal" if can_retry else "disabled")
        buttons["P"].configure(text="録画再開 [P]" if state == "paused" else "録画一時停止 [P]")
        descriptions = {
            "preview": "追従中・録画停止。R で録画開始。",
            "preparing": "追従中・録画の保存先を準備中。",
            "recording": f"追従中・録画中。上限 {args.episode_seconds:g} 秒（一時停止を除く）。P:一時停止 S:停止保存 D:保存せず破棄。",
            "pausing": "追従中・一時停止処理中。",
            "paused": "追従中・録画一時停止。P:同じ動画へ再開 D:保存せず破棄。",
            "saving": "追従中・保存中。保存後は R を押すまで録画しません。",
            "discarding": "追従中・今の録画を破棄しています。以前の保存は続きます。",
            "waiting": "追従中・録画停止。初期位置へ戻してから R。前の保存完了は待たず開始できます。",
            "finalizing": "追従中・データセットを確定しています。",
            "complete": "追従中・収集終了、保存完了。R で新しい収集を開始。",
            "closing": "追従を停止してモーター OFF・保存・終了処理中。",
            "closed": "終了処理完了。",
        }
        status.set(descriptions.get(state, "カメラ／アームを接続中。録画はまだ開始していません。"))
        if controller:
            if controller.last_discard and state == "waiting":
                status.set("今回の録画は学習対象から除外しました。追従・以前の保存は継続中。画像回復後、初期位置へ戻して R。")
            if controller.camera_warning:
                status.set("観測更新待ち・追従継続。新しい録画は画像の回復後に開始できます。" + controller.camera_warning)
            if controller.error:
                prefix = "追従停止" if controller.control_error else "追従継続・録画エラー"
                status.set(prefix + "。今回の録画を除外し、以前の保存分を残します。" + recovery["reason"])
            if controller.review_error:
                status.set("試行の削除に失敗しました。録画原本は残っています：" + controller.review_error)
            if controller.dataset_path:
                pending_text = f"バックグラウンド保存 {controller.pending_saves} 件" if controller.pending_saves else "保存待ちなし"
                saved_path.set(f"Dataset：{controller.dataset_path}\n動画：{controller.video_path}\n保存完了 {controller.saved_episodes} 回・{controller.total_frames} フレーム／{pending_text}、今回 {controller.frames} フレーム")
            if controller.latest is not None:
                for i, joint in enumerate(JOINTS):
                    table.item(joint, values=(joint, f"{controller.latest['observation.state'][i]:.2f}", f"{controller.latest['action'][i]:.2f}"))
            if state == "closed":
                shutdown.set()
        if error and controller is None:
            status.set("カメラ接続エラー。「接続をやり直す」で再接続できます。")
        if retry_connecting:
            status.set("以前のカメラ接続を閉じて、接続をやり直しています。")
        details = error or (controller.error if controller else None)
        if controller and not details:
            details = controller.review_error or (controller.camera_warning if controller.latest is not None else None)
        if details:
            if details != last_error:
                log_event({"phase": "ui_error", "error": details})
            last_error = details
        error_panel.set_error(last_error)
        with camera_lock:
            images = dict(camera_latest)
        for name, label in zip(CAMERAS, labels, strict=True):
            if name in images:
                frame, camera_timing = images[name]
                received = camera_timing["received_perf_counter_s"]
                if displayed_timestamps.get(name) != received:
                    photo = ImageTk.PhotoImage(Image.fromarray(frame).resize((480, 360)))
                    label.configure(image=photo, text="")
                    label.image = photo
                    displayed_timestamps[name] = received
        if shutdown.is_set() and not camera_thread.is_alive():
            window.destroy()
            return
        window.after(20, poll)
    window.after(40, poll)
    try:
        window.mainloop()
    finally:
        log_event({"phase": "ui_closed", "error": error or (controller.error if controller else None)})
        runtime_log.close()
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    return 1 if error or (controller and controller.error) else 0


