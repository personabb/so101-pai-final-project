"""Adapters for official remote ACT inference; no hardware opens during import."""
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import threading
import time
from queue import Empty

import numpy as np

from iphone_camera_profile import IPhoneCamera
from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.helpers import TimedObservation
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

USER_STOP_REASON = "User requested stop and save"


@CameraConfig.register_subclass("so101_iphone")
@dataclass
class SO101IPhoneCameraConfig(OpenCVCameraConfig):
    output_width: int = 640
    output_height: int = 480


class SO101IPhoneCamera(IPhoneCamera):
    def __init__(self, config):
        super().__init__(config, width=config.output_width, height=config.output_height)

    def connect(self, warmup=True):
        super().connect(warmup=warmup)
        # Same non-zero Camo warm-up criterion as the recording profile.
        try:
            deadline = time.monotonic() + 10
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("iPhone camera provided no non-zero image within 10 seconds")
                try:
                    frame = self.async_read(timeout_ms=min(200, remaining * 1000))
                except TimeoutError:
                    continue
                if np.any(frame):
                    return
        except BaseException:
            self.disconnect()
            raise


@RobotConfig.register_subclass("so101_remote_follower")
@dataclass
class SO101RemoteFollowerConfig(SO101FollowerConfig):
    pass


class SO101RemoteFollower(SO101Follower):
    config_class = SO101RemoteFollowerConfig

    def connect(self, calibrate=False):
        # Never start a new calibration from the inference launcher.
        super().connect(calibrate=False)

    def configure(self):
        if not self.is_calibrated:
            raise RuntimeError("Stored follower calibration does not match the arm; inference refused")
        super().configure()  # Official motor settings, with no extra motion limits.

    def disconnect(self):
        # Official whole-robot is_connected guard excludes partial connections.
        # Close each resource even if a camera failed during connect.
        errors = []
        if self.bus.is_connected:
            try:
                self.bus.disconnect(self.config.disable_torque_on_disconnect)
            except Exception as exc:
                errors.append(exc)
                try:
                    self.bus.port_handler.closePort()
                except Exception as close_error:
                    errors.append(close_error)
        elif getattr(self.bus.port_handler, "ser", None) is not None:
            try:
                self.bus.port_handler.closePort()
            except Exception as close_error:
                errors.append(close_error)
        for camera in self.cameras.values():
            if camera.is_connected or getattr(camera, "thread", None) is not None:
                try:
                    camera.disconnect()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError("Follower cleanup failed: " + "; ".join(map(str, errors)))


class _FailureAwareStub:
    def __init__(self, stub, on_error):
        self._stub, self._on_error = stub, on_error

    def __getattr__(self, name):
        method = getattr(self._stub, name)
        def call(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as exc:
                self._on_error(f"RPC {name}: {exc}")
                raise
        return call


class _FailureAwareRobot:
    def __init__(self, robot, on_error, on_success=None):
        self._robot, self._on_error = robot, on_error
        self._on_success = on_success

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def get_observation(self):
        try:
            result = self._robot.get_observation()
        except Exception as exc:
            self._on_error(f"Observation: {exc}")
            raise
        if self._on_success is not None:
            self._on_success("observation")
        return result

    def send_action(self, action):
        try:
            result = self._robot.send_action(action)
        except Exception as exc:
            self._on_error(f"Action: {exc}")
            raise
        if self._on_success is not None:
            self._on_success("action")
        return result


class FailStopRobotClient(RobotClient):
    """Keep official queues/control loop; stop and clean up on detected failures."""
    def __init__(self, config):
        self._dispatch_lock = threading.RLock()
        self._cleanup_lock = threading.Lock()
        self._cleaned_up = False
        self._cleanup_error = None
        self._metrics_lock = threading.Lock()
        self.successful_observation_count = 0
        self.successful_action_count = 0
        self.first_action_applied_at = None
        self._status_thread = None
        self._status_event = threading.Event()
        self._status_finished = threading.Event()
        self._recorder = None
        self._recording_stop = threading.Event()
        self._recording_thread = None
        self._receiver_thread = None
        self._recording_errors = 0
        self._recording_last_error = None
        self._stop_request_path = None
        self._stop_request_after = None
        self._console_stop_available = os.name == "nt"
        self._sender_lock = threading.Lock()
        self._sender_idle = threading.Event()
        self._sender_idle.set()
        self._sender_wake = threading.Event()
        self._sender_stop = threading.Event()
        self._sender_thread = None
        self._pending_observation = None
        self.stop_reason = None
        self.shutdown_event = threading.Event()
        try:
            super().__init__(config)
            self.stub = _FailureAwareStub(self.stub, self.request_stop)
            self.robot = _FailureAwareRobot(self.robot, self.request_stop, self._record_success)
        except BaseException:
            self.stop()
            raise

    def _record_success(self, operation):
        with self._metrics_lock:
            if operation == "observation":
                self.successful_observation_count += 1
                first_action = False
            else:
                self.successful_action_count += 1
                first_action = self.successful_action_count == 1
                if first_action:
                    self.first_action_applied_at = datetime.now().astimezone().isoformat()
        if first_action:
            self.logger.info("First motor action applied (successful_action_count=1)")
            self._status_event.set()

    def enable_recording(self, directory):
        """Record existing camera buffers and RPC data without opening extra devices."""
        from inference_session_recorder import InferenceSessionRecorder

        if self._recorder is not None:
            raise RuntimeError("Session recording is already enabled")
        metadata = {
            "policy_type": self.config.policy_type,
            "remote_model": self.config.pretrained_name_or_path,
            "task": self.config.task,
            "policy_device": self.config.policy_device,
            "action_features": list(self.robot.action_features),
            "observation_features": list(self.robot.observation_features),
            "input_scope": "Exact raw observations sent to PGX, before its image/tensor preprocessing and normalization.",
            "output_scope": "All returned action chunks before queue aggregation, and separately the targets actually sent to motors.",
            "video_scope": "RGB frames sampled from the two already connected camera buffers at nominal client fps; timestamps describe samples, not exposures.",
            "action_timestamp_scope": "Official TimedAction timestamp is observation timestamp plus action offset, not server send time.",
        }
        self._recorder = InferenceSessionRecorder(directory, fps=self.config.fps, metadata=metadata)
        self.logger.info("Camera videos and AI input/output: %s", self._recorder.directory)
        self._recording_thread = threading.Thread(target=self._camera_recording_worker, daemon=True)
        self._recording_thread.start()

    def _record_session(self, method, *args, **kwargs):
        """Keep recording failures outside robot/RPC failure handling."""
        if self._recorder is None:
            return
        try:
            return getattr(self._recorder, method)(*args, **kwargs)
        except Exception as exc:
            self._recording_errors += 1
            message = f"{method}: {exc}"
            if message != self._recording_last_error:
                self.logger.warning("Session recording failed: %s", message)
            self._recording_last_error = message

    def _camera_recording_worker(self):
        while not self._recording_stop.is_set():
            begin = time.perf_counter()
            frames, metadata = {}, {"sampled_at": time.time(), "cameras": {}}
            try:
                for name, camera in self.robot.cameras.items():
                    with camera.frame_lock:
                        frame, received = camera.latest_frame, camera.latest_timestamp
                    if frame is not None:
                        frames[name] = frame
                        metadata["cameras"][name] = {
                            "received_perf_counter_s": received,
                            "age_ms": (time.perf_counter() - received) * 1000 if received is not None else None,
                        }
                if frames:
                    self._record_session("record_video_frames", frames, metadata)
            except Exception as exc:
                self.logger.warning("Camera recording sample failed: %s", exc)
            self._recording_stop.wait(max(0, 1 / self.config.fps - (time.perf_counter() - begin)))

    def _send_observation_now(self, obs):
        """Run the official wire format/RPC and record its real completion."""
        begin = time.time()
        try:
            sent = super().send_observation(obs)
        except Exception:
            self._record_session("record_observation", obs, sent=False, started_at=begin, finished_at=time.time())
            raise
        self._record_session("record_observation", obs, sent=sent, started_at=begin, finished_at=time.time())
        return sent

    def send_observation(self, obs):
        """Accept one observation; the sender owns serialization and network waits."""
        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation")
        with self._sender_lock:
            if not self.running or self._sender_stop.is_set() or not self._sender_idle.is_set():
                return False
            self._sender_idle.clear()
            self._pending_observation = obs
            if self._sender_thread is None:
                self._sender_thread = threading.Thread(target=self._observation_sender_worker, daemon=True)
                self._sender_thread.start()
            self._sender_wake.set()
        return True

    def _observation_sender_worker(self):
        while not self._sender_stop.is_set():
            self._sender_wake.wait()
            self._sender_wake.clear()
            with self._sender_lock:
                obs = self._pending_observation
                self._pending_observation = None
            if obs is None:
                continue
            try:
                if self.running and not self._sender_stop.is_set():
                    sent = self._send_observation_now(obs)
                    if not sent and obs.must_go and self.running:
                        self.must_go.set()
                else:
                    now = time.time()
                    self._record_session("record_observation", obs, sent=False, started_at=now, finished_at=now)
            except Exception as exc:
                self.request_stop(f"Observation sender: {exc}")
            finally:
                self._sender_idle.set()

    def _ready_to_send_observation(self):
        return self._sender_idle.is_set() and not self._sender_stop.is_set() and super()._ready_to_send_observation()

    def control_loop_observation(self, task, verbose=False):
        """Keep device reads on the control thread; reserve must-go before enqueue."""
        try:
            raw = self.robot.get_observation()
            raw["task"] = task
            with self.latest_action_lock:
                step = max(self.latest_action, 0)
            obs = TimedObservation(timestamp=time.time(), timestep=step, observation=raw)
            with self._dispatch_lock:
                with self.action_queue_lock:
                    obs.must_go = self.must_go.is_set() and self.action_queue.empty()
                    if obs.must_go:
                        self.must_go.clear()
            accepted = self.send_observation(obs)
            if not accepted:
                now = time.time()
                self._record_session("record_observation", obs, sent=False, started_at=now, finished_at=now)
                if obs.must_go and self.running:
                    self.must_go.set()
            return raw
        except Exception as exc:
            self.logger.error("Error in observation sender: %s", exc)
            self.request_stop(f"Observation: {exc}")

    def enable_status(self, path):
        """Write passive status on a separate thread; never do file I/O in control ticks."""
        if self._status_thread is not None:
            raise RuntimeError("Runtime status is already enabled")
        self._status_path = Path(path)
        self._stop_request_path = self._status_path.parent / "stop-request.json"
        self._stop_request_after = time.time()
        self._status_event.set()
        self._status_thread = threading.Thread(target=self._status_worker, daemon=True)
        self._status_thread.start()

    def _check_stop_request(self):
        if self._stop_request_path is None or self.shutdown_event.is_set():
            return
        try:
            request = json.loads(self._stop_request_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return
        if (not isinstance(request, dict) or request.get("action") != "stop_and_save"
                or request.get("pid") != os.getpid()):
            return
        issued = request.get("requested_wall_time")
        if not isinstance(issued, (float, int)) or not math.isfinite(issued) or issued < self._stop_request_after:
            return
        self.logger.info("Stop-and-save request received; stopping motor dispatch and finishing recordings.")
        self.request_stop(USER_STOP_REASON)

    def _check_console_stop(self):
        if not self._console_stop_available or self.shutdown_event.is_set():
            return
        try:
            import msvcrt

            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                if key.lower() == "q":
                    self.logger.info("Q pressed: stopping motor dispatch and saving recordings.")
                    self.request_stop(USER_STOP_REASON)
                    return
        except (ImportError, OSError) as exc:
            self._console_stop_available = False
            self.logger.warning("Q-key stop is unavailable in this console: %s", exc)

    def _status_worker(self):
        previous_error = None
        last_write = 0.0
        while True:
            signaled = self._status_event.wait(timeout=0.05)
            self._status_event.clear()
            self._check_console_stop()
            now = time.monotonic()
            if not signaled and now - last_write < 1:
                continue
            last_write = now
            self._check_stop_request()
            with self._metrics_lock:
                finished = self._status_finished.is_set()
                result = {
                    "pid": os.getpid(), "updated_at": datetime.now().astimezone().isoformat(),
                    "phase": ("cleanup_failed" if self._cleanup_error else "stopped") if self._cleaned_up
                             else ("stopping" if self.shutdown_event.is_set() else "running"),
                    "successful_observation_count": self.successful_observation_count,
                    "successful_action_count": self.successful_action_count,
                    "first_action_applied_at": self.first_action_applied_at,
                    "stop_reason": self.stop_reason, "cleanup_error": self._cleanup_error,
                    "recording_directory": str(self._recorder.directory) if self._recorder else None,
                    "recording_integration_errors": self._recording_errors,
                    "recording_last_error": self._recording_last_error,
                    "recording_stats": self._recorder.stats if self._recorder else None,
                    "stop_and_save_supported": True,
                    "stop_request_path": str(self._stop_request_path),
                    "q_key_stop_available": self._console_stop_available,
                    "observation_send_busy": not self._sender_idle.is_set(),
                }
            try:
                self._status_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._status_path.with_name(self._status_path.name + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
                temporary.replace(self._status_path)
                previous_error = None
            except Exception as exc:
                if str(exc) != previous_error:
                    self.logger.warning("Runtime status could not be saved: %s", exc)
                    previous_error = str(exc)
            if finished:
                return

    def request_stop(self, reason="Stopped"):
        # Wait for any current send to finish, then prevent another send.
        with self._dispatch_lock:
            self.stop_reason = self.stop_reason or reason
            self.shutdown_event.set()
            self._status_event.set()
            if hasattr(self, "action_queue_lock"):
                with self.action_queue_lock:
                    while True:
                        try:
                            self.action_queue.get_nowait()
                        except Empty:
                            break
        self._sender_stop.set()
        with self._sender_lock:
            pending = self._pending_observation
            self._pending_observation = None
        self._sender_wake.set()
        if pending is not None:
            now = time.time()
            self._record_session("record_observation", pending, sent=False, started_at=now, finished_at=now)
        # Cancel outstanding RPCs so the receiver can join.
        if hasattr(self, "channel"):
            self.channel.close()
        if hasattr(self, "start_barrier"):
            self.start_barrier.abort()

    def control_loop_action(self, verbose=False):
        with self._dispatch_lock:
            if not self.running:
                return {}
            timed_action = None
            if self._recorder is not None:
                with self.action_queue_lock:
                    if not self.action_queue.empty():
                        timed_action = self.action_queue.queue[0]
            result = super().control_loop_action(verbose)
            if timed_action is not None:
                self._record_session("record_applied_action",
                    requested=self._action_tensor_to_action_dict(timed_action.get_action()),
                    applied=result, timestep=timed_action.get_timestep(), timestamp=time.time(),
                )
            return result

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        with self._dispatch_lock:
            if self._recorder is not None:
                self._record_session("record_action_chunk", incoming_actions, received_at=time.time())
            if self.running:
                return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def receive_actions(self, verbose=False):
        try:
            return super().receive_actions(verbose)
        except threading.BrokenBarrierError:
            if self.running:
                self.request_stop("Action receiver start barrier failed")
        except Exception as exc:
            self.request_stop(f"Action receiver: {exc}")

    def stop(self):
        with self._cleanup_lock:
            if self._cleaned_up:
                return
            self.request_stop()
            self._recording_stop.set()
            if self._recording_thread is not None:
                self._recording_thread.join(timeout=2)
            try:
                if hasattr(self, "robot"):
                    self.robot.disconnect()
            except Exception as exc:
                self._cleanup_error = str(exc)
                raise
            finally:
                self._cleaned_up = True
                self._status_finished.set()
                self._status_event.set()
                if self._status_thread is not None:
                    self._status_thread.join(timeout=2)
                if self._receiver_thread is not None and self._receiver_thread is not threading.current_thread():
                    self._receiver_thread.join()
                if (self._sender_thread is not None and self._sender_thread.ident is not None
                        and self._sender_thread is not threading.current_thread()):
                    self._sender_thread.join()
                self._sender_idle.set()
                if self._recorder is not None:
                    result = self._record_session("close", reason=self.stop_reason or "Client stopped")
                    if isinstance(result, dict):
                        if result.get("phase") == "closed" and not result.get("incomplete_recording"):
                            self.logger.info("Recordings saved: %s", self._recorder.directory)
                        else:
                            self.logger.warning("Recording is incomplete; see %s", self._recorder.directory / "manifest.json")


def run_client(config, *, client_class=FailStopRobotClient, status_path=None, recording_directory=None):
    client = client_class(config)
    receiver = None
    try:
        if status_path is not None:
            client.enable_status(status_path)
        if recording_directory is not None:
            client.enable_recording(recording_directory)
        if not client.start():
            raise RuntimeError("Policy server handshake failed; robot disconnected")
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        client._receiver_thread = receiver
        receiver.start()
        client.control_loop(task=config.task)
        if client.stop_reason is not None and client.stop_reason != USER_STOP_REASON:
            raise RuntimeError(client.stop_reason)
    finally:
        client.stop()
        if receiver is not None:
            receiver.join()

