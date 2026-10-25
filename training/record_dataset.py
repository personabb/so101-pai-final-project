"""Local LeRobot 0.6.1 dataset frames with measured state and actual sent targets."""
from __future__ import annotations

from serial import SerialException

import importlib.util
from datetime import datetime
from importlib.metadata import version
import math
from pathlib import Path
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERAS = ("camera0", "camera1")
DEFAULT_FPS = 30


def features(height=480, width=640):
    if height < 1 or width < 1:
        raise ValueError("Positive image dimensions required")
    names = [f"{joint}.pos" for joint in JOINTS]
    result = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names.copy()},
        "action": {"dtype": "float32", "shape": (6,), "names": names.copy()},
    }
    for camera in CAMERAS:
        result[f"observation.images.{camera}"] = {
            "dtype": "image", "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def create_local_dataset(path, repo_id="local/so101-demonstrations", fps=30, height=480, width=640):
    """Create a new image dataset; never upload, overwrite, or decode video."""
    if version("lerobot") != "0.6.1":
        raise RuntimeError("LeRobot 0.6.1 is required")
    if type(fps) is not int or fps < 1:
        raise ValueError("fps must be a positive integer")
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Dataset already exists; choose a new path: {destination}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset.create(
        repo_id=repo_id, root=destination, fps=fps, features=features(height, width),
        robot_type="so101_follower", use_videos=False, image_writer_processes=0,
        image_writer_threads=2, video_backend="pyav",
    )


def dataset_frame(observation, sent_action, task, *, height=480, width=640):
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A non-empty task description is required")
    frame = {"task": task.strip()}
    for name, values in (("observation.state", observation), ("action", sent_action)):
        positions = np.asarray([values[f"{joint}.pos"] for joint in JOINTS], dtype=np.float32)
        if not np.isfinite(positions).all():
            raise ValueError(f"Non-finite joint positions: {name}")
        frame[name] = positions
    for camera in CAMERAS:
        image = np.asarray(observation[camera])
        if image.dtype != np.uint8 or image.shape != (height, width, 3):
            raise ValueError(f"Expected RGB uint8 {height}x{width}x3 for {camera}; got {image.shape}/{image.dtype}")
        frame[f"observation.images.{camera}"] = image.copy()
    return frame


def load_follow():
    spec = importlib.util.spec_from_file_location("so101_record_follow", ROOT / "follow-so101.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ArmProxy:
    def __init__(self, arm):
        self._arm = arm

    def __getattr__(self, name):
        return getattr(self._arm, name)


def follow_with_observations(leader, follower, *, task, fps=60, record_fps=30,
                             stop_event=None, run_follow=None, emit=lambda row: None,
                             frame_emit=None, clock=time.perf_counter):
    """Keep official follow running; deliver measured snapshots without writing storage."""
    stop = stop_event if stop_event is not None else threading.Event()
    runner = run_follow or load_follow().run
    next_sample = clock()
    sample = None

    class Leader(_ArmProxy):
        def get_action(self):
            nonlocal next_sample, sample
            now = clock()
            sample = None
            if now >= next_sample and not stop.is_set():
                next_sample += 1 / record_fps
                if next_sample < now:
                    next_sample = now + 1 / record_fps
                try:
                    observation = follower.get_observation()
                    camera_metadata = {}
                    for name, camera in getattr(follower, "cameras", {}).items():
                        if hasattr(camera, "frame_lock"):
                            from iphone_camera_profile import read_latest_with_metadata
                            # Keep the frame and receive timestamp paired under the
                            # producer lock. This is arrival time, not exposure time.
                            observation[name], camera_metadata[name] = read_latest_with_metadata(camera)
                    sample = (observation, now, clock(), camera_metadata)
                except (SerialException, ConnectionError):
                    # Motor communication failure must reach the owner's OFF
                    # cleanup. Continuing would hide the original serial error
                    # and send another target through an incomplete transaction.
                    raise
                except Exception as exc:
                    emit({"phase": "capture_error", "error": str(exc)})
            return self._arm.get_action()

    class Follower(_ArmProxy):
        def send_action(self, action):
            sent_start = clock()
            sent = self._arm.send_action(action)
            sent_end = clock()
            if sample is not None and frame_emit is not None:
                try:
                    frame = dataset_frame(sample[0], sent, task)
                    timing = {"observation_start_s": sample[1], "observation_end_s": sample[2],
                              "action_send_start_s": sent_start, "action_sent_s": sent_end,
                              "acquisition_wall_time": datetime.now().astimezone().isoformat(),
                              "camera_timing": sample[3]}
                    frame_emit(frame, timing)
                except Exception as exc:
                    emit({"phase": "capture_error", "error": str(exc)})
            return sent

    return runner(Leader(leader), Follower(follower), fps=fps, stop_event=stop, emit=emit)


def record_episode(leader, follower, dataset, *, task, fps=30, duration_s=None,
                   stop_event=None, run_follow=None, emit=lambda row: None,
                   height=480, width=640, clock=time.perf_counter, frame_emit=None,
                   pause_event=None, control_lock=None, sleep=time.sleep):
    """Reuse follow's startup/OFF, observing before each official action.

    get_observation is the official measured Follower state plus camera images.
    send_action's return value is the official target actually sent to motors.
    No current-position-relative limiter, PID write, or temperature polling.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A task description is required")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration_s must be positive")
    stop = stop_event if stop_event is not None else threading.Event()
    run_follow = run_follow or load_follow().run
    sample = {}
    count = 0
    started = None
    paused_total = 0.
    lock = control_lock if control_lock is not None else threading.RLock()

    def wait_active():
        nonlocal paused_total
        if pause_event is not None and pause_event.is_set():
            pause_start = clock()
            while pause_event.is_set() and not stop.is_set():
                sleep(.01)
            if started is not None:
                paused_total += clock() - pause_start
        return not stop.is_set()

    class Leader(_ArmProxy):
        def get_action(self):
            nonlocal started
            if not wait_active():
                return {}
            sample["observed_start"] = clock()
            if started is None:
                started = sample["observed_start"]
            sample["observation"] = follower.get_observation()
            sample["observed_end"] = clock()
            return self._arm.get_action()

    class Follower(_ArmProxy):
        def send_action(self, action):
            nonlocal count
            with lock:
                if stop.is_set() or (pause_event is not None and pause_event.is_set()):
                    return action
                return self.send_active(action)

        def send_active(self, action):
            nonlocal count
            action_start = clock()
            sent = self._arm.send_action(action)
            action_end = clock()
            frame = dataset_frame(sample["observation"], sent, task, height=height, width=width)
            dataset.add_frame(frame)
            count += 1
            emit({"phase": "dataset_frame", "frame_index": count - 1,
                  "observation_start_s": sample["observed_start"] - started,
                  "observation_end_s": sample["observed_end"] - started,
                  "action_send_start_s": action_start - started,
                  "action_sent_s": action_end - started,
                  "dataset_added_s": clock() - started})
            if frame_emit is not None:
                frame_emit(frame)
            if duration_s is not None and action_end - started - paused_total >= duration_s:
                stop.set()
            return sent

    run_follow(Leader(leader), Follower(follower), fps=fps, stop_event=stop, emit=emit)
    return count


class RecordingPreview:
    """Display RGB images; N ends this episode, Q/Esc/close ends collection."""
    def __init__(self, on_stop, backend=None, on_episode_stop=None):
        if backend is None:
            import cv2
            backend = cv2
        self.cv = backend
        self.on_stop = on_stop
        self.on_episode_stop = on_episode_stop
        self.name = "SO-101: camera0 | camera1 - N saves episode; Q/Esc stops collection"
        self.opened = False

    def update(self, frame):
        images = [self.cv.cvtColor(frame[f"observation.images.{name}"], self.cv.COLOR_RGB2BGR)
                  for name in CAMERAS]
        self.cv.imshow(self.name, np.concatenate(images, axis=1))
        self.opened = True
        key = self.cv.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or self.cv.getWindowProperty(self.name, self.cv.WND_PROP_VISIBLE) < 1:
            self.on_stop()
        elif key in (ord("n"), ord("N")) and self.on_episode_stop is not None:
            self.on_episode_stop()

    def close(self):
        if self.opened:
            self.cv.destroyWindow(self.name)
            self.opened = False


def connect_cameras(follower, indices=None, fps=30, height=480, width=640, *, profile="pc"):
    """Connect only official OpenCV cameras; no whole robot.connect/configure."""
    if profile == "iphone":
        from iphone_camera_profile import connect_iphone_cameras
        return connect_iphone_cameras(follower, indices, fps=fps, height=height, width=width)
    if profile != "pc":
        raise ValueError("Unknown camera profile; choose pc or iphone")
    if indices is None:
        indices = (0, 1)
    if len(indices) != 2 or set(indices) != {0, 1}:
        raise ValueError("Select cameras 0 and 1 once each")
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.cameras.opencv.configuration_opencv import Cv2Backends
    cameras = {}
    try:
        for name, index in zip(CAMERAS, indices, strict=True):
            camera = OpenCVCamera(OpenCVCameraConfig(
                index_or_path=index, fps=fps, width=width, height=height, backend=Cv2Backends.DSHOW,
            ))
            cameras[name] = camera
            camera.connect()
    except BaseException as failure:
        for name, camera in cameras.items():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as cleanup_error:
                failure.add_note(f"Camera cleanup {name}: {cleanup_error!r}")
        raise
    follower.cameras = cameras
    return cameras


def close_cameras(cameras):
    errors = []
    for name, camera in cameras.items():
        try:
            if camera.is_connected:
                camera.disconnect()
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")
    if errors:
        raise RuntimeError("Camera cleanup failed: " + "; ".join(errors))

