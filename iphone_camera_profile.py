"""Real iPhone/USB cameras for the existing 640x480 recording pipeline.

Camo captures 1280x720 through MSMF. Its RGB image is scaled without cropping
to 640x360, then padded with 60 black rows above and below. Camera reads,
RGB conversion, frame validation and connection cleanup remain LeRobot's.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import Cv2Backends


def read_latest_with_metadata(camera, max_age_ms=500):
    """Peek at the shared latest frame and its local reception metadata, without copying.

    The official read_latest validation is retained. The timestamp is assigned
    by LeRobot after capture/postprocessing, not by Camo at exposure time.
    This helper opens no devices and does not consume new_frame_event.
    """
    camera.read_latest(max_age_ms=max_age_ms)
    with camera.frame_lock:
        frame, received = camera.latest_frame, camera.latest_timestamp
    if frame is None or received is None:
        raise RuntimeError("Camera latest frame was cleared while obtaining its metadata")
    observed = time.perf_counter()
    age_ms = (observed - received) * 1000
    if age_ms > max_age_ms:
        raise TimeoutError(f"Camera latest frame is too old: {age_ms:.1f} ms")
    return frame, {"received_perf_counter_s": received, "peek_perf_counter_s": observed,
                   "age_ms": age_ms, "timestamp_scope": "local postprocessing complete; not exposure"}


class IPhoneCamera(OpenCVCamera):
    """Keep native capture dimensions separate from delivered RGB dimensions."""

    def __init__(self, config: OpenCVCameraConfig, *, width: int = 640, height: int = 480):
        self.output_width, self.output_height = width, height
        super().__init__(config)

    def _reset_connection_settings(self) -> None:
        super()._reset_connection_settings()
        # The inherited validator still uses capture_width/capture_height=1280x720.
        self.width, self.height = self.output_width, self.output_height

    def _postprocess_image(self, image: np.ndarray) -> np.ndarray:
        rgb = super()._postprocess_image(image)
        source_height, source_width = rgb.shape[:2]
        scale = min(self.output_width / source_width, self.output_height / source_height)
        width, height = max(1, round(source_width * scale)), max(1, round(source_height * scale))
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        output = np.zeros((self.output_height, self.output_width, 3), dtype=rgb.dtype)
        left, top = (self.output_width - width) // 2, (self.output_height - height) // 2
        output[top:top + height, left:left + width] = resized
        return output

    def read_latest_with_metadata(self, max_age_ms=500):
        return read_latest_with_metadata(self, max_age_ms)


def camera_device_metadata(cameras):
    """Return the actual identities/indices attached during camera connection."""
    return {name: dict(camera.device_identity) for name, camera in cameras.items()
            if getattr(camera, "device_identity", None) is not None}


def connect_iphone_cameras(follower, indices=None, *, fps=30, height=480, width=640):
    """camera0=iPhone/MSMF, camera1=arm USB/DSHOW; no arm connection or writes."""
    if width < 1 or height < 1:
        raise ValueError("Output dimensions must be positive")
    from windows_camera_devices import resolve_iphone_camera_devices
    devices = resolve_iphone_camera_devices(indices)
    for name, device in devices.items():
        print(f"{name}: {device['friendly_name']} ({device['backend']} index {device['index']})", flush=True)
    cameras = {}
    try:
        cameras["camera0"] = IPhoneCamera(OpenCVCameraConfig(
            index_or_path=devices["camera0"]["index"], fps=fps, width=1280, height=720,
            backend=Cv2Backends.MSMF,
        ), width=width, height=height)
        cameras["camera0"].device_identity = dict(devices["camera0"])
        cameras["camera0"].connect()
        cameras["camera1"] = OpenCVCamera(OpenCVCameraConfig(
            index_or_path=devices["camera1"]["index"], fps=fps, width=width, height=height,
            backend=Cv2Backends.DSHOW,
        ))
        cameras["camera1"].device_identity = dict(devices["camera1"])
        cameras["camera1"].connect()
        # Camo can initially deliver an all-zero placeholder that passes the
        # official warm-up's "frame exists" check. Wait for real image pixels
        # before the GUI can start following; do not require scene motion.
        deadline = time.monotonic() + 10
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("iPhone camera provided no non-zero image within 10 seconds")
            try:
                frame = cameras["camera0"].async_read(timeout_ms=min(200, remaining * 1000))
            except TimeoutError:
                continue
            if np.any(frame):
                break
    except BaseException as failure:
        for name, camera in cameras.items():
            try:
                if camera.is_connected or camera.thread is not None:
                    camera.disconnect()
            except Exception as cleanup_error:
                failure.add_note(f"Camera cleanup {name}: {cleanup_error!r}")
        raise
    if follower is not None:
        follower.cameras = cameras
    return cameras

