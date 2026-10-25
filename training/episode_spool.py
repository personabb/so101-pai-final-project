"""Lossless recording spool, independent of LeRobot's expensive episode commit.

Only the recording worker writes these files. One separate worker submits whole
episodes to the public LeRobot save_episode(episode_data=...) API in order.
"""
from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
from PIL import Image


class EpisodeSpool:
    def __init__(self, directory, fps, camera_names):
        self.directory = Path(directory) / "pending_dataset"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fps, self.camera_names = fps, tuple(camera_names)
        self.rows = []
        self.paths = {name: [] for name in self.camera_names}
        for name in self.camera_names:
            (self.directory / name).mkdir()
        self.manifest_path = self.directory / "frames.jsonl"
        self.manifest = self.manifest_path.open("w", encoding="utf-8")
        self.raw_path = self.directory / "frames.rgb"
        self.raw = self.raw_path.open("wb")

    @classmethod
    def from_sealed(cls, manifest_path, fps, camera_names):
        """Read a closed spool in the export process without reopening writers."""
        spool = cls.__new__(cls)
        spool.manifest_path = Path(manifest_path)
        spool.directory = spool.manifest_path.parent
        spool.raw_path = spool.directory / "frames.rgb"
        spool.fps, spool.camera_names = fps, tuple(camera_names)
        spool.manifest = spool.raw = None
        with spool.manifest_path.open(encoding="utf-8") as source:
            spool.rows = [json.loads(line) for line in source if line.strip()]
        spool.paths = {name: [] for name in spool.camera_names}
        for index, row in enumerate(spool.rows):
            if row["frame_index"] != index:
                raise ValueError("Sealed spool frame indices must be contiguous")
            for name in spool.camera_names:
                path = Path(row["images"][name])
                expected = spool.directory / name / f"frame_{index:06d}.png"
                if path.resolve() != expected.resolve():
                    raise ValueError("Sealed spool image path does not match its frame")
                spool.paths[name].append(path)
        expected_bytes = len(spool.rows) * len(spool.camera_names) * 480 * 640 * 3
        if spool.raw_path.stat().st_size != expected_bytes:
            raise OSError("Sealed raw recording has unexpected size; original retained")
        return spool

    def add_frame(self, frame, timing):
        index = len(self.rows)
        image_paths = {}
        for name in self.camera_names:
            path = self.directory / name / f"frame_{index:06d}.png"
            image = frame[f"observation.images.{name}"]
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError("Spool requires RGB uint8 640x480")
            image_bytes = image.tobytes()
            if self.raw.write(image_bytes) != len(image_bytes):
                raise OSError("Incomplete raw camera frame write")
            self.paths[name].append(path)
            image_paths[name] = str(path)
        row = {"frame_index": index, "timestamp": index / self.fps,
               "task": frame["task"], "observation.state": frame["observation.state"].tolist(),
               "action": frame["action"].tolist(), "images": image_paths, **timing}
        self.manifest.write(json.dumps(row) + "\n")
        self.manifest.flush()
        self.raw.flush()
        self.rows.append(row)

    def close(self):
        errors = []
        for name in ("manifest", "raw"):
            stream = getattr(self, name)
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    setattr(self, name, None)
        if errors:
            for extra in errors[1:]:
                errors[0].add_note(f"Additional spool close error: {extra}")
            raise errors[0]

    def iter_frames(self):
        image_bytes = 480 * 640 * 3
        with self.raw_path.open("rb") as source:
            for row in self.rows:
                frame = {"task": row["task"], "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                         "action": np.asarray(row["action"], dtype=np.float32)}
                for name in self.camera_names:
                    data = source.read(image_bytes)
                    if len(data) != image_bytes:
                        raise OSError("Truncated raw recording; pending files retained")
                    frame[f"observation.images.{name}"] = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
                yield frame, row

    def save(self, dataset, episode_index, *, emit=lambda row: None, wait_for_capture=lambda: None):
        """Use the supported whole-episode API; do not mutate a live writer buffer."""
        started = time.perf_counter()
        emit({"phase": "png_start", "episode": episode_index, "perf_counter_s": started})
        # Encoding happens only here, away from live capture and motor control.
        def encode(image, path):
            wait_for_capture()
            Image.fromarray(image).save(path, compress_level=1)

        # Bound submitted image jobs as well as worker count: 8 RGB images use
        # about 7.4 MB, rather than queuing a whole episode in memory. Only PNG
        # encoding runs in parallel; the official dataset remains single-owner.
        pending = deque()
        with ThreadPoolExecutor(max_workers=4) as encoder:
            for index, (frame, _) in enumerate(self.iter_frames()):
                wait_for_capture()
                for name in self.camera_names:
                    if len(pending) >= 8:
                        pending.popleft().result()
                    pending.append(encoder.submit(encode, frame[f"observation.images.{name}"], self.paths[name][index]))
            while pending:
                pending.popleft().result()
        emit({"phase": "png_complete", "episode": episode_index, "perf_counter_s": time.perf_counter()})
        feature_names = dataset.features
        data = {key: [] for key in feature_names}
        data.update(size=len(self.rows), task=[row["task"] for row in self.rows],
                    episode_index=episode_index)
        for key in ("observation.state", "action"):
            data[key] = [np.asarray(row[key], dtype=np.float32) for row in self.rows]
        data["timestamp"] = [row["timestamp"] for row in self.rows]
        data["frame_index"] = [row["frame_index"] for row in self.rows]
        for name in self.camera_names:
            data[f"observation.images.{name}"] = [str(path) for path in self.paths[name]]
        emit({"phase": "official_commit_start", "episode": episode_index, "perf_counter_s": time.perf_counter()})
        dataset.save_episode(episode_data=data)
        emit({"phase": "official_commit_complete", "episode": episode_index, "perf_counter_s": time.perf_counter()})
        return time.perf_counter() - started

    def discard_committed_files(self):
        """Remove only files created by this spool, after successful embedding."""
        self.close()
        for name in self.camera_names:
            for path in self.paths[name]:
                if path.exists():
                    path.unlink()
            (self.directory / name).rmdir()
        self.manifest_path.unlink()
        self.raw_path.unlink()
        self.directory.rmdir()

