"""Asynchronous recording of existing-camera video and official AI I/O.

This module opens files, never cameras, robots, RPC channels, or CUDA contexts.
Video sampling and sent observations are separate streams. Submit calls perform
no disk I/O and never wait for queue space. Recording failures never alter control.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import uuid

import cv2
import numpy as np

CAMERAS = ('camera0', 'camera1')
JOINTS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos')


def _now():
    return datetime.now().astimezone().isoformat()


def _plain(value):
    """Convert small CPU-only metadata/tensors without importing torch."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    device = getattr(value, 'device', None)
    if device is not None and getattr(device, 'type', 'cpu') != 'cpu':
        raise ValueError('Recorder refuses a GPU tensor; official client transport must be CPU')
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported recording value: {type(value).__name__}')


class InferenceSessionRecorder:
    """Record into an exclusive session child beneath ``directory``.

    ``record_video_frames`` copies RGB arrays on the camera sampler thread.
    ``record_observation`` snapshots its raw dictionary and image references;
    LeRobot camera postprocessing must not mutate an already delivered array.
    Action dictionaries are small copies. Encoding, serialization, and filesystem
    work belong exclusively to the writer thread. ``close`` is called after the
    camera sampler and control/receiver threads have stopped.
    """

    def __init__(self, directory, *, fps=30, metadata=None,
                 event_queue_size=64, video_queue_size=4):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be positive')
        if event_queue_size < 1 or video_queue_size < 1:
            raise ValueError('Queue capacities must be positive')
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        name = 'session-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + uuid.uuid4().hex[:8]
        self.directory = base / name
        self.directory.mkdir(exist_ok=False)
        (self.directory / 'observations').mkdir(exist_ok=False)
        self.fps = float(fps)
        self.metadata = _plain(metadata or {})
        self._events = Queue(maxsize=event_queue_size)
        self._videos = Queue(maxsize=video_queue_size)
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._closing = False
        self._reason = None
        self._started_at = _now()
        self._started_wall = time.time()
        self._ended_at = None
        self._ended_wall = None
        self._sequence = 0
        self._submitted = Counter()
        self._accepted = Counter()
        self._written = Counter()
        self._dropped = Counter()
        self._failed = Counter()
        self._errors = []
        self._error_count = 0
        self._video_counts = Counter()
        self._video_sizes = {}
        self._writers = {}
        self._streams = {}
        self._thread = threading.Thread(target=self._worker, name='inference-session-writer', daemon=True)
        self._thread.start()

    @property
    def stats(self):
        with self._lock:
            return {'submitted': dict(self._submitted), 'accepted': dict(self._accepted),
                    'written': dict(self._written), 'dropped': dict(self._dropped),
                    'failed': dict(self._failed), 'error_count': self._error_count,
                    'errors': list(self._errors), 'video_frame_counts': dict(self._video_counts),
                    'closing': self._closing, 'writer_finished': self._finished.is_set(),
                    'event_queue_size': self._events.qsize(), 'video_queue_size': self._videos.qsize()}

    def _error(self, kind, exc):
        with self._lock:
            self._failed[kind] += 1
            self._error_count += 1
            if len(self._errors) < 100:
                self._errors.append({'at': _now(), 'kind': kind, 'error': str(exc)})

    def _submit(self, kind, value, *, video=False):
        queue = self._videos if video else self._events
        with self._lock:
            self._submitted[kind] += 1
            if self._closing:
                self._dropped[kind] += 1
                return False
            self._sequence += 1
            item = (kind, self._sequence, time.time(), value)
            try:
                queue.put_nowait(item)
            except Full:
                self._dropped[kind] += 1
                return False
            self._accepted[kind] += 1
            return True

    def record_video_frames(self, frames, metadata=None):
        try:
            with self._lock:
                if self._closing:
                    self._submitted['video_frames'] += 1
                    self._dropped['video_frames'] += 1
                    return False
            copied = {}
            for camera in CAMERAS:
                image = frames[camera]
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f'{camera} must be raw RGB uint8 HxWx3')
                copied[camera] = image.copy()
            return self._submit('video_frames', {'frames': copied, 'metadata': _plain(metadata or {})}, video=True)
        except Exception as exc:
            self._error('video_frames', exc)
            return False

    def record_observation(self, obs, sent, started_at, finished_at):
        try:
            value = {'raw': dict(obs.get_observation()), 'timestamp': obs.get_timestamp(),
                     'timestep': obs.get_timestep(), 'must_go': bool(getattr(obs, 'must_go', False)),
                     'sent': bool(sent), 'started_at': started_at, 'finished_at': finished_at}
            return self._submit('observation', value)
        except Exception as exc:
            self._error('observation', exc)
            return False

    def record_action_chunk(self, incoming_actions, received_at=None):
        try:
            actions = [{'timestamp': action.get_timestamp(), 'timestep': action.get_timestep(),
                        'action': action.get_action()} for action in incoming_actions]
            return self._submit('action_chunk', {'received_at': time.time() if received_at is None else received_at,
                                                 'actions': actions})
        except Exception as exc:
            self._error('action_chunk', exc)
            return False

    def record_applied_action(self, requested, applied, timestep, timestamp):
        try:
            return self._submit('applied_action', {'requested': dict(requested), 'applied': dict(applied),
                                                  'timestep': timestep, 'timestamp': timestamp})
        except Exception as exc:
            self._error('applied_action', exc)
            return False

    def _json_line(self, name, value):
        if name not in self._streams:
            self._streams[name] = (self.directory / name).open('x', encoding='utf-8')
        self._streams[name].write(json.dumps(_plain(value), ensure_ascii=False, allow_nan=False) + '\n')

    def _write_video(self, sequence, wall_time, value):
        for camera in CAMERAS:
            rgb = value['frames'][camera]
            size = (int(rgb.shape[1]), int(rgb.shape[0]))
            if camera not in self._writers:
                path = self.directory / (camera + '.avi')
                if path.exists():
                    raise FileExistsError(path)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), self.fps, size)
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f'MJPG VideoWriter could not open {camera}')
                self._writers[camera] = writer
                self._video_sizes[camera] = size
            if self._video_sizes[camera] != size:
                raise ValueError(f'{camera} changed frame dimensions')
            writer = self._writers[camera]
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            with self._lock:
                index = self._video_counts[camera]
                self._video_counts[camera] += 1
            self._json_line('camera_frames.jsonl', {
                'sequence': sequence, 'camera': camera, 'video': camera + '.avi',
                'frame_index': index, 'sampled_wall_time': value['metadata'].get('sampled_at', wall_time),
                'enqueued_wall_time': wall_time,
                'capture_metadata': value['metadata'].get('cameras', value['metadata']).get(camera, {}),
                'width': size[0], 'height': size[1], 'color_input': 'RGB',
            })

    def _write_observation(self, sequence, wall_time, value):
        raw = value['raw']
        state = {name: raw[name] for name in JOINTS if name in raw}
        record = {key: item for key, item in value.items() if key != 'raw'}
        record.update(sequence=sequence, recorded_wall_time=wall_time,
                      joint_state=state, task=raw.get('task'), camera_shapes={}, npz=None,
                      input_scope='Exact raw observation before PGX preprocessing/normalization; excludes internal policy tensors/latents.')
        arrays = {}
        for camera in CAMERAS:
            key = camera if camera in raw else 'observation.images.' + camera
            image = raw[key]
            if not isinstance(image, np.ndarray):
                raise ValueError(f'Raw observation {camera} must be numpy RGB array')
            arrays[camera] = image
            record['camera_shapes'][camera] = list(image.shape)
        if value['sent']:
            if 'observation.state' in raw:
                arrays['observation.state'] = np.asarray(raw['observation.state'])
            else:
                if len(state) != len(JOINTS):
                    raise ValueError('Successful observation missing a recorded joint state')
                arrays['observation.state'] = np.asarray([state[name] for name in JOINTS])
            arrays.update(timestamp=np.asarray(value['timestamp']), timestep=np.asarray(value['timestep']),
                          task=np.asarray(str(raw.get('task', ''))), must_go=np.asarray(value['must_go']))
            name = f'observations/observation-{sequence:09d}.npz'
            with (self.directory / name).open('xb') as handle:
                np.savez(handle, **arrays)
            record['npz'] = name
        record['send_duration_s'] = value['finished_at'] - value['started_at']
        self._json_line('observations.jsonl', record)

    def _process(self, item):
        kind, sequence, wall_time, value = item
        try:
            if kind == 'video_frames':
                self._write_video(sequence, wall_time, value)
            elif kind == 'observation':
                self._write_observation(sequence, wall_time, value)
            elif kind == 'action_chunk':
                value = {**value, 'sequence': sequence, 'recorded_wall_time': wall_time,
                         'count': len(value['actions']),
                         'timestamp_scope': 'TimedAction timestamp is planned execution time from observation timestamp plus step/fps, not server send time.'}
                self._json_line('action_chunks.jsonl', value)
            elif kind == 'applied_action':
                self._json_line('applied_actions.jsonl', {**value, 'sequence': sequence,
                                                        'recorded_wall_time': wall_time})
            else:
                raise ValueError('Unknown recording kind: ' + kind)
            with self._lock:
                self._written[kind] += 1
        except Exception as exc:
            self._error(kind, exc)

    def _manifest(self, phase):
        stats = self.stats
        return {'schema_version': 1, 'phase': phase, 'directory': str(self.directory),
                'started_at': self._started_at, 'started_wall_time': self._started_wall,
                'ended_at': self._ended_at, 'ended_wall_time': self._ended_wall,
                'duration_s': (self._ended_wall or time.time()) - self._started_wall,
                'updated_at': _now(), 'close_reason': self._reason, 'metadata': self.metadata,
                'stats': stats, 'queue_capacities': {'events': self._events.maxsize, 'video_pairs': self._videos.maxsize},
                'video': {'codec': 'MJPG', 'container': 'AVI', 'nominal_fps': self.fps,
                          'pixel_encoding': 'Lossy MJPEG video of raw delivered RGB; lossless successfully-sent observation pixels in NPZ.',
                          'timing': 'Use camera_frames.jsonl sampled_wall_time/capture_metadata; AVI nominal FPS is not proof of evenly spaced acquisition.'},
                'input_scope': 'Raw successfully sent PC observations; no PGX normalized/internal tensors or policy latents.',
                'streams': {'camera_video': ['camera0.avi', 'camera1.avi'], 'video_frame_timestamps': 'camera_frames.jsonl',
                            'sent_observations': 'observations.jsonl', 'raw_observation_arrays': 'observations/*.npz',
                            'returned_action_chunks_before_aggregation': 'action_chunks.jsonl',
                            'requested_and_applied_actions': 'applied_actions.jsonl'},
                'incomplete_recording': bool(stats['dropped'] or stats['failed'] or not stats['writer_finished']
                                             or stats['event_queue_size'] or stats['video_queue_size']),
                'robot_or_camera_opened_by_recorder': False}

    def _save_manifest(self, phase):
        try:
            value = self._manifest(phase)
            temporary = self.directory / ('manifest.' + uuid.uuid4().hex + '.tmp')
            with temporary.open('x', encoding='utf-8') as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, self.directory / 'manifest.json')
        except Exception as exc:
            self._error('manifest', exc)

    def _flush(self):
        for stream in self._streams.values():
            try:
                stream.flush()
            except Exception as exc:
                self._error('flush', exc)

    def _worker(self):
        self._save_manifest('recording')
        last_flush = time.monotonic()
        try:
            while not self._stop.is_set() or not self._events.empty() or not self._videos.empty():
                # A separate queue prevents camera flood from using action slots.
                for _ in range(8):
                    try:
                        item = self._events.get_nowait()
                    except Empty:
                        break
                    try:
                        self._process(item)
                    finally:
                        self._events.task_done()
                try:
                    item = self._videos.get_nowait()
                except Empty:
                    item = None
                if item is not None:
                    try:
                        self._process(item)
                    finally:
                        self._videos.task_done()
                if time.monotonic() - last_flush >= 1:
                    self._flush()
                    self._save_manifest('draining' if self._stop.is_set() else 'recording')
                    last_flush = time.monotonic()
                if self._events.empty() and self._videos.empty():
                    time.sleep(.005)
        except Exception as exc:
            self._error('writer', exc)
        finally:
            with self._lock:
                self._closing = True
            for queue in (self._events, self._videos):
                while True:
                    try:
                        item = queue.get_nowait()
                    except Empty:
                        break
                    with self._lock:
                        self._dropped[item[0]] += 1
                    queue.task_done()
            self._flush()
            for writer in self._writers.values():
                try:
                    writer.release()
                except Exception as exc:
                    self._error('video_release', exc)
            for stream in self._streams.values():
                try:
                    stream.close()
                except Exception as exc:
                    self._error('stream_close', exc)
            self._finished.set()
            self._ended_at = _now()
            self._ended_wall = time.time()
            self._save_manifest('closed')

    def close(self, *, timeout=10, reason='closed'):
        with self._close_lock:
            with self._lock:
                self._closing = True
                self._reason = self._reason or reason
            self._stop.set()
            if self._thread is not threading.current_thread():
                self._thread.join(timeout=max(0, timeout))
            if not self._finished.is_set():
                self._save_manifest('closing_timeout')
            else:
                self._save_manifest('closed')
            return self._manifest('closed' if self._finished.is_set() else 'closing_timeout')

