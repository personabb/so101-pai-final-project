"""Record locally: --execute starts continuous follow and the live-camera GUI.

R starts storage, P pauses storage, S saves, F finalizes collection; Q exits/OFF.

With --console --execute, collection begins immediately instead.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training"))
from record_dataset import RecordingPreview, close_cameras, connect_cameras, create_local_dataset, load_follow, record_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Describe one demonstrated task")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / f"demonstrations-{datetime.now():%Y%m%d-%H%M%S-%f}")
    parser.add_argument("--repo-id", default="local/so101-demonstrations")
    parser.add_argument("--leader-port", default="auto")
    parser.add_argument("--follower-port", default="auto")
    parser.add_argument("--camera-profile", choices=("pc", "iphone"), default="pc",
                        help="pc: original cameras; iphone: camera0=iPhone, camera1=arm USB")
    parser.add_argument("--camera-indices", nargs=2, type=int, default=None,
                        help="pc: defaults to 0 1; iphone: automatically identifies Camo/MSMF and arm USB/DSHOW, correcting stale indices")
    parser.add_argument("--fps", type=int, default=30, help="Official recording default: 30 Hz")
    parser.add_argument("--episode-seconds", type=float, default=60)
    parser.add_argument("--episodes", type=int, default=1, help="Episode limit for --console only; GUI recording is manual and unlimited")
    parser.add_argument("--execute", action="store_true", help="Start continuous follow and live cameras; R starts recording. With --console, start collection immediately")
    parser.add_argument("--no-preview", action="store_true", help="Disable the two-camera preview; Ctrl+C still stops and saves")
    parser.add_argument("--console", action="store_true", help="Use the previous console recording flow instead of the lifecycle GUI")
    args = parser.parse_args()
    if args.camera_indices is None and args.camera_profile == "pc":
        args.camera_indices = [0, 1]
    if not args.task.strip() or not 1 <= args.fps <= 120 or args.episodes < 1:
        parser.error("Non-empty task, positive episode count and fps 1..120 required")
    if args.camera_profile == "pc" and set(args.camera_indices) != {0, 1}:
        parser.error("The pc profile uses cameras 0 and 1 once each")
    if args.camera_indices is not None and min(args.camera_indices) < 0:
        parser.error("Choose non-negative camera indices")
    if not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0:
        parser.error("Use a positive finite episode duration")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "datasets"):
        parser.error("--output must be inside this project's datasets directory")
    if output.exists():
        parser.error("Choose a new output path; existing datasets are never overwritten")
    plan = {"task": args.task, "output": str(output), "fps": args.fps,
            "camera_profile": args.camera_profile, "camera_indices": args.camera_indices,
            "requested_camera_indices": args.camera_indices,
            "camera_selection": "device identity (Camo/MSMF, arm USB/DSHOW)" if args.camera_profile == "iphone" else "indices",
            "camera_mapping": ({"camera0": "Camera 1: iPhone/MSMF 1280x720 -> aspect-ratio-preserving 640x360 + 60 black rows top/bottom",
                                "camera1": "Camera 2: arm USB/DSHOW 640x480"}
                               if args.camera_profile == "iphone" else {"camera0": "Original camera 0", "camera1": "Original camera 1"}),
            "episodes": args.episodes, "episode_seconds": args.episode_seconds,
            "storage": "LeRobot image dataset (PNG), no upload",
            "state": "Follower.get_observation measured positions",
            "action": "Follower.send_action actual returned targets", "execute": args.execute,
            "mode": "console" if args.console else "GUI", "gui_episode_limit": None,
            "follow_fps": args.fps if args.console else 60,
            "gui_launch": "Continuous Leader-to-Follower follow and live cameras; no storage until R",
            "episodes_scope": "--console limit only; GUI recordings have no episode-count limit",
            "preview": not args.no_preview,
            "stop": ("N saves current episode then waits for reset; Q/Esc/close/Ctrl+C finishes collection"
                     if args.console else "R starts storage; P pauses storage only; S saves and waits for R; F finalizes collection with follow/cameras live; Q/Esc exits, OFF and closes ports/cameras")}
    if args.camera_profile == "iphone":
        plan["iphone_preprocessing"] = {"native_width": 1280, "native_height": 720,
            "output_width": 640, "output_height": 480, "resize_width": 640, "resize_height": 360,
            "padding_top": 60, "padding_bottom": 60, "color": "RGB",
            "method": "Aspect-ratio-preserving resize with black padding; no crop"}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not args.console:
        if args.no_preview:
            parser.error("--no-preview applies to --console; the recording GUI always shows live cameras")
        from so101_record_ui import run_ui
        return run_ui(args, ROOT)
    follow = load_follow()
    current = {"stop": threading.Event(), "waiting": False}
    cancelled = threading.Event()
    def request_stop(*_):
        cancelled.set()
        current["stop"].set()
        if current["waiting"]:
            raise KeyboardInterrupt()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    report = {**plan, "source": "https://huggingface.co/docs/lerobot/il_robots", "episodes_recorded": [],
              "timestamp_policy": "Dataset frame_index/fps; acquisition/action wall timing recorded separately",
              "camera_sync": "Official get_observation sequential latest RGB frames; not hardware-trigger synchronized",
              "calibration": {role: json.loads((ROOT / f"calibration/{role}/so101_{role}.json").read_text(encoding="utf-8"))
                              for role in ("leader", "follower")}}
    dataset = create_local_dataset(output, args.repo_id, fps=args.fps)
    exit_code = 0
    try:
        with (output / "acquisition.jsonl").open("w", encoding="utf-8") as log:
            for episode in range(args.episodes):
                if cancelled.is_set():
                    report["stopped_by_user"] = True
                    break
                if episode:
                    current["waiting"] = True
                    try:
                        answer = input("Next episode: reset task and press Enter; q finishes collection. ").strip().lower()
                    finally:
                        current["waiting"] = False
                    if answer == "q":
                        break
                if cancelled.is_set():
                    report["stopped_by_user"] = True
                    break
                current["stop"] = threading.Event()
                if cancelled.is_set():
                    current["stop"].set()
                    report["stopped_by_user"] = True
                    break
                cameras = {}
                preview = None
                rows = []
                def emit(row):
                    entry = {"episode": episode, "wall_time": datetime.now().astimezone().isoformat(), **row}
                    log.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                    log.flush()
                    if row.get("phase") in ("preflight", "stopped", "fault"):
                        rows.append(entry)
                    if row.get("phase") in ("following", "stopped", "fault"):
                        print(json.dumps(entry, ensure_ascii=False, default=str), flush=True)
                try:
                    if cancelled.is_set():
                        raise KeyboardInterrupt()
                    leader, follower = follow.build_arms(args.leader_port, args.follower_port)
                    cameras = connect_cameras(follower, args.camera_indices, fps=args.fps,
                                              profile=args.camera_profile)
                    if args.camera_profile == "iphone":
                        from iphone_camera_profile import camera_device_metadata
                        report["camera_devices"] = camera_device_metadata(cameras)
                        report["camera_indices"] = [report["camera_devices"][name]["index"]
                                                    for name in ("camera0", "camera1")]
                    if cancelled.is_set():
                        raise KeyboardInterrupt()
                    if not args.no_preview:
                        preview = RecordingPreview(request_stop, on_episode_stop=lambda: current["stop"].set())
                    count = record_episode(leader, follower, dataset, task=args.task, fps=args.fps,
                                           duration_s=args.episode_seconds, stop_event=current["stop"],
                                           run_follow=follow.run, emit=emit,
                                           frame_emit=preview.update if preview is not None else None)
                    if count < 1:
                        raise RuntimeError("No frames captured; no empty episode saved")
                    dataset.save_episode()
                    report["episodes_recorded"].append({"episode": episode, "frames": count,
                                                       "startup_and_stop": rows})
                finally:
                    try:
                        if preview is not None:
                            preview.close()
                    finally:
                        close_cameras(cameras)
                if cancelled.is_set():
                    report["stopped_by_user"] = True
                    break
    except KeyboardInterrupt:
        report["stopped_by_user"] = True
    except BaseException as exc:
        exit_code = 1
        report["error"] = repr(exc)
        print(f"Recording stopped: {exc}", file=sys.stderr)
    finally:
        try:
            dataset.finalize()
        except Exception as exc:
            exit_code = 1
            report["finalize_error"] = repr(exc)
        report["exit_code"] = exit_code
        (output / "collection-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    total_frames = sum(episode["frames"] for episode in report["episodes_recorded"])
    print(f"Saved: {len(report['episodes_recorded'])} episode(s), {total_frames} frames. Local dataset: {output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

