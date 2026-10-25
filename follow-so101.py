"""Continuous SO-101 following; Q or Stop stops and releases the follower.

Reference: https://huggingface.co/docs/lerobot/il_robots
Uses LeRobot 0.6.1's SO101Leader.get_action / SO101Follower.send_action.
Uses existing calibration and settings: no configure(), calibration or PID writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SOURCE = "https://huggingface.co/docs/lerobot/il_robots"


def read_raw(bus, register):
    values = bus.sync_read(register, normalize=False, num_retry=2)
    if set(values) != set(JOINTS):
        raise RuntimeError(f"Incomplete {register} response: {values}")
    return values


def preflight(arm, *, leader=False):
    if set(arm.calibration) != set(JOINTS) or not arm.is_calibrated:
        raise RuntimeError(f"{arm.bus.port}: existing calibration does not match motors; no recalibration performed")
    rows = {name: read_raw(arm.bus, name) for name in (
        "Operating_Mode", "Angular_Resolution", "Phase", "Torque_Enable",
        "Present_Position", "Goal_Position",
    )}
    for joint in JOINTS:
        calibration = arm.calibration[joint]
        if calibration.id != JOINTS.index(joint) + 1 or calibration.drive_mode != 0:
            raise RuntimeError(f"Invalid calibration: {joint}")
        if not 0 <= calibration.range_min < calibration.range_max <= 4095:
            raise RuntimeError(f"Invalid calibrated range: {joint}")
        if rows["Operating_Mode"][joint] != 0 or rows["Angular_Resolution"][joint] != 1:
            raise RuntimeError(f"{arm.bus.port}/{joint}: unexpected position mode or resolution")
        if rows["Phase"][joint] & 0x10:
            raise RuntimeError(f"{arm.bus.port}/{joint}: unsupported feedback mode; no settings changed")
        if rows["Torque_Enable"][joint] not in ((0,) if leader else (0, 1)):
            raise RuntimeError(f"{arm.bus.port}/{joint}: unexpected torque state")
        if not calibration.range_min <= rows["Present_Position"][joint] <= calibration.range_max:
            raise RuntimeError(f"{arm.bus.port}/{joint}: current position outside calibrated range")
    return rows


def _clear_failed_transaction(arm):
    """Release the SDK's local busy flag after the owner's request has unwound.

    scservo leaves is_using set when serial.write/flush raises. No hardware
    settings change here; the failed action is never retried. This owner then
    attempts its existing OFF/port-close cleanup.
    """
    port = getattr(arm.bus, "port_handler", None)
    if port is not None and getattr(port, "is_using", False):
        port.is_using = False


def run(leader, follower, *, fps=60, check_only=False, emit=lambda row: None,
        stop_event=None, clock=time.perf_counter, sleep=time.sleep, frame_emit=None):
    if not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("fps must be between 1 and 120")
    stop_event = stop_event if stop_event is not None else threading.Event()
    drive_started = False
    release_needed = False
    failure = None
    cleanup_errors = []
    off_verified = set()
    try:
        # Bus connect only performs PING/READ; arm.connect() would run configure().
        leader.bus.connect()
        follower.bus.connect()
        # Once the expected follower has been identified, normal-mode faults
        # must also release joints that were already holding before startup.
        release_needed = not check_only
        leader_rows = preflight(leader, leader=True)
        follower_rows = preflight(follower)
        emit({"phase": "preflight", "leader": leader_rows, "follower": follower_rows,
              "max_relative_target": None, "fps_target": fps})
        if check_only or stop_event.is_set():
            emit({"phase": "read_only_complete"})
            return
        # Retain an existing holding goal; initialize a released joint at Present.
        initial_goals = {j: follower_rows["Goal_Position"][j]
                         if follower_rows["Torque_Enable"][j] else follower_rows["Present_Position"][j]
                         for j in JOINTS}
        for j in JOINTS:
            c = follower.calibration[j]
            if not c.range_min <= initial_goals[j] <= c.range_max:
                raise RuntimeError(f"Initial goal outside calibrated range: {j}")
        drive_started = True
        follower.bus.sync_write("Goal_Position", initial_goals, normalize=False)
        if read_raw(follower.bus, "Goal_Position") != initial_goals:
            raise RuntimeError("Initial Goal_Position verification failed")
        for joint in JOINTS:
            if stop_event.is_set():
                return
            # Write torque only; enable_torque() would also modify Lock.
            follower.bus.write("Torque_Enable", joint, 1, num_retry=2)
        emit({"phase": "following", "stop": "Q key / Stop button", "relative_limit": None,
              "goal_step_limit": None, "initial_goals_raw": initial_goals})
        started = clock()
        interval_start = started
        interval_frames = 0
        while not stop_event.is_set():
            frame_start = clock()
            # The official tutorial's mirror loop: read Leader, send to Follower.
            raw_action = leader.get_action()
            if stop_event.is_set():
                break
            sent = follower.send_action(raw_action)
            interval_frames += 1
            now = clock()
            if frame_emit is not None:
                elapsed = now - interval_start
                frame_emit({"leader_action": raw_action, "sent_action": sent,
                            "actual_hz": round(interval_frames / elapsed, 2) if elapsed > 0 else None})
            if now - interval_start >= 1:
                emit({"phase": "following", "elapsed_s": round(now - started, 3),
                      "actual_hz": round(interval_frames / (now - interval_start), 2),
                      "leader_action": raw_action, "sent_action": sent})
                interval_start = now
                interval_frames = 0
            sleep(max(0., 1 / fps - (clock() - frame_start)))
    except KeyboardInterrupt:
        emit({"phase": "stop_requested", "reason": "Ctrl+C"})
    except BaseException as exc:
        failure = exc
        for arm in (follower, leader):
            _clear_failed_transaction(arm)
        emit({"phase": "fault", "error": repr(exc)})
        raise
    finally:
        if release_needed and follower.bus.is_connected:
            # Attempt every motor even if one write/read fails. Always close ports.
            for joint in JOINTS:
                try:
                    follower.bus.write("Torque_Enable", joint, 0, num_retry=2)
                    if follower.bus.read("Torque_Enable", joint, normalize=False, num_retry=2) != 0:
                        raise RuntimeError("OFF read-back mismatch")
                    off_verified.add(joint)
                except Exception as exc:
                    cleanup_errors.append(f"OFF {joint}: {exc!r}")
                    _clear_failed_transaction(follower)
        for arm in (follower, leader):
            try:
                if arm.bus.is_connected or getattr(arm.bus.port_handler, "ser", None) is not None:
                    arm.bus.port_handler.closePort()
            except Exception as exc:
                cleanup_errors.append(f"Close {arm.bus.port}: {exc!r}")
        emit({"phase": "stopped", "drive_started": drive_started,
              "release_needed": release_needed,
              "off_confirmed": len(off_verified) == len(JOINTS), "cleanup_errors": cleanup_errors})
        if cleanup_errors and failure is None:
            raise RuntimeError("Cleanup failed: " + "; ".join(cleanup_errors))


def resolve_ports(leader_port="auto", follower_port="auto", *, ports=None):
    """Resolve known USB adapters before importing robots or opening a serial port."""
    if ports is None:
        from serial.tools import list_ports
        ports = list_ports.comports()
    ports = list(ports)
    identities = {"Leader": os.environ.get("SO101_LEADER_SERIAL"),
                  "Follower": os.environ.get("SO101_FOLLOWER_SERIAL")}
    labels = {"Leader": "黒いLeader（つかむ爪がない側）", "Follower": "Follower（物をつかむ側）"}

    def known_role(port):
        if (getattr(port, "vid", None), getattr(port, "pid", None)) != (0x1A86, 0x55D3):
            return None
        return next((role for role, serial in identities.items()
                     if serial and getattr(port, "serial_number", None) == serial), None)

    details = []
    for port in ports:
        vid, pid = getattr(port, "vid", None), getattr(port, "pid", None)
        usb = f"{vid:04X}:{pid:04X}" if vid is not None and pid is not None else "不明"
        serial = getattr(port, "serial_number", None) or "不明"
        role = known_role(port)
        details.append(f"{port.device} [USB={usb}, SER={serial}" + (f", {role}]" if role else "]"))
    detected = "検出ポート: " + ("; ".join(details) if details else "なし")
    selected = {}
    for role, requested in (("Leader", leader_port), ("Follower", follower_port)):
        requested = requested.strip()
        if requested.lower() == "auto":
            if not identities[role]:
                raise ValueError(f"Set SO101_{role.upper()}_SERIAL or pass an explicit serial port")
            matches = [port for port in ports if known_role(port) == role]
            if not matches:
                raise ValueError(f"{labels[role]}のUSBが見つかりません（SER={identities[role]}）。USB接続を確認してください。{detected}")
        else:
            matches = [port for port in ports if port.device.upper() == requested.upper()]
            if not matches:
                raise ValueError(f"{labels[role]}に指定した{requested}が現在ありません。{detected}")
        if len(matches) != 1:
            raise ValueError(f"{labels[role]}の接続先を一つに特定できません。候補: "
                             + ", ".join(port.device for port in matches) + f"。{detected}")
        actual_role = known_role(matches[0])
        if actual_role is not None and actual_role != role:
            raise ValueError(f"{labels[role]}に指定した{matches[0].device}は{labels[actual_role]}のUSBです。{detected}")
        selected[role] = matches[0].device
    if selected["Leader"].upper() == selected["Follower"].upper():
        raise ValueError(f"LeaderとFollowerに同じ{selected['Leader']}は使えません。{detected}")
    return selected["Leader"], selected["Follower"]


def build_arms(leader_port, follower_port):
    leader_port, follower_port = resolve_ports(leader_port, follower_port)
    if version("lerobot") != "0.6.1":
        raise RuntimeError("This script requires LeRobot 0.6.1")
    if leader_port.upper() == follower_port.upper():
        raise ValueError("Leader and Follower ports must differ")
    for path in (ROOT / "calibration/leader/so101_leader.json", ROOT / "calibration/follower/so101_follower.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Existing calibration required: {path}")
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    leader = SO101Leader(SO101LeaderConfig(port=leader_port, id="so101_leader",
                         calibration_dir=ROOT / "calibration/leader", use_degrees=True))
    follower = SO101Follower(SO101FollowerConfig(port=follower_port, id="so101_follower",
                             calibration_dir=ROOT / "calibration/follower", use_degrees=True,
                             max_relative_target=None, disable_torque_on_disconnect=True))
    return leader, follower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-port", default="auto", help="auto selects the known black Leader USB serial")
    parser.add_argument("--follower-port", default="auto", help="auto selects the known Follower USB serial")
    parser.add_argument("--fps", type=float, default=60, help="Target loop frequency; official default is 60 Hz")
    parser.add_argument("--check-only", action="store_true", help="Read-only preflight, no motor writes")
    parser.add_argument("--camera-indices", nargs=2, type=int, default=[0, 1], help="Two camera device indices")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera frame rate")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output", help="Root for selected session saves")
    parser.add_argument("--preview-only", action="store_true", help="Camera and UI preview without connecting arms")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 120:
        parser.error("--fps must be 1 through 120")
    if args.camera_fps < 1:
        parser.error("--camera-fps must be positive")
    if set(args.camera_indices) != {0, 1}:
        parser.error("--camera-indices must select cameras 0 and 1 once each, in either order")
    if args.check_only and args.preview_only:
        parser.error("--check-only and --preview-only cannot be combined")
    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stop.set())
    if not args.check_only:
        from so101_follow_ui import run_ui
        args.stop_event = stop
        return run_ui(args, build_arms, run, ROOT)
    log_dir = ROOT / "tools/hardware-check"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}-continuous-follow.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        def emit(row):
            row = {"time": datetime.now().astimezone().isoformat(), **row}
            line = json.dumps(row, ensure_ascii=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
        emit({"phase": "starting", "source": SOURCE, "log": str(log_path), "check_only": args.check_only})
        run_started = False
        try:
            leader, follower = build_arms(args.leader_port, args.follower_port)
            emit({"phase": "ports_resolved", "leader_port": leader.bus.port,
                  "follower_port": follower.bus.port})
            run_started = True
            run(leader, follower, fps=args.fps, check_only=args.check_only, stop_event=stop, emit=emit)
        except Exception as exc:
            if not run_started:
                emit({"phase": "fault", "error": repr(exc)})
                emit({"phase": "stopped", "drive_started": False, "release_needed": False,
                      "off_confirmed": False, "cleanup_errors": []})
            message = str(exc)
            if run_started and isinstance(exc, (OSError, ConnectionError)):
                message = ("アームの通信を開始・継続できません。USB接続、モーター用電源、"
                           "他のソフトがポートを使用していないか確認してください。 " + message)
            print(f"エラー: {message}", file=sys.stderr, flush=True)
            print(f"ログ: {log_path}", file=sys.stderr, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

