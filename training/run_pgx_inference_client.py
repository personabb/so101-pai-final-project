"""Plan or run official PGX ACT inference on the local SO-101 follower."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from so101_remote_adapter import (
    SO101IPhoneCameraConfig, SO101RemoteFollowerConfig, run_client,
)
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.configuration_opencv import Cv2Backends


def inference_run_directory(directory):
    directory = Path(directory).resolve()
    return directory.parent.parent if directory.parent.name == "checkpoints" else directory


@contextmanager
def exclusive_session(directory, purpose):
    directory = Path(directory).resolve()
    run_directory = inference_run_directory(directory)
    path = run_directory / "exclusive-inference.lock"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "purpose": purpose,
                                 "checkpoint_dir": str(directory)}))
    try:
        # Older clients hold checkpoint-local leases. Never delete one or reset
        # the shared server while any such lease remains, including another step.
        legacy = list((run_directory / "checkpoints").glob("*/exclusive-inference.lock"))
        if directory != run_directory and (directory / "exclusive-inference.lock").exists():
            legacy.append(directory / "exclusive-inference.lock")
        if legacy:
            raise FileExistsError("Legacy checkpoint inference lease exists: " + str(legacy[0]))
        yield
    finally:
        path.unlink()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_checkpoint_ready(directory, remote_model, *, expected_step=20000):
    state = json.loads((directory / "monitor-state.json").read_text(encoding="utf-8"))
    role = json.loads((directory / "checkpoint-role.json").read_text(encoding="utf-8"))
    transfer = json.loads((directory / "transfer-verify.json").read_text(encoding="utf-8"))
    inference = json.loads((directory / "inference-cpu-windows.json").read_text(encoding="utf-8"))
    if state.get("phase") != "cpu_verified" or state.get("step") != expected_step:
        raise RuntimeError(f"{expected_step:,}-step checkpoint is not CPU verified")
    expected_role = "final_checkpoint" if expected_step == 100000 else "intermediate_checkpoint"
    if role.get("step") != expected_step or role.get("role") != expected_role:
        raise RuntimeError("Wrong checkpoint role or step")
    if role.get("remote_source") != remote_model:
        raise RuntimeError("Requested PGX model differs from verified checkpoint")
    if not transfer.get("all_hashes_match") or transfer.get("step") != expected_step:
        raise RuntimeError("Checkpoint transfer is not SHA256 verified")
    if not inference.get("checkpoint_cpu_inference_verified") or not inference.get("finite"):
        raise RuntimeError("CPU smoke did not produce finite actions")
    model = directory / "pretrained_model"
    if Path(inference["policy_path"]).resolve() != model.resolve():
        raise RuntimeError("CPU smoke checkpoint path differs")
    if "step" in inference and inference["step"] != expected_step:
        raise RuntimeError("CPU smoke checkpoint step differs")
    for entry in transfer["files"]:
        relative = Path(entry["relative"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid checkpoint manifest path")
        target = model / relative
        if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
            raise RuntimeError("Local verified checkpoint changed: " + str(relative))
    if sha256(model / "model.safetensors") != inference["model_sha256"]:
        raise RuntimeError("CPU smoke model SHA256 differs")
    stored = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if stored.get("type") != "act" or stored.get("chunk_size") != 100:
        raise RuntimeError("Expected original ACT chunk_size=100")
    for key in ("observation.images.camera0", "observation.images.camera1"):
        if stored["input_features"][key]["shape"] != [3, 480, 640]:
            raise RuntimeError("Checkpoint camera shape differs from recorded profile")
    return {"ready": True, "step": expected_step, "model_sha256": inference["model_sha256"]}


def require_gpu_checkpoint_ready(gpu, readiness, remote_model):
    if (gpu.get("phase") != "GPU_RPC_SAVED_OBSERVATION_VERIFIED"
            or gpu.get("step") != readiness["step"]
            or gpu.get("model_sha256") != readiness["model_sha256"]
            or gpu.get("remote_model") != remote_model
            or not gpu.get("finite")):
        raise RuntimeError("Saved-observation PGX GPU RPC verification is required for this checkpoint")


def build_config(follower_port, remote_model, server_address, task, *, camera1_index=1, camera0_index=2):
    if camera1_index < 0:
        raise ValueError("Arm camera index must be non-negative")
    return RobotClientConfig(
        policy_type="act", pretrained_name_or_path=remote_model,
        robot=SO101RemoteFollowerConfig(
            port=follower_port, id="so101_follower",
            calibration_dir=ROOT / "calibration/follower",
            use_degrees=True, max_relative_target=None,
            disable_torque_on_disconnect=True,
            cameras={
                "camera0": SO101IPhoneCameraConfig(
                    index_or_path=camera0_index, fps=30, width=1280, height=720,
                    backend=Cv2Backends.MSMF, output_width=640, output_height=480),
                "camera1": OpenCVCameraConfig(
                    index_or_path=camera1_index, fps=30, width=640, height=480,
                    backend=Cv2Backends.DSHOW),
            }),
        actions_per_chunk=50, server_address=server_address,
        policy_device="cuda", client_device="cpu", task=task,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=200000)
    parser.add_argument("--pretrained-path", required=True,
                        help="Absolute checkpoint path on PGX, not the Windows model path")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--skip-checkpoint-verification", action="store_true",
                        help="Use saved PGX weights without requiring prior CPU/GPU test records.")
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--overhead-camera-index", type=int, default=2)
    parser.add_argument("--arm-camera-index", type=int, default=1,
                        help="DirectShow arm USB camera index; independent of Camo's MSMF index")
    parser.add_argument("--status-path", type=Path,
                        help="Optional atomic runtime JSON written passively outside control ticks")
    parser.add_argument("--recording-directory", type=Path,
                        help="Save both camera videos and AI input/output in a unique session subdirectory.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--run", action="store_true",
                        help="Connect and drive the Follower; requires separate explicit approval")
    args = parser.parse_args()
    config = build_config(args.follower_port, args.pretrained_path, args.server_address, args.task,
                          camera1_index=args.arm_camera_index, camera0_index=args.overhead_camera_index)
    if args.skip_checkpoint_verification:
        readiness = {"ready": True, "step": args.checkpoint_step,
                     "verification": "skipped_at_user_request"}
    else:
        try:
            readiness = require_checkpoint_ready(args.checkpoint_dir, args.pretrained_path,
                                                 expected_step=args.checkpoint_step)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            readiness = {"ready": False, "reason": str(exc)}
    print(json.dumps({
        "mode": "run" if args.run else "plan_only",
        "checkpoint": readiness,
        "server_address": config.server_address,
        "remote_model": config.pretrained_name_or_path,
        "policy_type": config.policy_type, "policy_device": config.policy_device,
        "client_device": config.client_device, "fps": config.fps,
        "actions_per_chunk": config.actions_per_chunk,
        "chunk_size_threshold": config.chunk_size_threshold,
        "aggregate_fn_name": config.aggregate_fn_name,
        "follower_port": args.follower_port, "calibration_dir": str(config.robot.calibration_dir),
        "use_degrees": config.robot.use_degrees, "max_relative_target": config.robot.max_relative_target,
        "camera0": f"Camo {args.overhead_camera_index}/MSMF RGB 1280x720 -> 640x360 + top/bottom 60 black rows",
        "camera1": f"USB {args.arm_camera_index}/DSHOW RGB 640x480",
        "dataset_recording": False, "hub_upload": False,
        "status_path": str(args.status_path) if args.status_path else None,
        "recording_directory": str(args.recording_directory) if args.recording_directory else None,
        "stop": "Q in the launcher window: stop and save; Ctrl+C: interrupt. Detected RPC/camera/action errors stop, discard queue and disconnect.",
        "hardware_requested": args.run, "hardware_connected": False,
    }), flush=True)
    if not args.run:
        return
    if not readiness["ready"]:
        raise RuntimeError("Checkpoint readiness gate refused hardware connection")
    if not args.skip_checkpoint_verification:
        gpu = json.loads((args.checkpoint_dir / "inference-gpu-pgx.json").read_text(encoding="utf-8"))
        require_gpu_checkpoint_ready(gpu, readiness, args.pretrained_path)
    if not (config.robot.calibration_dir / "so101_follower.json").is_file():
        raise FileNotFoundError("Existing follower calibration is required")
    # Probe channel readiness before constructing the official robot client.
    import grpc
    channel = grpc.insecure_channel(config.server_address)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    finally:
        channel.close()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_session(args.checkpoint_dir, "robot_client"):
        run_client(config, status_path=args.status_path, recording_directory=args.recording_directory)


if __name__ == "__main__":
    main()

