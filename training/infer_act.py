"""Reload a saved ACT checkpoint and processors for dataset-only CPU inference."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy


def infer(policy_path: Path, dataset_root: Path, *, repo_id="local/so101-demonstrations", samples=3, output=None, num_threads=None, seed=1000):
    if version("lerobot") != "0.6.1": raise RuntimeError("Requires LeRobot 0.6.1")
    if samples < 1: raise ValueError("Positive sample count required")
    if not policy_path.is_dir(): raise FileNotFoundError(policy_path)
    torch.manual_seed(seed)
    if num_threads is not None: torch.set_num_threads(num_threads)
    stored_config = json.loads((policy_path / "config.json").read_text(encoding="utf-8"))
    saved_cfg = ACTConfig.from_pretrained(policy_path, local_files_only=True)
    training_device = stored_config.get("device")
    deserialized_device = saved_cfg.device
    saved_cfg.device = "cpu"
    # Full model.safetensors already contains backbone weights: no separate download is needed.
    saved_cfg.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(policy_path, config=saved_cfg, local_files_only=True, strict=True)
    policy.eval(); policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": "cpu"}, "normalizer_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}, "unnormalizer_processor": {"device": "cpu"}})
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    rows = []
    started = time.perf_counter()
    for index in range(min(samples, len(dataset))):
        observation = {key: value for key, value in dataset[index].items() if key in policy.config.input_features}
        input_digest = hashlib.sha256()
        for key in sorted(observation):
            value = observation[key].detach().cpu().contiguous()
            input_digest.update(key.encode("utf-8"))
            input_digest.update(str(value.dtype).encode("ascii"))
            input_digest.update(str(tuple(value.shape)).encode("ascii"))
            input_digest.update(value.numpy().tobytes())
        processed = preprocessor(observation)
        with torch.inference_mode():
            prediction = policy.select_action(processed)
            action = postprocessor(prediction).cpu()
        if tuple(action.shape) != (1, 6) or not torch.isfinite(action).all():
            raise RuntimeError(f"Expected finite CPU action [1,6], got {action.shape}")
        rows.append({"index": index, "input_sha256": input_digest.hexdigest(), "action_shape": list(action.shape), "action": action.tolist(),
            "finite": True, "device": str(action.device)})
    result = {"phase": "ACT_CPU_INFER_OK", "lerobot_version": version("lerobot"),
        "torch_version": torch.__version__, "training_device": training_device, "inference_device": "cpu",
        "config_device_after_library_fallback": deserialized_device,
        "policy_path": str(policy_path.resolve()), "dataset_root": str(dataset_root.resolve()),
        "seed": seed, "input_indices": [row["index"] for row in rows],
        "model_sha256": hashlib.sha256((policy_path / "model.safetensors").read_bytes()).hexdigest(),
        "processor_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(policy_path.iterdir()) if path.name.startswith("policy_")},
        "action_selection": "Official ACT select_action queue: first observation produces a chunk, next calls consume it until n_action_steps.",
        "samples": rows, "elapsed_s": time.perf_counter() - started,
        "saved_processors_restored": True, "normalization_stats_override": False,
        "separate_backbone_download": False, "robot_connected": False, "uploads": False,
        "real_task_performance_evaluated": False}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/so101-demonstrations")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    infer(args.policy_path, args.dataset_root, repo_id=args.repo_id, samples=args.samples,
          output=args.output, num_threads=args.num_threads, seed=args.seed)


if __name__ == "__main__": main()


