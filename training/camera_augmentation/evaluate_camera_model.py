"""Deterministic offline ACT stress tests on recorded data; never accesses hardware."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as VF

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.transforms.transforms import GaussianNoise, RandomShadow


CAMERAS = ("observation.images.camera0", "observation.images.camera1")
CASE_NAMES = ("clean", "camera0_shift", "camera0_crop", "camera0_color", "camera0_blur",
              "camera0_shadow", "camera1_color", "camera1_blur", "camera1_noise", "camera1_shadow")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choose_indices(meta, samples: int, seed: int) -> list[int]:
    """Spread selection across episodes, with seeded frame selection within each episode."""
    if samples < 1:
        raise ValueError("samples must be positive")
    starts = [int(x) for x in meta.episodes["dataset_from_index"]]
    ends = [int(x) for x in meta.episodes["dataset_to_index"]]
    if not starts or any(end <= start for start, end in zip(starts, ends, strict=True)):
        raise ValueError("Dataset episode boundaries are missing or empty")
    rng = random.Random(seed)
    count = min(samples, sum(end - start for start, end in zip(starts, ends, strict=True)))
    indices: list[int] = []
    seen: set[int] = set()
    for i in range(count):
        episode = (round(i * (len(starts) - 1) / max(count - 1, 1))
                   if count <= len(starts) else i % len(starts))
        index = rng.randrange(starts[episode], ends[episode])
        if index in seen:
            # Resolve duplicate draws without losing episode coverage.
            index = next((j for j in range(starts[episode], ends[episode]) if j not in seen), -1)
        if index >= 0:
            seen.add(index)
            indices.append(index)
    return indices


def parse_indices(value: str | None, meta, samples: int, seed: int) -> list[int]:
    if value is None:
        return choose_indices(meta, samples, seed)
    candidate = Path(value)
    if candidate.is_file():
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
        indices = loaded["indices"] if isinstance(loaded, dict) else loaded
    elif value.lstrip().startswith("["):
        indices = json.loads(value)
    else:
        indices = value.split(",")
    result = [int(index) for index in indices]
    if not result or len(result) != len(set(result)):
        raise ValueError("indices must be a nonempty list without duplicates")
    return result


def stress_image(image: torch.Tensor, case: str, seed: int) -> torch.Tensor:
    """Force one mild perturbation; seeded once per frame/case, identically across models."""
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float().clone()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected one RGB image, received {tuple(image.shape)}")
    if not bool(torch.isfinite(image).all()) or float(image.min()) < 0 or float(image.max()) > 1:
        raise ValueError("Expected finite RGB image values in [0,1]")
    h, w = image.shape[-2:]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if case.endswith("_shift"):
            px, py = max(1, round(w * 0.03)), max(1, round(h * 0.03))
            dx = px if seed % 2 else -px
            dy = py if (seed // 2) % 2 else -py
            image = VF.crop(VF.pad(image, [px, py], padding_mode="edge"), py - dy, px - dx, h, w)
        elif case.endswith("_crop"):
            image = v2.RandomResizedCrop([h, w], scale=(0.9, 1.0), ratio=(w / h, w / h),
                                         antialias=True)(image)
        elif case.endswith("_color"):
            image = v2.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                                   saturation=(0.85, 1.15), hue=(-0.02, 0.02))(image)
        elif case.endswith("_blur"):
            image = VF.gaussian_blur(image, [3, 3], [0.8, 0.8] if case.startswith("camera0") else [0.5, 0.5])
        elif case.endswith("_shadow"):
            image = RandomShadow(opacity=(0.05, 0.2))(image)
        elif case.endswith("_noise"):
            image = GaussianNoise(std=(1.275, 3.825))(image)
        else:
            raise ValueError(f"Unknown stress case: {case}")
    return image.clamp(0, 1)


def case_seed(seed: int, index: int, case: str) -> int:
    digest = hashlib.sha256(f"{seed}:{index}:{case}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "little")


def case_spec() -> dict:
    return {
        "application_probability": 1.0,
        "camera0_shift": {"fraction_xy": [0.03, 0.03], "directions": "seeded signed maximum x/y", "padding": "edge"},
        "camera0_crop": {"area_scale": [0.9, 1.0], "ratio": "original width/height"},
        "color_both_cameras": {"brightness": [0.8, 1.2], "contrast": [0.8, 1.2],
                               "saturation": [0.85, 1.15], "hue": [-0.02, 0.02]},
        "blur": {"kernel_size": 3, "camera0_sigma": 0.8, "camera1_sigma": 0.5},
        "shadow_both_cameras": {"type": "official RandomShadow", "opacity": [0.05, 0.2],
                                "note": "random darkening OR brightening band"},
        "camera1_noise": {"type": "official GaussianNoise", "std_pixel255": [1.275, 3.825]},
        "camera1_geometry": "never applied",
        "sampling": "perturbation parameters depend only on seed, dataset index and case; not model",
    }


def make_batch(sample: dict, cfg, case: str, seed: int, index: int) -> tuple[dict, str]:
    batch = {key: sample[key].clone() for key in cfg.input_features}
    for camera in CAMERAS:
        if camera not in batch:
            raise ValueError(f"Expected configured camera: {camera}")
        if batch[camera].dtype == torch.uint8:
            batch[camera] = batch[camera].float() / 255.0
    if case != "clean":
        camera = CAMERAS[0] if case.startswith("camera0_") else CAMERAS[1]
        batch[camera] = stress_image(batch[camera], case, case_seed(seed, index, case))
    digest = hashlib.sha256()
    for camera in CAMERAS:
        digest.update(batch[camera].contiguous().numpy().tobytes())
    batch["action"] = sample["action"].clone()
    batch["action_is_pad"] = sample["action_is_pad"].clone()
    # Explicit batching is needed for action chunks: the default processor only
    # adds an action batch dimension to a one-dimensional action.
    batch = {key: value.unsqueeze(0) for key, value in batch.items()}
    return batch, digest.hexdigest()


def evaluate(args) -> dict:
    model = Path(args.model).resolve()
    root = Path(args.dataset).resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    cfg = ACTConfig.from_pretrained(model, local_files_only=True)
    cfg.device = args.device
    cfg.pretrained_backbone_weights = None  # All backbone weights come from the checkpoint.
    policy = ACTPolicy.from_pretrained(model, config=cfg, local_files_only=True, strict=True)
    policy.to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        cfg, pretrained_path=str(model),
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "normalizer_processor": {"device": args.device}},
    )
    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    dataset = LeRobotDataset(args.repo_id, root=root, image_transforms=None,
                             delta_timestamps={"action": [i / meta.fps for i in cfg.action_delta_indices]},
                             return_uint8=False, video_backend=args.video_backend)
    indices = parse_indices(args.indices, meta, args.samples, args.seed)
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("An evaluation index is outside the dataset")
    selected_cases = args.cases.split(",") if args.cases else list(CASE_NAMES)
    if "clean" not in selected_cases or any(case not in CASE_NAMES for case in selected_cases):
        raise ValueError(f"cases must include clean and belong to {CASE_NAMES}")
    selected_cases = ["clean"] + [case for case in selected_cases if case != "clean"]
    rows = []
    accumulators: dict[str, dict] = {}
    for index in indices:
        sample = dataset[index]
        clean_prediction = None
        episode = int(sample["episode_index"])
        for case in selected_cases:
            raw_batch, input_hash = make_batch(sample, cfg, case, args.seed, index)
            valid = ~raw_batch["action_is_pad"].bool().cpu()
            batch = preprocessor(raw_batch)
            policy.reset()
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(batch).cpu().float()
            target = batch["action"].detach().cpu().float()
            if prediction.shape != target.shape or prediction.shape[0] != 1:
                raise ValueError(f"Action shapes differ: prediction {prediction.shape}, target {target.shape}")
            if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
                raise ValueError("Non-finite model prediction or normalized target")
            if valid.shape != prediction.shape[:2] or not bool(valid.any()):
                raise ValueError("Invalid or empty action padding mask")
            errors = (prediction - target).abs()[valid]
            if case == "clean":
                clean_prediction = prediction.clone()
            deltas = (prediction - clean_prediction).abs()[valid]
            count = int(valid.sum())
            accumulator = accumulators.setdefault(case, {
                "error_sum": torch.zeros(prediction.shape[-1], dtype=torch.float64),
                "delta_sum": torch.zeros(prediction.shape[-1], dtype=torch.float64),
                "first_sum": torch.zeros(prediction.shape[-1], dtype=torch.float64),
                "valid_actions": 0, "samples": 0, "input_digest": hashlib.sha256(),
            })
            accumulator["error_sum"] += errors.double().sum(dim=0)
            accumulator["delta_sum"] += deltas.double().sum(dim=0)
            accumulator["first_sum"] += (prediction[0, 0] - target[0, 0]).abs().double()
            accumulator["valid_actions"] += count
            accumulator["samples"] += 1
            accumulator["input_digest"].update(input_hash.encode("ascii"))
            rows.append({"index": index, "episode_index": episode, "case": case,
                         "valid_action_steps": count, "input_sha256": input_hash,
                         "normalized_mae": float(errors.mean()),
                         "normalized_mae_per_joint": errors.mean(dim=0).tolist(),
                         "normalized_first_action_mae": float((prediction[0, 0] - target[0, 0]).abs().mean()),
                         "normalized_delta_from_clean": float(deltas.mean())})
        print(json.dumps({"completed_samples": len(rows) // len(selected_cases),
                          "total_samples": len(indices), "index": index}), flush=True)
    aggregates = {}
    for case, item in accumulators.items():
        means = item["error_sum"] / item["valid_actions"]
        deltas = item["delta_sum"] / item["valid_actions"]
        aggregates[case] = {"normalized_mae": float(means.mean()), "normalized_mae_per_joint": means.tolist(),
                            "normalized_first_action_mae": float(item["first_sum"].mean() / item["samples"]),
                            "normalized_delta_from_clean": float(deltas.mean()),
                            "normalized_delta_per_joint": deltas.tolist(),
                            "valid_action_steps": item["valid_actions"], "samples": item["samples"],
                            "input_sha256": item["input_digest"].hexdigest()}
    for case, item in aggregates.items():
        item["normalized_mae_increase_over_clean"] = item["normalized_mae"] - aggregates["clean"]["normalized_mae"]
    return {"completed_at": datetime.now().astimezone().isoformat(), "model": str(model),
            "model_sha256": file_hash(model / "model.safetensors"), "config_sha256": file_hash(model / "config.json"),
            "processor_files_sha256": {path.name: file_hash(path) for path in model.iterdir()
                                       if path.is_file() and ("processor" in path.name or path.name.startswith("step_"))},
            "dataset": str(root), "repo_id": args.repo_id, "seed": args.seed, "device": args.device,
            "indices": indices, "sample_selection": "episode-spread, seeded frame sampling; explicit indices override",
            "case_spec": case_spec(), "metric_units": "ACT normalized action units using this checkpoint's saved official processor",
            "joint_order": meta.features["action"].get("names"), "chunk_size": cfg.chunk_size,
            "masking": "action_is_pad excluded from full-chunk metrics; valid action count weighted aggregation",
            "scope": "in-domain recorded-data stress test; source 200k model was trained on all 66 episodes",
            "limitations": ["Not a genuinely held-out or unseen-environment evaluation.",
                            "No closed-loop robot execution; MAE and sensitivity do not measure task success.",
                            "One seeded perturbation per sampled frame/case; not comprehensive robustness coverage.",
                            "Extra fine-tuning and augmentation effects are confounded without a matched non-augmented continuation."],
            "robot_access": False, "aggregates": aggregates, "rows": rows, "exit_code": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local pretrained_model directory")
    parser.add_argument("--dataset", required=True, help="Local LeRobot dataset root")
    parser.add_argument("--repo-id", default="local/so101-merged-20261012")
    parser.add_argument("--output", required=True, help="New JSON report path; existing files are refused")
    parser.add_argument("--indices", help="Comma-separated indices, JSON list, or JSON file with indices")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1012026)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--cases", help="Comma-separated subset of stress cases, including clean")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation report: {output}")
    result = evaluate(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "samples": len(result["indices"]),
                      "cases": len(result["aggregates"]), "exit_code": 0}), flush=True)


if __name__ == "__main__":
    main()
