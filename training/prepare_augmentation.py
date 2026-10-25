"""Preview LeRobot image transforms and prepare separate, unexecuted train configs.

Run from the project's configured uv environment. Only --output is written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig


def profiles(height: int, width: int) -> dict[str, dict]:
    color = {"weight": 1.0, "type": "ColorJitter", "kwargs": {
        "brightness": [0.8, 1.2], "contrast": [0.8, 1.2],
        "saturation": [0.85, 1.15], "hue": [-0.02, 0.02],
    }}
    crop = {"weight": 1.0, "type": "RandomResizedCrop", "kwargs": {
        "size": [height, width], "scale": [0.9, 1.0],
        "ratio": [width / height, width / height], "antialias": True,
    }}
    return {
        "baseline": {"enable": False, "max_num_transforms": 1, "random_order": False, "tfs": {}},
        "color": {"enable": True, "max_num_transforms": 1, "random_order": False, "tfs": {"color": color}},
        "crop": {"enable": True, "max_num_transforms": 1, "random_order": False, "tfs": {"crop": crop}},
        "color_crop": {"enable": True, "max_num_transforms": 2, "random_order": False,
                       "tfs": {"crop": crop, "color": color}},
    }


def transform(config: dict) -> ImageTransforms:
    cfg = copy.deepcopy(config)
    cfg["tfs"] = {key: ImageTransformConfig(**value) for key, value in cfg["tfs"].items()}
    return ImageTransforms(ImageTransformsConfig(**cfg))


def pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dtype != torch.uint8:
        tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 150, 300])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--train-output-root", help="Remote root for new experiment outputs; no training is started")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    dataset_root, output = args.dataset.resolve(), args.output.resolve()
    if output == dataset_root or output.is_relative_to(dataset_root):
        raise ValueError("--output must be outside the original dataset")
    output.mkdir(parents=True, exist_ok=False)
    dataset = LeRobotDataset("local/augmentation-preview", root=dataset_root,
                             video_backend="pyav", return_uint8=True)
    report = {"seed": args.seed, "dataset": str(dataset_root), "dataset_length": len(dataset),
              "source_data_modified": False, "training_started": False,
              "camera_transform_sampling": "Independent per camera, matching DatasetReader",
              "profiles": {}, "samples": [], "checks": []}
    first = dataset[args.indices[0]]
    height, width = first[dataset.meta.camera_keys[0]].shape[-2:]
    configs = profiles(height, width)
    for name, cfg in configs.items():
        (output / f"image-transforms-{name}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    report["profiles"] = configs
    for index in args.indices:
        item = dataset[index]
        original = {key: value.clone() for key, value in item.items() if isinstance(value, torch.Tensor)}
        for camera_number, camera in enumerate(dataset.meta.camera_keys):
            source = item[camera]
            if tuple(source.shape[-2:]) != (height, width):
                raise ValueError("Camera sizes differ; configure a per-camera pipeline before using crops")
            sheet = Image.new("RGB", (320 * len(configs), 264), "white")
            draw = ImageDraw.Draw(sheet)
            sample = {"index": index, "camera": camera,
                      "source_tensor_sha256": hashlib.sha256(source.numpy().tobytes()).hexdigest(),
                      "source_shape": list(source.shape), "source_dtype": str(source.dtype), "variants": {}}
            for column, (name, cfg) in enumerate(configs.items()):
                # Keep profiles reproducible while sampling each camera separately.
                torch.manual_seed(args.seed + index * 100 + camera_number)
                result = transform(cfg)(source.clone())
                assert result.shape == source.shape and result.dtype == source.dtype
                assert bool(torch.isfinite(result).all())
                if name == "baseline":
                    assert torch.equal(result, source)
                else:
                    assert not torch.equal(result, source), f"{name} did not alter pixels"
                filename = f"frame-{index:06d}-{camera.rsplit('.', 1)[-1]}-{name}.png"
                rendered = pil(result)
                rendered.save(output / filename)
                sheet.paste(rendered.resize((320, 240)), (column * 320, 24))
                draw.text((column * 320 + 6, 5), name, fill="black")
                sample["variants"][name] = filename
            filename = f"comparison-{index:06d}-{camera.rsplit('.', 1)[-1]}.png"
            sheet.save(output / filename)
            sample["comparison"] = filename
            report["samples"].append(sample)
        for key, tensor in original.items():
            assert torch.equal(item[key], tensor), f"Original tensor changed: {key}"
    report["checks"] = ["all output shapes/dtypes preserved", "all outputs finite",
                        "baseline pixels identical", "enabled transforms change pixels",
                        "input image/state/action tensors unchanged"]
    if args.base_config:
        if not args.train_output_root:
            raise ValueError("--train-output-root is required with --base-config")
        base = json.loads(args.base_config.read_text(encoding="utf-8"))
        for name, image_config in configs.items():
            cfg = copy.deepcopy(base)
            cfg["dataset"]["image_transforms"] = image_config
            cfg.update(resume=False, checkpoint_path=None, steps=args.steps,
                       output_dir=f"{args.train_output_root.rstrip('/')}/{name}/output",
                       job_name=f"act_augmentation_{name}", save_freq=args.steps,
                       log_freq=min(20, args.steps), save_checkpoint_to_hub=False)
            cfg["policy"]["push_to_hub"] = False
            cfg["policy"]["pretrained_path"] = None
            cfg["wandb"]["enable"] = False
            (output / f"train-{name}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        report["train_configs"] = {"status": "prepared, not run", "steps": args.steps,
                                   "initialization": "fresh policy with base config's pretrained vision backbone",
                                   "base_config": str(args.base_config.resolve())}
    report["exit_code"] = 0
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(report["samples"]),
                      "checks": report["checks"], "exit_code": 0}))


if __name__ == "__main__":
    main()
