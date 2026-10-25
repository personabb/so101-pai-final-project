"""Camera-specific SO-101 training augmentation; no hardware or network access."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from lerobot.transforms.transforms import GaussianNoise, RandomShadow


PROFILES = ("combined", "color", "crop", "shift", "blur", "shadow", "noise", "none")


def augmentation_spec(profile: str = "combined", height: int = 480, width: int = 640) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    return {
        "profile": profile,
        "enabled_types": ["color", "crop", "shift", "blur", "shadow", "noise"] if profile == "combined" else ([] if profile == "none" else [profile]),
        "individual_profile_probability": "Same per-type probability as combined; disabled types are omitted. No forced p=1.",
        "kind": "custom experiment using official torchvision and LeRobot transforms",
        "official_default": "LeRobot v0.6.1 dataset.image_transforms.enable=false",
        "dataset_image_transforms_enable": False,
        "camera_mapping": {"observation.images.camera0": "overhead", "observation.images.camera1": "wrist"},
        "size": [height, width],
        "camera0_geometry": {
            "crop": {"p": 0.5, "scale": [0.9, 1.0], "ratio": [width / height, width / height]},
            "shift": {"p": 0.5, "padding": [round(width * 0.03), round(height * 0.03)],
                      "mode": "edge pad then random crop; no scale change"}},
        "camera1_geometry": "Identity: no crop, shift, rotation, or flip",
        "both_color": {"p": 0.8, "brightness": [0.8, 1.2], "contrast": [0.8, 1.2],
                       "saturation": [0.85, 1.15], "hue": [-0.02, 0.02]},
        "camera0_blur": {"p": 0.15, "kernel_size": 3, "sigma": [0.1, 0.8]},
        "camera1_blur": {"p": 0.1, "kernel_size": 3, "sigma": [0.1, 0.5]},
        "both_shadow": {"p": 0.2, "opacity": [0.05, 0.2],
                        "behavior": "Official RandomShadow randomly darkens OR brightens a smooth vertical band by up to 20%."},
        "camera1_noise": {"p": 0.2, "std_float01": [0.005, 0.015], "official_std_pixel255": [1.275, 3.825]},
        "order": ["geometry", "color", "blur", "shadow", "noise on wrist only"],
        "temporal_consistency": "One sample shares all parameters and noise realization across leading time dimensions; different samples and cameras draw independently.",
        "dtype": "Convert uint8 or float32 [0,1] to float32 [0,1], apply transforms, restore original dtype.",
        "actions_states": "Not transformed; original dataset is never written.",
        "limitations": ["2D augmentation does not reproduce camera rotation, perspective, or depth parallax.",
                        "Cropping and shifts can remove border content; success must be evaluated on held-out real conditions.",
                        "Probabilities and strengths are custom experiment settings, not official defaults."],
    }


class TemporalSharedGaussianNoise(torch.nn.Module):
    """Use official noise with identical random samples across a sample's time axis."""
    def __init__(self) -> None:
        super().__init__()
        self.noise = GaussianNoise(std=(1.275, 3.825))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        params = self.noise.make_params([image])
        if image.ndim == 3:
            return self.noise.transform(image, params)
        frames = image.reshape(-1, *image.shape[-3:])
        return torch.stack([self.noise.transform(frame, params) for frame in frames]).reshape(image.shape)


class CameraSpecificTransforms:
    def __init__(self, height: int = 480, width: int = 640, profile: str = "combined") -> None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown augmentation profile: {profile}")
        self.height, self.width = height, width
        self.profile = profile
        enabled = lambda kind: profile in ("combined", kind)
        geometry = []
        if enabled("crop"):
            geometry.append(v2.RandomApply([v2.RandomResizedCrop([height, width], scale=(0.9, 1.0),
                                            ratio=(width / height, width / height), antialias=True)], p=0.5))
        if enabled("shift"):
            geometry.append(v2.RandomApply([v2.Pad([round(width * 0.03), round(height * 0.03)], padding_mode="edge"),
                                           v2.RandomCrop([height, width])], p=0.5))
        self.geometry = {
            "observation.images.camera0": v2.Compose(geometry) if geometry else v2.Identity(),
            "observation.images.camera1": v2.Identity(),
        }
        color = lambda: v2.RandomApply([v2.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                                                      saturation=(0.85, 1.15), hue=(-0.02, 0.02))], p=0.8)
        self.appearance = {}
        for camera_number in (0, 1):
            operations = []
            if enabled("color"):
                operations.append(color())
            if enabled("blur"):
                operations.append(v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 0.8 if camera_number == 0 else 0.5))],
                                                p=0.15 if camera_number == 0 else 0.1))
            if enabled("shadow"):
                operations.append(v2.RandomApply([RandomShadow(opacity=(0.05, 0.2))], p=0.2))
            if enabled("noise") and camera_number == 1:
                operations.append(v2.RandomApply([TemporalSharedGaussianNoise()], p=0.2))
            self.appearance[f"observation.images.camera{camera_number}"] = v2.Compose(operations) if operations else v2.Identity()

    def __call__(self, camera: str, image: torch.Tensor) -> torch.Tensor:
        if camera not in self.geometry:
            raise KeyError(f"Unconfigured camera: {camera}")
        if image.dtype not in (torch.uint8, torch.float32):
            raise TypeError(f"Unsupported dtype: {image.dtype}")
        if image.ndim < 3 or image.shape[-3:] != (3, self.height, self.width):
            raise ValueError(f"Unexpected image shape: {tuple(image.shape)}")
        dtype = image.dtype
        working = image.to(torch.float32).clone()
        if dtype == torch.uint8:
            working = working / 255.0
        if not bool(torch.isfinite(working).all()) or float(working.min()) < 0 or float(working.max()) > 1:
            raise ValueError("Input must contain finite RGB values in [0,1] or uint8")
        result = self.appearance[camera](self.geometry[camera](working)).clamp(0, 1)
        if dtype == torch.uint8:
            return (result * 255).round().to(dtype)
        return result.to(dtype)


class CameraSpecificDataset(Dataset):
    """Spawn-pickleable dataset adapter; only configured image values are replaced."""
    def __init__(self, dataset: Dataset, height: int = 480, width: int = 640, profile: str = "combined") -> None:
        if getattr(dataset, "image_transforms", None) is not None:
            raise ValueError("Disable the underlying dataset's global image transforms before wrapping")
        self.dataset = dataset
        self.transforms = CameraSpecificTransforms(height, width, profile)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[index])
        for camera in self.transforms.geometry:
            if camera not in item:
                raise KeyError(f"Expected camera is missing: {camera}")
            item[camera] = self.transforms(camera, item[camera])
        return item

    def __getitems__(self, indices: list[int]) -> list[dict]:
        return [self[index] for index in indices]

    def __getattr__(self, name: str):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)
