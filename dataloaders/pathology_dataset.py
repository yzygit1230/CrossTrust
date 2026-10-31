import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PathologyPair:
    image_path: Path
    mask_path: Path
    dataset_name: str
    organ: str
    group: str


def _extract_group(stem, pattern):
    stem = _strip_aug_suffix(stem)
    if not pattern:
        return stem
    match = re.match(pattern, stem)
    if match is None:
        raise ValueError(f"Filename '{stem}' does not match group_pattern '{pattern}'.")
    return match.group(1)


def _strip_aug_suffix(stem):
    return re.sub(r"_aug\d+$", "", stem)


def _extract_organ(stem, task_config):
    stem = _strip_aug_suffix(stem)
    if task_config.get("organ_from_suffix", False):
        return stem.rsplit("_", 1)[-1]
    return task_config.get("organ", task_config["name"])


def discover_pathology_pairs(task_config, split):
    root = Path(task_config["root"])
    image_dir = root / task_config.get(f"{split}_image_dir", f"{split}/img")
    mask_dir = root / task_config.get(f"{split}_mask_dir", f"{split}/gt")
    if not image_dir.exists() or not mask_dir.exists():
        return []

    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    pairs = []
    missing_masks = []
    max_foreground_ratio = float(task_config.get("max_foreground_ratio", 1.0))
    min_foreground_ratio = float(task_config.get("min_foreground_ratio", 0.0))

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        mask = np.asarray(Image.open(mask_path).convert("L"))
        foreground_ratio = float((mask != 0).mean())
        if not min_foreground_ratio <= foreground_ratio <= max_foreground_ratio:
            continue

        pairs.append(
            PathologyPair(
                image_path=image_path,
                mask_path=mask_path,
                dataset_name=task_config["name"],
                organ=_extract_organ(image_path.stem, task_config),
                group=_extract_group(
                    image_path.stem,
                    task_config.get("group_pattern"),
                ),
            )
        )

    if missing_masks and not task_config.get("allow_unpaired_images", False):
        examples = ", ".join(missing_masks[:5])
        raise FileNotFoundError(
            f"{task_config['name']}:{split} has {len(missing_masks)} images without "
            f"masks. Examples: {examples}"
        )
    return pairs


def group_train_validation_split(pairs, validation_fraction, seed):
    groups = sorted({pair.group for pair in pairs})
    if len(groups) < 2:
        raise ValueError("At least two source groups are required for a grouped split.")

    random.Random(seed).shuffle(groups)
    validation_group_count = max(1, int(round(len(groups) * validation_fraction)))
    validation_group_count = min(validation_group_count, len(groups) - 1)
    validation_groups = set(groups[:validation_group_count])
    train_pairs = [pair for pair in pairs if pair.group not in validation_groups]
    validation_pairs = [pair for pair in pairs if pair.group in validation_groups]
    return train_pairs, validation_pairs


def split_labeled_unlabeled(pairs, label_percent, seed):
    if not 0 < label_percent <= 100:
        raise ValueError("label_percent must be in (0, 100].")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    labeled_count = max(1, int(np.ceil(len(indices) * label_percent / 100.0)))
    labeled_indices = set(indices[:labeled_count])
    labeled = [pair for index, pair in enumerate(pairs) if index in labeled_indices]
    unlabeled = [pair for index, pair in enumerate(pairs) if index not in labeled_indices]
    return labeled, unlabeled


class PathologyNucleiDataset(Dataset):
    def __init__(
        self,
        pairs,
        training,
        augmentation=None,
        return_strong_view=False,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        self.pairs = list(pairs)
        self.training = training
        self.augmentation = augmentation or {}
        self.return_strong_view = return_strong_view
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.pairs)

    def _geometric_augment(self, image, mask):
        if random.random() < float(self.augmentation.get("horizontal_flip", 0.5)):
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < float(self.augmentation.get("vertical_flip", 0.5)):
            image, mask = TF.vflip(image), TF.vflip(mask)
        if random.random() < float(self.augmentation.get("rotate90", 0.75)):
            angle = random.choice((90, 180, 270))
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        degrees = float(self.augmentation.get("rotation_degrees", 10.0))
        scale_range = self.augmentation.get("scale", [0.9, 1.1])
        translate_fraction = float(self.augmentation.get("translate", 0.04))
        angle = random.uniform(-degrees, degrees)
        scale = random.uniform(float(scale_range[0]), float(scale_range[1]))
        shear_degrees = float(self.augmentation.get("shear_degrees", 0.0))
        shear = [
            random.uniform(-shear_degrees, shear_degrees),
            random.uniform(-shear_degrees, shear_degrees),
        ]
        translate = [
            int(round(random.uniform(-translate_fraction, translate_fraction) * image.width)),
            int(round(random.uniform(-translate_fraction, translate_fraction) * image.height)),
        ]
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=(255, 255, 255),
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        return image, mask

    def _stain_augment(self, image, strength):
        config = self.augmentation
        probability = float(config.get("stain_probability", 0.0)) * strength
        if random.random() >= probability:
            return image

        array = np.asarray(image).astype(np.float32) / 255.0
        optical_density = -np.log(np.clip(array, 1e-4, 1.0))
        alpha = float(config.get("stain_alpha", 0.0)) * strength
        beta = float(config.get("stain_beta", 0.0)) * strength
        cross = float(config.get("stain_cross", 0.0)) * strength

        scale = np.random.uniform(1.0 - alpha, 1.0 + alpha, size=(1, 1, 3))
        bias = np.random.uniform(-beta, beta, size=(1, 1, 3))
        if cross > 0:
            matrix = np.eye(3, dtype=np.float32)
            matrix += np.random.uniform(-cross, cross, size=(3, 3)).astype(np.float32)
            np.fill_diagonal(matrix, np.diag(matrix).clip(0.8, 1.2))
            optical_density = np.tensordot(optical_density, matrix, axes=([2], [0]))

        optical_density = np.clip(optical_density * scale + bias, 0.0, 6.0)
        augmented = np.exp(-optical_density).clip(0.0, 1.0)
        return Image.fromarray((augmented * 255.0).round().astype(np.uint8))

    def _coarse_dropout(self, tensor, strength):
        config = self.augmentation
        probability = float(config.get("cutout_probability", 0.0)) * strength
        if random.random() >= probability:
            return tensor

        _, height, width = tensor.shape
        count_config = config.get("cutout_count", [1, 1])
        if isinstance(count_config, (list, tuple)):
            count = random.randint(int(count_config[0]), int(count_config[-1]))
        else:
            count = int(count_config)
        size_config = config.get("cutout_size", [0.05, 0.15])
        min_size = float(size_config[0])
        max_size = float(size_config[-1])
        fill = float(config.get("cutout_fill", 1.0))

        for _ in range(max(0, count)):
            box_h = max(1, int(round(random.uniform(min_size, max_size) * height)))
            box_w = max(1, int(round(random.uniform(min_size, max_size) * width)))
            top = random.randint(0, max(0, height - box_h))
            left = random.randint(0, max(0, width - box_w))
            tensor[:, top : top + box_h, left : left + box_w] = fill
        return tensor

    def _color_augment(self, image, strength):
        config = self.augmentation
        settings = (
            ("brightness", TF.adjust_brightness),
            ("contrast", TF.adjust_contrast),
            ("saturation", TF.adjust_saturation),
        )
        operations = []
        for key, operation in settings:
            magnitude = float(config.get(key, 0.0)) * strength
            if magnitude > 0:
                factor = random.uniform(1.0 - magnitude, 1.0 + magnitude)
                operations.append(lambda value, op=operation, f=factor: op(value, f))

        hue = float(config.get("hue", 0.0)) * strength
        if hue > 0:
            factor = random.uniform(-hue, hue)
            operations.append(lambda value, f=factor: TF.adjust_hue(value, f))

        gamma = float(config.get("gamma", 0.0)) * strength
        if gamma > 0:
            factor = random.uniform(max(0.2, 1.0 - gamma), 1.0 + gamma)
            operations.append(lambda value, f=factor: TF.adjust_gamma(value, f))

        random.shuffle(operations)
        for operation in operations:
            image = operation(image)

        image = self._stain_augment(image, strength)

        blur_probability = float(config.get("blur_probability", 0.0)) * strength
        if random.random() < blur_probability:
            radius = random.uniform(
                0.1,
                max(0.1, float(config.get("blur_radius", 0.8)) * strength),
            )
            image = image.filter(ImageFilter.GaussianBlur(radius))
        return image

    def _to_tensor(self, image, noise_strength):
        tensor = TF.to_tensor(image)
        noise_std = float(self.augmentation.get("noise_std", 0.0)) * noise_strength
        if self.training:
            tensor = self._coarse_dropout(tensor, noise_strength)
        if self.training and noise_std > 0:
            tensor = (tensor + torch.randn_like(tensor) * noise_std).clamp(0.0, 1.0)
        return TF.normalize(tensor, self.mean, self.std)

    def __getitem__(self, index):
        pair = self.pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        mask = Image.open(pair.mask_path).convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch: {pair.image_path} {image.size} vs "
                f"{pair.mask_path} {mask.size}"
            )
        if self.training:
            image, mask = self._geometric_augment(image, mask)

        strong_image = None
        if self.training and self.return_strong_view:
            strong_image = self._color_augment(image.copy(), strength=1.0)
            image = self._color_augment(image, strength=0.25)
        elif self.training:
            image = self._color_augment(image, strength=0.7)

        label = torch.from_numpy(
            (np.asarray(mask, dtype=np.uint8) != 0).astype(np.int64)
        )
        sample = {
            "image": self._to_tensor(
                image,
                noise_strength=0.25 if self.return_strong_view else 0.7,
            ),
            "label": label,
            "dataset": pair.dataset_name,
            "organ": pair.organ,
            "group": pair.group,
            "name": pair.image_path.name,
        }
        if strong_image is not None:
            sample["strong_image"] = self._to_tensor(strong_image, noise_strength=1.0)
        return sample
