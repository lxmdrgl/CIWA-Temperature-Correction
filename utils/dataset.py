from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class TemperatureDataset(Dataset):
    def __init__(self, root, split, crop, model, image, mask, interpolation, target,
                 reg_output="canopy", bound=5.0):
        if image not in {"thermal", "visual", "both"}:
            raise ValueError("image must be thermal, visual, or both")
        if mask not in {"none", "hard", "soft", "both"}:
            raise ValueError("mask must be none, hard, soft, or both")
        if reg_output not in {"canopy", "cwsi"}:
            raise ValueError("reg_output must be canopy or cwsi")
        if not np.isfinite(bound) or not 0 < bound <= 50:
            raise ValueError("bound must be a percentage greater than 0 and at most 50")
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.model = model
        self.image = image
        self.mask = mask
        self.interpolation = interpolation
        self.target = target
        self.reg_output = reg_output
        self.bound = bound

        (
            self.thermal_images,
            self.visual_images,
            self.visual_label_masks,
            self.visual_hard_masks,
            self.visual_soft_masks,
        ) = self.get_data()

    def __len__(self):
        return len(self.thermal_images)

    def __getitem__(self, index):
        data, target, _ = self.get_sample(index)
        return data, target

    def _temperature_targets(self, thermal_image, mask):
        """Return canopy mean or [canopy mean, upper-tail mean, lower-tail mean]."""
        canopy = np.sum(thermal_image * mask) / mask.sum()
        if self.reg_output == "canopy":
            return canopy
        temperatures = np.sort(thermal_image[mask > 0])
        # Round up so even small masks contribute at least one pixel per tail.
        count = max(1, int(np.ceil(temperatures.size * self.bound / 100.0)))
        return np.array([canopy, temperatures[-count:].mean(),
                         temperatures[:count].mean()], dtype=np.float32)

    def get_sample(self, index):
        """Return inputs, target, and thermal label mask/baseline for visualization."""
        thermal_image = self.thermal_images[index]
        height, width = thermal_image.shape
        size = (width, height)

        interpolation_method = {
            "nearest": cv2.INTER_NEAREST,
            "area": cv2.INTER_AREA,
        }[self.interpolation]

        thermal_label_mask = cv2.resize(
            self.visual_label_masks[index].astype(np.float32),
            size,
            interpolation=interpolation_method,
        )
        thermal_hard_mask = cv2.resize(
            self.visual_hard_masks[index].astype(np.float32),
            size,
            interpolation=interpolation_method,
        )
        # Input order: temperature, optional hard mask, optional soft mask (H, W, C).
        channels = [thermal_image]
        if self.mask in {"hard", "both"}:
            channels.append(thermal_hard_mask)
        if self.mask in {"soft", "both"}:
            channels.append(cv2.resize(
                self.visual_soft_masks[index].astype(np.float32), size,
                interpolation=interpolation_method,
            ))
        thermal_data = np.stack(channels, axis=-1)

        weight_sum_label = thermal_label_mask.sum()
        weight_sum_hard = thermal_hard_mask.sum()

        if weight_sum_label == 0 or weight_sum_hard == 0:
            raise ValueError(f"Empty target mask at index {index}")

        thermal_label_target = self._temperature_targets(thermal_image, thermal_label_mask)
        thermal_hard_target = self._temperature_targets(thermal_image, thermal_hard_mask)

        if self.target == "direct":
            thermal_target = thermal_label_target
        elif self.target == "correction":
            thermal_target = thermal_label_target - thermal_hard_target

        if self.image == "thermal":
            data = thermal_data
        elif self.image == "visual":
            # (H, W, 3), RGB at the original visual resolution.
            data = cv2.cvtColor(self.visual_images[index], cv2.COLOR_BGR2RGB)
        elif self.image == "both":
            # OpenCV loads BGR; vision models expect RGB before thermal channels.
            rgb = cv2.cvtColor(self.visual_images[index], cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            data = np.concatenate((rgb, thermal_data), axis=-1)
        return data, thermal_target, {
            "thermal_label_mask": thermal_label_mask,
            "baseline": thermal_hard_target,
        }
        

    def get_data(self):
        crop_dir = self.root / self.crop

        thermal_dir = crop_dir / "original" / self.split / "thermal"
        visual_dir = crop_dir / "original" / self.split / "visual"
        
        label_dir = crop_dir / "original" / self.split / "label"
        model_dir = crop_dir / self.model / self.split / "logit"

        # (N, H, W)
        thermal_images = np.stack([
            np.load(p)
            for p in sorted(thermal_dir.iterdir())
            if p.is_file()
        ])

        # (N, H, W, 3)
        visual_images = [
            cv2.imread(str(p))
            for p in sorted(visual_dir.iterdir())
            if p.is_file()
        ]

        # (N, H, W)
        # (0: nonsunlit, 1: sunlit)
        visual_label_masks = [
            np.all(cv2.imread(str(p)) == (0, 255, 0), axis=-1).astype(np.uint8)
            for p in sorted(label_dir.glob("*.png"))
        ]

        # (N, 2, H, W)
        logits = [
            torch.from_numpy(np.load(path)).float()
            for path in sorted(model_dir.glob("*.npy"))
        ]

        # (N, H, W)
        # (0: nonsunlit, 1: sunlit)
        visual_hard_masks = [x.argmax(dim=0).numpy() for x in logits]
        # (0-1: sunlit probabilities)
        visual_soft_masks = [x.softmax(dim=0)[1].numpy() for x in logits]

        return thermal_images, visual_images, visual_label_masks, visual_hard_masks, visual_soft_masks
