from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class TemperatureDataset(Dataset):
    def __init__(self, root, split, crop, model, image, mask, interpolation, target):
        if mask not in {"none", "hard", "soft", "both"}:
            raise ValueError("mask must be none, hard, soft, or both")
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.model = model
        self.image = image
        self.mask = mask
        self.interpolation = interpolation
        self.target = target

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

        thermal_label_target = (
            np.sum(thermal_image * thermal_label_mask)
            / weight_sum_label
        )
        thermal_hard_target = (
            np.sum(thermal_image * thermal_hard_mask)
            / weight_sum_hard
        )

        if self.target == "direct":
            thermal_target = thermal_label_target
        elif self.target == "correction":
            thermal_target = thermal_label_target - thermal_hard_target

        if self.image == "thermal":
            data = thermal_data
        elif self.image == "visual":
            # (W, H, 3)
            data = self.visual_images[index]
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

        
        #if self.image == "thermal":
        # (N, H, W)
        thermal_images = np.stack([
            np.load(p)
            for p in sorted(thermal_dir.iterdir())
            if p.is_file()
        ])

        #if self.image == "visual":
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
