from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

class TemperatureDataset(Dataset):
    def __init__(self, root, split, crop, model, image, mask, interpolation):
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.model = model
        self.image = image
        self.mask = mask
        self.interpolation = interpolation
        self.data = self.get_data()

    def __len__(self):

    def __getitem__(self, index):
        if self.image == "thermal":
            height, width = np.load(images[index]).shape
            size = (width, height)

            interpolation_method = {
                "nearest": cv2.INTER_NEAREST,
                "area": cv2.INTER_AREA,
            }[self.interpolation]

            thermal_label_mask = [
                cv2.resize(cv2.resize(x.numpy().astype(np.uint8)), size, interpolation=interpolation_method)
                for x in visual_label_mask
            ]
            thermal_hard_mask = [
                cv2.resize(cv2.resize(x.numpy().astype(np.uint8)), size, interpolation=interpolation_method)
                for x in visual_hard_mask
            ]
            thermal_soft_mask = [
                cv2.resize(cv2.resize(x.numpy().astype(np.uint8)), size, interpolation=interpolation_method)
                for x in visual_soft_mask
            ]

        # if mask == all: (N, W, H, 3)
        # if mask == hard: (N, W, H, 2)
        # if mask == soft: (N, W, H, 2)
        thermal_data = 

        # (N, W, H, 3)
        visual_data = visual_images
            

        return data, label

    def get_data(self):
        crop_dir = self.root / self.crop

        if self.image == "all":
            raise ValueError("Invalid image type")

        original_dir = crop_dir / "original" / self.split / self.image
        
        label_dir = crop_dir / "original" / self.split / "label"
        model_dir = crop_dir / self.model / self.split / "logit"

        
        if self.image == "thermal":
            # (N, H, W)
            thermal_images = np.stack([
                np.load(p)
                for p in sorted(original_dir.iterdir())
                if p.is_file()
            ])
        if self.image == "visual"
            # (N, H, W, 3)
            visual_images = np.stack([
                cv2.imread(str(p))
                for p in sorted(original_dir.iterdir())
                if p.is_file()
            ])


        # (N, H, W)
        # (0: nonsunlit, 1: sunlit)
        visual_label_mask = np.stack([
            np.all(cv2.imread(str(p)) == (0, 255, 0), axis=-1).astype(np.uint8)
            for p in sorted(label_dir.glob("*.png"))
        ])

        # (N, 2, H, W)
        logits = np.stack([
            torch.from_numpy(np.load(path)).float()
            for path in sorted(model_dir.glob("*.npy"))
        ])

        # (N, H, W)
        # (0: nonsunlit, 1: sunlit)
        visual_hard_mask = np.stack([x.argmax(dim=0) for x in logits])
        # (0-1: sunlit probabilities)
        visual_soft_mask = np.stack([x.softmax(dim=0)[1] for x in logits])

            
