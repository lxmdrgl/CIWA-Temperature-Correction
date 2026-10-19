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
            

        return data, label

    def get_data(self):
        crop_dir = self.root / self.crop

        if self.image != "all":
            original_dir = crop_dir / "original" / self.split / self.image
        
        label_dir = crop_dir / "original" / self.split / "label"
        model_dir = crop_dir / self.model / self.split / "logit"

        # images can be thermal or visual
        images = sorted(p for p in original_dir.iterdir() if p.is_file())
        visual_label_mask = sorted(label_dir.glob("*.png"))

        # logits shape (2 channels, height, width)
        logits = sorted(model_dir.glob("*.npy"))
        logit_arrays = [
            torch.from_numpy(np.load(path)).float()
            for path in logits
        ]

        # mask shape (height, width).
        # (0: nonsunlit, 1: sunlit)
        visual_hard_mask = [x.argmax(dim=0) for x in logit_arrays]
        # (0-1: sunlit probabilities)
        visual_soft_mask = [x.softmax(dim=0)[1] for x in logit_arrays]

        if self.image == "thermal"
            if self.interpolation = "nearest"
            elif self.interpolation = "area"
            thermal_label_mask =
            thermal_hard_mask = 
            thermal_soft_mask =  

            
