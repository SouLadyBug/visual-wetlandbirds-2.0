from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class DetectionFrameDataset(Dataset):
    def __init__(self, samples, label_to_idx, image_size: int, augment: bool = False):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        original_w, original_h = image.size
        image = image.resize((self.image_size, self.image_size))
        boxes = np.asarray(sample["boxes"], dtype=np.float32).reshape(-1, 4)
        boxes[:, [0, 2]] *= self.image_size / original_w
        boxes[:, [1, 3]] *= self.image_size / original_h
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            old = boxes.copy(); boxes[:, 0] = self.image_size - old[:, 2]; boxes[:, 2] = self.image_size - old[:, 0]
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor([self.label_to_idx[x] for x in sample["labels"]], dtype=torch.int64), "image_id": torch.tensor([idx])}
        return pil_to_tensor(image).float() / 255.0, target


def collate_fn(batch):
    return tuple(zip(*batch))
