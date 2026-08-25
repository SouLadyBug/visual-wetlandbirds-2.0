from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import DetectionFrameDataset, collate_fn
from .metrics import detection_metrics
from .models import build_detector


def train_eval(train_samples, val_samples, label_to_idx, cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector(len(label_to_idx), int(cfg["image_size"]), pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=float(cfg["learning_rate"]), momentum=float(cfg["momentum"]), weight_decay=float(cfg["weight_decay"]))
    train_loader = DataLoader(DetectionFrameDataset(train_samples, label_to_idx, int(cfg["image_size"]), True), batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(DetectionFrameDataset(val_samples, label_to_idx, int(cfg["image_size"]), False), batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn)
    start = time.perf_counter()
    model.train()
    for _ in range(int(cfg["epochs"])):
        for images, targets in train_loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    elapsed = time.perf_counter() - start
    model.eval(); preds, gts = {}, {}; image_id = 0
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model([x.to(device) for x in images])
            for output, target in zip(outputs, targets):
                preds[image_id] = {k: v.cpu().numpy() for k, v in output.items()}
                gts[image_id] = {k: v.cpu().numpy() for k, v in target.items()}
                image_id += 1
    metrics = detection_metrics(preds, gts, sorted(label_to_idx.values()), float(cfg["score_threshold"]))
    metrics.update({"train_time_sec": round(elapsed, 2), "n_train": len(train_samples), "n_val": len(val_samples), "num_params": sum(p.numel() for p in model.parameters())})
    return metrics
