from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import CropDataset
from .evaluation import classification_metrics
from .models import build_model
from ..statistics.significance import holm_bonferroni, mcnemar


def run_kfold(samples: list[dict], task: str, model_names: list[str], cfg: dict, output_dir: Path, seed: int):
    if len(samples) < 2:
        raise ValueError(f"Not enough samples for {task} classification")
    labels = sorted({s["label"] for s in samples})
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y = np.asarray([label_to_idx[s["label"]] for s in samples])
    paths = np.asarray([s["image_path"] for s in samples])
    groups = np.asarray([s.get("video_name", s["image_path"]) for s in samples])
    k = min(int(cfg["folds"]), int(np.bincount(y).min()))
    if k < 2:
        raise ValueError(f"Task {task} has a class with fewer than two samples")

    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["learning_rate"])
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = {name: np.full(len(samples), -1, dtype=int) for name in model_names}
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(paths, y, groups=groups), start=1):
        train_loader = DataLoader(CropDataset(paths[train_idx].tolist(), y[train_idx].tolist(), train_tf), batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(CropDataset(paths[val_idx].tolist(), y[val_idx].tolist(), val_tf), batch_size=batch_size, shuffle=False, num_workers=0)
        for model_name in model_names:
            torch.manual_seed(seed + fold)
            model = build_model(model_name, len(labels), pretrained=bool(cfg["pretrained"])).to(device)
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
            criterion = nn.CrossEntropyLoss()
            started = time.perf_counter()
            model.train()
            for _ in range(epochs):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
            elapsed = time.perf_counter() - started
            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            preds = np.asarray(preds)
            oof[model_name][val_idx] = preds
            metrics = classification_metrics(y[val_idx], preds)
            fold_rows.append({"task": task, "fold": fold, "model": model_name, "train_samples": len(train_idx), "val_samples": len(val_idx), "train_time_sec": round(elapsed, 2), **metrics})

    if any(set(groups[train_idx]) & set(groups[val_idx]) for train_idx, val_idx in splitter.split(paths, y, groups=groups)):
        raise RuntimeError("Video leakage detected in classification folds")

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / f"kfold_results_{task}.csv", index=False)

    summary_rows = []
    for model_name in model_names:
        sub = fold_df[fold_df.model == model_name]
        row = {"task": task, "model": model_name, "k_folds": len(sub)}
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
            row[f"mean_{metric}"] = sub[metric].mean()
            row[f"std_{metric}"] = sub[metric].std(ddof=1) if len(sub) > 1 else 0.0
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / f"kfold_summary_{task}.csv", index=False)

    oof_df = pd.DataFrame({"sample_index": np.arange(len(samples)), "true_label": y})
    for name, pred in oof.items():
        oof_df[f"pred_{name}"] = pred
    oof_df.to_csv(output_dir / f"oof_predictions_{task}.csv", index=False)

    pairs = list(itertools.combinations(model_names, 2))
    pair_rows = []
    pvals = []
    for a, b in pairs:
        valid = (oof[a] >= 0) & (oof[b] >= 0)
        if not valid.any():
            continue
        result = mcnemar(y[valid], oof[a][valid], oof[b][valid])
        row = {"task": task, "model_a": a, "model_b": b, "n_compared": int(valid.sum()), **result}
        pair_rows.append(row)
        pvals.append(result["p_value"])
    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        pair_df["p_value_holm"] = holm_bonferroni(pvals)
        pair_df["significant_at_alpha"] = pair_df["p_value_holm"] < float(cfg["alpha"])
        pair_df.to_csv(output_dir / f"mcnemar_{task}.csv", index=False)
    else:
        pair_df = pd.DataFrame()
    return {"folds": fold_df, "summary": summary_df, "oof": oof_df, "pairwise": pair_df}
