from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

from ..data.bbox import parse_bbox_cell
from .splits import grouped_folds, grouped_holdout
from .training import train_eval


def build_video_index(video_dir: Path) -> dict[str, Path]:
    if not video_dir.exists(): return {}
    return {p.name: p for p in video_dir.rglob("*.mp4")}


def build_detection_samples(bundle, video_dir: Path, output_dir: Path, frames_per_video: int, max_frames: int, seed: int):
    cols = {"video": "video_name", "frame": "frame", "species": "species", "boxes": "bounding_boxes"}
    if not all(c in bundle.bboxes.columns for c in cols.values()):
        raise ValueError("Expected bounding_boxes.csv columns are missing")
    grouped = defaultdict(list)
    for _, row in bundle.bboxes.iterrows():
        grouped[(str(row[cols["video"]]), int(row[cols["frame"]]))].append(row)
    per_video = defaultdict(list)
    for key in grouped: per_video[key[0]].append(key)
    rng = random.Random(seed)
    keys = []
    for video, values in per_video.items():
        rng.shuffle(values); keys.extend(values[:frames_per_video])
    rng.shuffle(keys); keys = keys[:max_frames]
    index = build_video_index(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, labels = [], set()
    captures = {}
    for video, frame in keys:
        path = index.get(Path(video).name)
        if path is None: continue
        cap = captures.setdefault(str(path), cv2.VideoCapture(str(path)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame - 1, 0)); ok, image = cap.read()
        if not ok: continue
        h, w = image.shape[:2]; boxes, species = [], []
        for row in grouped[(video, frame)]:
            for parsed in parse_bbox_cell(row[cols["boxes"]]):
                x1, y1, x2, y2 = parsed["box"]
                x1, x2 = sorted((max(0, min(int(x1), w)), max(0, min(int(x2), w))))
                y1, y2 = sorted((max(0, min(int(y1), h)), max(0, min(int(y2), h))))
                if x2 - x1 < 5 or y2 - y1 < 5: continue
                boxes.append((x1, y1, x2, y2)); species.append(str(row[cols["species"]])); labels.add(str(row[cols["species"]]))
        if not boxes: continue
        name = re.sub(r"[^A-Za-z0-9_-]", "_", video) + f"_f{frame}.jpg"
        path_out = output_dir / name; cv2.imwrite(str(path_out), image)
        samples.append({"image_path": str(path_out), "video_name": video, "boxes": boxes, "labels": species})
    for cap in captures.values(): cap.release()
    return samples, {label: i + 1 for i, label in enumerate(sorted(labels))}


def run_detection(samples, label_to_idx, cfg: dict, output_dir: Path, seed: int):
    if len(samples) < 10: raise ValueError("Not enough detection frames")
    train_idx, val_idx = grouped_holdout(samples, float(cfg["test_fraction"]), seed)
    single = train_eval([samples[i] for i in train_idx], [samples[i] for i in val_idx], label_to_idx, cfg)
    pd.DataFrame([single]).to_csv(output_dir / "single_split_species_detection.csv", index=False)

    dominant = [s["labels"][0] for s in samples]
    folds = min(int(cfg["folds"]), pd.Series(dominant).value_counts().min())
    fold_rows = []
    for fold, (tr, va) in enumerate(grouped_folds(samples, dominant, folds, seed), start=1):
        tr_videos = {samples[i]["video_name"] for i in tr}; va_videos = {samples[i]["video_name"] for i in va}
        if tr_videos & va_videos: raise RuntimeError("Video leakage detected in detection fold")
        metrics = train_eval([samples[i] for i in tr], [samples[i] for i in va], label_to_idx, cfg)
        fold_rows.append({"fold": fold, **metrics})
    folds_df = pd.DataFrame(fold_rows); folds_df.to_csv(output_dir / "kfold_results_species_detection.csv", index=False)
    summary = {"k_folds": len(folds_df)}
    for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
        summary[f"mean_{metric}"] = folds_df[metric].mean(); summary[f"std_{metric}"] = folds_df[metric].std(ddof=1) if len(folds_df) > 1 else 0.0
    pd.DataFrame([summary]).to_csv(output_dir / "kfold_summary_species_detection.csv", index=False)
    return single, summary
