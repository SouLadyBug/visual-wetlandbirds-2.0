from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

from .bbox import parse_bbox_cell
from .loaders import DatasetBundle, schema_columns
from .sampling import stratified_sample
from .video import build_video_index, resolve_video_path


def _largest_box(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda item: max(0.0, item["box"][2] - item["box"][0]) * max(0.0, item["box"][3] - item["box"][1]))


def _save_crop(cap, frame_number: int, box, destination: Path) -> bool:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box["box"]
    x1, x2 = sorted((max(0, min(int(x1), width)), max(0, min(int(x2), width))))
    y1, y2 = sorted((max(0, min(int(y1), height)), max(0, min(int(y2), height))))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(destination), frame[y1:y2, x1:x2]))


def build_species_samples(bundle: DatasetBundle, video_dirs: list[Path], output_dir: Path, samples_per_class: int, max_total: int, seed: int) -> list[dict]:
    cols = schema_columns(bundle)["bbox"]
    required = [cols[k] for k in ("species", "video", "frame", "boxes")]
    if any(x is None for x in required):
        raise ValueError("bounding_boxes.csv is missing required species crop columns")
    work = bundle.bboxes[required].dropna().copy()
    work.columns = ["species", "video_name", "frame", "boxes"]
    sampled = stratified_sample(work, "species", samples_per_class * 3, max_total * 3, seed)
    index = build_video_index(video_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int); samples = []; captures = {}
    for row_number, row in sampled.iterrows():
        label = str(row["species"])
        if counts[label] >= samples_per_class or len(samples) >= max_total:
            continue
        box = _largest_box(parse_bbox_cell(row["boxes"]))
        path = resolve_video_path(str(row["video_name"]), index)
        if box is None or path is None:
            continue
        capture = captures.setdefault(str(path), cv2.VideoCapture(str(path)))
        safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label)
        destination = output_dir / f"{safe_label}_{len(samples)}.jpg"
        if _save_crop(capture, int(row["frame"]), box, destination):
            samples.append({"image_path": str(destination), "label": label, "video_name": str(row["video_name"]), "source_frame": int(row["frame"])})
            counts[label] += 1
    for capture in captures.values(): capture.release()
    return samples


def build_behavior_samples(bundle: DatasetBundle, video_dirs: list[Path], output_dir: Path, samples_per_class: int, max_total: int, seed: int) -> list[dict]:
    cols = schema_columns(bundle)
    c = cols["crop"]; b = cols["bbox"]
    required = [c[k] for k in ("video", "action_id", "start_frame", "end_frame")]
    if any(x is None for x in required) or any(b[k] is None for k in ("video", "frame", "boxes")):
        raise ValueError("Dataset is missing required behavior-crop columns")
    behavior_map = {}
    if "behavior_id" in bundle.behaviors.columns and "behavior" in bundle.behaviors.columns:
        behavior_map = dict(zip(bundle.behaviors["behavior_id"], bundle.behaviors["behavior"]))
    elif len(bundle.behaviors.columns) >= 2:
        behavior_map = dict(zip(bundle.behaviors.iloc[:, 0], bundle.behaviors.iloc[:, 1]))

    clips = bundle.crops[required].dropna().copy()
    clips.columns = ["video_name", "action_id", "start_frame", "end_frame"]
    clips["behavior"] = clips["action_id"].map(behavior_map).fillna(clips["action_id"].astype(str))
    sampled = stratified_sample(clips, "behavior", samples_per_class * 3, max_total * 3, seed)

    bbox_index: dict[str, dict[int, object]] = defaultdict(dict)
    for _, row in bundle.bboxes[[b["video"], b["frame"], b["boxes"]]].dropna().iterrows():
        bbox_index[str(row.iloc[0])][int(row.iloc[1])] = row.iloc[2]

    index = build_video_index(video_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int); samples = []; captures = {}
    for _, row in sampled.iterrows():
        label = str(row["behavior"])
        if counts[label] >= samples_per_class or len(samples) >= max_total:
            continue
        video = str(row["video_name"]); midpoint = (int(row["start_frame"]) + int(row["end_frame"])) // 2
        frames = bbox_index.get(video)
        if not frames:
            continue
        frame_number = min(frames, key=lambda f: abs(f - midpoint))
        if abs(frame_number - midpoint) > 60:
            continue
        box = _largest_box(parse_bbox_cell(frames[frame_number]))
        path = resolve_video_path(video, index)
        if box is None or path is None:
            continue
        capture = captures.setdefault(str(path), cv2.VideoCapture(str(path)))
        safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label)
        destination = output_dir / f"{safe_label}_{len(samples)}.jpg"
        if _save_crop(capture, frame_number, box, destination):
            samples.append({"image_path": str(destination), "label": label, "video_name": video, "source_frame": frame_number})
            counts[label] += 1
    for capture in captures.values(): capture.release()
    return samples
