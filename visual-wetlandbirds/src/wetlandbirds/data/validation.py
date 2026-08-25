from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bbox import parse_bbox_cell
from .loaders import DatasetBundle, schema_columns


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 3) if total else 0.0


def quality_report(bundle: DatasetBundle, bbox_sample_size: int = 15000, seed: int = 42) -> pd.DataFrame:
    cols = schema_columns(bundle)
    rows: list[dict] = []

    for name, frame in [("bounding_boxes.csv", bundle.bboxes), ("crops.csv", bundle.crops)]:
        missing = int(frame.isna().sum().sum())
        duplicates = int(frame.duplicated().sum())
        rows += [
            {"file": name, "check": "missing_values", "count": missing, "pct": _pct(missing, frame.size)},
            {"file": name, "check": "duplicate_rows", "count": duplicates, "pct": _pct(duplicates, len(frame))},
        ]

    bcols = cols["bbox"]
    if bcols["boxes"]:
        sample = bundle.bboxes.sample(min(len(bundle.bboxes), bbox_sample_size), random_state=seed)
        invalid = tiny = empty = 0
        for value in sample[bcols["boxes"]]:
            parsed = parse_bbox_cell(value)
            if not parsed:
                empty += 1
            for item in parsed:
                x1, y1, x2, y2 = item["box"]
                if x2 <= x1 or y2 <= y1:
                    invalid += 1
                if max(0.0, x2 - x1) * max(0.0, y2 - y1) < 25:
                    tiny += 1
        rows.extend([
            {"file": "bounding_boxes.csv", "check": "empty_bbox_rows_sample", "count": empty, "pct": _pct(empty, len(sample))},
            {"file": "bounding_boxes.csv", "check": "invalid_boxes_sample", "count": invalid, "pct": _pct(invalid, len(sample))},
            {"file": "bounding_boxes.csv", "check": "tiny_boxes_sample", "count": tiny, "pct": _pct(tiny, len(sample))},
        ])

    if bcols["video"] and cols["crop"]["video"]:
        bbox_videos = set(bundle.bboxes[bcols["video"]].dropna().astype(str))
        crop_videos = set(bundle.crops[cols["crop"]["video"]].dropna().astype(str))
        only_bbox = bbox_videos - crop_videos
        only_crop = crop_videos - bbox_videos
        rows.extend([
            {"file": "cross-file", "check": "videos_only_in_bounding_boxes", "count": len(only_bbox), "pct": None},
            {"file": "cross-file", "check": "videos_only_in_crops", "count": len(only_crop), "pct": None},
        ])

    return pd.DataFrame(rows)
