from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    behaviors: pd.DataFrame
    species: pd.DataFrame
    bboxes: pd.DataFrame
    crops: pd.DataFrame
    splits: dict[str, list[str]]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    return pd.read_csv(path)


def load_dataset(root: Path) -> DatasetBundle:
    split_path = root / "splits.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    return DatasetBundle(
        behaviors=_read_csv(root / "behaviors_ID.csv"),
        species=_read_csv(root / "species_ID.csv"),
        bboxes=_read_csv(root / "bounding_boxes.csv"),
        crops=_read_csv(root / "crops.csv"),
        splits=splits,
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def schema_columns(bundle: DatasetBundle) -> dict[str, dict[str, str | None]]:
    return {
        "bbox": {
            "species": find_column(bundle.bboxes, ["species"]),
            "species_id": find_column(bundle.bboxes, ["species_id"]),
            "video": find_column(bundle.bboxes, ["video_name", "video_id", "video"]),
            "frame": find_column(bundle.bboxes, ["frame", "frame_id"]),
            "boxes": find_column(bundle.bboxes, ["bounding_boxes", "bbox", "boxes"]),
        },
        "crop": {
            "video": find_column(bundle.crops, ["video_name", "video_id", "video"]),
            "bird_id": find_column(bundle.crops, ["bird_id", "subject_id", "individual_id"]),
            "species_id": find_column(bundle.crops, ["species_id"]),
            "action_id": find_column(bundle.crops, ["action_id", "behavior_id", "behaviour_id"]),
            "start_frame": find_column(bundle.crops, ["start_frame", "start"]),
            "end_frame": find_column(bundle.crops, ["end_frame", "end"]),
        },
    }
