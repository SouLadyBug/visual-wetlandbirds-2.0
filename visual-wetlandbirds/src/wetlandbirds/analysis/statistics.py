from __future__ import annotations

import pandas as pd

from ..data.loaders import DatasetBundle, schema_columns
from ..data.splits import make_split_map


def class_balance(bundle: DatasetBundle) -> dict[str, pd.DataFrame]:
    cols = schema_columns(bundle)
    out = {}
    if cols["bbox"]["species"]:
        out["species"] = bundle.bboxes[cols["bbox"]["species"]].value_counts().rename_axis("species").reset_index(name="count")
    action = cols["crop"]["action_id"]
    if action:
        out["behavior"] = bundle.crops[action].value_counts().rename_axis("action_id").reset_index(name="count")
    return out


def clip_lengths(bundle: DatasetBundle) -> pd.DataFrame:
    c = schema_columns(bundle)["crop"]
    if not c["start_frame"] or not c["end_frame"]:
        return pd.DataFrame(columns=["clip_length"])
    result = bundle.crops[[c["start_frame"], c["end_frame"]]].copy()
    result["clip_length"] = pd.to_numeric(result[c["end_frame"]], errors="coerce") - pd.to_numeric(result[c["start_frame"]], errors="coerce") + 1
    return result


def split_balance(bundle: DatasetBundle) -> pd.DataFrame:
    c = schema_columns(bundle)["crop"]
    if not c["video"]:
        return pd.DataFrame()
    mapping = make_split_map(bundle.splits)
    work = bundle.crops.copy()
    work["split"] = work[c["video"]].astype(str).map(mapping)
    return work["split"].value_counts(dropna=False).rename_axis("split").reset_index(name="count")
