"""
================================================================================
 VISUAL WETLANDBIRDS DATASET - STATISTICAL STUDY + PRETRAINED CNN COMPARISON
================================================================================
Matches the REAL schema of the released files:

  bounding_boxes.csv : species_id, species, video_name, frame, bounding_boxes
                        -> "bounding_boxes" is ONE cell containing the box
                           coordinates (as a string: a list of 4 numbers, or
                           a list of boxes if several individuals of the
                           species are visible in that frame). There is NO
                           per-row behavior label here.

  crops.csv          : video_name, bird_id, species_id, action_id,
                        start_frame, end_frame
                        -> this is where BEHAVIOR (action_id) lives: one row
                           per behavior clip (a bird doing one action for a
                           range of frames).

  species_ID.csv      : id <-> species name
  behaviors_ID.csv     : id <-> behavior/action name   (mapped onto action_id)
  splits.json          : {"train_set": [...video_names...], "val_set": [...], "test_set": [...]}
  videos.zip / videos/ : the raw .mp4 files, named exactly as in video_name

What this script does
----------------------
1. Loads everything, auto-detecting column names but defaulting to the
   confirmed real names above.
2. Data-quality report: missing values, duplicates, malformed / tiny / zero
   area bounding boxes, video_name mismatches between files.
3. Statistics focused on what's WRONG with the data: class balance for
   species (from bounding_boxes.csv) and behavior (from crops.csv), clip
   length distribution, species x behavior co-occurrence, train/val/test
   balance.
4. Builds a small stratified sample of labelled bird crops:
     - species task -> sampled from bounding_boxes.csv directly
     - behavior task -> sampled clips from crops.csv, middle frame located
       back in bounding_boxes.csv to get the box
   Frames are read straight from the actual video files.
5. Fine-tunes a handful of PRETRAINED torchvision CNNs (frozen backbone,
   few epochs -> fast) on that sample for both tasks and compares them.
6. Saves every figure / table / log into a new folder under C:\\stage.

Requirements (install once)
----------------------------
pip install pandas numpy matplotlib scikit-learn opencv-python pillow torch torchvision

This script is defensive: if one stage fails it logs a warning and moves on
instead of crashing, so you always get partial, usable results.
================================================================================
"""

import os
import re
import ast
import sys
import json
import time
import zipfile
import warnings
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = r"C:\stage"

FILES = {
    "behaviors":  os.path.join(BASE_DIR, "behaviors_ID.csv"),
    "species":    os.path.join(BASE_DIR, "species_ID.csv"),
    "bboxes":     os.path.join(BASE_DIR, "bounding_boxes.csv"),
    "crops":      os.path.join(BASE_DIR, "crops.csv"),
    "splits":     os.path.join(BASE_DIR, "splits.json"),
    "videos_zip": os.path.join(BASE_DIR, "videos.zip"),
}
# Candidate folders where the actual .mp4 files might already live
VIDEO_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "videos"),
    os.path.join(BASE_DIR, "Videos"),
    os.path.join(BASE_DIR, "videos_extracted"),
]
VIDEOS_EXTRACT = os.path.join(BASE_DIR, "videos_extracted")  # fallback extraction target

OUTPUT_DIR    = os.path.join(BASE_DIR, "analysis_output")
FIG_DIR       = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR     = os.path.join(OUTPUT_DIR, "model_results")
CROPS_OUT_DIR = os.path.join(OUTPUT_DIR, "sample_crops")

# ---- sampling / training config (kept small on purpose -> fast run) -------
SAMPLES_PER_CLASS   = 35
MAX_TOTAL_SAMPLES   = 700
IMG_SIZE            = 160
BATCH_SIZE          = 16
EPOCHS              = 5
VAL_FRACTION        = 0.25
RANDOM_SEED         = 42
MODELS_TO_TRY  = ["resnet18", "mobilenet_v2", "efficientnet_b0", "densenet121"]
TASKS_TO_RUN   = ["species", "behavior"]
BBOX_STATS_SAMPLE_SIZE = 15000   # rows sampled for bbox geometry stats (speed)

np.random.seed(RANDOM_SEED)

# ============================================================================
# LOGGING
# ============================================================================
_LOG_LINES = []

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    _LOG_LINES.append(line)

def save_log():
    try:
        with open(os.path.join(OUTPUT_DIR, "run_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(_LOG_LINES))
    except Exception as e:
        print(f"Could not save log: {e}")

def make_dirs():
    for d in [OUTPUT_DIR, FIG_DIR, MODEL_DIR, CROPS_OUT_DIR]:
        os.makedirs(d, exist_ok=True)
    log(f"Output folder ready at: {OUTPUT_DIR}")

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log(f"  saved figure -> {path}")
    except Exception as e:
        log(f"  WARNING: could not save figure {name}: {e}")
    finally:
        plt.close(fig)


# ============================================================================
# GENERIC HELPERS
# ============================================================================
def find_col(df, candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None


def safe_read_csv(path, **kwargs):
    if not os.path.exists(path):
        log(f"  WARNING: file not found: {path}")
        return None
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, **kwargs)
            if df.shape[1] > 1:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            continue
    log(f"  WARNING: could not parse {path} with common separators.")
    return None


def build_id_map(df, id_candidates, name_candidates):
    if df is None:
        return {}
    id_col = find_col(df, id_candidates)
    name_col = find_col(df, name_candidates)
    if id_col is None or name_col is None:
        if df.shape[1] >= 2:
            id_col, name_col = df.columns[0], df.columns[1]
        else:
            return {}
    return dict(zip(df[id_col], df[name_col]))


def parse_bbox_cell(cell):
    """
    Parse a 'bounding_boxes' cell into a list of dicts:
        [{"bird_id": <id or None>, "box": (x1, y1, x2, y2)}, ...]
    Handles several plausible serializations robustly:
        "[x1, y1, x2, y2]"
        "[[x1,y1,x2,y2], [x1,y1,x2,y2]]"
        "{'x1':..,'y1':..,'x2':..,'y2':..}"
        "[{'bird_id':1,'x1':..,...}, ...]"
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple)):
        parsed = cell
    else:
        s = str(cell).strip()
        if not s or s.lower() in ("nan", "none", "[]", "{}"):
            return []
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            nums = re.findall(r"-?\d+\.?\d*", s)
            nums = [float(n) for n in nums]
            boxes = []
            for i in range(0, len(nums) - 3, 4):
                boxes.append({"bird_id": None, "box": tuple(nums[i:i + 4])})
            return boxes

    boxes = []

    def handle_dict(d):
        keys_lower = {str(k).lower(): k for k in d.keys()}

        def getkey(cands):
            for c in cands:
                if c in keys_lower:
                    return d[keys_lower[c]]
            return None

        x1 = getkey(["x1", "xmin", "x_min", "left"])
        y1 = getkey(["y1", "ymin", "y_min", "top"])
        x2 = getkey(["x2", "xmax", "x_max", "right"])
        y2 = getkey(["y2", "ymax", "y_max", "bottom"])
        bird_id = getkey(["bird_id", "id", "subject_id", "subject"])
        if None not in (x1, y1, x2, y2):
            try:
                boxes.append({"bird_id": bird_id,
                               "box": (float(x1), float(y1), float(x2), float(y2))})
            except (TypeError, ValueError):
                pass

    def handle_list_of_nums(lst):
        nums = [v for v in lst if isinstance(v, (int, float))]
        if len(nums) >= 4:
            boxes.append({"bird_id": None, "box": tuple(float(v) for v in nums[:4])})

    if isinstance(parsed, dict):
        handle_dict(parsed)
    elif isinstance(parsed, (list, tuple)):
        if len(parsed) > 0 and isinstance(parsed[0], dict):
            for item in parsed:
                if isinstance(item, dict):
                    handle_dict(item)
        elif len(parsed) > 0 and isinstance(parsed[0], (list, tuple)):
            for item in parsed:
                handle_list_of_nums(list(item))
        else:
            handle_list_of_nums(list(parsed))
    return boxes


def stratified_sample(df, group_col, n_per_group, seed=RANDOM_SEED):
    """
    Sample up to n_per_group rows per value of group_col.
    Implemented as a plain loop + concat (NOT groupby().apply()) because
    recent pandas versions (2.2+/3.0) changed groupby().apply() to drop the
    grouping column from the group by default, which caused KeyError on the
    very column we grouped by.
    """
    parts = []
    for _, g in df.groupby(group_col, sort=False):
        n = min(len(g), n_per_group)
        if n > 0:
            parts.append(g.sample(n, random_state=seed))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=False)


def biggest_box(boxes):
    """From a list of parsed boxes, return the one with the largest area."""
    best, best_area = None, -1
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        area = abs((x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area = area
            best = b
    return best


# ============================================================================
# STEP 1 - LOAD ALL DATA
# ============================================================================
def load_data():
    log("=" * 70)
    log("STEP 1: Loading dataset files")
    log("=" * 70)

    behaviors_raw = safe_read_csv(FILES["behaviors"])
    species_raw   = safe_read_csv(FILES["species"])
    bbox_df       = safe_read_csv(FILES["bboxes"])
    crops_df      = safe_read_csv(FILES["crops"])

    splits = None
    if os.path.exists(FILES["splits"]):
        try:
            with open(FILES["splits"], "r", encoding="utf-8") as f:
                splits = json.load(f)
            log(f"  loaded splits.json (keys: {list(splits.keys()) if isinstance(splits, dict) else 'list'})")
        except Exception as e:
            log(f"  WARNING: could not read splits.json: {e}")
    else:
        log("  WARNING: splits.json not found")

    species_map = build_id_map(
        species_raw,
        id_candidates=["id", "species_id", "class_id", "specie_id"],
        name_candidates=["species", "name", "species_name", "specie", "class_name", "common_name"])
    behavior_map = build_id_map(
        behaviors_raw,
        id_candidates=["id", "behavior_id", "behaviour_id", "activity_id", "action_id", "class_id"],
        name_candidates=["behavior", "behaviour", "name", "activity", "action", "label", "class_name"])

    log(f"  species map entries: {len(species_map)}")
    log(f"  behavior map entries: {len(behavior_map)}")
    if bbox_df is not None:
        log(f"  bounding_boxes.csv: {bbox_df.shape[0]} rows, columns: {list(bbox_df.columns)}")
    if crops_df is not None:
        log(f"  crops.csv: {crops_df.shape[0]} rows, columns: {list(crops_df.columns)}")

    return {
        "bbox_df": bbox_df,
        "crops_df": crops_df,
        "species_map": species_map,
        "behavior_map": behavior_map,
        "splits": splits,
    }


def detect_bbox_columns(bbox_df):
    cols = {
        "species_id": find_col(bbox_df, ["species_id", "specie_id"]),
        "species":    find_col(bbox_df, ["species", "specie", "species_name"]),
        "video":      find_col(bbox_df, ["video_name", "video_id", "video"]),
        "frame":      find_col(bbox_df, ["frame", "frame_id", "frame_num", "frame_number"]),
        "boxes":      find_col(bbox_df, ["bounding_boxes", "bounding_box", "bbox", "boxes"]),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        log(f"  WARNING: could not auto-detect these bbox columns: {missing}. "
            f"Available columns: {list(bbox_df.columns)}")
    return cols


def detect_crops_columns(crops_df):
    cols = {
        "video":      find_col(crops_df, ["video_name", "video_id", "video"]),
        "bird_id":    find_col(crops_df, ["bird_id", "subject_id", "individual_id"]),
        "species_id": find_col(crops_df, ["species_id", "specie_id"]),
        "action_id":  find_col(crops_df, ["action_id", "behavior_id", "behaviour_id", "activity_id"]),
        "start":      find_col(crops_df, ["start_frame", "frame_start", "start"]),
        "end":        find_col(crops_df, ["end_frame", "frame_end", "end"]),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        log(f"  WARNING: could not auto-detect these crops.csv columns: {missing}. "
            f"Available columns: {list(crops_df.columns)}")
    return cols


# ============================================================================
# STEP 2 - DATA QUALITY / CLEANING REPORT
# ============================================================================
def data_quality_report(data):
    log("=" * 70)
    log("STEP 2: Data quality checks")
    log("=" * 70)
    report_rows = []
    bbox_df = data["bbox_df"]
    crops_df = data["crops_df"]

    if bbox_df is None:
        log("  Skipping: bounding_boxes.csv not available.")
        return None

    bcols = detect_bbox_columns(bbox_df)
    data["bbox_cols"] = bcols

    ccols = None
    if crops_df is not None:
        ccols = detect_crops_columns(crops_df)
        data["crops_cols"] = ccols

    # -- missing values -------------------------------------------------
    na_counts = bbox_df.isna().sum()
    for c in bbox_df.columns:
        pct = round(na_counts[c] / len(bbox_df) * 100, 2)
        report_rows.append({"file": "bounding_boxes.csv", "check": "missing_values",
                             "column": c, "count": int(na_counts[c]), "pct": pct})
    log(f"  bounding_boxes.csv missing values total: {int(na_counts.sum())}")

    # -- duplicates -------------------------------------------------------
    n_dupes = bbox_df.duplicated().sum()
    report_rows.append({"file": "bounding_boxes.csv", "check": "duplicate_rows", "column": "ALL",
                         "count": int(n_dupes), "pct": round(n_dupes / len(bbox_df) * 100, 2)})
    log(f"  bounding_boxes.csv duplicate rows: {n_dupes}")

    # -- malformed / invalid / tiny boxes (sampled for speed) -------------
    if bcols["boxes"]:
        try:
            sample_n = min(BBOX_STATS_SAMPLE_SIZE, len(bbox_df))
            sample_df = bbox_df.sample(sample_n, random_state=RANDOM_SEED)
            n_empty, n_multi, n_invalid, n_tiny, widths, heights = 0, 0, 0, 0, [], []
            for cell in sample_df[bcols["boxes"]]:
                boxes = parse_bbox_cell(cell)
                if not boxes:
                    n_empty += 1
                    continue
                if len(boxes) > 1:
                    n_multi += 1
                for b in boxes:
                    x1, y1, x2, y2 = b["box"]
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        n_invalid += 1
                        continue
                    widths.append(w); heights.append(h)
                    if w * h < 100:
                        n_tiny += 1
            pct = lambda n: round(n / sample_n * 100, 2)
            report_rows.append({"file": "bounding_boxes.csv", "check": "unparseable_or_empty_box_cell",
                                 "column": bcols["boxes"], "count": n_empty, "pct": pct(n_empty)})
            report_rows.append({"file": "bounding_boxes.csv", "check": "frames_with_multiple_birds",
                                 "column": bcols["boxes"], "count": n_multi, "pct": pct(n_multi)})
            report_rows.append({"file": "bounding_boxes.csv", "check": "invalid_boxes(w<=0 or h<=0)",
                                 "column": bcols["boxes"], "count": n_invalid, "pct": pct(n_invalid)})
            report_rows.append({"file": "bounding_boxes.csv", "check": "tiny_boxes(<10x10px)",
                                 "column": bcols["boxes"], "count": n_tiny, "pct": pct(n_tiny)})
            log(f"  (on a {sample_n}-row sample) empty/unparseable box cells: {n_empty} ({pct(n_empty)}%)")
            log(f"  (on a {sample_n}-row sample) frames with >1 bird box: {n_multi} ({pct(n_multi)}%)")
            log(f"  (on a {sample_n}-row sample) invalid boxes: {n_invalid} ({pct(n_invalid)}%), "
                f"tiny boxes (<10x10px): {n_tiny} ({pct(n_tiny)}%)")
            if widths:
                log(f"  box width  -> mean={np.mean(widths):.1f}px, median={np.median(widths):.1f}px")
                log(f"  box height -> mean={np.mean(heights):.1f}px, median={np.median(heights):.1f}px")
        except Exception as e:
            log(f"  WARNING: could not evaluate box geometry: {e}")

    # -- video_name consistency across files -------------------------------
    try:
        if bcols["video"]:
            bbox_videos = set(bbox_df[bcols["video"]].dropna().unique())
            if crops_df is not None and ccols and ccols["video"]:
                crops_videos = set(crops_df[ccols["video"]].dropna().unique())
                only_in_bbox = bbox_videos - crops_videos
                only_in_crops = crops_videos - bbox_videos
                report_rows.append({"file": "cross-check", "check": "videos_only_in_bounding_boxes",
                                     "column": "video_name", "count": len(only_in_bbox), "pct": None})
                report_rows.append({"file": "cross-check", "check": "videos_only_in_crops",
                                     "column": "video_name", "count": len(only_in_crops), "pct": None})
                log(f"  videos present in bounding_boxes.csv but not in crops.csv: {len(only_in_bbox)}")
                log(f"  videos present in crops.csv but not in bounding_boxes.csv: {len(only_in_crops)}")
            splits = data.get("splits")
            if splits and isinstance(splits, dict):
                split_videos = set()
                for v in splits.values():
                    if isinstance(v, list):
                        split_videos.update(v)
                unresolved = bbox_videos - split_videos
                report_rows.append({"file": "cross-check", "check": "videos_not_found_in_splits_json",
                                     "column": "video_name", "count": len(unresolved), "pct": None})
                log(f"  videos in bounding_boxes.csv not found in any split: {len(unresolved)}")
    except Exception as e:
        log(f"  WARNING: could not cross-check video names: {e}")

    # -- orphan species / behavior IDs --------------------------------------
    if bcols["species_id"] and data["species_map"]:
        orphan = set(bbox_df[bcols["species_id"]].dropna().unique()) - set(data["species_map"].keys())
        if orphan:
            log(f"  WARNING: species_id values not found in species_ID.csv: {orphan}")
    if crops_df is not None and ccols and ccols["action_id"] and data["behavior_map"]:
        orphan = set(crops_df[ccols["action_id"]].dropna().unique()) - set(data["behavior_map"].keys())
        if orphan:
            log(f"  WARNING: action_id values in crops.csv not found in behaviors_ID.csv: {orphan}")

    report_df = pd.DataFrame(report_rows)
    try:
        report_df.to_csv(os.path.join(OUTPUT_DIR, "data_quality_report.csv"), index=False)
        log("  saved data_quality_report.csv")
    except Exception as e:
        log(f"  WARNING: could not save data quality report: {e}")

    return report_df


# ============================================================================
# STEP 3 - STATISTICAL STUDY / EDA
# ============================================================================
def eda_study(data):
    log("=" * 70)
    log("STEP 3: Statistical study (class balance & distributions)")
    log("=" * 70)
    bbox_df = data["bbox_df"]
    crops_df = data["crops_df"]
    bcols = data.get("bbox_cols")
    ccols = data.get("crops_cols")
    species_map = data["species_map"]
    behavior_map = data["behavior_map"]
    balance_rows = []

    # ---- 3a. species class balance (annotated frames per species) --------
    if bbox_df is not None and bcols and bcols["species"]:
        counts = bbox_df[bcols["species"]].value_counts()
        _bar_plot(counts, "Annotated frames per species (bounding_boxes.csv)",
                  "Species", "count", "01_species_class_balance.png")
        imb = counts.max() / max(counts.min(), 1)
        log(f"  Species imbalance ratio (frames): {imb:.1f}x "
            f"(most: {counts.idxmax()}={counts.max()}, least: {counts.idxmin()}={counts.min()})")
        balance_rows.append({"task": "species (frames)", "num_classes": counts.shape[0],
                              "max_class": counts.idxmax(), "max_count": int(counts.max()),
                              "min_class": counts.idxmin(), "min_count": int(counts.min()),
                              "imbalance_ratio": round(float(imb), 2)})

    # ---- 3b. behavior class balance (from crops.csv = real clips) --------
    clip_len = None
    if crops_df is not None and ccols and ccols["action_id"] and ccols["start"] and ccols["end"]:
        try:
            cdf = crops_df.copy()
            cdf["behavior_name"] = cdf[ccols["action_id"]].map(behavior_map).fillna(cdf[ccols["action_id"]].astype(str))
            counts = cdf["behavior_name"].value_counts()
            _bar_plot(counts, "Number of behavior clips per class (crops.csv)",
                      "Behavior", "clip count", "02_behavior_class_balance.png")
            imb = counts.max() / max(counts.min(), 1)
            log(f"  Behavior imbalance ratio (clips): {imb:.1f}x "
                f"(most: {counts.idxmax()}={counts.max()}, least: {counts.idxmin()}={counts.min()})")
            balance_rows.append({"task": "behavior (clips)", "num_classes": counts.shape[0],
                                  "max_class": counts.idxmax(), "max_count": int(counts.max()),
                                  "min_class": counts.idxmin(), "min_count": int(counts.min()),
                                  "imbalance_ratio": round(float(imb), 2)})

            # true clip length (frames) per behavior
            clip_len = (cdf[ccols["end"]] - cdf[ccols["start"]] + 1).clip(lower=0)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(clip_len.dropna(), bins=40, color="seagreen")
            ax.set_title("Distribution of behavior clip length (frames)")
            ax.set_xlabel("frames in clip"); ax.set_ylabel("number of clips")
            savefig(fig, "03_clip_length_distribution.png")
            log(f"  clip length -> mean={clip_len.mean():.1f} frames, median={clip_len.median():.1f}, "
                f"min={clip_len.min():.0f}, max={clip_len.max():.0f}")

            mean_len_per_behavior = (clip_len.groupby(cdf["behavior_name"]).mean().sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(9, 5))
            mean_len_per_behavior.plot(kind="bar", ax=ax, color="darkorange")
            ax.set_title("Mean clip duration (frames) per behavior")
            ax.set_ylabel("mean frames")
            plt.xticks(rotation=45, ha="right")
            savefig(fig, "04_mean_clip_length_per_behavior.png")

            # species x behavior co-occurrence
            cdf["species_name"] = cdf[ccols["species_id"]].map(species_map).fillna(cdf[ccols["species_id"]].astype(str)) \
                if ccols["species_id"] else None
            if cdf["species_name"] is not None:
                pivot = pd.crosstab(cdf["species_name"], cdf["behavior_name"])
                fig, ax = plt.subplots(figsize=(9, 7))
                im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
                ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
                ax.set_title("Species x Behavior co-occurrence (clip counts)")
                fig.colorbar(im, ax=ax, label="clips")
                savefig(fig, "05_species_behavior_heatmap.png")

            data["crops_df_enriched"] = cdf
        except Exception as e:
            log(f"  WARNING: could not compute behavior statistics from crops.csv: {e}")

    if balance_rows:
        pd.DataFrame(balance_rows).to_csv(os.path.join(OUTPUT_DIR, "class_balance_summary.csv"), index=False)
        log("  saved class_balance_summary.csv")

    # ---- 3c. train/val/test split balance ----------------------------------
    splits = data.get("splits")
    if splits and isinstance(splits, dict) and "crops_df_enriched" in data and ccols["video"]:
        try:
            split_map = {}
            for split_name, vids in splits.items():
                if isinstance(vids, list):
                    for v in vids:
                        split_map[v] = split_name
            cdf = data["crops_df_enriched"].copy()
            cdf["split"] = cdf[ccols["video"]].map(split_map)
            unresolved = cdf["split"].isna().sum()
            if unresolved:
                log(f"  WARNING: {unresolved} clip rows could not be matched to a split "
                    f"(video_name mismatch between crops.csv and splits.json)")
            pivot = pd.crosstab(cdf["behavior_name"], cdf["split"])
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(9, 5))
            pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
            ax.set_title("Train/Val/Test proportion per behavior (%)")
            ax.set_ylabel("% of clips")
            plt.xticks(rotation=45, ha="right")
            savefig(fig, "06_split_balance_per_behavior.png")
            log(f"  split distribution (clips): {cdf['split'].value_counts().to_dict()}")
        except Exception as e:
            log(f"  WARNING: could not analyze split balance: {e}")

    log("  EDA finished. See figures/ for plots and *.csv for tables.")


def _bar_plot(counts, title, xlabel, ylabel, filename):
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        counts.sort_values(ascending=False).plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        savefig(fig, filename)
    except Exception as e:
        log(f"  WARNING: could not plot {filename}: {e}")


# ============================================================================
# VIDEO ACCESS HELPERS
# ============================================================================
_VIDEO_INDEX = None  # cache: {video_name: local_path}

def build_video_index():
    """Locate every video file once, either from an already-extracted folder
    or by extracting videos.zip into VIDEOS_EXTRACT. Returns dict name->path."""
    global _VIDEO_INDEX
    if _VIDEO_INDEX is not None:
        return _VIDEO_INDEX

    index = {}
    # 1) look in any already-extracted folder
    for d in VIDEO_DIR_CANDIDATES:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        index[f] = os.path.join(root, f)
            if index:
                log(f"  found {len(index)} video files already extracted in {d}")
                break

    # 2) fall back to extracting the zip (only if nothing found above)
    if not index and os.path.exists(FILES["videos_zip"]):
        try:
            log("  extracting videos.zip (first run only, please wait)...")
            os.makedirs(VIDEOS_EXTRACT, exist_ok=True)
            with zipfile.ZipFile(FILES["videos_zip"], "r") as zf:
                zf.extractall(VIDEOS_EXTRACT)
            for root, _, files in os.walk(VIDEOS_EXTRACT):
                for f in files:
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        index[f] = os.path.join(root, f)
            log(f"  extracted {len(index)} video files to {VIDEOS_EXTRACT}")
        except Exception as e:
            log(f"  WARNING: could not extract videos.zip: {e}")

    _VIDEO_INDEX = index
    return index


def resolve_video_path(video_name, video_index):
    """Match a video_name value from the CSVs to an actual file path found by
    build_video_index(), tolerating path prefixes, case differences, and
    missing file extensions."""
    vn = os.path.basename(str(video_name)).strip()
    if vn in video_index:
        return video_index[vn]

    vn_lower = vn.lower()
    for name, path in video_index.items():
        if name.lower() == vn_lower:
            return path

    # maybe the CSV value has no extension (e.g. "0012-VIDEO.Ardea")
    if not os.path.splitext(vn)[1]:
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            cand_lower = (vn_lower + ext)
            for name, path in video_index.items():
                if name.lower() == cand_lower:
                    return path

    # last resort: substring match either way
    for name, path in video_index.items():
        nl = name.lower()
        if vn_lower in nl or nl in vn_lower:
            return path
    return None


# ============================================================================
# STEP 4 - BUILD SAMPLE DATASETS OF LABELLED CROPS
# ============================================================================
def build_species_samples(data):
    log("  Building sample dataset for task='species' (from bounding_boxes.csv) ...")
    bbox_df = data["bbox_df"]
    bcols = data.get("bbox_cols")
    if bbox_df is None or not bcols or not all([bcols["species"], bcols["video"], bcols["frame"], bcols["boxes"]]):
        log("    Cannot build species samples: required columns missing.")
        return []

    # only keep rows whose box cell actually parses to something usable
    work = bbox_df[[bcols["species"], bcols["video"], bcols["frame"], bcols["boxes"]]].dropna(
        subset=[bcols["species"], bcols["video"], bcols["frame"]])

    try:
        sampled = stratified_sample(work, bcols["species"], SAMPLES_PER_CLASS * 3)
    except Exception as e:
        log(f"    WARNING: stratified sampling failed ({e}), falling back to random sample.")
        sampled = work.sample(min(len(work), MAX_TOTAL_SAMPLES * 3), random_state=RANDOM_SEED)

    video_index = build_video_index()
    if not video_index:
        log("    WARNING: no video files found/extracted, cannot build species crops.")
        return []
    log(f"    matching sampled boxes against {len(video_index)} available video files...")
    n_video_miss = 0

    out_dir = os.path.join(CROPS_OUT_DIR, "species")
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    per_class_count = defaultdict_int()
    cap_cache = {}
    try:
        import cv2
    except ImportError:
        log("    WARNING: opencv-python not installed. Skipping species crop extraction.")
        return []

    for _, r in sampled.iterrows():
        label = r[bcols["species"]]
        if per_class_count[label] >= SAMPLES_PER_CLASS or len(samples) >= MAX_TOTAL_SAMPLES:
            continue
        boxes = parse_bbox_cell(r[bcols["boxes"]])
        if not boxes:
            continue
        box = biggest_box(boxes)
        if box is None:
            continue

        video_name = r[bcols["video"]]
        vpath = resolve_video_path(video_name, video_index)
        if vpath is None:
            n_video_miss += 1
            continue

        try:
            if vpath not in cap_cache:
                cap_cache[vpath] = cv2.VideoCapture(vpath)
            cap = cap_cache[vpath]
            frame_idx = int(r[bcols["frame"]])
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx - 1, 0))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            h_img, w_img = frame.shape[:2]
            x1, y1, x2, y2 = box["box"]
            x1, x2 = sorted((max(0, min(int(x1), w_img)), max(0, min(int(x2), w_img))))
            y1, y2 = sorted((max(0, min(int(y1), h_img)), max(0, min(int(y2), h_img))))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = frame[y1:y2, x1:x2]
            safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", str(label))
            fname = f"{safe_label}_{len(samples)}.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, crop)
            samples.append({"image_path": fpath, "label": label})
            per_class_count[label] += 1
        except Exception:
            continue

    for cap in cap_cache.values():
        cap.release()

    log(f"    -> built {len(samples)} species crops across {len(per_class_count)} classes"
        + (f" ({n_video_miss} rows skipped: video file not found)" if n_video_miss else ""))
    return samples


def build_behavior_samples(data):
    log("  Building sample dataset for task='behavior' (from crops.csv) ...")
    crops_df = data.get("crops_df_enriched", data.get("crops_df"))
    ccols = data.get("crops_cols")
    bbox_df = data["bbox_df"]
    bcols = data.get("bbox_cols")
    if crops_df is None or ccols is None or bbox_df is None or bcols is None:
        log("    Cannot build behavior samples: required files/columns missing.")
        return []
    if "behavior_name" not in crops_df.columns:
        behavior_map = data["behavior_map"]
        crops_df = crops_df.copy()
        crops_df["behavior_name"] = crops_df[ccols["action_id"]].map(behavior_map).fillna(crops_df[ccols["action_id"]].astype(str))

    try:
        sampled = stratified_sample(crops_df, "behavior_name", SAMPLES_PER_CLASS * 3)
    except Exception as e:
        log(f"    WARNING: stratified sampling failed ({e}), falling back to random sample.")
        sampled = crops_df.sample(min(len(crops_df), MAX_TOTAL_SAMPLES * 3), random_state=RANDOM_SEED)

    # index bounding_boxes.csv by (video_name, frame) for fast lookup
    log("    indexing bounding_boxes.csv by (video, frame) for lookup...")
    bbox_index = {}
    for video_name, frame, boxcell in zip(bbox_df[bcols["video"]], bbox_df[bcols["frame"]], bbox_df[bcols["boxes"]]):
        bbox_index.setdefault(video_name, {})[int(frame)] = boxcell

    video_index = build_video_index()
    if not video_index:
        log("    WARNING: no video files found/extracted, cannot build behavior crops.")
        return []
    log(f"    matching sampled clips against {len(video_index)} available video files...")
    n_video_miss = 0
    n_no_frame_match = 0

    out_dir = os.path.join(CROPS_OUT_DIR, "behavior")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import cv2
    except ImportError:
        log("    WARNING: opencv-python not installed. Skipping behavior crop extraction.")
        return []

    samples = []
    per_class_count = defaultdict_int()
    cap_cache = {}

    for _, r in sampled.iterrows():
        label = r["behavior_name"]
        if per_class_count[label] >= SAMPLES_PER_CLASS or len(samples) >= MAX_TOTAL_SAMPLES:
            continue
        video_name = r[ccols["video"]]
        start_f, end_f = int(r[ccols["start"]]), int(r[ccols["end"]])
        mid_f = (start_f + end_f) // 2

        frames_for_video = bbox_index.get(video_name)
        if not frames_for_video:
            n_no_frame_match += 1
            continue
        # find the closest annotated frame to the clip midpoint
        candidate_frame = min(frames_for_video.keys(), key=lambda f: abs(f - mid_f)) if frames_for_video else None
        if candidate_frame is None or abs(candidate_frame - mid_f) > 60:
            n_no_frame_match += 1
            continue
        boxes = parse_bbox_cell(frames_for_video[candidate_frame])
        if not boxes:
            continue
        box = biggest_box(boxes)
        if box is None:
            continue

        vpath = resolve_video_path(video_name, video_index)
        if vpath is None:
            n_video_miss += 1
            continue

        try:
            if vpath not in cap_cache:
                cap_cache[vpath] = cv2.VideoCapture(vpath)
            cap = cap_cache[vpath]
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(candidate_frame - 1, 0))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            h_img, w_img = frame.shape[:2]
            x1, y1, x2, y2 = box["box"]
            x1, x2 = sorted((max(0, min(int(x1), w_img)), max(0, min(int(x2), w_img))))
            y1, y2 = sorted((max(0, min(int(y1), h_img)), max(0, min(int(y2), h_img))))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = frame[y1:y2, x1:x2]
            safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", str(label))
            fname = f"{safe_label}_{len(samples)}.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, crop)
            samples.append({"image_path": fpath, "label": label})
            per_class_count[label] += 1
        except Exception:
            continue

    for cap in cap_cache.values():
        cap.release()

    log(f"    -> built {len(samples)} behavior crops across {len(per_class_count)} classes"
        + (f" ({n_video_miss} video misses, {n_no_frame_match} no nearby annotated frame)"
           if (n_video_miss or n_no_frame_match) else ""))
    return samples


def defaultdict_int():
    from collections import defaultdict
    return defaultdict(int)


# ============================================================================
# STEP 5 - PRETRAINED CNN TRAINING & COMPARISON
# ============================================================================
def train_and_compare_models(samples, task_name):
    log("=" * 70)
    log(f"STEP 5: Pretrained CNN comparison for task = '{task_name}'")
    log("=" * 70)

    if len(samples) < 20:
        log(f"  Not enough samples ({len(samples)}) to train models for '{task_name}'. Skipping.")
        return None

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, models
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        from PIL import Image
    except ImportError as e:
        log(f"  WARNING: missing library ({e}). Install torch, torchvision, scikit-learn, pillow. Skipping models.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Using device: {device}")

    labels_all = sorted(set(s["label"] for s in samples))
    label_to_idx = {l: i for i, l in enumerate(labels_all)}
    num_classes = len(labels_all)
    log(f"  {len(samples)} samples across {num_classes} classes: {labels_all}")

    paths = [s["image_path"] for s in samples]
    y = [label_to_idx[s["label"]] for s in samples]

    try:
        train_paths, val_paths, train_y, val_y = train_test_split(
            paths, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=y)
    except ValueError:
        train_paths, val_paths, train_y, val_y = train_test_split(
            paths, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED)

    log(f"  train size={len(train_paths)}, val size={len(val_paths)}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class CropDataset(Dataset):
        def __init__(self, paths, labels, tf):
            self.paths, self.labels, self.tf = paths, labels, tf

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            return self.tf(img), self.labels[idx]

    train_loader = DataLoader(CropDataset(train_paths, train_y, train_tf),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val_paths, val_y, val_tf),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    def build_model(name, num_classes):
        try:
            if name == "resnet18":
                m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif name == "mobilenet_v2":
                m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            elif name == "efficientnet_b0":
                m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif name == "densenet121":
                m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unknown model {name}")
        except Exception as e:
            log(f"    WARNING: could not download pretrained weights for {name} "
                f"({e}). Falling back to random initialization for this model "
                f"(check your internet connection / firewall for download.pytorch.org "
                f"if this is unexpected).")
            if name == "resnet18":
                m = models.resnet18(weights=None)
            elif name == "mobilenet_v2":
                m = models.mobilenet_v2(weights=None)
            elif name == "efficientnet_b0":
                m = models.efficientnet_b0(weights=None)
            elif name == "densenet121":
                m = models.densenet121(weights=None)

        for p in m.parameters():
            p.requires_grad = False
        if name == "resnet18":
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif name == "mobilenet_v2":
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        elif name == "efficientnet_b0":
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        elif name == "densenet121":
            m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    results = []
    best_model_name, best_acc, best_preds, best_val_y = None, -1, None, None

    for model_name in MODELS_TO_TRY:
        log(f"  --- training {model_name} ---")
        try:
            model = build_model(model_name, num_classes).to(device)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            t0 = time.time()
            model.train()
            for epoch in range(EPOCHS):
                running_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * xb.size(0)
                epoch_loss = running_loss / max(len(train_paths), 1)
                log(f"    epoch {epoch+1}/{EPOCHS} - loss={epoch_loss:.4f}")
            train_time = time.time() - t0

            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model(xb)
                    preds = out.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_true.extend(yb.numpy().tolist())

            acc = accuracy_score(all_true, all_preds)
            f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
            n_params = sum(p.numel() for p in model.parameters())

            log(f"    {model_name}: val_accuracy={acc:.3f}, val_macro_f1={f1:.3f}, "
                f"train_time={train_time:.1f}s, params={n_params:,}")

            results.append({
                "task": task_name, "model": model_name, "val_accuracy": round(acc, 4),
                "val_macro_f1": round(f1, 4), "train_time_sec": round(train_time, 1),
                "num_params": n_params, "num_classes": num_classes,
                "train_samples": len(train_paths), "val_samples": len(val_paths),
            })

            if acc > best_acc:
                best_acc = acc
                best_model_name = model_name
                best_preds, best_val_y = all_preds, all_true

        except Exception as e:
            log(f"    WARNING: training {model_name} failed: {e}")
            traceback.print_exc()
            continue

    if not results:
        log("  No model trained successfully.")
        return None

    results_df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
    results_df.to_csv(os.path.join(MODEL_DIR, f"model_comparison_{task_name}.csv"), index=False)
    log(f"  Saved model_comparison_{task_name}.csv")
    log(f"  BEST MODEL for '{task_name}': {best_model_name} (accuracy={best_acc:.3f})")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(results_df["model"], results_df["val_accuracy"], color="teal")
        axes[0].set_title(f"Validation accuracy by model ({task_name})")
        axes[0].set_ylim(0, 1)
        plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
        axes[1].bar(results_df["model"], results_df["train_time_sec"], color="indianred")
        axes[1].set_title(f"Training time (s) by model ({task_name})")
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
        savefig(fig, f"07_model_comparison_{task_name}.png")
    except Exception as e:
        log(f"  WARNING: could not plot model comparison: {e}")

    if best_preds is not None:
        try:
            cm = confusion_matrix(best_val_y, best_preds)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_title(f"Confusion matrix - best model ({best_model_name}, {task_name})")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            fig.colorbar(im, ax=ax)
            savefig(fig, f"08_confusion_matrix_{task_name}.png")
        except Exception as e:
            log(f"  WARNING: could not plot confusion matrix: {e}")

    return results_df


# ============================================================================
# MAIN
# ============================================================================
def main():
    t_start = time.time()
    make_dirs()
    log("VISUAL WETLANDBIRDS - ANALYSIS RUN STARTED")
    log(f"Base data folder: {BASE_DIR}")

    try:
        data = load_data()
    except Exception as e:
        log(f"FATAL: could not load data: {e}")
        traceback.print_exc()
        save_log()
        return

    try:
        data_quality_report(data)
    except Exception as e:
        log(f"WARNING: data quality step failed: {e}")
        traceback.print_exc()

    try:
        eda_study(data)
    except Exception as e:
        log(f"WARNING: EDA step failed: {e}")
        traceback.print_exc()

    all_model_results = []
    for task in TASKS_TO_RUN:
        try:
            if task == "species":
                samples = build_species_samples(data)
            else:
                samples = build_behavior_samples(data)
            res = train_and_compare_models(samples, task_name=task)
            if res is not None:
                all_model_results.append(res)
        except Exception as e:
            log(f"WARNING: modeling step failed for task '{task}': {e}")
            traceback.print_exc()

    if all_model_results:
        combined = pd.concat(all_model_results, ignore_index=True)
        combined.to_csv(os.path.join(MODEL_DIR, "model_comparison_ALL_TASKS.csv"), index=False)
        best_overall = combined.sort_values("val_accuracy", ascending=False).iloc[0]
        log("=" * 70)
        log(f"OVERALL BEST MODEL: {best_overall['model']} on task '{best_overall['task']}' "
            f"with accuracy={best_overall['val_accuracy']}")
        log("=" * 70)

    try:
        with open(os.path.join(OUTPUT_DIR, "SUMMARY.txt"), "w", encoding="utf-8") as f:
            f.write("VISUAL WETLANDBIRDS - ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write("This run performed:\n")
            f.write("1. Data quality checks (missing values, duplicates, invalid/tiny boxes, orphan IDs, cross-file video_name consistency)\n")
            f.write("2. Statistical study of class balance (species from bounding_boxes.csv, behavior from crops.csv), clip length, co-occurrence, split balance\n")
            f.write("3. A quick pretrained-CNN benchmark on a small stratified sample of crops for species and behavior classification\n\n")
            f.write("See data_quality_report.csv, class_balance_summary.csv, figures/, and model_results/\n")
            f.write("for details. run_log.txt has the full console log of this run.\n")
        log("Saved SUMMARY.txt")
    except Exception as e:
        log(f"WARNING: could not save summary: {e}")

    total_time = time.time() - t_start
    log(f"DONE. Total run time: {total_time/60:.1f} minutes.")
    log(f"All results saved under: {OUTPUT_DIR}")
    save_log()


if __name__ == "__main__":
    main()
