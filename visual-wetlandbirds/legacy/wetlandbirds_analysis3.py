"""
================================================================================
 VISUAL WETLANDBIRDS DATASET - INDEPENDENT BENCHMARK & CROSS-VALIDATED
 COMPARISON AGAINST Rodriguez-Juan et al. (Scientific Data, 2025)
================================================================================
The paper introduced this dataset and reported single-run baselines: YOLOv9
for species (detection), five video models for behavior. Single split, no
repeats, no confidence intervals, no significance testing between models.
This script asks the next question -- how stable are those numbers, and how
far can a lightweight, everyday CNN get on the same data -- and answers it
with the same rigor a reviewer would ask for: repeated cross-validation,
confidence intervals, and pairwise significance testing.

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

What this script does, step by step
------------------------------------
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
5. Fine-tunes four PRETRAINED torchvision CNNs (frozen backbone, few epochs)
   on that sample for both tasks, but properly this time: stratified 5-fold
   CV instead of one split, out-of-fold predictions for every sample, and
   pairwise McNemar significance testing (Holm-Bonferroni corrected) so
   "model X beat model Y" is a statistical claim, not a coin flip read off
   one table.
6. Takes the winner of step 5 (MobileNetV2) and points it at the problem
   the crop-classification benchmark couldn't touch: full-frame species
   DETECTION -- localization AND recognition together, same shape of task
   as the paper's YOLOv9 baseline, scored on the same four metrics
   (Precision, Recall, mAP50, mAP50-95). Run twice, on purpose:
     6a. one grouped train/val split, no repeats -- the fair, eye-to-eye
         number that sits directly next to the paper's single-split result.
     6b. 5-fold group cross-validation on the identical pipeline -- the
         number that shows whether 6a was a stable estimate or a lucky
         draw, and by how much.
     6c. a final side-by-side report: paper vs. our single-split vs. our
         k-fold mean, plus a plain-numbers answer to "what did 5-fold
         actually buy us here."
7. Saves every figure / table / log into a new folder under C:\\stage.

Requirements (install once)
----------------------------
pip install pandas numpy matplotlib scikit-learn opencv-python pillow torch torchvision scipy

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
import random
import zipfile
import warnings
import traceback
import itertools

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

# ---- k-fold CV / significance testing config -------------------------------
# K_FOLDS is a *requested* number of folds; the actual number used for a given
# task is capped at the smallest class's sample count (StratifiedKFold cannot
# split a class into more folds than it has members). This is logged clearly
# so a silent reduction never hides in the output.
K_FOLDS = 5
MCNEMAR_EXACT_THRESHOLD = 25   # below this many discordant pairs, use the
                                # exact binomial test instead of the chi-square
                                # approximation (standard McNemar guidance)
ALPHA = 0.05                   # significance level for the Holm-Bonferroni
                                # corrected pairwise model comparisons

# ---- full-frame species DETECTION config (localization + recognition) -----
# separate knobs from the crop-classification setup above on purpose: full
# frames are way bigger than 160x160 crops and a Faster R-CNN training step
# costs a lot more than a frozen classifier head, so these stay conservative
# to keep the whole thing runnable without a beefy GPU.
DETECTION_BACKBONE     = "mobilenet_v2"  # winner of the classification benchmark above
DET_IMG_SIZE           = 480
DET_BATCH_SIZE         = 2
DET_EPOCHS             = 6
DET_FRAMES_PER_VIDEO   = 6      # cap so one heavily-annotated video can't dominate the set
DET_MAX_FRAMES         = 450
DET_TEST_FRACTION      = 0.20   # single-split mode: video-level holdout size
DET_SCORE_THRESH       = 0.5    # confidence cutoff used for the Precision/Recall point estimate
DET_LR                 = 0.005
DET_MOMENTUM           = 0.9
DET_WEIGHT_DECAY       = 0.0005
# RPN/ROI sizing kept modest -- default torchvision settings (2000 proposals
# pre/post NMS) are tuned for COCO-scale scenes with dozens of objects; our
# frames have at most a handful of birds, so trimming this saves a lot of
# memory and time with no real accuracy cost.
DET_RPN_PRE_NMS_TRAIN  = 400
DET_RPN_POST_NMS_TRAIN = 200
DET_RPN_PRE_NMS_TEST   = 200
DET_RPN_POST_NMS_TEST  = 100
DET_BOX_BATCH_PER_IMG  = 64
DET_RPN_BATCH_PER_IMG  = 64
# paper's own YOLOv9 species baseline (Table 6) -- hardcoded literature
# values we compare against, never computed by this script
PAPER_YOLOV9_METRICS = {
    "precision": 0.835,
    "recall": 0.759,
    "mAP50": 0.801,
    "mAP50_95": 0.556,
}

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
    for d in [OUTPUT_DIR, FIG_DIR, MODEL_DIR, CROPS_OUT_DIR, os.path.join(CROPS_OUT_DIR, "species_detection")]:
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
# STATISTICAL SIGNIFICANCE HELPERS (McNemar's test, Holm-Bonferroni)
# ============================================================================
def mcnemar_test(y_true, pred_a, pred_b):
    """
    Paired McNemar's test between two classifiers evaluated on the SAME set
    of samples (this is why out-of-fold predictions matter: every sample must
    have been predicted by both models being compared, on data neither model
    was trained on).

    Only the discordant pairs matter:
        n10 = # samples model A got right and model B got wrong
        n01 = # samples model A got wrong and model B got right
    Samples where both models agree (both right or both wrong) carry no
    information about which model is better and are correctly ignored.

    Uses the exact binomial form when the number of discordant pairs is small
    (< MCNEMAR_EXACT_THRESHOLD, following standard guidance that the
    chi-square approximation is unreliable in that regime) and the
    continuity-corrected chi-square form otherwise.
    """
    from scipy import stats as spstats

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n10 = int(np.sum(correct_a & (~correct_b)))   # A right, B wrong
    n01 = int(np.sum((~correct_a) & correct_b))   # A wrong, B right
    n_discordant = n01 + n10

    if n_discordant == 0:
        # models agree on every sample -> nothing to distinguish them
        return {"n10": n10, "n01": n01, "n_discordant": 0,
                "test": "no_discordant_pairs", "statistic": np.nan, "p_value": 1.0}

    if n_discordant < MCNEMAR_EXACT_THRESHOLD:
        k = min(n01, n10)
        result = spstats.binomtest(k, n_discordant, 0.5, alternative="two-sided")
        return {"n10": n10, "n01": n01, "n_discordant": n_discordant,
                "test": "exact_binomial", "statistic": float(k), "p_value": float(result.pvalue)}
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / n_discordant   # Edwards' continuity correction
        p = float(1 - spstats.chi2.cdf(stat, df=1))
        return {"n10": n10, "n01": n01, "n_discordant": n_discordant,
                "test": "chi2_continuity_corrected", "statistic": float(stat), "p_value": p}


def holm_bonferroni(pvals):
    """
    Holm-Bonferroni step-down correction for a family of p-values.
    Less conservative than plain Bonferroni while still controlling the
    family-wise error rate; appropriate here because with 4 models we run
    C(4,2) = 6 pairwise McNemar tests per task, and testing all 6 at raw
    alpha=0.05 would inflate the false-positive rate for "model X beats
    model Y" claims.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * pvals[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


# ============================================================================
# STEP 5 - PRETRAINED CNN TRAINING & COMPARISON (STRATIFIED K-FOLD CV
#           WITH OUT-OF-FOLD PREDICTIONS + PAIRWISE McNEMAR TESTING)
# ============================================================================
#
# WHY THIS REPLACES A SINGLE TRAIN/VAL SPLIT
# --------------------------------------------------------------------------
# A single 75/25 split evaluates each model on ~60-115 samples. At that size
# a difference of a handful of samples (which is all that separated the
# models in the original run) is well within sampling noise: at p~0.5 and
# n=62, the standard error on a single accuracy estimate is already
# sqrt(0.5*0.5/62) ~= 6.4 points, i.e. a 95% CI of roughly +/-12.5 points.
# Two things fix this:
#   1. K-FOLD CV gives K independent accuracy estimates per model, so we can
#      report mean +/- std (and a proper across-fold 95% CI) instead of one
#      point estimate.
#   2. OUT-OF-FOLD (OOF) PREDICTIONS: because every sample is held out
#      exactly once (in exactly one fold, for every model), we can assemble
#      one prediction per model for every sample in the dataset, all made on
#      data that model never trained on. That is exactly the paired setup
#      McNemar's test requires: same samples, two classifiers, count how
#      often they disagree and in which direction.
# ============================================================================
def train_and_compare_models_kfold(samples, task_name):
    log("=" * 70)
    log(f"STEP 5: Stratified {K_FOLDS}-fold CNN comparison + McNemar testing "
        f"for task = '{task_name}'")
    log("=" * 70)

    if len(samples) < 20:
        log(f"  Not enough samples ({len(samples)}) to train models for '{task_name}'. Skipping.")
        return None

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, models
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        from scipy import stats as spstats
        from PIL import Image
    except ImportError as e:
        log(f"  WARNING: missing library ({e}). Install torch, torchvision, scikit-learn, scipy, pillow. Skipping models.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Using device: {device}")

    labels_all = sorted(set(s["label"] for s in samples))
    label_to_idx = {l: i for i, l in enumerate(labels_all)}
    num_classes = len(labels_all)
    log(f"  {len(samples)} samples across {num_classes} classes: {labels_all}")

    paths = np.array([s["image_path"] for s in samples])
    y = np.array([label_to_idx[s["label"]] for s in samples])

    # ---- pick a safe fold count -------------------------------------------
    # StratifiedKFold requires every class to have at least n_splits members.
    # Rather than crash on a scarce class (e.g. Preening/Flying in the
    # behavior task), cap K_FOLDS at the smallest class count and say so.
    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min())
    k_folds = min(K_FOLDS, min_class_count)
    if k_folds < K_FOLDS:
        log(f"  WARNING: smallest class ('{labels_all[int(np.argmin(class_counts))]}') has only "
            f"{min_class_count} samples; reducing folds from {K_FOLDS} to {k_folds} so every "
            f"fold still contains at least one example of every class.")
    if k_folds < 2:
        log(f"  Cannot run k-fold CV for '{task_name}': smallest class has fewer than 2 samples.")
        return None

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    log(f"  running {k_folds}-fold stratified CV "
        f"({k_folds} folds x {len(MODELS_TO_TRY)} models = {k_folds * len(MODELS_TO_TRY)} training runs)")

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

    # Out-of-fold (OOF) prediction storage: exactly one prediction per
    # sample per model, made on a fold that model never trained on. This is
    # the array McNemar's test will run on.
    oof_preds = {name: np.full(len(samples), -1, dtype=int) for name in MODELS_TO_TRY}
    oof_fold = np.full(len(samples), -1, dtype=int)
    per_fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths, y)):
        log(f"  --- fold {fold_idx + 1}/{k_folds}  (train={len(train_idx)}, val={len(val_idx)}) ---")
        oof_fold[val_idx] = fold_idx

        train_loader = DataLoader(
            CropDataset(paths[train_idx].tolist(), y[train_idx].tolist(), train_tf),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(
            CropDataset(paths[val_idx].tolist(), y[val_idx].tolist(), val_tf),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        for model_name in MODELS_TO_TRY:
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
                    epoch_loss = running_loss / max(len(train_idx), 1)
                train_time = time.time() - t0

                model.eval()
                fold_preds = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        out = model(xb)
                        fold_preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
                fold_preds = np.array(fold_preds)
                oof_preds[model_name][val_idx] = fold_preds

                fold_true = y[val_idx]
                acc = accuracy_score(fold_true, fold_preds)
                f1 = f1_score(fold_true, fold_preds, average="macro", zero_division=0)
                n_params = sum(p.numel() for p in model.parameters())

                log(f"    fold {fold_idx+1} {model_name}: acc={acc:.3f}, macro_f1={f1:.3f}, "
                    f"train_time={train_time:.1f}s (final epoch loss={epoch_loss:.4f})")

                per_fold_rows.append({
                    "task": task_name, "fold": fold_idx, "model": model_name,
                    "val_accuracy": round(float(acc), 4), "val_macro_f1": round(float(f1), 4),
                    "train_time_sec": round(train_time, 1), "num_params": n_params,
                    "num_classes": num_classes,
                    "train_samples": len(train_idx), "val_samples": len(val_idx),
                })
            except Exception as e:
                log(f"    WARNING: fold {fold_idx+1} training failed for {model_name}: {e}")
                traceback.print_exc()
                continue

    if not per_fold_rows:
        log("  No model trained successfully in any fold.")
        return None

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(os.path.join(MODEL_DIR, f"kfold_results_{task_name}.csv"), index=False)
    log(f"  Saved kfold_results_{task_name}.csv "
        f"({k_folds} folds x {per_fold_df['model'].nunique()} models = {len(per_fold_df)} rows)")

    # ---- aggregate per-fold metrics: mean, std, across-fold 95% CI --------
    summary_rows = []
    for model_name in MODELS_TO_TRY:
        sub = per_fold_df[per_fold_df["model"] == model_name]
        if sub.empty:
            continue
        accs, f1s = sub["val_accuracy"].values, sub["val_macro_f1"].values
        n = len(accs)
        mean_acc, std_acc = float(accs.mean()), float(accs.std(ddof=1)) if n > 1 else 0.0
        mean_f1, std_f1 = float(f1s.mean()), float(f1s.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            tcrit = float(spstats.t.ppf(0.975, df=n - 1))
            ci_acc, ci_f1 = tcrit * std_acc / np.sqrt(n), tcrit * std_f1 / np.sqrt(n)
        else:
            ci_acc = ci_f1 = float("nan")
        summary_rows.append({
            "task": task_name, "model": model_name, "k_folds": n,
            "mean_accuracy": round(mean_acc, 4), "std_accuracy": round(std_acc, 4),
            "acc_95ci_halfwidth": round(ci_acc, 4) if ci_acc == ci_acc else None,
            "mean_macro_f1": round(mean_f1, 4), "std_macro_f1": round(std_f1, 4),
            "f1_95ci_halfwidth": round(ci_f1, 4) if ci_f1 == ci_f1 else None,
            "mean_train_time_sec": round(float(sub["train_time_sec"].mean()), 1),
            "num_params": int(sub["num_params"].iloc[0]),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_accuracy", ascending=False)
    summary_df.to_csv(os.path.join(MODEL_DIR, f"kfold_summary_{task_name}.csv"), index=False)
    log(f"  Saved kfold_summary_{task_name}.csv")
    log(f"  --- {task_name}: mean +/- std accuracy across {k_folds} folds ---")
    for _, row in summary_df.iterrows():
        ci_txt = f", 95% CI +/-{row['acc_95ci_halfwidth']:.4f}" if row["acc_95ci_halfwidth"] is not None else ""
        log(f"    {row['model']}: acc={row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f}{ci_txt}  "
            f"macro_f1={row['mean_macro_f1']:.4f} +/- {row['std_macro_f1']:.4f}")

    # ---- OOF prediction table: every sample, every model, exactly once ----
    oof_rows = []
    for i in range(len(samples)):
        row = {
            "sample_idx": i, "image_path": paths[i],
            "true_label_idx": int(y[i]), "true_label": labels_all[int(y[i])],
            "fold": int(oof_fold[i]),
        }
        for model_name in MODELS_TO_TRY:
            pred_idx = int(oof_preds[model_name][i])
            row[f"pred_{model_name}"] = pred_idx
            row[f"pred_{model_name}_label"] = labels_all[pred_idx] if pred_idx >= 0 else None
        oof_rows.append(row)
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(os.path.join(MODEL_DIR, f"oof_predictions_{task_name}.csv"), index=False)
    log(f"  Saved oof_predictions_{task_name}.csv "
        f"({len(oof_df)} samples, each with one held-out prediction per model)")

    # ---- pairwise McNemar tests on OOF predictions, Holm-Bonferroni corrected --
    pair_rows = []
    for model_a, model_b in itertools.combinations(MODELS_TO_TRY, 2):
        valid = (oof_preds[model_a] >= 0) & (oof_preds[model_b] >= 0)
        if valid.sum() == 0:
            continue
        result = mcnemar_test(y[valid], oof_preds[model_a][valid], oof_preds[model_b][valid])
        acc_a = accuracy_score(y[valid], oof_preds[model_a][valid])
        acc_b = accuracy_score(y[valid], oof_preds[model_b][valid])
        pair_rows.append({
            "task": task_name, "model_a": model_a, "model_b": model_b,
            "n_compared": int(valid.sum()),
            "oof_acc_a": round(float(acc_a), 4), "oof_acc_b": round(float(acc_b), 4),
            "n_a_right_b_wrong": result["n10"], "n_a_wrong_b_right": result["n01"],
            "n_discordant": result["n_discordant"], "test_type": result["test"],
            "statistic": result["statistic"], "p_value": result["p_value"],
        })

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["p_value_holm"] = holm_bonferroni(pair_df["p_value"].values)
        pair_df[f"significant_at_{ALPHA}_holm"] = pair_df["p_value_holm"] < ALPHA
        pair_df = pair_df.sort_values("p_value").reset_index(drop=True)
        pair_df.to_csv(os.path.join(MODEL_DIR, f"mcnemar_{task_name}.csv"), index=False)
        log(f"  Saved mcnemar_{task_name}.csv "
            f"({len(pair_df)} pairwise comparisons, Holm-Bonferroni corrected across the family)")
        log(f"  --- pairwise McNemar results for '{task_name}' (OOF predictions) ---")
        for _, row in pair_df.iterrows():
            sig = "SIGNIFICANT" if row[f"significant_at_{ALPHA}_holm"] else "not significant"
            log(f"    {row['model_a']} (acc={row['oof_acc_a']:.3f}) vs "
                f"{row['model_b']} (acc={row['oof_acc_b']:.3f}): "
                f"discordant n={row['n_discordant']} "
                f"[{row['model_a']}-only-right={row['n_a_right_b_wrong']}, "
                f"{row['model_b']}-only-right={row['n_a_wrong_b_right']}], "
                f"{row['test_type']}, p={row['p_value']:.4f}, "
                f"Holm-adjusted p={row['p_value_holm']:.4f} -> {sig}")
    else:
        log("  No valid pairwise McNemar comparisons could be computed (no overlapping predictions).")

    # ---- plot: mean accuracy / macro-F1 with across-fold error bars -------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].bar(summary_df["model"], summary_df["mean_accuracy"],
                    yerr=summary_df["std_accuracy"], capsize=5, color="teal")
        axes[0].set_title(f"K-fold accuracy: mean +/- std (k={k_folds}) - {task_name}")
        axes[0].set_ylim(0, 1)
        plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
        axes[1].bar(summary_df["model"], summary_df["mean_macro_f1"],
                    yerr=summary_df["std_macro_f1"], capsize=5, color="indianred")
        axes[1].set_title(f"K-fold macro-F1: mean +/- std (k={k_folds}) - {task_name}")
        axes[1].set_ylim(0, 1)
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
        savefig(fig, f"07_kfold_comparison_{task_name}.png")
    except Exception as e:
        log(f"  WARNING: could not plot k-fold comparison: {e}")

    # ---- pooled OOF confusion matrix for the best model by mean accuracy --
    try:
        best_model_name = summary_df.iloc[0]["model"]
        valid = oof_preds[best_model_name] >= 0
        cm = confusion_matrix(y[valid], oof_preds[best_model_name][valid])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Pooled OOF confusion matrix - best model ({best_model_name}, {task_name})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)
        savefig(fig, f"08_confusion_matrix_oof_{task_name}.png")
    except Exception as e:
        log(f"  WARNING: could not plot confusion matrix: {e}")

    log(f"  BEST MODEL for '{task_name}' by mean k-fold accuracy: "
        f"{summary_df.iloc[0]['model']} ({summary_df.iloc[0]['mean_accuracy']:.4f} "
        f"+/- {summary_df.iloc[0]['std_accuracy']:.4f})")

    return {"summary": summary_df, "pairwise": pair_df if not pair_df.empty else None,
            "oof": oof_df, "per_fold": per_fold_df}


# ============================================================================
# MAIN
# ============================================================================
# ============================================================================
# STEP 6 - FULL-FRAME SPECIES DETECTION (LOCALIZATION + RECOGNITION)
# ============================================================================
# Everything above this line classifies a bird crop that's already been
# handed to it -- a strictly easier problem than what the paper's YOLOv9
# baseline actually solves, which is finding the bird somewhere in a raw
# frame AND naming it. This section closes that gap so the species numbers
# can finally sit next to the paper's on the same metrics: Precision,
# Recall, mAP50, mAP50-95.
#
# The detector uses MobileNetV2 as its backbone because that's what won the
# crop-classification benchmark above -- no point dragging three more
# backbones through a much slower detection training loop just to re-learn
# what we already know from Step 5.
#
# Two runs, deliberately:
#   1. a single grouped train/val split, no repeats. This is the number
#      that lines up directly against the paper's own single-split result
#      -- same protocol, same shot at the data, genuinely eye to eye.
#   2. stratified group k-fold CV on top of the same pipeline, which is
#      where our actual added value shows up: does the single-split number
#      hold up, or was it a lucky (or unlucky) draw?
# ============================================================================
def build_species_detection_samples(data):
    """
    Assemble full-frame detection ground truth from bounding_boxes.csv.

    Unlike build_species_samples() above (one crop = one row = the biggest
    box in that row), this groups every row sharing the same (video, frame)
    together, so a frame with several birds keeps all of them as separate
    object instances. That's the part that makes this a real detection
    problem instead of a classification problem wearing a detection hat.

    Frames are sampled per video (not globally) so one heavily-annotated
    video can't dominate the training set, and everything is capped at
    DET_MAX_FRAMES to keep training time sane on a laptop-class machine.
    """
    log("  Building full-frame species detection dataset (bounding_boxes.csv) ...")
    from collections import defaultdict
    bbox_df = data["bbox_df"]
    bcols = data.get("bbox_cols")
    if bbox_df is None or not bcols or not all([bcols["species"], bcols["video"], bcols["frame"], bcols["boxes"]]):
        log("    Cannot build detection samples: required columns missing.")
        return [], {}

    work = bbox_df[[bcols["video"], bcols["frame"], bcols["species"], bcols["boxes"]]].dropna(
        subset=[bcols["video"], bcols["frame"], bcols["species"]])

    # one key per (video, frame) -- every row that shares a key contributes
    # its own box(es)+species to that frame's ground truth
    group_keys = list(zip(work[bcols["video"]], work[bcols["frame"]]))
    work = work.assign(_group_key=group_keys)

    groups_per_video = defaultdict(list)
    for key, _ in work.groupby("_group_key", sort=False):
        groups_per_video[key[0]].append(key)

    rng = random.Random(RANDOM_SEED)
    selected_keys = []
    for video_name, keys in groups_per_video.items():
        rng.shuffle(keys)
        selected_keys.extend(keys[:DET_FRAMES_PER_VIDEO])
    rng.shuffle(selected_keys)
    selected_keys = selected_keys[:DET_MAX_FRAMES]
    selected_set = set(selected_keys)
    log(f"    sampled {len(selected_set)} frames across {len(groups_per_video)} videos "
        f"(cap: {DET_FRAMES_PER_VIDEO}/video, {DET_MAX_FRAMES} total)")

    video_index = build_video_index()
    if not video_index:
        log("    WARNING: no video files found/extracted, cannot build detection frames.")
        return [], {}

    out_dir = os.path.join(CROPS_OUT_DIR, "species_detection")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import cv2
    except ImportError:
        log("    WARNING: opencv-python not installed. Skipping detection frame extraction.")
        return [], {}

    grouped = work[work["_group_key"].isin(selected_set)].groupby("_group_key", sort=False)
    label_set = set()
    samples = []
    cap_cache = {}
    n_video_miss = 0
    n_no_valid_box = 0

    for (video_name, frame_idx), rows in grouped:
        vpath = resolve_video_path(video_name, video_index)
        if vpath is None:
            n_video_miss += 1
            continue
        try:
            if vpath not in cap_cache:
                cap_cache[vpath] = cv2.VideoCapture(vpath)
            cap = cap_cache[vpath]
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_idx) - 1, 0))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            h_img, w_img = frame.shape[:2]

            boxes, labels = [], []
            for _, r in rows.iterrows():
                species_label = r[bcols["species"]]
                for parsed in parse_bbox_cell(r[bcols["boxes"]]):
                    x1, y1, x2, y2 = parsed["box"]
                    x1, x2 = sorted((max(0, min(int(x1), w_img)), max(0, min(int(x2), w_img))))
                    y1, y2 = sorted((max(0, min(int(y1), h_img)), max(0, min(int(y2), h_img))))
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue
                    boxes.append((float(x1), float(y1), float(x2), float(y2)))
                    labels.append(species_label)

            if not boxes:
                n_no_valid_box += 1
                continue

            safe_video = re.sub(r"[^A-Za-z0-9_-]", "_", str(video_name))
            fname = f"{safe_video}_f{int(frame_idx)}.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, frame)

            samples.append({
                "image_path": fpath,
                "video_name": str(video_name),
                "boxes": boxes,
                "labels": labels,
            })
            label_set.update(labels)
        except Exception:
            continue

    for cap in cap_cache.values():
        cap.release()

    label_to_idx = {name: i + 1 for i, name in enumerate(sorted(label_set))}  # 0 is background
    n_boxes_total = sum(len(s["boxes"]) for s in samples)
    log(f"    -> built {len(samples)} full-frame samples, {n_boxes_total} bird instances, "
        f"{len(label_to_idx)} species"
        + (f" ({n_video_miss} frames skipped: video not found)" if n_video_miss else "")
        + (f" ({n_no_valid_box} frames skipped: no valid box after clipping)" if n_no_valid_box else ""))
    return samples, label_to_idx


def _detection_collate(batch):
    # detection targets are variable-length -- can't stack them like a
    # normal batch, so just hand back a tuple of (images, targets)
    return tuple(zip(*batch))


def build_species_detector(num_classes):
    """Faster R-CNN on a MobileNetV2 backbone. num_classes excludes the
    background class -- torchvision's detection heads reserve index 0 for
    it internally, so we pass num_classes + 1 down to FasterRCNN.

    RPN/ROI batch sizes are trimmed from the torchvision defaults (which
    assume COCO-style scenes with dozens of objects) since our frames have
    at most a handful of birds -- this keeps memory and training time down
    with no meaningful accuracy trade-off for this dataset.
    """
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    try:
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
    except Exception:
        backbone = mobilenet_v2(weights=None).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=DET_IMG_SIZE,
        max_size=DET_IMG_SIZE,
        rpn_pre_nms_top_n_train=DET_RPN_PRE_NMS_TRAIN,
        rpn_post_nms_top_n_train=DET_RPN_POST_NMS_TRAIN,
        rpn_pre_nms_top_n_test=DET_RPN_PRE_NMS_TEST,
        rpn_post_nms_top_n_test=DET_RPN_POST_NMS_TEST,
        box_batch_size_per_image=DET_BOX_BATCH_PER_IMG,
        rpn_batch_size_per_image=DET_RPN_BATCH_PER_IMG,
    )
    return model


# ---- home-grown detection metrics ------------------------------------------
# No pycocotools dependency -- this is a plain VOC-style AP (all-point
# interpolation) computed per class, then averaged across classes for mAP50,
# and averaged again across the 10 COCO IoU thresholds (0.50:0.05:0.95) for
# mAP50-95. Precision/Recall are reported at a single fixed confidence
# operating point (DET_SCORE_THRESH), which is the number that's actually
# comparable to a single Precision/Recall pair like the paper reports.
def _iou_xyxy(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)


def _average_precision(recall, precision):
    # standard all-point interpolation (Pascal VOC 2010+ / COCO style)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_for_class_iou(preds_by_image, gts_by_image, class_id, iou_thr):
    npos = 0
    gt_boxes_cache, gt_used = {}, {}
    for img_idx, g in gts_by_image.items():
        mask = g["labels"] == class_id
        boxes = g["boxes"][mask]
        gt_boxes_cache[img_idx] = boxes
        gt_used[img_idx] = np.zeros(len(boxes), dtype=bool)
        npos += len(boxes)
    if npos == 0:
        return None  # class doesn't appear in this eval set at all -- skip, don't score as 0

    dets = []
    for img_idx, p in preds_by_image.items():
        mask = p["labels"] == class_id
        for box, score in zip(p["boxes"][mask], p["scores"][mask]):
            dets.append((img_idx, box, score))
    if not dets:
        return 0.0

    dets.sort(key=lambda d: -d[2])
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i, (img_idx, box, score) in enumerate(dets):
        gts = gt_boxes_cache.get(img_idx, np.zeros((0, 4)))
        if len(gts) == 0:
            fp[i] = 1
            continue
        ious = _iou_xyxy(box, gts)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not gt_used[img_idx][best]:
            tp[i] = 1
            gt_used[img_idx][best] = True
        else:
            fp[i] = 1

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / npos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _average_precision(recall, precision)


def compute_detection_metrics(preds_by_image, gts_by_image, class_ids, score_thresh=DET_SCORE_THRESH):
    """Returns precision, recall (at score_thresh, IoU 0.5, pooled across
    classes) plus mAP50 and mAP50-95 (per-class AP averaged over classes,
    and for mAP50-95 also over IoU 0.50:0.05:0.95) -- the same four numbers
    the paper reports for YOLOv9."""
    iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    ap50_per_class, ap_multi_per_class = [], []
    for c in class_ids:
        a50 = _ap_for_class_iou(preds_by_image, gts_by_image, c, 0.5)
        if a50 is not None:
            ap50_per_class.append(a50)
        aps = [_ap_for_class_iou(preds_by_image, gts_by_image, c, t) for t in iou_thresholds]
        aps = [a for a in aps if a is not None]
        if aps:
            ap_multi_per_class.append(float(np.mean(aps)))

    tp = fp = fn = 0
    for img_idx, g in gts_by_image.items():
        p = preds_by_image.get(img_idx, {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,)), "scores": np.zeros((0,))})
        keep = p["scores"] >= score_thresh
        pred_boxes, pred_labels, pred_scores = p["boxes"][keep], p["labels"][keep], p["scores"][keep]
        order = np.argsort(-pred_scores)
        matched = np.zeros(len(g["boxes"]), dtype=bool)
        for idx in order:
            same_class = np.where(g["labels"] == pred_labels[idx])[0]
            if len(same_class) == 0:
                fp += 1
                continue
            ious = _iou_xyxy(pred_boxes[idx], g["boxes"][same_class])
            best_local = int(np.argmax(ious))
            best_global = same_class[best_local]
            if ious[best_local] >= 0.5 and not matched[best_global]:
                tp += 1
                matched[best_global] = True
            else:
                fp += 1
        fn += int((~matched).sum())

    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "mAP50": float(np.mean(ap50_per_class)) if ap50_per_class else 0.0,
        "mAP50_95": float(np.mean(ap_multi_per_class)) if ap_multi_per_class else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def _train_eval_one_detector_split(train_samples, val_samples, label_to_idx, epochs, tag):
    """Train the MobileNetV2 detector on one train/val split and return its
    val-set metrics. Shared by both the single-split run and every fold of
    the k-fold run below, so the two protocols stay identical apart from
    how the split itself was produced."""
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        from PIL import Image
    except ImportError as e:
        log(f"    WARNING: missing library ({e}). Skipping detection training.")
        return None

    # kept local (not module-level) so the rest of this script still runs
    # fine on a machine without torch installed -- same reasoning as
    # CropDataset in the classification benchmark above
    class DetectionFrameDataset(Dataset):
        """Full frame + its list of (box, species) pairs, resized to a
        fixed square so batches build without per-image padding. Boxes get
        rescaled by the same x/y ratio as the image."""

        def __init__(self, samples, label_to_idx, img_size, augment=False):
            self.samples = samples
            self.label_to_idx = label_to_idx
            self.img_size = img_size
            self.augment = augment
            self.to_tensor = transforms.ToTensor()

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            img = Image.open(s["image_path"]).convert("RGB")
            orig_w, orig_h = img.size
            img_r = img.resize((self.img_size, self.img_size))

            sx, sy = self.img_size / orig_w, self.img_size / orig_h
            boxes = np.array(s["boxes"], dtype=np.float32).reshape(-1, 4)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

            # cheap augmentation: horizontal flip, boxes flipped to match --
            # skipped for eval loaders (augment=False)
            if self.augment and random.random() < 0.5:
                img_r = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                flipped = boxes.copy()
                flipped[:, 0] = self.img_size - boxes[:, 2]
                flipped[:, 2] = self.img_size - boxes[:, 0]
                boxes = flipped

            labels = np.array([self.label_to_idx[l] for l in s["labels"]], dtype=np.int64)
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }
            return self.to_tensor(img_r), target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_to_idx)
    model = build_species_detector(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=DET_LR, momentum=DET_MOMENTUM, weight_decay=DET_WEIGHT_DECAY)

    train_loader = DataLoader(
        DetectionFrameDataset(train_samples, label_to_idx, DET_IMG_SIZE, augment=True),
        batch_size=DET_BATCH_SIZE, shuffle=True, collate_fn=_detection_collate)
    val_loader = DataLoader(
        DetectionFrameDataset(val_samples, label_to_idx, DET_IMG_SIZE, augment=False),
        batch_size=DET_BATCH_SIZE, shuffle=False, collate_fn=_detection_collate)

    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        log(f"    [{tag}] epoch {epoch+1}/{epochs} - avg loss={running/max(n_batches,1):.3f}")
    train_time = time.time() - t0

    model.eval()
    preds_by_image, gts_by_image = {}, {}
    with torch.no_grad():
        img_counter = 0
        for images, targets in val_loader:
            images_dev = [img.to(device) for img in images]
            outputs = model(images_dev)
            for out, tgt in zip(outputs, targets):
                preds_by_image[img_counter] = {k: v.cpu().numpy() for k, v in out.items()}
                gts_by_image[img_counter] = {k: v.cpu().numpy() for k, v in tgt.items()}
                img_counter += 1

    class_ids = sorted(set(label_to_idx.values()))
    metrics = compute_detection_metrics(preds_by_image, gts_by_image, class_ids)
    metrics.update({
        "train_time_sec": round(train_time, 1),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "num_params": sum(p.numel() for p in model.parameters()),
    })
    log(f"    [{tag}] done: precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"mAP50={metrics['mAP50']:.4f} mAP50-95={metrics['mAP50_95']:.4f} "
        f"({train_time:.0f}s, {len(train_samples)} train / {len(val_samples)} val frames)")
    return metrics


def run_species_detection_single_split(samples, label_to_idx):
    """One grouped train/val split, no repeats -- deliberately mirrors the
    paper's own single-split protocol so this number can sit directly next
    to YOLOv9's reported Precision/Recall/mAP without any k-fold averaging
    smoothing things over. This is the 'eye to eye' comparison."""
    log("=" * 70)
    log("STEP 6a: single-split species detection (fair, eye-to-eye vs. the paper)")
    log("=" * 70)
    if len(samples) < 10:
        log(f"  Not enough frames ({len(samples)}) for a detection run. Skipping.")
        return None

    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as e:
        log(f"  WARNING: missing library ({e}). Skipping.")
        return None

    groups = [s["video_name"] for s in samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=DET_TEST_FRACTION, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gss.split(samples, groups=groups))
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    log(f"  video-level split: {len(set(groups[i] for i in train_idx))} train videos / "
        f"{len(set(groups[i] for i in val_idx))} val videos "
        f"({len(train_samples)} / {len(val_samples)} frames)")

    metrics = _train_eval_one_detector_split(train_samples, val_samples, label_to_idx,
                                              epochs=DET_EPOCHS, tag="single-split")
    if metrics is None:
        return None

    row = {"run": "single_split", "backbone": DETECTION_BACKBONE, **metrics}
    pd.DataFrame([row]).to_csv(os.path.join(MODEL_DIR, "single_split_species_detection.csv"), index=False)
    log("  Saved single_split_species_detection.csv")
    return metrics


def run_species_detection_kfold(samples, label_to_idx):
    """Stratified group k-fold on top of the exact same pipeline -- this is
    the part the paper's single-split methodology can't give you: does the
    single-split number above hold up across different train/val draws, or
    was it a lucky (or unlucky) split of the videos?

    Grouped by video (no frame from the same video ever appears in both
    train and val) and stratified on each frame's dominant species so class
    coverage stays reasonable across folds despite the species imbalance."""
    log("=" * 70)
    log(f"STEP 6b: {K_FOLDS}-fold species detection (our added statistical value)")
    log("=" * 70)
    if len(samples) < 10:
        log(f"  Not enough frames ({len(samples)}) for k-fold detection. Skipping.")
        return None

    from scipy import stats as spstats

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        stratified = True
    except ImportError:
        from sklearn.model_selection import GroupKFold
        stratified = False
        log("  NOTE: StratifiedGroupKFold not available in this sklearn version, "
            "falling back to plain GroupKFold (still no video leakage, just no class balancing).")

    groups = [s["video_name"] for s in samples]
    # one dominant label per frame purely for the stratifier -- training
    # still sees every box in the frame regardless of this simplification
    dominant_label = [s["labels"][0] for s in samples]

    class_counts = pd.Series(dominant_label).value_counts()
    min_class_count = int(class_counts.min())
    k_folds = min(K_FOLDS, min_class_count) if stratified else K_FOLDS
    if k_folds < K_FOLDS:
        log(f"  WARNING: smallest dominant-species group has only {min_class_count} frames; "
            f"reducing folds from {K_FOLDS} to {k_folds}.")
    if k_folds < 2:
        log("  Cannot run k-fold detection: not enough frames per class. Skipping.")
        return None

    if stratified:
        splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(samples, dominant_label, groups=groups)
    else:
        splitter = GroupKFold(n_splits=k_folds)
        split_iter = splitter.split(samples, groups=groups)

    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        train_videos = set(groups[i] for i in train_idx)
        val_videos = set(groups[i] for i in val_idx)
        assert train_videos.isdisjoint(val_videos), "video leaked across train/val -- this should never happen"

        log(f"  --- fold {fold_idx+1}/{k_folds}: {len(train_videos)} train videos / "
            f"{len(val_videos)} val videos ---")
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        metrics = _train_eval_one_detector_split(train_samples, val_samples, label_to_idx,
                                                  epochs=DET_EPOCHS, tag=f"fold {fold_idx+1}")
        if metrics is None:
            continue
        fold_rows.append({"fold": fold_idx, "backbone": DETECTION_BACKBONE, **metrics})

    if not fold_rows:
        log("  No fold completed successfully.")
        return None

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(MODEL_DIR, "kfold_results_species_detection.csv"), index=False)
    log(f"  Saved kfold_results_species_detection.csv ({len(fold_df)} folds)")

    summary = {"backbone": DETECTION_BACKBONE, "k_folds": len(fold_df)}
    for metric_name in ["precision", "recall", "mAP50", "mAP50_95"]:
        vals = fold_df[metric_name].values
        n = len(vals)
        mean_v, std_v = float(vals.mean()), float(vals.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            tcrit = float(spstats.t.ppf(0.975, df=n - 1))
            ci = tcrit * std_v / np.sqrt(n)
        else:
            ci = float("nan")
        summary[f"mean_{metric_name}"] = round(mean_v, 4)
        summary[f"std_{metric_name}"] = round(std_v, 4)
        summary[f"{metric_name}_95ci_halfwidth"] = round(ci, 4) if ci == ci else None

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(MODEL_DIR, "kfold_summary_species_detection.csv"), index=False)
    log("  Saved kfold_summary_species_detection.csv")
    log(f"  --- {k_folds}-fold species detection: mean +/- std across folds ---")
    for metric_name in ["precision", "recall", "mAP50", "mAP50_95"]:
        log(f"    {metric_name}: {summary[f'mean_{metric_name}']:.4f} +/- {summary[f'std_{metric_name}']:.4f} "
            f"(95% CI +/-{summary[f'{metric_name}_95ci_halfwidth']})")

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        metric_names = ["precision", "recall", "mAP50", "mAP50_95"]
        means = [summary[f"mean_{m}"] for m in metric_names]
        stds = [summary[f"std_{m}"] for m in metric_names]
        ax.bar(metric_names, means, yerr=stds, capsize=5, color="darkorange")
        ax.set_ylim(0, 1)
        ax.set_title(f"Species detection: {k_folds}-fold mean +/- std ({DETECTION_BACKBONE})")
        savefig(fig, "09_species_detection_kfold.png")
    except Exception as e:
        log(f"  WARNING: could not plot detection k-fold summary: {e}")

    return summary, fold_df


def build_detection_comparison_report(single_split_metrics, kfold_result):
    """Lines up all three numbers side by side: the paper's YOLOv9 result,
    our single-split MobileNetV2 detector (same protocol shape as the
    paper), and our k-fold mean +/- CI (our added statistical rigor). Also
    reports where the single-split point estimate falls relative to the
    k-fold spread, which is the concrete answer to 'did 5-fold change
    anything for us here.'"""
    log("=" * 70)
    log("STEP 6c: final species detection comparison (paper vs. our work)")
    log("=" * 70)

    rows = [{"source": "Paper (YOLOv9, full dataset)", "run": "reported", **PAPER_YOLOV9_METRICS}]

    if single_split_metrics is not None:
        rows.append({
            "source": f"Ours ({DETECTION_BACKBONE}, single split)",
            "run": "single_split",
            "precision": round(single_split_metrics["precision"], 4),
            "recall": round(single_split_metrics["recall"], 4),
            "mAP50": round(single_split_metrics["mAP50"], 4),
            "mAP50_95": round(single_split_metrics["mAP50_95"], 4),
        })

    if kfold_result is not None:
        summary, _ = kfold_result
        rows.append({
            "source": f"Ours ({DETECTION_BACKBONE}, {summary['k_folds']}-fold mean)",
            "run": "kfold_mean",
            "precision": summary["mean_precision"],
            "recall": summary["mean_recall"],
            "mAP50": summary["mean_mAP50"],
            "mAP50_95": summary["mean_mAP50_95"],
        })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(os.path.join(MODEL_DIR, "detection_comparison_vs_paper.csv"), index=False)
    log("  Saved detection_comparison_vs_paper.csv")
    log("  " + report_df.to_string(index=False))

    # the actual "what did 5-fold buy us" answer, in plain numbers
    if single_split_metrics is not None and kfold_result is not None:
        summary, fold_df = kfold_result
        for metric_name in ["mAP50", "mAP50_95"]:
            single_val = single_split_metrics[metric_name]
            fold_vals = fold_df[metric_name].values
            mean_v, std_v = fold_vals.mean(), fold_vals.std(ddof=1) if len(fold_vals) > 1 else 0.0
            z = (single_val - mean_v) / std_v if std_v > 0 else 0.0
            rank = int((fold_vals < single_val).sum()) + 1
            log(f"  IMPACT OF 5-FOLD on {metric_name}: single-split={single_val:.4f}, "
                f"fold mean={mean_v:.4f} (std={std_v:.4f}). Single-split sits {z:+.2f} "
                f"std-devs from the fold mean, ranking {rank}/{len(fold_vals)} among the "
                f"individual folds -- {'a fairly typical draw' if abs(z) < 1 else 'a noticeably lucky/unlucky draw'} "
                f"for this protocol.")

    log("  Read this table as: row 1 is what the dataset's authors measured on the full "
        "858-clip release; row 2 is us running the identical single-split idea on our "
        "sampled frames; row 3 is what a properly cross-validated estimate looks like. "
        "The gap between rows 2 and 3 is the whole point of doing 5-fold in the first place.")
    return report_df


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

    all_summaries = []
    all_pairwise = []
    for task in TASKS_TO_RUN:
        try:
            if task == "species":
                samples = build_species_samples(data)
            else:
                samples = build_behavior_samples(data)
            res = train_and_compare_models_kfold(samples, task_name=task)
            if res is not None:
                all_summaries.append(res["summary"])
                if res["pairwise"] is not None:
                    all_pairwise.append(res["pairwise"])
        except Exception as e:
            log(f"WARNING: modeling step failed for task '{task}': {e}")
            traceback.print_exc()

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(os.path.join(MODEL_DIR, "kfold_summary_ALL_TASKS.csv"), index=False)
        best_overall = combined.sort_values("mean_accuracy", ascending=False).iloc[0]
        log("=" * 70)
        log(f"OVERALL BEST MODEL (by mean k-fold accuracy): {best_overall['model']} "
            f"on task '{best_overall['task']}' with "
            f"acc={best_overall['mean_accuracy']:.4f} +/- {best_overall['std_accuracy']:.4f}")
        log("=" * 70)

    if all_pairwise:
        combined_pairwise = pd.concat(all_pairwise, ignore_index=True)
        combined_pairwise.to_csv(os.path.join(MODEL_DIR, "mcnemar_ALL_TASKS.csv"), index=False)
        n_sig = int(combined_pairwise[f"significant_at_{ALPHA}_holm"].sum())
        n_total = len(combined_pairwise)
        log(f"McNemar summary across all tasks: {n_sig}/{n_total} pairwise model comparisons "
            f"were statistically significant after Holm-Bonferroni correction (alpha={ALPHA}).")

    # ---- full-frame species detection: localization + recognition, the
    # part of the paper's YOLOv9 baseline the crop-classification benchmark
    # above deliberately couldn't touch -----------------------------------
    single_split_metrics, kfold_detection_result = None, None
    try:
        det_samples, det_label_to_idx = build_species_detection_samples(data)
        if det_samples and det_label_to_idx:
            single_split_metrics = run_species_detection_single_split(det_samples, det_label_to_idx)
            kfold_detection_result = run_species_detection_kfold(det_samples, det_label_to_idx)
            build_detection_comparison_report(single_split_metrics, kfold_detection_result)
        else:
            log("Skipping species detection: no usable full-frame samples were built.")
    except Exception as e:
        log(f"WARNING: species detection step failed: {e}")
        traceback.print_exc()

    try:
        with open(os.path.join(OUTPUT_DIR, "SUMMARY.txt"), "w", encoding="utf-8") as f:
            f.write("VISUAL WETLANDBIRDS - ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write("This run performed:\n")
            f.write("1. Data quality checks (missing values, duplicates, invalid/tiny boxes, orphan IDs, cross-file video_name consistency)\n")
            f.write("2. Statistical study of class balance (species from bounding_boxes.csv, behavior from crops.csv), clip length, co-occurrence, split balance\n")
            f.write(f"3. A stratified {K_FOLDS}-fold pretrained-CNN benchmark (capped at the smallest class's "
                     f"sample count per task) for species and behavior classification, reporting mean +/- std "
                     f"accuracy/macro-F1 across folds instead of a single train/val split\n")
            f.write("4. Out-of-fold (OOF) predictions saved per sample per model, and pairwise McNemar's tests "
                     "(exact binomial below 25 discordant pairs, continuity-corrected chi-square otherwise) "
                     "between every pair of models, with Holm-Bonferroni correction across the resulting "
                     "family of 6 tests per task\n")
            f.write(f"5. A full-frame species DETECTION benchmark (localization + recognition, not just "
                     f"classification) using a Faster R-CNN with a MobileNetV2 backbone -- run once as a "
                     f"single grouped train/val split (directly comparable to the paper's own single-split "
                     f"YOLOv9 protocol), then again as {K_FOLDS}-fold group cross-validation, so the report "
                     f"can show exactly what k-fold changes versus a one-shot split\n\n")
            f.write("See data_quality_report.csv, class_balance_summary.csv, figures/, and model_results/\n")
            f.write("(kfold_results_*.csv = per-fold rows, kfold_summary_*.csv = mean+/-std per model, "
                     "oof_predictions_*.csv = one prediction per sample per model, "
                     "mcnemar_*.csv = pairwise significance tests, "
                     "single_split_species_detection.csv / kfold_*_species_detection.csv = detection results, "
                     "detection_comparison_vs_paper.csv = paper vs. our single-split vs. our k-fold) for details.\n")
            f.write("run_log.txt has the full console log of this run.\n")
        log("Saved SUMMARY.txt")
    except Exception as e:
        log(f"WARNING: could not save summary: {e}")

    total_time = time.time() - t_start
    log(f"DONE. Total run time: {total_time/60:.1f} minutes.")
    log(f"All results saved under: {OUTPUT_DIR}")
    save_log()


if __name__ == "__main__":
    main()
