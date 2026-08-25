# Visual WetlandBirds — Reproducible Benchmark and Statistical Analysis

## Status

This repository is the cleaned research-codebase derived from three supplied analysis scripts. The source scripts contain data-quality analysis, crop-based CNN benchmarking, cross-validation, OOF predictions, McNemar/Holm-Bonferroni testing, and full-frame Faster R-CNN detection.


## Research focus

The project is organized around four connected questions:

1. What statistical and annotation properties characterize Visual WetlandBirds?
2. How do lightweight pretrained CNNs perform on crop-based species and behavior classification?
3. Are apparent differences between classifiers supported by paired out-of-fold statistical testing?
4. How does a lightweight full-frame species detector behave under video-grouped evaluation, and how should that result be interpreted relative to the published YOLOv9 baseline?

## Methodology

```text
Dataset
   ↓
Schema + data-quality validation
   ↓
Exploratory statistical analysis
   ↓
Crop-based classification benchmark
   ↓
Grouped / stratified validation where appropriate
   ↓
Out-of-fold predictions
   ↓
McNemar + Holm-Bonferroni
   ↓
Full-frame species detection
   ↓
Video-grouped single split + grouped cross-validation
```

## Dataset

The expected dataset directory contains:

- `bounding_boxes.csv` — species/frame/bounding-box annotations.
- `crops.csv` — video/bird/action temporal annotations.
- `species_ID.csv` — species ID mapping.
- `behaviors_ID.csv` — behavior/action mapping.
- `splits.json` — official video-level train/validation/test lists.
- `videos/` — raw `.mp4` files, when required by the crop/detection pipeline.

The repository keeps species information and behavior information separate. It does not infer behavior labels from `bounding_boxes.csv`.

## Repository structure

```text
visual-wetlandbirds/
├── configs/
├── data/
├── docs/
├── results/
├── reports/
├── legacy/
├── scripts/
├── src/wetlandbirds/
└── tests/
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,vision]"
```

## Dataset setup

Place the dataset files under `data/` or change `paths.dataset_root` in `configs/config.yaml`.

Do not commit the raw dataset or video files to Git.

## Running data-quality analysis

```bash
python scripts/run_quality_analysis.py
```

Outputs are written to `results/data_quality/`.

## Running detection

```bash
python scripts/run_detection.py
```

The detection pipeline uses video-level grouping for its train/validation procedures. It reports a single grouped split and grouped cross-validation.

## Classification

The supplied scripts construct crop samples by reading video frames and bounding-box annotations. That sample-construction rule should be made an explicit configurable stage before running the final classification benchmark. The current entry point intentionally stops rather than silently inventing a different sample-selection protocol.

## Reproducibility

The default seed is `42`. Experiment settings live in `configs/config.yaml`. Results should be treated as protocol-specific estimates, not universal model rankings.


