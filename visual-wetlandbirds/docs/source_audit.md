# Source audit

This audit records what was found in the three supplied Python files before refactoring. The original files are preserved under `legacy/` for traceability.

## Version progression

### `wetlandbirds_analysis (3).py`

This is the earliest supplied version. It contains the data-quality/EDA pipeline, crop construction, and a single-split CNN comparison. It has approximately 1,182 lines and 30 functions.

### `wetlandbirds_analysis2.py`

This version adds the statistical-comparison layer: five-fold CNN evaluation, out-of-fold predictions, McNemar testing, and Holm-Bonferroni correction. It has approximately 1,424 lines and 32 functions.

### `wetlandbirds_analysis3.py`

This version incorporates the strongest parts of the previous two and adds full-frame species detection with Faster R-CNN/MobileNetV2, grouped detection evaluation, and a paper-comparison report. It has approximately 2,151 lines and 46 functions.

## Keep

- Bounding-box parsing.
- Explicit separation of species annotations (`bounding_boxes.csv`) and behavior annotations (`crops.csv`).
- Dataset-quality checks and class-balance analysis.
- Crop construction from actual video frames.
- ResNet18, MobileNetV2, EfficientNet-B0, and DenseNet121 benchmark.
- Out-of-fold predictions.
- McNemar + Holm-Bonferroni analysis, subject to a correct paired evaluation set.
- Full-frame detection as a distinct task.
- Video-grouped detection splits.

## Modify

### 1. Classification fold construction

The supplied classification implementation uses `StratifiedKFold` over individual crop samples. Because each sample carries a source video, this can place samples from the same video into both training and validation folds. The refactored repository therefore uses `StratifiedGroupKFold` for classification whenever video provenance is available.

### 2. Configuration

The source uses a fixed `C:\\stage` directory and writes all outputs there. The refactored repository moves paths and experiment settings to `configs/config.yaml`.

### 3. Error handling

The source catches broad exceptions around major stages and continues. The refactored pipeline raises fatal data/configuration errors instead of presenting a partial run as complete.

### 4. Logging

The source maintains a global list of log strings. The refactored repository uses Python's `logging` module with console and file handlers.

### 5. Detection evaluation

The source implements its own AP calculation. This is retained as an explicitly local metric implementation, but the repository does not silently claim that it is identical to the metric implementation used for the published YOLOv9 numbers. Exact metric/protocol verification remains a prerequisite for strong quantitative paper comparisons.

### 6. Detection sampling

The source samples a capped number of frames per video, then performs video-grouped evaluation. The refactored implementation preserves this basic strategy while making it a configurable stage.

## Remove

- Machine-specific absolute paths.
- Duplicated data loaders.
- Global output state.
- Decorative/overly verbose comments.
- Silent fallback from failed experiments to partial results.
- The implication that sample-level classification CV is automatically leakage-free.

## Important unresolved issue

The supplied report was not among the three files available for this build. Therefore this repository deliberately does not claim that its README or methodology exactly matches the report. Add the report before the final report-to-code consistency audit.
