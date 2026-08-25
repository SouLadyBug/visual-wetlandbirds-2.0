# Limitations

1. The supplied report was not available in the workspace used to construct this repository.
2. Crop classification in the original code uses sample-level stratification; shared video provenance may compromise independence.
3. The detection metrics are implemented locally and should not be assumed identical to the metric implementation used for the published YOLOv9 baseline.
4. The supplied experiments intentionally use small training budgets and sampled frames/crops for practical runtime. They are controlled benchmarks, not exhaustive hyperparameter searches.
5. Literature baseline values are external references and require exact source/protocol verification before quantitative claims of superiority or inferiority.
