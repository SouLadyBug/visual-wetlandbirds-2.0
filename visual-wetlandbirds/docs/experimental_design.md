# Experimental design

## Classification

The supplied implementation evaluates ResNet18, MobileNetV2, EfficientNet-B0, and DenseNet121 using frozen pretrained feature extractors with a newly initialized classification head. The default configuration retains the original project's conservative 5-epoch benchmark scale.

A key unresolved issue is grouping: the supplied classification implementation uses `StratifiedKFold` over crop samples, not video groups. If multiple samples originate from the same video, fold independence is not guaranteed. This must be corrected or explicitly justified before final scientific claims are made.

## Detection

The full-frame task is distinct from crop classification. Detection predicts both the location and species of birds. The repository uses Faster R-CNN with a MobileNetV2 feature backbone, with video-grouped holdout and grouped cross-validation.

## Statistical comparison

McNemar's test is applied to paired out-of-fold classification predictions. Holm-Bonferroni correction is applied within the family of pairwise comparisons for each task.

## Literature comparison

Published YOLOv9 values are external reference values. They are never treated as locally generated results. Exact comparability depends on matching dataset version, split, preprocessing, detector implementation, and metric definition.
