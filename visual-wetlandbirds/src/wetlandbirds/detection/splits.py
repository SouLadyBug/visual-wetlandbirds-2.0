from __future__ import annotations

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def grouped_holdout(samples, fraction: float, seed: int):
    groups = [s["video_name"] for s in samples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    return next(splitter.split(samples, groups=groups))


def grouped_folds(samples, labels, folds: int, seed: int):
    groups = [s["video_name"] for s in samples]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return splitter.split(samples, labels, groups=groups)
