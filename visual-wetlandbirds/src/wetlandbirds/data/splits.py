from __future__ import annotations

from collections import defaultdict


def make_split_map(splits: dict[str, list[str]]) -> dict[str, str]:
    mapping = {}
    for split_name, videos in splits.items():
        for video in videos:
            if video in mapping and mapping[video] != split_name:
                raise ValueError(f"Video appears in multiple official splits: {video}")
            mapping[video] = split_name
    return mapping
