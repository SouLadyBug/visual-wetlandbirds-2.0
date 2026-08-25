from __future__ import annotations

from pathlib import Path


def build_video_index(video_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for directory in video_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                index.setdefault(path.name, path)
        if index:
            break
    return index


def resolve_video_path(video_name: str, index: dict[str, Path]) -> Path | None:
    target = Path(str(video_name)).name.strip()
    if target in index:
        return index[target]
    lower = target.lower()
    for name, path in index.items():
        if name.lower() == lower:
            return path
    if Path(target).suffix == "":
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            for name, path in index.items():
                if name.lower() == (lower + ext):
                    return path
    return None
