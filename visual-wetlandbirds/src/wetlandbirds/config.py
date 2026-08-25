from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw=raw, root=path.parent.parent)

    def path(self, key: str) -> Path:
        value = self.raw["paths"][key]
        p = Path(value)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def seed(self) -> int:
        return int(self.raw["experiment"]["seed"])
