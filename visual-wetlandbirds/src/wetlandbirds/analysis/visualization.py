from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_class_bar(counts: pd.DataFrame, label_col: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts[label_col].astype(str), counts["count"])
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
