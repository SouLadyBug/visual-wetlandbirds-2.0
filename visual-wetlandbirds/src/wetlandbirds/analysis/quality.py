from __future__ import annotations

from pathlib import Path

from ..data.loaders import DatasetBundle
from ..data.validation import quality_report
from .statistics import class_balance, clip_lengths, split_balance
from .visualization import save_class_bar


def run_analysis(bundle: DatasetBundle, output_dir: Path, bbox_sample_size: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = quality_report(bundle, bbox_sample_size=bbox_sample_size, seed=seed)
    report.to_csv(output_dir / "data_quality_report.csv", index=False)

    balances = class_balance(bundle)
    for name, table in balances.items():
        table.to_csv(output_dir / f"{name}_balance.csv", index=False)
        label = table.columns[0]
        save_class_bar(table, label, f"{name.title()} distribution", output_dir / f"{name}_distribution.png")

    lengths = clip_lengths(bundle)
    lengths.to_csv(output_dir / "clip_lengths.csv", index=False)
    split_balance(bundle).to_csv(output_dir / "split_balance.csv", index=False)
