from pathlib import Path

from wetlandbirds.analysis.quality import run_analysis
from wetlandbirds.config import Config
from wetlandbirds.data.loaders import load_dataset
from wetlandbirds.scripts_helpers import set_seed

cfg = Config.from_yaml("configs/config.yaml")
set_seed(cfg.seed)
bundle = load_dataset(cfg.path("dataset_root"))
out = cfg.path("output_root") / "data_quality"
run_analysis(bundle, out, int(cfg.raw["analysis"]["bbox_stats_sample_size"]), cfg.seed)
print(f"Saved analysis outputs to {out}")
