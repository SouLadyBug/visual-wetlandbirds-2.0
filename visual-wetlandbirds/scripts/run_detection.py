from wetlandbirds.config import Config
from wetlandbirds.data.loaders import load_dataset
from wetlandbirds.detection.pipeline import build_detection_samples, run_detection
from wetlandbirds.scripts_helpers import set_seed

cfg = Config.from_yaml("configs/config.yaml")
set_seed(cfg.seed)
bundle = load_dataset(cfg.path("dataset_root"))
out = cfg.path("output_root") / "detection"
samples, label_to_idx = build_detection_samples(bundle, cfg.path("videos_dir"), out / "frames", int(cfg.raw["detection"]["frames_per_video"]), int(cfg.raw["detection"]["max_frames"]), cfg.seed)
run_detection(samples, label_to_idx, {**cfg.raw["detection"], "folds": cfg.raw["experiment"]["folds"]}, out, cfg.seed)
print(f"Saved detection outputs to {out}")
