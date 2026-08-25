from pathlib import Path

from wetlandbirds.classification.cross_validation import run_kfold
from wetlandbirds.config import Config
from wetlandbirds.data.crops import build_behavior_samples, build_species_samples
from wetlandbirds.data.loaders import load_dataset
from wetlandbirds.logging_utils import configure_logging
from wetlandbirds.scripts_helpers import set_seed


def main():
    cfg = Config.from_yaml("configs/config.yaml")
    set_seed(cfg.seed)
    logger = configure_logging(cfg.path("output_root") / "logs")
    bundle = load_dataset(cfg.path("dataset_root"))
    video_dirs = [cfg.path("videos_dir")]
    crop_cfg = cfg.raw["classification"]
    output = cfg.path("output_root") / "classification"
    for task in cfg.raw["experiment"]["tasks"]:
        if task == "species":
            samples = build_species_samples(bundle, video_dirs, output / "crops" / "species", int(crop_cfg["samples_per_class"]), int(crop_cfg["max_total_samples"]), cfg.seed)
        elif task == "behavior":
            samples = build_behavior_samples(bundle, video_dirs, output / "crops" / "behavior", int(crop_cfg["samples_per_class"]), int(crop_cfg["max_total_samples"]), cfg.seed)
        else:
            raise ValueError(f"Unknown task: {task}")
        logger.info("%s: prepared %d labeled crops", task, len(samples))
        result = run_kfold(samples, task, cfg.raw["experiment"]["models"], {**crop_cfg, "folds": cfg.raw["experiment"]["folds"], "alpha": cfg.raw["experiment"]["alpha"]}, output / task, cfg.seed)
        result["summary"].to_csv(output / f"summary_{task}.csv", index=False)


if __name__ == "__main__":
    main()
