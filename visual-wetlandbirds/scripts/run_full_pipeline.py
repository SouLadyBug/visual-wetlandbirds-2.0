from pathlib import Path

from wetlandbirds.analysis.quality import run_analysis
from wetlandbirds.config import Config
from wetlandbirds.data.loaders import load_dataset
from wetlandbirds.detection.pipeline import build_detection_samples, run_detection
from wetlandbirds.logging_utils import configure_logging
from wetlandbirds.scripts_helpers import set_seed


def main():
    cfg = Config.from_yaml("configs/config.yaml")
    set_seed(cfg.seed)
    out = cfg.path("output_root")
    logger = configure_logging(out / "logs")
    logger.info("Starting Visual WetlandBirds pipeline")
    bundle = load_dataset(cfg.path("dataset_root"))
    run_analysis(bundle, out / "data_quality", int(cfg.raw["analysis"]["bbox_stats_sample_size"]), cfg.seed)
    logger.info("Data-quality and EDA stage completed")
    logger.info("Classification is kept as a separate entry point until crop-sample construction is explicitly configured")
    det_cfg = {**cfg.raw["detection"], "folds": cfg.raw["experiment"]["folds"]}
    samples, label_to_idx = build_detection_samples(bundle, cfg.path("videos_dir"), out / "detection" / "frames", int(det_cfg["frames_per_video"]), int(det_cfg["max_frames"]), cfg.seed)
    if samples:
        run_detection(samples, label_to_idx, det_cfg, out / "detection", cfg.seed)
    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
