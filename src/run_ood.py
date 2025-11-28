"""
OOD Layer: Docker-Optimized Local Cache

- Copies large input files/models to /tmp for fast I/O.
- Executes Spark + Torch computations.
- Cleans up /tmp cache automatically.
- Preserves bind mount outputs for Windows visibility.
"""

import os
import time
import shutil
from pathlib import Path

from pyspark.sql import SparkSession

from src.configs import load_config, ExperimentConfig
from src.pipeline.ood_layer import run_ood
from src.utils.telemetry import ResourceMonitor, ExperimentLogger, logger


def setup_hadoop_windows() -> None:
    if os.name == "nt":
        hadoop_home = Path("C:/hadoop")
        if hadoop_home.exists():
            os.environ["HADOOP_HOME"] = str(hadoop_home)


def build_spark(cfg: ExperimentConfig) -> SparkSession:
    builder = SparkSession.builder.appName(cfg.spark.app_name).master(cfg.spark.master)
    for k, v in cfg.spark.to_dict().items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def cache_to_tmp(src: Path, dst: Path) -> None:
    if not src.exists():
        logger.warning(f"Source path {src} does not exist. Skipping cache.")
        return
    logger.info(f"Caching {src.name} to fast local storage...")
    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to cache {src}: {e}")


def main() -> None:
    setup_hadoop_windows()
    try:
        cfg = load_config("configuration/config.yaml")
    except Exception as e:
        logger.error(f"Config load error: {e}")
        return

    FAST_ROOT = Path("/tmp/ki_xai_cache")
    fast_images = FAST_ROOT / "source"
    fast_models = FAST_ROOT / "adapters"

    slow_images = Path(cfg.paths.input)
    slow_models = Path(cfg.paths.models)

    try:
        cache_to_tmp(slow_images, fast_images)
        cache_to_tmp(slow_models, fast_models)

        object.__setattr__(cfg.paths, "input", str(fast_images))
        object.__setattr__(cfg.paths, "models", str(fast_models))

        spark = build_spark(cfg)

        monitor = ResourceMonitor(interval=1.0)
        monitor.start()
        start_time = time.time()

        run_ood(spark, cfg)

        duration = time.time() - start_time
        monitor.stop()
        monitor.join()

        df_result = spark.read.parquet(cfg.paths.ood)
        count = df_result.count()
        device_used = df_result.select("device").limit(1).collect()[0]["device"] if count else "Unknown"

        exp_logger = ExperimentLogger(cfg.paths.logs)
        exp_logger.log(duration, count, cfg.model.batch_size, device_used, monitor.get_stats(), cfg.spark.to_dict())

        logger.info("--- OOD Layer Sample ---")
        df_result.select("image_path", "corruption_type", "adapter_rank", "is_correct").show(5, truncate=False)

    except Exception as e:
        logger.critical(f"OOD Layer Failed: {e}", exc_info=True)
        if "monitor" in locals():
            monitor.stop()

    finally:
        if "spark" in locals():
            spark.stop()
        if FAST_ROOT.exists():
            try:
                shutil.rmtree(FAST_ROOT)
                logger.info("Temporary /tmp cache removed.")
            except Exception as e:
                logger.warning(f"Could not remove /tmp cache: {e}")


if __name__ == "__main__":
    main()
