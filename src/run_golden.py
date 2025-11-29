"""
Gold Layer: Docker-Optimized Local Cache

- Copies large input files to /tmp for fast I/O.
- Executes Spark + sklearn computations.
- Cleans up /tmp cache automatically.
"""

import os
import time
import shutil
from pathlib import Path

from pyspark.sql import SparkSession

from src.configs import load_config, ExperimentConfig
from src.pipeline.golden_layer import run_gold
from src.utils.telemetry import ResourceMonitor, ExperimentLogger, logger


def setup_hadoop_windows() -> None:
    if os.name == "nt":
        hadoop_home = Path("C:/hadoop")
        if hadoop_home.exists():
            os.environ["HADOOP_HOME"] = str(hadoop_home)


def build_spark(cfg: ExperimentConfig) -> SparkSession:
    builder = SparkSession.builder.appName(cfg.spark.app_name + "_Gold").master(cfg.spark.master)
    for k, v in cfg.spark.to_dict().items():
        builder = builder.config(k, v)
    # Arrow optimization
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    builder = builder.config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
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
    fast_silver = FAST_ROOT / "silver_parquet"
    fast_ood = FAST_ROOT / "ood_parquet"

    slow_silver = Path(cfg.paths.silver)
    slow_ood = Path(cfg.paths.ood)

    try:
        cache_to_tmp(slow_silver, fast_silver)
        cache_to_tmp(slow_ood, fast_ood)

        object.__setattr__(cfg.paths, "silver", str(fast_silver))
        object.__setattr__(cfg.paths, "ood", str(fast_ood))

        spark = build_spark(cfg)

        monitor = ResourceMonitor(interval=1.0)
        monitor.start()
        start_time = time.time()

        run_gold(spark, cfg)

        duration = time.time() - start_time
        monitor.stop()
        monitor.join()

        exp_logger = ExperimentLogger(cfg.paths.logs)
        exp_logger.log(duration, 0, cfg.model.batch_size, "cpu", monitor.get_stats(), cfg.spark.to_dict())

        logger.info(f"Gold Layer completed in {duration:.2f}s")

    except Exception as e:
        logger.critical(f"Gold Layer Failed: {e}", exc_info=True)
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