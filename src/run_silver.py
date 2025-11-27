"""
Silver Layer: Docker-Optimized Local Cache

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
from pyspark.sql.functions import regexp_replace, col

from src.configs import load_config, ExperimentConfig
from src.pipeline.silver_layer import run_silver
from src.utils.telemetry import ResourceMonitor, ExperimentLogger, logger


def setup_hadoop_windows() -> None:
    """Setup HADOOP_HOME if running on Windows."""
    if os.name == 'nt':
        hadoop_home = Path("C:/hadoop")
        if hadoop_home.exists():
            os.environ['HADOOP_HOME'] = str(hadoop_home)


def build_spark(cfg: ExperimentConfig) -> SparkSession:
    """Initialize SparkSession with project config."""
    builder = SparkSession.builder.appName(cfg.spark.app_name).master(cfg.spark.master)
    for k, v in cfg.spark.to_dict().items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def cache_to_tmp(src: Path, dst: Path) -> None:
    """Copy data/models to /tmp quickly; overwrite only if necessary."""
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
    fast_bronze = FAST_ROOT / "bronze_parquet"
    fast_images = FAST_ROOT / "source"
    fast_models = FAST_ROOT / "adapters"

    slow_bronze = Path(cfg.paths.bronze)
    slow_images = Path(cfg.paths.input)
    slow_models = Path(cfg.paths.models)

    try:
        # Cache large assets to /tmp
        cache_to_tmp(slow_bronze, fast_bronze)
        cache_to_tmp(slow_images, fast_images)
        cache_to_tmp(slow_models, fast_models)

        # Update config paths for fast runtime
        object.__setattr__(cfg.paths, 'bronze', str(fast_bronze))
        object.__setattr__(cfg.paths, 'input', str(fast_images))
        object.__setattr__(cfg.paths, 'models', str(fast_models))

        spark = build_spark(cfg)

        df_bronze = spark.read.parquet(str(fast_bronze))
        old_path = str(slow_images).replace("\\", "/")
        new_path = str(fast_images).replace("\\", "/")
        if "/app/" not in old_path:
            old_path = "/app/data/raw/source"

        df_bronze = df_bronze.withColumn(
            "image_path",
            regexp_replace(col("image_path"), old_path, new_path)
        )

        # Optional quick check
        count = df_bronze.count()
        logger.info(f"Processing {count} records from fast cache.")

        monitor = ResourceMonitor(interval=1.0)
        monitor.start()
        start_time = time.time()

        run_silver(spark, df_bronze, cfg)

        duration = time.time() - start_time
        monitor.stop()
        monitor.join()

        df_result = spark.read.parquet(cfg.paths.silver)
        device_used = df_result.select("device").limit(1).collect()[0]["device"] if df_result.count() else "Unknown"

        exp_logger = ExperimentLogger(cfg.paths.logs)
        exp_logger.log(duration, count, cfg.model.batch_size, device_used, monitor.get_stats(), cfg.spark.to_dict())

        logger.info("--- Silver Layer Sample ---")
        df_result.select("image_path", "adapter_rank", "entropy", "deletion_score", "true_label", "predicted_class").show(5, truncate=False)

    except Exception as e:
        logger.critical(f"Silver Layer Failed: {e}", exc_info=True)
        if 'monitor' in locals(): monitor.stop()

    finally:
        if 'spark' in locals(): spark.stop()
        if FAST_ROOT.exists():
            try:
                shutil.rmtree(FAST_ROOT)
                logger.info("Temporary /tmp cache removed.")
            except Exception as e:
                logger.warning(f"Could not remove /tmp cache: {e}")


if __name__ == "__main__":
    main()