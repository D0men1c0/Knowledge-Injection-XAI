"""
Pipeline Entry Point (Bronze Layer).

Loads YAML config, sets up environment, and executes the Bronze Layer extraction.
"""
import os
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession

from src.configs import load_config, ExperimentConfig
from src.pipeline.bronze_layer import run_bronze
from src.utils.telemetry import ResourceMonitor, ExperimentLogger, logger


def setup_windows_hadoop() -> None:
    """Sets up Hadoop environment for Windows."""
    hadoop_home = Path("C:/hadoop")
    if hadoop_home.exists() and (hadoop_home / "bin/hadoop.dll").exists():
        os.environ['HADOOP_HOME'] = str(hadoop_home)
        sys.path.append(str(hadoop_home / "bin"))
        os.environ['PATH'] += os.pathsep + str(hadoop_home / "bin")
    else:
        logger.warning("Hadoop Utils not found. Parquet write may fail.")


def build_spark_session(config: ExperimentConfig) -> SparkSession:
    """Builds Spark Session from config object."""
    builder = SparkSession.builder.appName(config.spark.app_name).master(config.spark.master)
    
    for key, value in config.spark.to_dict().items():
        builder = builder.config(key, value)
        
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def main() -> None:
    # 1. Environment Setup
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    setup_windows_hadoop()

    # 2. Config Loading
    try:
        cfg = load_config("configuration/config.yaml")
        logger.info(f"Loaded config. Batch: {cfg.model.batch_size}, Master: {cfg.spark.master}")
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        return

    # 3. Spark Init
    spark = build_spark_session(cfg)
    
    # 4. Data Preparation
    # Recursive search for images in the configured input path
    paths = [str(p.absolute()) for p in cfg.paths.input.rglob("*.jpg")]
    
    if not paths:
        logger.error(f"No images found in {cfg.paths.input}")
        spark.stop()
        return

    df_input = spark.createDataFrame([(p,) for p in paths], schema=["image_path"])
    logger.info(f"Pipeline Input: {len(paths)} images.")

    # 5. Execution
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    start_time = time.time()

    try:
        # Pass the entire config object to run_bronze
        run_bronze(spark, df_input, cfg)
        
        duration = time.time() - start_time
        monitor.stop()
        monitor.join()

        # 6. Logging & Verification
        stats = monitor.get_stats()
        
        df_result = spark.read.parquet(cfg.paths.bronze)
        device_row = df_result.select("device").limit(1).collect()
        device_used = device_row[0]["device"] if device_row else "Unknown"

        exp_logger = ExperimentLogger(cfg.paths.logs)
        exp_logger.log(
            duration, len(paths), cfg.model.batch_size, device_used,
            stats, cfg.spark.to_dict()
        )

        logger.info("--- Top 5 Bronze Records ---")
        df_result.select("image_path", "device").show(5, truncate=False)

    except Exception as e:
        logger.critical(f"Pipeline crashed: {e}", exc_info=True)
        monitor.stop()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()