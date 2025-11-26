"""
Silver Layer Entry Point.

Loads Bronze Layer output (Parquet), applies LoRA adapters,
computes XAI metrics, and logs performance telemetry.
"""
import os
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.errors import AnalysisException

from src.configs import load_config, ExperimentConfig
from src.pipeline.silver_layer import run_silver
from src.utils.telemetry import ResourceMonitor, ExperimentLogger, logger


def setup_windows_hadoop() -> None:
    """Sets up Hadoop environment variables for Windows execution."""
    hadoop_home = Path("C:/hadoop")
    if hadoop_home.exists() and (hadoop_home / "bin/hadoop.dll").exists():
        os.environ['HADOOP_HOME'] = str(hadoop_home)
        sys.path.append(str(hadoop_home / "bin"))
        os.environ['PATH'] += os.pathsep + str(hadoop_home / "bin")
    else:
        logger.warning("Hadoop Utils not found. Parquet write may fail.")


def build_spark_session(config: ExperimentConfig) -> SparkSession:
    """Configures and builds the Spark Session."""
    builder = SparkSession.builder.appName("KnowledgeInjection_Silver").master(config.spark.master)
    
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

    # 2. Configuration Loading
    try:
        cfg = load_config("configuration/config.yaml")
        logger.info(f"Loaded config. Batch: {cfg.model.batch_size}, Device: {cfg.spark.master}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # 3. Initialize Spark
    spark = build_spark_session(cfg)

    # 4. Load Bronze Data (Input for Silver Layer)
    bronze_path = cfg.paths.bronze
    logger.info(f"Reading Bronze Layer data from: {bronze_path}")
    
    try:
        df_bronze = spark.read.parquet(bronze_path)
        input_count = df_bronze.count()
        logger.info(f"Loaded {input_count} records from Bronze Layer.")
        
        if input_count == 0:
            logger.error("Bronze dataset is empty. Run Bronze Layer first.")
            spark.stop()
            return
            
    except AnalysisException:
        logger.error(f"Bronze data not found at {bronze_path}. Run 'src.run_pipeline' first.")
        spark.stop()
        return

    # 5. Execution & Monitoring
    # Silver layer is computationally intensive (multiple adapters per image)
    monitor = ResourceMonitor(interval=1.0)
    monitor.start()
    start_time = time.time()

    try:
        # Execute Distributed Inference & XAI
        run_silver(spark, df_bronze, cfg)
        
        duration = time.time() - start_time
        monitor.stop()
        monitor.join()

        # 6. Logging & Verification
        stats = monitor.get_stats()
        
        # Read the generated Silver output
        silver_path = cfg.paths.silver
        df_result = spark.read.parquet(silver_path)
        
        # Extract metadata for logging
        device_row = df_result.select("device").limit(1).collect()
        device_used = device_row[0]["device"] if device_row else "Unknown"

        # Log System Performance
        exp_logger = ExperimentLogger(cfg.paths.logs)
        exp_logger.log(
            duration, input_count, cfg.model.batch_size, device_used,
            stats, cfg.spark.to_dict()
        )

        logger.info("--- Top 5 Silver Layer Records ---")
        df_result.select("image_path", "adapter_rank", "entropy", "deletion_score").show(5, truncate=False)

    except Exception as e:
        logger.critical(f"Silver Layer crashed: {e}", exc_info=True)
        monitor.stop()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()