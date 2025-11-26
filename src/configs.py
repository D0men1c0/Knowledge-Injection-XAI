"""
Configuration management.

Loads settings from YAML files into typed Dataclasses for safety and autocompletion.
"""
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class SparkConfig:
    """Spark Session resource parameters."""
    app_name: str
    master: str
    driver_mem: str
    executor_mem: str
    off_heap_enabled: str
    off_heap_size: str
    network_timeout: str
    heartbeat_interval: str
    max_result_size: str
    arrow_enabled: str

    def to_dict(self) -> Dict[str, str]:
        """Maps config attributes to Spark property keys."""
        return {
            "spark.driver.memory": self.driver_mem,
            "spark.executor.memory": self.executor_mem,
            "spark.memory.offHeap.enabled": self.off_heap_enabled,
            "spark.memory.offHeap.size": self.off_heap_size,
            "spark.network.timeout": self.network_timeout,
            "spark.executor.heartbeatInterval": self.heartbeat_interval,
            "spark.driver.maxResultSize": self.max_result_size,
            "spark.sql.execution.arrow.pyspark.enabled": self.arrow_enabled,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Pipeline experiment parameters."""
    backbone_name: str
    batch_size: int
    input_path: Path
    output_path: str
    log_file: str
    spark: SparkConfig


def load_config(path: str = "config.yaml") -> ExperimentConfig:
    """
    Parses the YAML configuration file.

    Args:
        path: Path to the .yaml file.

    Returns:
        ExperimentConfig: Populated configuration object.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    spark_conf = SparkConfig(**raw_config["spark"])
    
    # Convert input_path string to Path object here
    exp_data = raw_config["experiment"]
    return ExperimentConfig(
        backbone_name=exp_data["backbone_name"],
        batch_size=exp_data["batch_size"],
        input_path=Path(exp_data["input_path"]),
        output_path=exp_data["output_path"],
        log_file=exp_data["log_file"],
        spark=spark_conf
    )