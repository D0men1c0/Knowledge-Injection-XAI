"""
Configuration management.

Loads settings from YAML files into hierarchical Dataclasses.
Ensures type safety and autocompletion for nested configurations.
"""
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any


@dataclass(frozen=True)
class SparkConfig:
    """Spark Session parameters."""
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
        """Returns Spark configuration dictionary."""
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
class PathsConfig:
    """Project directory paths."""
    input: Path
    bronze: str
    silver: str
    gold: str
    logs: str


@dataclass(frozen=True)
class ModelConfig:
    """Base model parameters."""
    backbone_name: str
    batch_size: int
    patch_size: int


@dataclass(frozen=True)
class AdapterConfig:
    """LoRA Adapter Zoo configuration."""
    ranks: List[int]
    alpha: int
    dropout: float
    target_modules: List[str]


@dataclass(frozen=True)
class XAIConfig:
    """XAI Metric parameters."""
    perturbation_steps: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Main configuration container."""
    spark: SparkConfig
    paths: PathsConfig
    model: ModelConfig
    adapters: AdapterConfig
    xai: XAIConfig


def load_config(path: str = "configuration/config.yaml") -> ExperimentConfig:
    """
    Parses the YAML configuration file into typed dataclasses.

    Args:
        path: Path to the .yaml file.

    Returns:
        ExperimentConfig: Fully populated config object.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Validate and instantiate nested configs
    return ExperimentConfig(
        spark=SparkConfig(**raw["spark"]),
        paths=PathsConfig(
            input=Path(raw["paths"]["input"]),
            bronze=raw["paths"]["bronze"],
            silver=raw["paths"]["silver"],
            gold=raw["paths"]["gold"],
            logs=raw["paths"]["logs"]
        ),
        model=ModelConfig(**raw["model"]),
        adapters=AdapterConfig(**raw["adapters"]),
        xai=XAIConfig(**raw["xai"])
    )