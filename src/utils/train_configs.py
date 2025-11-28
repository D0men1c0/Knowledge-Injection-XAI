"""
Training Configuration Management.

Defines the structure for loading training-specific configurations
from YAML files, separating them from the general pipeline config.
"""
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for data augmentation strategies."""
    enable: bool
    rotation_degrees: int
    horizontal_flip_prob: float
    color_jitter_brightness: float
    color_jitter_contrast: float
    resize_size: int
    crop_size: int


@dataclass(frozen=True)
class AdapterTrainConfig:
    """LoRA specific training settings."""
    ranks: List[int]
    alpha_scaling: float
    dropout: float
    use_dora: bool
    use_rslora: bool
    target_modules: List[str]


@dataclass(frozen=True)
class TrainingHyperparameters:
    """General training hyperparameters."""
    epochs: int
    batch_size: int
    learning_rate: float
    grad_accumulation: int
    logging_steps: int
    save_steps: int


@dataclass(frozen=True)
class DataConfig:
    """Dataset loading and processing parameters."""
    input_path: Path
    test_size: float
    num_workers: int


@dataclass(frozen=True)
class ExperimentSettings:
    """Global experiment settings."""
    name: str
    seed: int
    output_dir: Path
    metrics_file: Path


@dataclass(frozen=True)
class ModelSettings:
    """Base model settings."""
    backbone: str
    num_labels: int


@dataclass(frozen=True)
class TrainConfig:
    """Root configuration object for the training pipeline."""
    experiment: ExperimentSettings
    data: DataConfig
    model: ModelSettings
    training: TrainingHyperparameters
    adapters: AdapterTrainConfig
    augmentation: AugmentationConfig


def load_train_config(path: str = "configuration/train_config.yaml") -> TrainConfig:
    """
    Parses the YAML training configuration file into typed dataclasses.

    Args:
        path: Path to the .yaml file.

    Returns:
        TrainConfig: Fully populated configuration object.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return TrainConfig(
        experiment=ExperimentSettings(
            name=raw["experiment"]["name"],
            seed=raw["experiment"]["seed"],
            output_dir=Path(raw["experiment"]["output_dir"]),
            metrics_file=Path(raw["experiment"]["metrics_file"])
        ),
        data=DataConfig(
            input_path=Path(raw["data"]["input_path"]),
            test_size=raw["data"]["test_size"],
            num_workers=raw["data"]["num_workers"]
        ),
        model=ModelSettings(**raw["model"]),
        training=TrainingHyperparameters(**raw["training"]),
        adapters=AdapterTrainConfig(**raw["adapters"]),
        augmentation=AugmentationConfig(**raw["augmentation"])
    )