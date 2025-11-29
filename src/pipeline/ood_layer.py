"""
OOD Layer: Out-of-Distribution Evaluation.

Executes distributed OOD inference:
1. Joins source images with corruptions and adapters.
2. Distributed Inference: Applies LoRA adapters on corrupted images.
3. Outputs accuracy ground truth for meta-learner training.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import logging as hf_logging

from src.configs import ExperimentConfig
from src.utils.telemetry import logger
from peft import PeftModel

# Suppress HuggingFace warnings at module level
hf_logging.set_verbosity_error()


def get_ood_schema() -> StructType:
    """Returns the schema for the OOD Layer output DataFrame."""
    return StructType([
        StructField("image_path", StringType(), False),
        StructField("corruption_type", StringType(), False),
        StructField("corruption_level", StringType(), False),
        StructField("adapter_rank", StringType(), False),
        StructField("predicted_class", StringType(), False),
        StructField("true_label", StringType(), False),
        StructField("is_correct", IntegerType(), False),
        StructField("device", StringType(), False),
    ])


def apply_gaussian_noise(img: Image.Image, sigma: float = 25.0) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_contrast_change(img: Image.Image, factor: float = 0.5) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    mean = arr.mean()
    adjusted = (arr - mean) * factor + mean
    return Image.fromarray(np.clip(adjusted, 0, 255).astype(np.uint8))


# Corruption levels: shallow, medium, heavy
CORRUPTION_LEVELS = {
    "gaussian_noise": {"shallow": 15.0, "medium": 40.0, "heavy": 80.0},
    "blur": {"shallow": 1.0, "medium": 3.0, "heavy": 6.0},
    "contrast": {"shallow": 0.7, "medium": 0.4, "heavy": 0.15},
}

CORRUPTION_FNS = {
    "gaussian_noise": apply_gaussian_noise,
    "blur": apply_blur,
    "contrast": apply_contrast_change,
}


def _load_base_model_and_processor(
    model_name: str, num_labels: int
) -> Tuple[Any, Any, str]:
    """Load base DINOv2 model and image processor."""
    import warnings
    warnings.simplefilter("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        output_attentions=False,
        return_dict=True,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device).eval()

    return processor, model, device


class OODDataset(Dataset):
    """Dataset wrapper for OOD corrupted images."""

    def __init__(
        self,
        image_paths: List[str],
        corruption_types: List[str],
        corruption_levels: List[str],
        true_labels: List[str],
        processor: Any,
    ):
        self.image_paths = image_paths
        self.corruption_types = corruption_types
        self.corruption_levels = corruption_levels
        self.true_labels = true_labels
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any] | None:
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            corruption_type = self.corruption_types[idx]
            corruption_level = self.corruption_levels[idx]
            
            # Get the parameter value for this corruption+level
            param = CORRUPTION_LEVELS[corruption_type][corruption_level]
            img_corrupted = CORRUPTION_FNS[corruption_type](img, param)
            
            inputs = self.processor(images=img_corrupted, return_tensors="pt")
            return {
                "image_path": self.image_paths[idx],
                "corruption_type": corruption_type,
                "corruption_level": corruption_level,
                "true_label": self.true_labels[idx],
                "pixel_values": inputs["pixel_values"].squeeze(0),
            }
        except Exception:
            return None


def collate_fn(batch: List[Dict]) -> Dict[str, Any] | None:
    """Custom collator to filter None values from loading errors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "image_paths": [b["image_path"] for b in batch],
        "corruption_types": [b["corruption_type"] for b in batch],
        "corruption_levels": [b["corruption_level"] for b in batch],
        "true_labels": [b["true_label"] for b in batch],
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
    }


def _process_ood_partition(
    iterator: Iterator[pd.DataFrame],
    backbone_name: str,
    models_path_str: str,
    num_classes: int,
    batch_size: int,
    num_workers: int = 2,
    label2id: Dict[str, int] = None,
) -> Iterator[pd.DataFrame]:
    """Process a Spark partition with optimized GroupBy logic."""
    try:
        processor, base_model, device = _load_base_model_and_processor(
            backbone_name, num_classes
        )
        id2label = base_model.config.id2label
    except Exception as e:
        print(f"CRITICAL: Worker initialization failed: {e}")
        return

    active_rank = -1
    peft_model = None

    def switch_adapter(rank: int) -> PeftModel | None:
        nonlocal active_rank, peft_model
        if rank == active_rank and peft_model is not None:
            return peft_model

        path = Path(models_path_str) / f"lora_r{rank}"
        if not (path / "adapter_config.json").exists():
            print(f"WARNING: Adapter config missing at {path}")
            return None

        try:
            peft_model = PeftModel.from_pretrained(
                base_model, str(path), is_trainable=False
            ).eval()
            peft_model.to(device)
            active_rank = rank
            return peft_model
        except Exception as e:
            print(f"Error loading adapter rank {rank}: {e}")
            return None

    for batch_df in iterator:
        results: List[Dict[str, Any]] = []

        for adapter_rank_val, group_df in batch_df.groupby("adapter_rank"):
            adapter_rank = int(adapter_rank_val)
            peft_model = switch_adapter(adapter_rank)
            if peft_model is None:
                continue

            image_paths = group_df["image_path"].tolist()
            corruption_types = group_df["corruption_type"].tolist()
            corruption_levels = group_df["corruption_level"].tolist()
            true_labels = group_df["true_label"].tolist()

            dataset = OODDataset(
                image_paths, corruption_types, corruption_levels, true_labels, processor
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=(device == "cuda"),
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )

            for batch in tqdm(dataloader, desc=f"OOD r={adapter_rank}", leave=False):
                if batch is None:
                    continue

                pixel_values = batch["pixel_values"].to(device, non_blocking=True)

                with torch.inference_mode():
                    outputs = peft_model(pixel_values)
                    preds = torch.argmax(outputs.logits, dim=-1)

                for i, img_path in enumerate(batch["image_paths"]):
                    pred_idx = preds[i].item()
                    pred_label = id2label.get(pred_idx, str(pred_idx))
                    true_label = batch["true_labels"][i]
                    
                    true_idx = label2id.get(true_label, -1) if label2id else -1
                    is_correct = 1 if pred_idx == true_idx else 0

                    results.append({
                        "image_path": str(img_path),
                        "corruption_type": batch["corruption_types"][i],
                        "corruption_level": batch["corruption_levels"][i],
                        "adapter_rank": str(adapter_rank),
                        "predicted_class": str(pred_label),
                        "true_label": str(true_label),
                        "is_correct": is_correct,
                        "device": device,
                    })

                del pixel_values, outputs

        if results:
            yield pd.DataFrame(results)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def run_ood(spark: SparkSession, cfg: ExperimentConfig) -> None:
    """Execute OOD Layer logic."""
    logger.info("Starting OOD Layer")

    input_path = Path(cfg.paths.input)
    
    # Build label mapping (alphabetical order, same as load_dataset)
    folder_names = sorted([f.name for f in input_path.iterdir() if f.is_dir()])
    label2id = {name: i for i, name in enumerate(folder_names)}
    logger.info(f"Found {len(folder_names)} classes, label2id sample: {dict(list(label2id.items())[:5])}")
    
    image_paths = [str(p) for p in input_path.rglob("*.jpg")]
    if not image_paths:
        image_paths = [str(p) for p in input_path.rglob("*.png")]
    logger.info(f"Found {len(image_paths)} source images")

    df_images = spark.createDataFrame(
        [(p, Path(p).parent.name) for p in image_paths], ["image_path", "true_label"]
    )
    df_corruptions = spark.createDataFrame(
        [(c, lvl) for c in CORRUPTION_FNS.keys() for lvl in ["shallow", "medium", "heavy"]],
        ["corruption_type", "corruption_level"]
    )
    df_adapters = spark.createDataFrame(
        [(str(r),) for r in cfg.adapters.ranks], ["adapter_rank"]
    )

    df_workload = df_images.crossJoin(df_corruptions).crossJoin(df_adapters)
    estimated_size = len(image_paths) * len(CORRUPTION_FNS) * 3 * len(cfg.adapters.ranks)
    logger.info(f"OOD workload size (estimated): {estimated_size}")

    bc_backbone = spark.sparkContext.broadcast(cfg.model.backbone_name)
    bc_models_path = spark.sparkContext.broadcast(str(cfg.paths.models))
    bc_batch_size = spark.sparkContext.broadcast(cfg.model.batch_size)
    bc_num_workers = spark.sparkContext.broadcast(2)
    bc_label2id = spark.sparkContext.broadcast(label2id)

    NUM_CLASSES = len(folder_names)

    def execute_ood_iter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        return _process_ood_partition(
            iterator,
            backbone_name=bc_backbone.value,
            models_path_str=bc_models_path.value,
            num_classes=NUM_CLASSES,
            batch_size=bc_batch_size.value,
            num_workers=bc_num_workers.value,
            label2id=bc_label2id.value,
        )

    ood_output_path = cfg.paths.ood

    (
        df_workload
        .repartition(2)
        .sortWithinPartitions("adapter_rank")
        .mapInPandas(execute_ood_iter, schema=get_ood_schema())
        .write
        .mode("overwrite")
        .parquet(ood_output_path)
    )

    logger.info(f"OOD Layer completed. Output: {ood_output_path}")
