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

from src.configs import ExperimentConfig
from src.utils.telemetry import logger
from peft import PeftModel


def get_ood_schema() -> StructType:
    """Returns the schema for the OOD Layer output DataFrame."""
    return StructType([
        StructField("image_path", StringType(), False),
        StructField("corruption_type", StringType(), False),
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
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
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

    def __init__(self, records: List[Dict], processor: Any):
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any] | None:
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            img_corrupted = CORRUPTION_FNS[rec["corruption_type"]](img)
            inputs = self.processor(images=img_corrupted, return_tensors="pt")
            return {
                "image_path": rec["image_path"],
                "corruption_type": rec["corruption_type"],
                "true_label": rec["true_label"],
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

            dataset = OODDataset(group_df.to_dict("records"), processor)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=(device == "cuda"),
            )

            for batch in tqdm(dataloader, desc=f"OOD r={adapter_rank}", leave=False):
                if batch is None:
                    continue

                pixel_values = batch["pixel_values"].to(device)

                with torch.no_grad():
                    outputs = peft_model(pixel_values)
                    preds = torch.argmax(outputs.logits, dim=-1)

                for i, img_path in enumerate(batch["image_paths"]):
                    pred_idx = preds[i].item()
                    pred_label = id2label.get(pred_idx, str(pred_idx))
                    true_label = batch["true_labels"][i]
                    pred_clean = pred_label.replace("LABEL_", "")

                    results.append({
                        "image_path": str(img_path),
                        "corruption_type": batch["corruption_types"][i],
                        "adapter_rank": str(adapter_rank),
                        "predicted_class": str(pred_label),
                        "true_label": str(true_label),
                        "is_correct": 1 if pred_clean == true_label else 0,
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
    image_paths = [str(p) for p in input_path.rglob("*.jpg")]
    if not image_paths:
        image_paths = [str(p) for p in input_path.rglob("*.png")]
    logger.info(f"Found {len(image_paths)} source images")

    df_images = spark.createDataFrame(
        [(p, Path(p).parent.name) for p in image_paths], ["image_path", "true_label"]
    )
    df_corruptions = spark.createDataFrame(
        [(c,) for c in CORRUPTION_FNS.keys()], ["corruption_type"]
    )
    df_adapters = spark.createDataFrame(
        [(str(r),) for r in cfg.adapters.ranks], ["adapter_rank"]
    )

    df_workload = df_images.crossJoin(df_corruptions).crossJoin(df_adapters)
    logger.info(f"OOD workload size: {df_workload.count()}")

    bc_backbone = spark.sparkContext.broadcast(cfg.model.backbone_name)
    bc_models_path = spark.sparkContext.broadcast(str(cfg.paths.models))
    bc_batch_size = spark.sparkContext.broadcast(cfg.model.batch_size)
    bc_num_workers = spark.sparkContext.broadcast(2)

    NUM_CLASSES = 37

    def execute_ood_iter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        return _process_ood_partition(
            iterator,
            backbone_name=bc_backbone.value,
            models_path_str=bc_models_path.value,
            num_classes=NUM_CLASSES,
            batch_size=bc_batch_size.value,
            num_workers=bc_num_workers.value,
        )

    ood_output_path = cfg.paths.ood
    Path(ood_output_path).mkdir(parents=True, exist_ok=True)

    (
        df_workload
        .repartition(2)
        .sortWithinPartitions("adapter_rank")
        .mapInPandas(execute_ood_iter, schema=get_ood_schema())
        .write
        .mode("overwrite")
        .parquet(ood_output_path)
    )
