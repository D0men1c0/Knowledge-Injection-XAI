"""
Silver Layer: Distributed XAI & Adapter Evaluation.

Executes the core logic of the framework:
1. Joins Bronze data with Adapter Zoo (LoRA configurations).
2. Distributed Inference: Applies LoRA adapters dynamically.
3. Computes XAI Metrics: Entropy, Deletion, Insertion, Sparsity.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from src.configs import ExperimentConfig
from src.xai.metrics import XAIEvaluator
from peft import PeftModel


def get_silver_schema() -> StructType:
    """Returns the schema for the Silver Layer output DataFrame."""
    return StructType([
        StructField("image_path", StringType(), True),
        StructField("adapter_rank", StringType(), True),
        StructField("entropy", FloatType(), True),
        StructField("sparsity", FloatType(), True),
        StructField("deletion_score", FloatType(), True),
        StructField("insertion_score", FloatType(), True),
        StructField("predicted_class", StringType(), True),
        StructField("true_label", StringType(), True),
        StructField("device", StringType(), True),
    ])


def _load_base_model_and_processor(
    model_name: str, num_labels: int
) -> Tuple[Any, Any, str]:
    """
    Load base DINOv2 model and image processor.

    Args:
        model_name: Pretrained model identifier.
        num_labels: Number of classification labels.

    Returns:
        processor: AutoImageProcessor
        model: AutoModelForImageClassification
        device: str ("cuda" or "cpu")
    """
    import torch
    import warnings
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    warnings.simplefilter("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        output_attentions=True,
        return_dict=True,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).to(device).eval()

    return processor, model, device

def _process_partition(
    iterator: Iterator[pd.DataFrame],
    backbone_name: str,
    models_path_str: str,
    num_classes: int,
    batch_size: int,
    num_workers: int = 2,
) -> Iterator[pd.DataFrame]:
    """
    Process a Spark partition using internal DataLoader batching.
    DRY + SoC: adapter loading, dataset, inference, metrics separated.
    """
    import torch

    # --- 1. Initialize model and evaluator
    try:
        processor, base_model, device = _load_base_model_and_processor(backbone_name, num_classes)
        evaluator = XAIEvaluator()
        id2label = base_model.config.id2label
    except Exception as e:
        print(f"CRITICAL: Worker initialization failed: {e}")
        return

    active_rank = -1
    peft_model = None

    # --- Dataset wrapper
    class ImageDataset(Dataset):
        def __init__(self, paths: List[str], processor: Any):
            self.paths = paths
            self.processor = processor

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int) -> Dict[str, Any] | None:
            path = self.paths[idx]
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze(0)  # [3, H, W]
                return {"path": path, "pixel_values": pixel_values}
            except Exception:
                return None

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {
            "paths": [b["path"] for b in batch],
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),  # [B, 3, H, W]
        }

    # --- Helper: load adapter
    def switch_adapter(rank: int) -> PeftModel | None:
        nonlocal active_rank, peft_model
        if rank == active_rank and peft_model is not None:
            return peft_model
        path = Path(models_path_str) / f"lora_r{rank}"
        if not (path / "adapter_config.json").exists():
            print(f"WARNING: Adapter config missing at {path}")
            return None
        try:
            peft_model = PeftModel.from_pretrained(base_model, str(path), is_trainable=False).eval()
            active_rank = rank
            return peft_model
        except Exception as e:
            print(f"Error loading adapter rank {rank}: {e}")
            return None

    # --- Main loop
    for batch_df in iterator:
        results: List[Dict[str, Any]] = []
        rows = batch_df.to_dict("records")

        with tqdm(total=len(rows), desc="Silver Worker", mininterval=1) as pbar:
            i = 0
            while i < len(rows):
                chunk = rows[i:i + batch_size]
                i += batch_size

                adapter_rank = int(chunk[0]["adapter_rank"])
                peft_model = switch_adapter(adapter_rank)
                if peft_model is None:
                    pbar.update(len(chunk))
                    continue

                image_paths = [r["image_path"] for r in chunk]
                dataset = ImageDataset(image_paths, processor)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                        collate_fn=collate_fn, pin_memory=True)

                # --- Inference loop
                for batch in dataloader:
                    if batch is None:
                        continue
                    paths, pixel_values = batch["paths"], batch["pixel_values"].to(device)

                    with torch.no_grad():
                        outputs = peft_model(pixel_values)
                        logits = outputs.logits  # [B, num_classes]
                        preds = torch.argmax(logits, dim=-1)
                        attentions = outputs.attentions

                    # --- Metrics + results
                    for b_idx, path in enumerate(paths):
                        pred_idx = preds[b_idx].item()
                        pred_label = id2label.get(pred_idx, str(pred_idx))
                        metrics = evaluator.evaluate(
                            model=peft_model,
                            pixel_values=pixel_values[b_idx:b_idx + 1],  # [1,3,H,W]
                            attentions=tuple(a[b_idx:b_idx + 1] for a in attentions),
                            target_class=pred_idx
                        )
                        results.append({
                            "image_path": str(path),
                            "adapter_rank": str(adapter_rank),
                            "entropy": float(metrics["entropy"]),
                            "sparsity": float(metrics["sparsity"]),
                            "deletion_score": float(metrics["deletion"]),
                            "insertion_score": float(metrics["insertion"]),
                            "predicted_class": str(pred_label),
                            "true_label": str(Path(path).parent.name),
                            "device": device
                        })

                    del pixel_values, outputs, attentions
                    if device == "cuda":
                        torch.cuda.empty_cache()

                    pbar.update(len(paths))

        yield pd.DataFrame(results)
        gc.collect()


def run_silver(spark: SparkSession, df_bronze: DataFrame, cfg: ExperimentConfig) -> None:
    """
    Execute Silver Layer with internal DataLoader batching and single Spark worker.

    Args:
        spark: SparkSession.
        df_bronze: Bronze Layer DataFrame.
        cfg: Experiment configuration.
    """
    ranks = cfg.adapters.ranks
    df_ranks = spark.createDataFrame([(r,) for r in ranks], ["adapter_rank"])
    df_input = df_bronze.select("image_path").distinct()
    df_workload = df_input.crossJoin(df_ranks)

    bc_backbone = spark.sparkContext.broadcast(cfg.model.backbone_name)
    bc_models_path = spark.sparkContext.broadcast(str(cfg.paths.models))
    bc_batch_size = spark.sparkContext.broadcast(cfg.model.batch_size)
    bc_num_workers = spark.sparkContext.broadcast(2)

    NUM_CLASSES = 37

    def execute_silver_iter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        return _process_partition(
            iterator,
            backbone_name=bc_backbone.value,
            models_path_str=bc_models_path.value,
            num_classes=NUM_CLASSES,
            batch_size=bc_batch_size.value,
            num_workers=bc_num_workers.value
        )

    silver_output_path = cfg.paths.silver

    (
        df_workload
        .repartition(8)  # single Spark worker for GPU saturation
        .mapInPandas(execute_silver_iter, schema=get_silver_schema())
        .write
        .mode("overwrite")
        .parquet(silver_output_path)
    )