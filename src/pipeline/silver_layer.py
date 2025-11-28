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
import torch
from src.configs import ExperimentConfig
from src.xai.metrics import XAIEvaluator
from peft import PeftModel
from transformers import AutoImageProcessor, AutoModelForImageClassification


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
    """Load base DINOv2 model and image processor."""
    import warnings
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


class ImageDataset(Dataset):
    """Simple Dataset wrapper for batched loading."""
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
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Custom collator to handle potential None values from loading errors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "paths": [b["path"] for b in batch],
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),  # [B, 3, H, W]
    }


def _process_partition(
    iterator: Iterator[pd.DataFrame],
    backbone_name: str,
    models_path_str: str,
    num_classes: int,
    batch_size: int,
    patch_size: int,
    perturbation_steps: int,
    num_workers: int = 2,
) -> Iterator[pd.DataFrame]:
    """
    Process a Spark partition using optimized GroupBy logic.
    Ensures adapters are switched safely and DataLoaders are persistent per-rank.
    """
    
    # 1. Initialize Global Resources (Model & XAI)
    try:
        processor, base_model, device = _load_base_model_and_processor(backbone_name, num_classes)
        evaluator = XAIEvaluator(perturbation_steps=perturbation_steps, patch_size=patch_size)
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
            # Reload adapter onto base_model
            peft_model = PeftModel.from_pretrained(base_model, str(path), is_trainable=False).eval()
            peft_model.to(device) # Ensure adapter is on device
            active_rank = rank
            return peft_model
        except Exception as e:
            print(f"Error loading adapter rank {rank}: {e}")
            return None

    # 2. Iterate Spark Partition Batches (Pandas DataFrames)
    for batch_df in iterator:
        results: List[Dict[str, Any]] = []
        
        for adapter_rank_val, group_df in batch_df.groupby("adapter_rank"):
            adapter_rank = int(adapter_rank_val)
            
            # A. Switch Adapter
            peft_model = switch_adapter(adapter_rank)
            if peft_model is None:
                continue
                
            # B. Prepare Efficient DataLoader
            image_paths = group_df["image_path"].tolist()
            dataset = ImageDataset(image_paths, processor)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                num_workers=num_workers,
                collate_fn=collate_fn, 
                pin_memory=(device == "cuda")
            )
            
            # C. Inference Loop
            for batch in tqdm(dataloader, desc=f"Rank {adapter_rank}", leave=False):
                if batch is None: continue
                
                paths = batch["paths"]
                pixel_values = batch["pixel_values"].to(device) # [B, 3, H, W]

                with torch.no_grad():
                    # Forward Pass
                    outputs = peft_model(pixel_values)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1) # [B]
                    attentions = outputs.attentions      # Tuple of [B, H, Seq, Seq]

                    # XAI Evaluation
                    metrics = evaluator.evaluate(
                        model=peft_model,
                        pixel_values=pixel_values,
                        attentions=attentions,
                        target_classes=preds,
                    )

                # D. Collect Results
                for b_idx, path in enumerate(paths):
                    pred_idx = preds[b_idx].item()
                    pred_label = id2label.get(pred_idx, str(pred_idx))
                    
                    results.append({
                        "image_path": str(path),
                        "adapter_rank": str(adapter_rank),
                        "entropy": float(metrics["entropy"][b_idx].item()),
                        "sparsity": float(metrics["sparsity"][b_idx].item()),
                        "deletion_score": float(metrics["deletion"][b_idx].item()),
                        "insertion_score": float(metrics["insertion"][b_idx].item()),
                        "predicted_class": str(pred_label),
                        "true_label": str(Path(path).parent.name),
                        "device": device,
                    })
                
                # Cleanup VRAM
                del pixel_values, outputs, attentions, metrics
        
        # Yield results for this Spark partition chunk
        if results:
            yield pd.DataFrame(results)
        
        # Garbage Collection
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def run_silver(spark: SparkSession, df_bronze: DataFrame, cfg: ExperimentConfig) -> None:
    """Execute Silver Layer logic."""
    ranks = cfg.adapters.ranks
    
    # Create workload: Cross Join Images x Adapters
    df_ranks = spark.createDataFrame([(r,) for r in ranks], ["adapter_rank"])
    df_input = df_bronze.select("image_path").distinct()
    df_workload = df_input.crossJoin(df_ranks)

    # Broadcast configuration
    bc_backbone = spark.sparkContext.broadcast(cfg.model.backbone_name)
    bc_models_path = spark.sparkContext.broadcast(str(cfg.paths.models))
    bc_batch_size = spark.sparkContext.broadcast(cfg.model.batch_size)
    bc_patch_size = spark.sparkContext.broadcast(cfg.model.patch_size)
    bc_perturbation_steps = spark.sparkContext.broadcast(cfg.xai.perturbation_steps)
    bc_num_workers = spark.sparkContext.broadcast(2) 

    # TODO: Make dynamic or load from config
    NUM_CLASSES = 37 

    def execute_silver_iter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        return _process_partition(
            iterator,
            backbone_name=bc_backbone.value,
            models_path_str=bc_models_path.value,
            num_classes=NUM_CLASSES,
            batch_size=bc_batch_size.value,
            patch_size=bc_patch_size.value,
            perturbation_steps=bc_perturbation_steps.value,
            num_workers=bc_num_workers.value
        )

    silver_output_path = cfg.paths.silver

    (
        df_workload
        .repartition(8)
        .sortWithinPartitions("adapter_rank")
        .mapInPandas(execute_silver_iter, schema=get_silver_schema())
        .write
        .mode("overwrite")
        .parquet(silver_output_path)
    )