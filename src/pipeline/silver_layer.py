"""
Silver Layer: Distributed XAI & Adapter Evaluation.

Executes the core logic:
1. Joins Bronze data with Adapter Zoo configurations.
2. Distributed Inference: Loads PRE-TRAINED LoRA adapters from disk.
3. Computes XAI Metrics (Entropy, Deletion, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Dict, Any, Tuple
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, FloatType

from src.configs import ExperimentConfig
from src.xai.metrics import XAIEvaluator
from peft import PeftModel  # Required for loading trained adapters


def get_silver_schema() -> StructType:
    """Returns the schema for the Silver Layer output."""
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


def _load_base_model_and_processor(model_name: str, num_labels: int) -> Tuple[Any, Any, str]:
    """Initializes the base DINOv2 model."""
    import torch
    import warnings
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    warnings.simplefilter("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    
    # Base model loaded with correct head size but standard weights
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
    models_path_str: str,  # Path to artifacts/adapters
    num_classes: int
) -> Iterator[pd.DataFrame]:
    """
    Scalar Iterator Logic: Loads Base Model once, swaps LoRA adapters dynamically.
    """
    import torch
    from PIL import Image

    # 1. Initialize Resources
    processor, base_model, device = _load_base_model_and_processor(backbone_name, num_classes)
    evaluator = XAIEvaluator()
    
    # Helper to map ID back to Label string (if available in config)
    id2label = base_model.config.id2label

    active_rank = -1
    peft_model = None

    for batch_df in iterator:
        results: List[Dict[str, Any]] = []

        for _, row in batch_df.iterrows():
            image_path = row["image_path"]
            adapter_rank = int(row["adapter_rank"])

            # 2. Dynamic Adapter Switching (FROM DISK)
            if adapter_rank != active_rank:
                adapter_path = Path(models_path_str) / f"lora_r{adapter_rank}"
                
                try:
                    if not adapter_path.exists():
                        print(f"WARNING: Adapter not found at {adapter_path}")
                        continue
                        
                    # Load trained weights onto the base model
                    peft_model = PeftModel.from_pretrained(
                        base_model,
                        str(adapter_path),
                        is_trainable=False
                    )
                    peft_model.eval()
                    active_rank = adapter_rank
                    
                except Exception as e:
                    print(f"Error loading adapter {adapter_rank}: {e}")
                    continue

            # Skip if no model loaded
            if peft_model is None: continue

            try:
                # 3. Data Loading
                image = Image.open(image_path).convert("RGB")
                
                # Extract True Label (Folder Name)
                true_label_str = Path(image_path).parent.name
                
                # Map string label to index (Naive assumption: folders are named cleanly)
                # In robust systems, pass a mapping dict. Here we use simple heuristics or 0.
                # For XAI metrics, we usually need the index of the predicted class.
                
                inputs = processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                # 4. Inference
                with torch.no_grad():
                    outputs = peft_model(pixel_values)
                    logits = outputs.logits
                    pred_idx = torch.argmax(logits, dim=-1).item()
                    pred_label = id2label.get(pred_idx, str(pred_idx))
                    attentions = outputs.attentions

                # 5. Compute XAI Metrics
                metrics = evaluator.evaluate(
                    model=peft_model,
                    pixel_values=pixel_values,
                    attentions=attentions,
                    target_class=pred_idx
                )

                results.append({
                    "image_path": image_path,
                    "adapter_rank": str(adapter_rank),
                    "entropy": metrics["entropy"],
                    "sparsity": metrics["sparsity"],
                    "deletion_score": metrics["deletion"],
                    "insertion_score": metrics["insertion"],
                    "predicted_class": str(pred_label),
                    "true_label": str(true_label_str),
                    "device": device
                })

            except Exception as e:
                print(f"Error processing img {image_path}: {e}")
                continue

        yield pd.DataFrame(results)


def run_silver(spark: SparkSession, df_bronze: DataFrame, cfg: ExperimentConfig) -> None:
    """Orchestrates the Silver Layer pipeline."""
    
    ranks = cfg.adapters.ranks
    df_ranks = spark.createDataFrame([(r,) for r in ranks], ["adapter_rank"])

    df_input = df_bronze.select("image_path").distinct()
    df_workload = df_input.crossJoin(df_ranks)

    # Broadcasts
    bc_backbone = spark.sparkContext.broadcast(cfg.model.backbone_name)
    bc_models_path = spark.sparkContext.broadcast(str(cfg.paths.models))
    
    # Hardcoded for now, ideally passed via config or detected
    # This must match what was used in Training
    NUM_CLASSES = 37 

    def execute_silver_iter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        return _process_partition(
            iterator, 
            backbone_name=bc_backbone.value,
            models_path_str=bc_models_path.value,
            num_classes=NUM_CLASSES
        )

    silver_output_path = cfg.paths.silver
    
    (
        df_workload
        .repartition(8) 
        .mapInPandas(execute_silver_iter, schema=get_silver_schema())
        .write
        .mode("overwrite")
        .parquet(silver_output_path)
    )