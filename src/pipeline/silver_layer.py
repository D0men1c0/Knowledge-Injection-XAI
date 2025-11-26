"""
Silver Layer: Distributed XAI & Adapter Evaluation.

Executes the core logic of the framework:
1. Joins Bronze data with Adapter Zoo (LoRA configurations).
2. Distributed Inference: Applies LoRA adapters dynamically.
3. Computes XAI Metrics: Entropy, Deletion, Insertion, Sparsity.

Reference: Proposal 2.1 (Silver Layer), 3.2 (Entropy), 3.3 (Deletion).
"""
from __future__ import annotations

from typing import Iterator, List, Dict, Any, Tuple
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, FloatType

from src.configs import ExperimentConfig
from src.xai.metrics import XAIEvaluator
from src.ml.adapters import LoRAFactory


def get_silver_schema() -> StructType:
    """Returns the schema for the Silver Layer output."""
    return StructType([
        StructField("image_path", StringType(), True),
        StructField("adapter_rank", StringType(), True),
        StructField("entropy", FloatType(), True),
        StructField("sparsity", FloatType(), True),
        StructField("deletion_score", FloatType(), True),
        StructField("insertion_score", FloatType(), True),
        StructField("predicted_class", FloatType(), True),
        StructField("device", StringType(), True),
    ])


def _load_base_model_and_processor(model_name: str) -> Tuple[Any, Any, str]:
    """
    Initializes the base DINOv2 model and processor on the worker.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tuple[Processor, Model, Device string]
    """
    import torch
    import warnings
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import logging as hf_logging

    # Suppress library noise
    hf_logging.set_verbosity_error()
    warnings.simplefilter("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    
    # Load model with attention output enabled for XAI
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        output_attentions=True,
        return_dict=True
    ).to(device).eval()

    return processor, model, device


def _process_partition(
    iterator: Iterator[pd.DataFrame],
    backbone_name: str
) -> Iterator[pd.DataFrame]:
    """
    Scalar Iterator Logic.
    Loads model ONCE per partition, then iterates over batches.
    """
    import torch
    from PIL import Image

    # 1. Initialize Resources (Once per Executor/Partition)
    processor, base_model, device = _load_base_model_and_processor(backbone_name)
    evaluator = XAIEvaluator()

    # Cache for LoRA adapters to avoid re-initialization overhead
    active_rank = -1
    peft_model = None

    for batch_df in iterator:
        results: List[Dict[str, Any]] = []

        for _, row in batch_df.iterrows():
            image_path = row["image_path"]
            adapter_rank = int(row["adapter_rank"])

            # 2. Dynamic Adapter Switching
            if adapter_rank != active_rank:
                # Injects LoRA layers dynamically
                peft_model = LoRAFactory.inject_adapter(base_model, adapter_rank)
                peft_model.eval()
                active_rank = adapter_rank

            try:
                # 3. Data Loading (Raw Image required for Deletion Score)
                image = Image.open(image_path).convert("RGB")
                
                inputs = processor(images=image, return_tensors="pt")
                # Tensor Shape: [1, 3, height, width]
                pixel_values = inputs["pixel_values"].to(device)

                # 4. Inference (Forward Pass)
                with torch.no_grad():
                    outputs = peft_model(pixel_values)
                    
                    # Tensor Shape: [1, num_classes]
                    logits = outputs.logits
                    pred_class = torch.argmax(logits, dim=-1).item()
                    
                    # Tuple of Attentions: [num_layers, 1, num_heads, seq_len, seq_len]
                    attentions = outputs.attentions

                # 5. Compute XAI Metrics
                metrics = evaluator.evaluate(
                    model=peft_model,
                    pixel_values=pixel_values,
                    attentions=attentions,
                    target_class=pred_class
                )

                results.append({
                    "image_path": image_path,
                    "adapter_rank": str(adapter_rank),
                    "entropy": metrics["entropy"],
                    "sparsity": metrics["sparsity"],
                    "deletion_score": metrics["deletion"],
                    "insertion_score": metrics["insertion"],
                    "predicted_class": float(pred_class),
                    "device": device
                })

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        yield pd.DataFrame(results)


def run_silver(spark: SparkSession, df_bronze: DataFrame, cfg: ExperimentConfig) -> None:
    """
    Orchestrates the Silver Layer pipeline.
    
    Strategy:
    1. Define Adapter Zoo (list of ranks).
    2. CrossJoin Bronze Data with Adapter Zoo (Cartesian Product).
    3. Execute Distributed XAI Extraction.
    """
    # 1. Adapter Zoo (From Config)
    ranks = cfg.adapters.ranks
    
    df_ranks = spark.createDataFrame(
        [(r,) for r in ranks], ["adapter_rank"]
    )

    # 2. Expand Dataset: Each image processed by Each Adapter
    # We select only image_path to keep shuffle light
    df_input = df_bronze.select("image_path").distinct()
    df_workload = df_input.crossJoin(df_ranks)

    # Broadcast backbone name to workers
    bc_backbone = spark.sparkContext.broadcast(cfg.model.backbone_name)

    # 3. Define Wrapper Function (NO @pandas_udf DECORATOR)
    def execute_silver_iter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        return _process_partition(iterator, bc_backbone.value)

    # 4. Execution
    # Repartitioning ensures optimal parallelism for the heavy XAI workload
    silver_output_path = cfg.paths.silver
    
    (
        df_workload
        .repartition(8) 
        .mapInPandas(execute_silver_iter, schema=get_silver_schema())
        .write
        .mode("overwrite")
        .parquet(silver_output_path)
    )