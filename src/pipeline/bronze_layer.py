"""
Bronze Layer Logic.

Distributed feature extraction using PyTorch SDPA and Spark Pandas UDFs.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple, Any, Dict
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

from src.configs import ExperimentConfig


def get_schema() -> StructType:
    return StructType([
        StructField("image_path", StringType(), True),
        StructField("cls", ArrayType(FloatType()), True),
        StructField("patches", ArrayType(ArrayType(FloatType())), True),
        StructField("device", StringType(), True),
    ])


def _load_model(model_name: str) -> Tuple[Any, Any, str]:
    """Initializes model on executor with FP16 and SDPA."""
    import torch
    import warnings
    from transformers import AutoImageProcessor, AutoModel
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    warnings.simplefilter("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa"
    ).to(device).eval()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    return processor, model, device


def _process_batch(
    paths: List[str], processor: Any, model: Any, device: str
) -> List[Dict[str, Any]]:
    """Runs inference on a batch of images."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from PIL import Image

    valid_imgs: List[Image.Image] = []
    valid_paths: List[str] = []

    for p in paths:
        try:
            valid_imgs.append(Image.open(p).convert("RGB"))
            valid_paths.append(str(p))
        except Exception:
            continue

    if not valid_imgs:
        return []

    try:
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        
        with torch.no_grad(), sdpa_kernel(backends):
            inputs = processor(images=valid_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if device == "cuda":
                # Shape: [batch_size, 3, H, W] -> Float16
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

            outputs = model(**inputs)

            # Last hidden: [batch_size, seq_len, hidden_dim]
            last_hidden = outputs.last_hidden_state
            
            cls = last_hidden[:, 0, :].float().cpu().numpy()
            patches = last_hidden[:, 1:, :].float().cpu().numpy()

        return [
            {
                "image_path": p,
                "cls": c.tolist(),
                "patches": pm.tolist(),
                "device": device
            }
            for p, c, pm in zip(valid_paths, cls, patches)
        ]

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"ERROR: VRAM OOM (Batch Size: {len(paths)})")
        return []


def run_bronze(spark: SparkSession, df: DataFrame, cfg: ExperimentConfig) -> None:
    """Executes the Bronze Layer pipeline."""
    bc_model = spark.sparkContext.broadcast(cfg.model.backbone_name)
    batch_size = cfg.model.batch_size

    @pandas_udf(get_schema())
    def extract_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        processor, model, device = _load_model(bc_model.value)

        for batch_paths in iterator:
            paths = batch_paths.tolist()
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                yield pd.DataFrame.from_records(
                    _process_batch(chunk, processor, model, device)
                )

    df.repartition(4).select(col("image_path").alias("path")) \
      .select(extract_udf(col("path")).alias("data")) \
      .select("data.*") \
      .write.mode("overwrite").parquet(cfg.paths.bronze)