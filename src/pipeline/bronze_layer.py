"""
Bronze layer: Feature extraction pipeline using Spark and PyTorch.
Follows PEP 8 standards with strict type hinting and import isolation.
"""
from __future__ import annotations
from typing import Iterator, List, Tuple, Any

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType


class BronzeConfig:
    def __init__(self, backbone: str, img_col: str, out_path: str, batch_size: int = 32):
        self.backbone = backbone
        self.img_col = img_col
        self.out_path = out_path
        self.batch_size = batch_size


def get_schema() -> StructType:
    return StructType([
        StructField("image_path", StringType(), True),
        StructField("cls", ArrayType(FloatType()), True),
        StructField("patches", ArrayType(ArrayType(FloatType())), True),
        StructField("device", StringType(), True),
    ])


def _load_model(model_name: str) -> Tuple[Any, Any, str]:
    """Initializes the model inside the worker process."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    return processor, model, device


def _process_batch(
    paths: List[str], processor: Any, model: Any, device: str
) -> List[dict]:
    """Performs inference on a single batch of images."""
    import torch
    from PIL import Image

    valid_imgs, valid_paths = [], []

    for p in paths:
        try:
            # Convert to RGB to handle PNG/Grayscale inconsistencies
            valid_imgs.append(Image.open(p).convert("RGB"))
            valid_paths.append(str(p))
        except Exception:
            continue

    if not valid_imgs:
        return []

    try:
        with torch.no_grad():
            inputs = processor(images=valid_imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)

            # Move tensors to CPU immediately to free VRAM
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            patches = outputs.last_hidden_state[:, 1:, :].cpu().numpy()

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
            print(f"ERROR: CUDA OOM. Batch size {len(paths)} is too large.")
        return []


def run_bronze(spark: SparkSession, df: DataFrame, cfg: BronzeConfig) -> None:
    """Executes the distributed Bronze Layer pipeline."""
    
    # Broadcast large/static variables
    bc_model_name = spark.sparkContext.broadcast(cfg.backbone)
    batch_size = cfg.batch_size

    @pandas_udf(get_schema())
    def extract_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        # Import Isolation: libraries loaded only on workers
        processor, model, device = _load_model(bc_model_name.value)

        for batch_paths in iterator:
            paths = batch_paths.tolist()
            
            # Chunk processing to respect memory limits
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                results = _process_batch(chunk, processor, model, device)
                yield pd.DataFrame.from_records(results)

    # Execution Plan
    df_processed = (
        df.repartition(8)
        .select(col(cfg.img_col).alias("path"))
        .select(extract_udf(col("path")).alias("data"))
        .select(
            col("data.image_path"),
            col("data.cls"),
            col("data.patches"),
            col("data.device")
        )
    )

    df_processed.write.mode("overwrite").parquet(cfg.out_path)