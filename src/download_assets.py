"""Utility script to download backbone, adapters (stubs), and dataset.

This keeps downloads separate from the Spark pipeline.
"""

from pathlib import Path

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel


def download_backbone(model_name: str) -> None:
    """Download and cache a ViT/DINOv2 backbone locally."""

    AutoImageProcessor.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)


def prepare_directories() -> None:
    """Create minimal directories for adapters and raw data."""

    for path in [
        Path("artifacts/adapters/lora_r4"),
        Path("artifacts/adapters/lora_r32"),
        Path("data/raw/source"),
        Path("data/raw/ood"),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def download_oxford_iiit_pet() -> None:
    """Download the Oxford-IIIT Pet dataset via Hugging Face Datasets."""

    ds = load_dataset("timm/oxford_iiit_pet")
    out_dir = Path("data/raw/source")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save only a small subset to disk as a starting point.
    split = ds["train"]
    for idx, row in enumerate(split):
        image = row["image"]
        label = row["label"]
        cls_dir = out_dir / str(label)
        cls_dir.mkdir(parents=True, exist_ok=True)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(cls_dir / f"{idx}.jpg")


def main() -> None:
    """Download backbone and dataset, and create adapter dirs."""

    model_name = "facebook/dinov2-base"
    prepare_directories()
    download_backbone(model_name)
    download_oxford_iiit_pet()


if __name__ == "__main__":
    main()
