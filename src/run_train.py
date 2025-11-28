"""
Platinum Layer: Enhanced LoRA Training Pipeline.

Trains LoRA adapters with robust data augmentation, stratified validation,
and automated documentation generation.
"""
import csv
import shutil
import time
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    TrainerCallback
)
from transformers.image_processing_utils import BaseImageProcessor
from peft import get_peft_model, LoraConfig
from torchvision import transforms

from src.utils.train_configs import load_train_config, TrainConfig

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*flash attention.*"
)

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)

CONFIG_PATH = "configuration/train_config.yaml"

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class LogCallback(TrainerCallback):
    """Custom callback to log training metrics (loss) directly to the logger."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", None)
            if loss is not None:
                logger.info(f"Step {state.global_step}: Loss = {loss:.4f}")


def get_transforms(cfg: TrainConfig, processor: BaseImageProcessor):
    """Creates training and validation transforms with robust augmentation."""
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = cfg.augmentation.crop_size

    train_transforms_list = [
        transforms.Resize(cfg.augmentation.resize_size),
        transforms.RandomRotation(cfg.augmentation.rotation_degrees),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=cfg.augmentation.horizontal_flip_prob),
        transforms.ColorJitter(
            brightness=cfg.augmentation.color_jitter_brightness,
            contrast=cfg.augmentation.color_jitter_contrast
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ]

    test_transforms_list = [
        transforms.Resize(cfg.augmentation.resize_size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ]

    _train_compose = transforms.Compose(train_transforms_list)
    _test_compose = transforms.Compose(test_transforms_list)

    def train_transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        examples["pixel_values"] = [
            _train_compose(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    def test_transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        examples["pixel_values"] = [
            _test_compose(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    return train_transform, test_transform


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Computes accuracy, precision, recall, and f1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_metrics_to_csv(file_path: Path, rank: int, metrics: Dict[str, float], train_loss: float, duration: float) -> None:
    """Appends training results to a CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = file_path.exists()
    
    header = ['timestamp', 'rank', 'train_loss', 'eval_loss', 'accuracy', 'precision', 'recall', 'f1', 'duration_min']
    row = [
        datetime.datetime.now().isoformat(),
        rank,
        f"{train_loss:.4f}",
        f"{metrics.get('eval_loss', 0):.4f}",
        f"{metrics.get('eval_accuracy', 0):.4f}",
        f"{metrics.get('eval_precision', 0):.4f}",
        f"{metrics.get('eval_recall', 0):.4f}",
        f"{metrics.get('eval_f1', 0):.4f}",
        f"{duration:.2f}"
    ]

    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def generate_readme(
    output_dir: Path,
    rank: int,
    cfg,
    metrics: dict,
    model,
    duration_min: float | None = None,
) -> None:
    """Generates a simple, self-contained Model Card README."""

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- Hardware detection ----
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        precision = "fp16"
    else:
        device_name = "CPU"
        precision = "fp32"

    # ---- Trainable parameters ----
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # ---- YAML Header ----
    header = f"""---
base_model: {cfg.model.backbone}
library_name: peft
tags:
- base_model:adapter:{cfg.model.backbone}
- lora
- computer-vision
- image-classification
- transformers
---

"""

    # ---- Main Content ----
    content = f"""{header}
# LoRA Adapter (Rank {rank})

Fine-tuned LoRA adapter for `{cfg.model.backbone}`.

- **Training Date:** {timestamp}
- **Dataset Path:** `{cfg.data.input_path}`
- **Hardware:** {device_name}
- **Use DoRA:** {cfg.adapters.use_dora}
- **Use RsLoRA:** {cfg.adapters.use_rslora}

## Training

- **Backbone:** `{cfg.model.backbone}`
- **LoRA Rank:** {rank}
- **LoRA Alpha:** {rank * cfg.adapters.alpha_scaling}
- **LoRA Dropout:** {cfg.adapters.dropout}
- **Use DoRA:** {cfg.adapters.use_dora}
- **Use RsLoRA:** {cfg.adapters.use_rslora}
- **Target Modules:** {cfg.adapters.target_modules}
- **Batch Size:** {cfg.training.batch_size}
- **Learning Rate:** {cfg.training.learning_rate}
- **Epochs:** {cfg.training.epochs}
- **Precision:** {precision}
- **Training Time (min):** {f"{duration_min:.2f}" if duration_min else "N/A"}

## Parameters

- **Trainable Parameters:** {trainable_params:,}
- **Total Parameters:** {total_params:,}
- **Trainable %:** {(trainable_params / total_params) * 100:.4f}%

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | {metrics.get('eval_accuracy', 0):.4f} |
| **Precision** | {metrics.get('eval_precision', 0):.4f} |
| **Recall** | {metrics.get('eval_recall', 0):.4f} |
| **F1 Score** | {metrics.get('eval_f1', 0):.4f} |
| **Eval Loss** | {metrics.get('eval_loss', 0):.4f} |

## How to Use

```python
from transformers import AutoModelForImageClassification
from peft import PeftModel

base_model = AutoModelForImageClassification.from_pretrained("{cfg.model.backbone}")
model = PeftModel.from_pretrained(base_model, "./lora_r{rank}")
model.eval()
```
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(content)


class AdapterTrainer:
    """Manages the lifecycle of training a single LoRA adapter rank."""

    def __init__(self, cfg: TrainConfig, num_labels: int, label2id: Dict, id2label: Dict):
        self.cfg = cfg
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self, rank: int):
        model = AutoModelForImageClassification.from_pretrained(
            self.cfg.model.backbone,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True
        )

        peft_config = LoraConfig(
            inference_mode=False,
            r=rank,
            lora_alpha=rank * self.cfg.adapters.alpha_scaling,
            lora_dropout=self.cfg.adapters.dropout,
            target_modules=self.cfg.adapters.target_modules,
            bias="lora_only",
            modules_to_save=["classifier"],
            use_dora=self.cfg.adapters.use_dora,
            use_rslora=self.cfg.adapters.use_rslora
        )

        model = get_peft_model(model, peft_config)
        model.to(self.device)
        logger.info(f"Model device: {next(model.parameters()).device}")
        model.print_trainable_parameters()
        return model

    def train(self, rank: int, dataset_splits: DatasetDict) -> None:
        """Runs the training loop for a specific rank."""
        logger.info(f"--- Starting Training for Rank {rank} ---")

        model = self._init_model(rank)
        
        # Prepare Output Dir
        output_dir = self.cfg.experiment.output_dir / f"lora_r{rank}"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(output_dir),
            no_cuda=not torch.cuda.is_available(), 
            remove_unused_columns=False,
            eval_strategy="steps",
            eval_steps=self.cfg.training.logging_steps,
            save_strategy="steps",
            save_steps=self.cfg.training.save_steps,
            learning_rate=self.cfg.training.learning_rate,
            per_device_train_batch_size=self.cfg.training.batch_size,
            per_device_eval_batch_size=self.cfg.training.batch_size,
            gradient_accumulation_steps=self.cfg.training.grad_accumulation,
            num_train_epochs=self.cfg.training.epochs,
            logging_steps=self.cfg.training.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=str(output_dir / "logs"),
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=self.cfg.data.num_workers,
            save_total_limit=2
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset_splits["train"],
            eval_dataset=dataset_splits["test"],
            compute_metrics=compute_metrics,
            data_collator=DefaultDataCollator(),
            callbacks=[LogCallback()]  # Inject custom logger
        )

        start_time = time.time()
        train_result = trainer.train()
        duration = (time.time() - start_time) / 60.0
        train_loss = train_result.training_loss

        logger.info(f"Training Rank {rank} finished in {duration:.2f} mins.")

        metrics = trainer.evaluate()
        logger.info(f"Rank {rank} Results: Accuracy={metrics.get('eval_accuracy', 0):.4f}")

        model.save_pretrained(str(output_dir))
        save_metrics_to_csv(self.cfg.experiment.metrics_file, rank, metrics, train_loss, duration)
        generate_readme(
            output_dir=output_dir,
            rank=rank,
            cfg=self.cfg,
            metrics=metrics,
            model=model,
            duration_min=duration,
        )


def main() -> None:
    try:
        cfg = load_train_config(CONFIG_PATH)
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        return

    set_seed(cfg.experiment.seed)
    logger.info(f"Loading dataset from: {cfg.data.input_path}")

    # Load Data
    try:
        raw_dataset = load_dataset("imagefolder", data_dir=str(cfg.data.input_path))
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    if "label" not in raw_dataset["train"].features:
        logger.error("Dataset missing 'label' column.")
        return

    # Split
    dataset_splits = raw_dataset["train"].train_test_split(
        test_size=cfg.data.test_size,
        seed=cfg.experiment.seed,
        stratify_by_column="label"
    )

    labels = raw_dataset["train"].features["label"].names
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    logger.info(f"Classes found: {len(labels)}")

    # Transforms
    processor = AutoImageProcessor.from_pretrained(cfg.model.backbone, use_fast=True)
    train_tf, test_tf = get_transforms(cfg, processor)
    
    dataset_splits["train"] = dataset_splits["train"].with_transform(train_tf)
    dataset_splits["test"] = dataset_splits["test"].with_transform(test_tf)

    # Manager
    trainer_manager = AdapterTrainer(cfg, len(labels), label2id, id2label)

    # Execution Loop
    for rank in cfg.adapters.ranks:
        try:
            trainer_manager.train(rank, dataset_splits)
        except Exception as e:
            logger.error(f"CRITICAL ERROR training Rank {rank}: {e}", exc_info=True)

    logger.info("All training tasks completed successfully.")


if __name__ == "__main__":
    main()