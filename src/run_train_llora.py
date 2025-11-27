"""
Platinum Layer: LoRA Training Pipeline.

Trains LoRA adapters with stratified validation, extended epochs,
and comprehensive metrics (Accuracy, Precision, Recall, Loss).
"""
import os
import shutil
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from peft import get_peft_model, LoraConfig
from src.configs import load_config

# Ensure deterministic behavior
torch.manual_seed(42)


def compute_metrics(eval_pred) -> dict:
    """
    Computes Accuracy, Precision, and Recall for validation.
    
    Args:
        eval_pred: Tuple containing (predictions, labels).
                   predictions shape: [batch_size, num_classes]
                   labels shape: [batch_size]
    
    Returns:
        dict: Calculated metrics.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall
    }


def train_adapter(rank: int, cfg, dataset_splits, num_labels: int) -> None:
    """
    Executes the training loop for a specific LoRA rank.
    
    Args:
        rank: LoRA rank parameter.
        cfg: Experiment configuration.
        dataset_splits: Dictionary with 'train' and 'test' datasets.
        num_labels: Number of target classes.
    """
    print(f"\n{'='*40}")
    print(f"   STARTING TRAINING: LoRA Rank {rank}")
    print(f"{'='*40}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Base Model
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model.backbone_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # 2. Inject LoRA with higher capacity parameters
    peft_config = LoraConfig(
        inference_mode=False,
        r=rank,
        lora_alpha=cfg.adapters.alpha * 2,  # Scaling alpha with rank for stability
        lora_dropout=cfg.adapters.dropout,
        target_modules=cfg.adapters.target_modules,
        bias="lora_only"      # Train biases for better convergence
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    # 3. Output Directory
    output_dir = cfg.paths.models / f"lora_r{rank}"
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = 300

    # 4. Extended Training Configuration
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        num_train_epochs=20,
        
        # --- LOGGING CONFIGURATION ---
        logging_strategy="steps",
        logging_steps=steps,
        
        # --- EVALUATION CONFIGURATION ---
        eval_strategy="steps",
        eval_steps=steps,
        
        # Saving
        save_strategy="steps",
        save_steps=steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        
        fp16=(device.type == "cuda"),
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False 
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    # 6. Train
    start_time = time.time()
    trainer.train()
    duration = (time.time() - start_time) / 60
    print(f"\n[DONE] Rank {rank} trained in {duration:.1f} minutes.")
    
    # 7. Save Adapter
    print(f"Saving best model to: {output_dir}")
    model.save_pretrained(str(output_dir))
    peft_config.save_pretrained(str(output_dir))


def main() -> None:
    # 1. Load Config
    try:
        cfg = load_config("configuration/config.yaml")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. Load & Split Dataset (Stratified)
    print(f"Loading dataset from {cfg.paths.input}...")
    raw_dataset = load_dataset("imagefolder", data_dir=str(cfg.paths.input))
    
    # Perform Stratified Split (80% Train, 20% Test)
    dataset_splits = raw_dataset["train"].train_test_split(
        test_size=0.2, 
        seed=42, 
        stratify_by_column="label"
    )
    
    labels = raw_dataset["train"].features["label"].names
    num_labels = len(labels)
    print(f"Dataset Split: {len(dataset_splits['train'])} Train, {len(dataset_splits['test'])} Test.")
    print(f"Classes: {num_labels}")

    # 3. Preprocessing
    processor = AutoImageProcessor.from_pretrained(cfg.model.backbone_name)
    
    def transforms(examples):
        images = [x.convert("RGB") for x in examples["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["label"] = examples["label"]
        return inputs

    # Apply transforms to both splits
    processed_splits = {
        "train": dataset_splits["train"].with_transform(transforms),
        "test": dataset_splits["test"].with_transform(transforms)
    }

    # 4. Training Loop
    for rank in cfg.adapters.ranks:
        try:
            train_adapter(rank, cfg, processed_splits, num_labels)
        except Exception as e:
            print(f"CRITICAL ERROR Training Rank {rank}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()