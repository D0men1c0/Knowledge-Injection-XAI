"""
Platinum Layer: LoRA Training Pipeline.

Trains multiple LoRA adapters based on the configuration zoo
and saves them to the artifacts directory for the Silver Layer.
"""
import os
import time
import shutil
from pathlib import Path
import torch
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

# Force standard deterministic behavior
torch.manual_seed(42)

def train_adapter(rank: int, cfg, dataset, num_labels: int):
    print(f"\n{'='*40}")
    print(f"   STARTING TRAINING: LoRA Rank {rank}")
    print(f"{'='*40}\n")
    
    # 1. Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Base Model
    print("Loading Base Model...")
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model.backbone_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # 3. Inject LoRA Config
    peft_config = LoraConfig(
        inference_mode=False,
        r=rank,
        lora_alpha=cfg.adapters.alpha,
        lora_dropout=cfg.adapters.dropout,
        target_modules=cfg.adapters.target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    
    print(f"Moving model to {device}...")
    model.to(device)

    model.print_trainable_parameters()

    # 4. Output Directory
    output_dir = cfg.paths.models / f"lora_r{rank}"
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        num_train_epochs=5,
        fp16=True if torch.cuda.is_available() else False,
        no_cuda=False,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False 
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DefaultDataCollator(),
    )

    # 7. Execute
    start = time.time()
    trainer.train()
    duration = (time.time() - start) / 60
    print(f"\n[DONE] Rank {rank} trained in {duration:.1f} minutes.")
    
    # 8. Save
    print(f"Saving adapter to: {output_dir}")
    model.save_pretrained(str(output_dir))
    peft_config.save_pretrained(str(output_dir))


def main():
    try:
        cfg = load_config("configuration/config.yaml")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    print(f"Loaded Config. Models path: {cfg.paths.models}")

    # 1. Load Dataset
    print(f"Loading dataset from {cfg.paths.input}...")
    raw_dataset = load_dataset("imagefolder", data_dir=str(cfg.paths.input))
    
    labels = raw_dataset["train"].features["label"].names
    num_labels = len(labels)
    print(f"Detected {num_labels} classes.")

    # 2. Preprocessing
    processor = AutoImageProcessor.from_pretrained(cfg.model.backbone_name)
    
    def transforms(examples):
        images = [x.convert("RGB") for x in examples["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["label"] = examples["label"]
        return inputs

    # Apply transform
    processed_dataset = raw_dataset["train"].with_transform(transforms)

    # 3. Loop through the "Zoo"
    for rank in cfg.adapters.ranks:
        try:
            train_adapter(rank, cfg, processed_dataset, num_labels)
        except Exception as e:
            print(f"CRITICAL ERROR Training Rank {rank}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()