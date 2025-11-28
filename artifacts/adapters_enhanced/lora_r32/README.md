---
base_model: facebook/dinov2-base
library_name: peft
tags:
- base_model:adapter:facebook/dinov2-base
- lora
- computer-vision
- image-classification
- transformers
---


# LoRA Adapter (Rank 32)

Fine-tuned LoRA adapter for `facebook/dinov2-base`.

- **Training Date:** 2025-11-28 09:33
- **Dataset Path:** `data\raw\source`
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Use DoRA:** True
- **Use RsLoRA:** True

## Training

- **Backbone:** `facebook/dinov2-base`
- **LoRA Rank:** 32
- **LoRA Alpha:** 64.0
- **LoRA Dropout:** 0.1
- **Use DoRA:** True
- **Use RsLoRA:** True
- **Target Modules:** ['query', 'value', 'fc1', 'fc2']
- **Batch Size:** 16
- **Learning Rate:** 0.0003
- **Epochs:** 15
- **Precision:** fp16
- **Training Time (min):** 38.07

## Parameters

- **Trainable Parameters:** 4,314,661
- **Total Parameters:** 90,887,498
- **Trainable %:** 4.7473%

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.9538 |
| **Precision** | 0.9560 |
| **Recall** | 0.9538 |
| **F1 Score** | 0.9537 |
| **Eval Loss** | 0.1689 |

## How to Use

```python
from transformers import AutoModelForImageClassification
from peft import PeftModel

base_model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base")
model = PeftModel.from_pretrained(base_model, "./lora_r32")
model.eval()
```
