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


# LoRA Adapter (Rank 4)

Fine-tuned LoRA adapter for `facebook/dinov2-base`.

- **Training Date:** 2025-11-28 08:17
- **Dataset Path:** `data\raw\source`
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Use DoRA:** True
- **Use RsLoRA:** True

## Training

- **Backbone:** `facebook/dinov2-base`
- **LoRA Rank:** 4
- **LoRA Alpha:** 8.0
- **LoRA Dropout:** 0.1
- **Use DoRA:** True
- **Use RsLoRA:** True
- **Target Modules:** ['query', 'value', 'fc1', 'fc2']
- **Batch Size:** 16
- **Learning Rate:** 0.0003
- **Epochs:** 15
- **Precision:** fp16
- **Training Time (min):** 37.41

## Parameters

- **Trainable Parameters:** 701,989
- **Total Parameters:** 87,274,826
- **Trainable %:** 0.8043%

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.9606 |
| **Precision** | 0.9649 |
| **Recall** | 0.9606 |
| **F1 Score** | 0.9604 |
| **Eval Loss** | 0.1195 |

## How to Use

```python
from transformers import AutoModelForImageClassification
from peft import PeftModel

base_model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base")
model = PeftModel.from_pretrained(base_model, "./lora_r4")
model.eval()
```
