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


# LoRA Adapter (Rank 16)

Fine-tuned LoRA adapter for `facebook/dinov2-base`.

- **Training Date:** 2025-11-28 08:54
- **Dataset Path:** `data\raw\source`
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Use DoRA:** True
- **Use RsLoRA:** True

## Training

- **Backbone:** `facebook/dinov2-base`
- **LoRA Rank:** 16
- **LoRA Alpha:** 32.0
- **LoRA Dropout:** 0.1
- **Use DoRA:** True
- **Use RsLoRA:** True
- **Target Modules:** ['query', 'value', 'fc1', 'fc2']
- **Batch Size:** 16
- **Learning Rate:** 0.0003
- **Epochs:** 15
- **Precision:** fp16
- **Training Time (min):** 36.77

## Parameters

- **Trainable Parameters:** 2,250,277
- **Total Parameters:** 88,823,114
- **Trainable %:** 2.5334%

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.9674 |
| **Precision** | 0.9688 |
| **Recall** | 0.9674 |
| **F1 Score** | 0.9672 |
| **Eval Loss** | 0.1215 |

## How to Use

```python
from transformers import AutoModelForImageClassification
from peft import PeftModel

base_model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base")
model = PeftModel.from_pretrained(base_model, "./lora_r16")
model.eval()
```
