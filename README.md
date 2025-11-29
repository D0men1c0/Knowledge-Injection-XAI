# Knowledge Injection via XAI: Predicting OOD Robustness

**Research Question:** Can Explainability (XAI) metrics computed on clean images predict model robustness under Out-of-Distribution (OOD) corruptions?

A distributed framework for evaluating Vision Transformer adapters using Explainable AI metrics. Built with **Apache Spark** for scalable inference and **PyTorch (PEFT/LoRA)** for parameter-efficient fine-tuning.

---

## Overview

This project investigates whether attention-based XAI metrics (Entropy, Deletion Score) extracted from clean images can serve as early indicators of model robustness under distribution shift.

### Key Findings

| Finding | Result |
|---------|--------|
| **XAI predicts robustness** | ROC-AUC ~0.74 using Meta-Learner |
| **Entropy is most informative** | Higher entropy → less robust (r = -0.17) |
| **Lower LoRA rank generalizes better** | Rank 4 (95%) vs Rank 32 (76%) on OOD data |
| **Blur is most challenging** | Up to 81% accuracy drop at heavy level |

---

## Architecture

### Medallion Pipeline (Spark-based)

```
Bronze Layer    →    Silver Layer    →    OOD Layer    →    Gold Layer
(Embeddings)         (XAI Metrics)        (Corruptions)      (Meta-Learner)
```

| Layer | Purpose | Output |
|-------|---------|--------|
| **Bronze** | Extract DINOv2 embeddings (CLS + Patch tokens) | `bronze_parquet/` |
| **Silver** | Compute XAI metrics on clean images | `silver_parquet/` |
| **OOD** | Apply corruptions and evaluate robustness | `ood_parquet/` |
| **Gold** | Train Meta-Learner, compute correlations | `gold_parquet/` |

### Model Architecture

- **Backbone:** `facebook/dinov2-base` (ViT-B/14, 86M parameters)
- **Adaptation:** LoRA with DoRA + RsLoRA (ranks 4, 16, 32)
- **Target Modules:** query, value, fc1, fc2

### XAI Metrics

| Metric | Description |
|--------|-------------|
| **Attention Entropy** | Normalized Shannon entropy of attention weights |
| **Sparsity** | Gini coefficient measuring attention concentration |
| **Deletion Score** | AUC when removing high-attention patches (RISE) |
| **Insertion Score** | AUC when revealing patches from blank image |

### OOD Corruptions

| Type | Severity Levels | Parameters |
|------|-----------------|------------|
| Gaussian Noise | shallow, medium, heavy | σ ∈ {15, 40, 80} |
| Blur | shallow, medium, heavy | radius ∈ {1.0, 3.0, 6.0} |
| Contrast | shallow, medium, heavy | factor ∈ {0.7, 0.4, 0.15} |

---

## Adapter Zoo: LoRA Training

The **Adapter Zoo** contains three LoRA adapters with varying capacities, trained via `src/run_train.py`.

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| **Technique** | DoRA (Weight-Decomposed) + RsLoRA (Rank-Stabilized) |
| **Ranks** | 4, 16, 32 |
| **Alpha Scaling** | α = 2 × rank |
| **Dropout** | 0.1 |
| **Target Modules** | query, value, fc1, fc2 |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 3×10⁻⁴ |
| **Epochs** | 15 |
| **Batch Size** | 16 (effective 32 with grad accumulation) |
| **Gradient Accumulation** | 2 steps |
| **Precision** | FP16 (mixed precision) |
| **Validation Split** | 20% stratified |

### Data Augmentation

| Augmentation | Parameter |
|--------------|-----------|
| Random Rotation | ±30° |
| Horizontal Flip | p=0.5 |
| Color Jitter | brightness/contrast 0.2 |
| Random Resized Crop | 224×224 from 256×256 |

### Training Results

| Rank | Trainable Params | Train Loss | Eval Loss | Accuracy | F1 Score |
|------|------------------|------------|-----------|----------|----------|
| 4 | 702K (0.80%) | 0.251 | 0.120 | 96.1% | 96.0% |
| 16 | 2.25M (2.53%) | 0.307 | 0.122 | 96.7% | 96.7% |
| 32 | 4.31M (4.75%) | 0.477 | 0.169 | 95.4% | 95.4% |

Configuration: `configuration/train_config.yaml`  
Output: `artifacts/adapters_enhanced/lora_r{4,16,32}/`

---

## Project Structure

```
Knowledge-Injection-XAI/
├── src/
│   ├── pipeline/
│   │   ├── bronze_layer.py      # Distributed feature extraction
│   │   ├── silver_layer.py      # XAI metrics computation
│   │   ├── ood_layer.py         # Corruption-based evaluation
│   │   └── golden_layer.py      # Meta-Learner training
│   ├── xai/
│   │   └── metrics.py           # XAI metric implementations
│   └── utils/
│       └── train_configs.py     # Training configurations
├── artifacts/
│   └── adapters_enhanced/       # Trained LoRA adapters (r4, r16, r32)
├── data/
│   ├── raw/source/              # Input images (37 classes)
│   └── processed/
│       ├── bronze_parquet/      # Extracted embeddings
│       ├── silver_parquet/      # XAI features
│       ├── ood_parquet/         # OOD evaluation results
│       └── gold_parquet/        # Final analytics
├── notebooks/
│   └── results_presentation.ipynb  # Interactive results
├── configuration/
│   ├── config.yaml              # Pipeline configuration
│   └── train_config.yaml        # Training hyperparameters
└── logging/
    └── training_metrics.csv     # Adapter training results
```

---

## Quick Start

### Prerequisites

- Docker Desktop with WSL 2 and NVIDIA GPU support
- NVIDIA drivers updated on host

### 1. Start Environment

```bash
cd /mnt/c/PythonProjects/Knowledge-Injection-XAI
docker compose up -d --build
docker exec -it xai_container bash
```

### 2. Run Pipeline

```bash
# Bronze: Extract embeddings
python3 -m src.run_bronze

# Train adapters (optional - pre-trained included)
python3 -m src.run_train

# Silver: Compute XAI metrics
python3 -m src.run_silver

# OOD: Evaluate robustness
python3 -m src.run_ood

# Gold: Train Meta-Learner
python3 -m src.run_golden
```

### 3. View Results

Open `notebooks/results_presentation.ipynb` in VS Code or Jupyter.

---

## Configuration

Edit `configuration/config.yaml`:

```yaml
spark:
  driver_memory: "8g"
  executor_memory: "4g"

model:
  batch_size: 32        # Reduce if OOM
  lora_ranks: [4, 16, 32]

corruptions:
  types: ["gaussian_noise", "blur", "contrast"]
  levels: ["shallow", "medium", "heavy"]
```

---

## Results Summary

### Adapter Performance (OOD)

| Adapter | Clean Accuracy | OOD Accuracy | Gap |
|---------|---------------|--------------|-----|
| Rank 4 | 96.1% | 95.0% | 1.1pp |
| Rank 16 | 96.7% | 89.2% | 7.5pp |
| Rank 32 | 95.4% | 75.9% | 19.5pp |

### Meta-Learner Performance

| Model | ROC-AUC | CV AUC |
|-------|---------|--------|
| XGBoost_Tuned | 0.739 | 0.737 ± 0.006 |
| RandomForest | 0.731 | 0.729 ± 0.005 |
| LogisticRegression | 0.718 | 0.717 ± 0.005 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU Memory Error | Reduce `batch_size` in config.yaml |
| Spark OOM | Reduce `driver_memory` or partition data |
| Container conflict | `docker rm -f xai_container && docker compose up -d` |

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.