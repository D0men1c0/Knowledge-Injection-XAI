# Knowledge Injection via XAI: Distributed Framework

A Spark-based framework for evaluating Vision Transformer adapters using Explainable AI (XAI) metrics. This project uses **Apache Spark** for distributed inference and **PyTorch (PEFT/LoRA)** for model adaptation, containerized via Docker for stability and reproducibility.

---

## Environment Setup (Docker)

You don't need to install Python, CUDA, or Spark manually. Everything runs inside a container.

### Prerequisites
* **Docker Desktop** installed on Windows.
* **WSL 2 Integration** enabled in Docker Settings.
* **NVIDIA Drivers** updated on Windows.

### 1. Build & Start the Environment
Open your terminal in the project root and run:

```bash
# Build and start the container in background
docker-compose up -d --build
```
## 2. Enter the Container

To run scripts, you must be inside the Linux shell:

```bash
docker exec -it xai_container bash
```

Once inside, verify GPU access:

```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'NO GPU')"
```

---

## Pipeline Execution

Run these commands inside the container. The pipeline consists of **4 sequential layers**.

---

### 1. Bronze Layer (Feature Extraction)

Extracts static features (CLS token, Patch Embeddings) from the frozen DINOv2 backbone.

**Input:** Raw images in `data/raw/source`  
**Output:** Parquet files in `data/processed/bronze_parquet`

```bash
python3 -m src.run_bronze
```

---

### 2. Platinum Layer (LoRA Training)

Trains the Adapter Zoo (Rank 4, 16, 32) on your dataset. This creates the models used for evaluation.

**Input:** Raw images  
**Output:** Trained weights in `artifacts/adapters/`

```bash
python3 -m src.run_training
```

---

### 3. Silver Layer (Distributed XAI)

The core analysis engine. Loads the trained LoRA adapters, performs inference on the Bronze data, and calculates XAI metrics (Entropy, Deletion Score, etc.).

**Input:** Bronze Parquet + Trained Adapters  
**Output:** Analytical table in `data/processed/silver_parquet`

```bash
python3 -m src.run_silver
```

---

### 4. Gold Layer (Meta-Dataset)

Aggregates the Silver results to produce the final Meta-Dataset (Model-level statistics).

**Input:** Silver Parquet  
**Output:** Final CSV/Parquet in `data/processed/gold_parquet`

```bash
python3 -m src.run_gold
```

---

## Data Management

**Where are my files?**  
The container uses a **Bind Mount**.

Any file generated inside `/app/data` in Docker appears instantly in your Windows `data/` folder.  
You do **not** need to copy files manually.

---

## Configuration

Edit `configuration/config.yaml` to change:

- Batch sizes (reduce if OOM)
- Spark resources (Driver/Executor memory)
- LoRA Ranks to train/evaluate

---

## Troubleshooting

- **GPU Memory Error:** Reduce `batch_size` in `config.yaml` (e.g., from `64` to `16`)
- **Changes not applying:**  
  - Python code → instant  
  - Requirements / Dockerfile →  
    ```bash
    docker-compose up -d --build
    ```
- **Stop everything:**
  ```bash
  docker-compose down
  ```
