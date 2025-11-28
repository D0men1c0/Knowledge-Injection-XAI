# Knowledge Injection via XAI: Distributed Framework

A Spark-based framework for evaluating Vision Transformer adapters using Explainable AI (XAI) metrics. This project uses **Apache Spark** for distributed inference and **PyTorch (PEFT/LoRA)** for model adaptation, containerized via Docker for stability and reproducibility.

---

## Environment Setup (Docker)

You don't need to install Python, CUDA, or Spark manually. Everything runs inside a container.

### Prerequisites
* **Docker Desktop** installed on Windows.
* **WSL 2 Integration** enabled in Docker Settings.
* **NVIDIA Drivers** updated on Windows.

---

## 1. Navigate to Project Folder

⚠️ **Important:** For the bind mount to work correctly, you **must run Docker commands from the folder that contains the `docker-compose.yml` file**.  

On Windows/WSL2, the correct folder is:

```bash
cd /mnt/c/PythonProjects/Knowledge-Injection-XAI
```

Check that the compose file exists:

```bash
ls docker-compose.yml
```

---

## 2. Build & Start the Environment

From the project root, run:

```bash
docker compose up -d --build
```

> If there is a conflict with the container name, first remove old containers:
```bash
docker rm -f xai_container
docker compose down -v --remove-orphans
```

---

## 3. Stop or Restart (Clean Temporary Files)
If you need to stop the environment or free up space (e.g., clear Spark temporary files in /tmp inside the container), perform a clean restart:

1. Stop and remove the container (cleans runtime trash)
    ```bash
    docker compose down
    ```

2. Start a fresh instance (fast start, no rebuild)
    ```bash
    docker compose up -d
    ```

---

## 4. Enter the Container

To run scripts, you must be inside the Linux shell:

```bash
docker exec -it xai_container bash
```

Once inside, verify GPU access:

```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'NO GPU')"
```

---

## Bind Mount Behavior

All files created in `/app` inside the container are **saved directly in your Windows folder**:

/app ↔ C:\PythonProjects\Knowledge-Injection-XAI

- No duplication occurs.  
- Modifying files in the container instantly updates the Windows folder.  
- Modifying files on Windows instantly updates `/app` in the container.  
- The Docker image itself **does not store these files**, so your data is safe outside Docker.

Test:

```bash
echo "BIND OK" > /app/test_bind.txt
```

Check on Windows:  
`C:\PythonProjects\Knowledge-Injection-XAI\test_bind.txt` should appear.

---

## Pipeline Execution

Run these commands **inside the container**. The pipeline has 4 sequential layers.

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

Trains the Adapter Zoo (Rank 4, 16, 32) on your dataset.

**Input:** Raw images  
**Output:** Trained weights in `artifacts/adapters/`

```bash
python3 -m src.run_training
```

---

### 3. Silver Layer (Distributed XAI)

Loads trained LoRA adapters, performs inference on Bronze data, calculates XAI metrics (Entropy, Deletion Score, etc.)

**Input:** Bronze Parquet + Trained Adapters  
**Output:** Analytical table in `data/processed/silver_parquet`

```bash
python3 -m src.run_silver
```

---

### 4. Gold Layer (Meta-Dataset)

Aggregates Silver results to produce final Meta-Dataset (model-level statistics).

**Input:** Silver Parquet  
**Output:** Final CSV/Parquet in `data/processed/gold_parquet`

```bash
python3 -m src.run_gold
```

---

## Configuration

Edit `configuration/config.yaml` to change:

- Batch sizes (reduce if OOM)
- Spark resources (Driver/Executor memory)
- LoRA Ranks to train/evaluate

---

## Troubleshooting

- **GPU Memory Error:** Reduce `batch_size` in `config.yaml` (e.g., 64 → 16)
- **Changes not applying:**  
  - Python code → instant  
  - Requirements / Dockerfile →  
    ```bash
    docker compose up -d --build
    ```
- **Stop everything:**
    ```bash
    docker compose down
    ```
- **Bind mount not working:** Make sure:
  1. You are in the correct folder: `/mnt/c/PythonProjects/Knowledge-Injection-XAI`
  2. The container was recreated after modifying compose: `docker compose up -d --build`
  3. You are using `/mnt/c/...` in the compose file, not `/home/...` inside WSL