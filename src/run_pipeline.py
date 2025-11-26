import os
import sys
import time
import csv
import threading
import psutil
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession
from src.pipeline.bronze_layer import run_bronze, BronzeConfig

# --- CONFIGURATION ---
SPARK_DRIVER_MEM = "4g"
BATCH_SIZE = 32
LOG_FILE = "experiments_log.csv"

class ResourceMonitor(threading.Thread):
    """Background thread to track system resource usage."""
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_log = []
        self.ram_log = []

    def run(self):
        while not self.stop_event.is_set():
            self.cpu_log.append(psutil.cpu_percent())
            self.ram_log.append(psutil.virtual_memory().percent)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

    def get_stats(self):
        if not self.cpu_log: return 0.0, 0.0
        avg_cpu = sum(self.cpu_log) / len(self.cpu_log)
        max_ram = max(self.ram_log)
        return avg_cpu, max_ram

def log_experiment(duration, num_imgs, batch, device, cpu_avg, ram_max):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Images", "Batch", "Sec", "Img/Sec", "Device", "Avg_CPU%", "Max_RAM%"])
        
        img_per_sec = num_imgs / duration if duration > 0 else 0
        writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            num_imgs, batch, f"{duration:.1f}", f"{img_per_sec:.1f}",
            device, f"{cpu_avg:.1f}", f"{ram_max:.1f}"
        ])
    print(f"\n[INFO] Metrics saved: {duration:.1f}s | RAM Peak: {ram_max}% | CPU Avg: {cpu_avg}%")

def main():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    print(f"[INIT] Starting Spark with {SPARK_DRIVER_MEM} JVM Heap limit...")
    
    spark = SparkSession.builder \
        .appName("BronzeLayer_Optimized") \
        .master("local[*]") \
        .config("spark.driver.memory", SPARK_DRIVER_MEM) \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Load Data
    input_dir = Path("data/raw/source")
    paths = [str(p.absolute()) for p in input_dir.rglob("*.jpg")]
    if not paths:
        print("No images found."); return

    df_input = spark.createDataFrame([(p,) for p in paths], schema=["image_path"])
    config = BronzeConfig("facebook/dinov2-base", "image_path", "data/processed/bronze_parquet", BATCH_SIZE)

    # 2. Start Monitor
    monitor = ResourceMonitor()
    monitor.start()

    # 3. Run Pipeline
    print(f"[RUN] Processing {len(paths)} images on GPU (Batch: {BATCH_SIZE})...")
    start = time.time()
    
    try:
        run_bronze(spark, df_input, config)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
    finally:
        monitor.stop()
        monitor.join()

    duration = time.time() - start
    
    # 4. Check & Log
    try:
        df_res = spark.read.parquet(config.output_path)
        device = df_res.select("device").first()["device"]
        cpu, ram = monitor.get_stats()
        log_experiment(duration, len(paths), BATCH_SIZE, device, cpu, ram)
    except Exception as e:
        print(f"[WARN] Could not verify output: {e}")

    spark.stop()

if __name__ == "__main__":
    main()