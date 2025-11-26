"""
Telemetry and logging utilities.

Handles background resource monitoring (CPU, RAM, GPU) and experiment logging.
"""
import csv
import logging
import os
import statistics
import subprocess
import threading
import time
import psutil
from datetime import datetime
from typing import List, Dict


# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Telemetry")


class ResourceMonitor(threading.Thread):
    """Background thread to track system and GPU resources."""

    def __init__(self, interval: float = 0.5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_log: List[float] = []
        self.ram_log: List[float] = []
        self.gpu_log: List[float] = []

    def _get_gpu_load(self) -> float:
        """Queries nvidia-smi for GPU utilization."""
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            return float(result.strip())
        except Exception:
            return 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.cpu_log.append(psutil.cpu_percent())
            self.ram_log.append(psutil.virtual_memory().percent)
            self.gpu_log.append(self._get_gpu_load())
            time.sleep(self.interval)

    def stop(self) -> None:
        self.stop_event.set()

    def get_stats(self) -> Dict[str, float]:
        """Computes aggregate statistics (Avg, Max)."""
        def _calc(data: List[float], prefix: str) -> Dict[str, float]:
            if not data:
                return {f"{prefix}_Avg": 0.0, f"{prefix}_Max": 0.0}
            return {
                f"{prefix}_Avg": round(statistics.mean(data), 2),
                f"{prefix}_Max": round(max(data), 2),
            }

        stats = {}
        stats.update(_calc(self.cpu_log, "CPU"))
        stats.update(_calc(self.ram_log, "RAM"))
        stats.update(_calc(self.gpu_log, "GPU"))
        return stats


class ExperimentLogger:
    """Handles writing experiment metrics to CSV."""

    def __init__(self, log_file: str):
        self.log_file = log_file

    def log(
        self,
        duration: float,
        num_imgs: int,
        batch_size: int,
        device: str,
        res_stats: Dict[str, float],
        spark_conf: Dict[str, str]
    ) -> None:
        """Writes a summary row to the CSV log."""
        file_exists = os.path.isfile(self.log_file)
        
        img_per_sec = num_imgs / duration if duration > 0 else 0
        
        headers = [
            "Timestamp", "Images", "Batch", "Device",
            "Duration_Sec", "Img_Per_Sec",
            "GPU_Avg", "GPU_Max", "RAM_Avg", "RAM_Max", "CPU_Avg",
            "Spark_Driver_Mem", "Spark_OffHeap"
        ]

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_imgs, batch_size, device,
            f"{duration:.2f}", f"{img_per_sec:.2f}",
            res_stats.get("GPU_Avg", 0), res_stats.get("GPU_Max", 0),
            res_stats.get("RAM_Avg", 0), res_stats.get("RAM_Max", 0),
            res_stats.get("CPU_Avg", 0),
            spark_conf.get("spark.driver.memory", "N/A"),
            spark_conf.get("spark.memory.offHeap.size", "N/A")
        ]

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)

        logger.info(f"Logged: {img_per_sec:.1f} img/s | GPU Max: {res_stats.get('GPU_Max')}%")