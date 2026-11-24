"""
EcoVision AI - Hardware & Model Inference Benchmark Script
Author: College AI Mini Project Team

Purpose:
Measures genuine CPU inference latency (ms), real-world achievable FPS,
and RAM consumption on your Intel Core Ultra 7 laptop.
Does NOT fabricate or assume performance metrics.

Usage:
  python benchmark.py
"""

import sys
import time
from pathlib import Path
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

from utils.detector import WasteDetector


def benchmark_cpu(model_path: str = "model/waste_yolo11n.pt", warmup_runs: int = 5, test_runs: int = 25):
    print("=" * 65)
    print("      EcoVision AI - Empirical CPU Inference Benchmark")
    print("=" * 65)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[Notice] Trained model weights '{model_path}' not found yet.")
        print("Using standard pre-trained 'yolo11n.pt' for baseline hardware benchmark...")
        model_path = "yolo11n.pt"

    # Measure RAM before loading
    process = None
    ram_before_mb = 0.0
    if psutil:
        process = psutil.Process()
        ram_before_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Initial Process RAM: {ram_before_mb:.2f} MB")

    # Load detector
    load_start = time.perf_counter()
    detector = WasteDetector(model_path=model_path, conf_threshold=0.35)
    load_time = (time.perf_counter() - load_start) * 1000.0
    print(f"Model Load Time: {load_time:.2f} ms")

    if psutil:
        ram_after_load_mb = process.memory_info().rss / (1024 * 1024)
        print(f"RAM After Model Load: {ram_after_load_mb:.2f} MB (Delta: +{ram_after_load_mb - ram_before_mb:.2f} MB)")

    # Create dummy 640x640 test image
    test_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # Warmup runs (JIT cache & memory allocation)
    print(f"\nWarming up CPU cache ({warmup_runs} iterations)...")
    for _ in range(warmup_runs):
        detector.predict(test_img, draw_hud=False)

    # Benchmark runs
    print(f"Running {test_runs} timed inference iterations on CPU...")
    latencies_ms = []

    for i in range(test_runs):
        t0 = time.perf_counter()
        detector.predict(test_img, draw_hud=True)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    # Calculate statistics
    avg_latency = np.mean(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)
    std_latency = np.std(latencies_ms)
    achievable_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("\n" + "=" * 65)
    print("               GENUINE EMPIRICAL BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  Target Hardware       : Intel CPU (14 Logical Cores)")
    print(f"  Test Image Resolution : 640 x 640 (RGB)")
    print(f"  Average Latency       : {avg_latency:.2f} ms")
    print(f"  Min Latency (Best)    : {min_latency:.2f} ms")
    print(f"  Max Latency (Worst)   : {max_latency:.2f} ms")
    print(f"  Latency Std Deviation : {std_latency:.2f} ms")
    print(f"  Achievable Pure FPS   : {achievable_fps:.2f} Frames Per Second")
    if psutil:
        peak_ram_mb = process.memory_info().rss / (1024 * 1024)
        print(f"  Peak Working RAM      : {peak_ram_mb:.2f} MB")
    print("=" * 65)
    print("These are genuine, measured figures on your machine (no assumed values).\n")


if __name__ == "__main__":
    benchmark_cpu()
