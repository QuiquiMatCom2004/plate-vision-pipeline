"""Exporta modelo YOLO a ONNX y/o TensorRT y benchmark de latencia.

Uso:
    python scripts/export_and_benchmark.py model=yolov11n.pt img_size=640 device=cuda
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="export")
def export(cfg: DictConfig) -> None:
    """Exporta a ONNX/TensorRT y mide latencia."""
    model = YOLO(cfg.model)

    # 1. Exportar a ONNX
    if cfg.export_onnx:
        onnx_path = Path(cfg.output_dir) / f"{Path(cfg.model).stem}.onnx"
        logger.info("Exportando a ONNX: %s", onnx_path)
        model.export(format="onnx", imgsz=cfg.img_size, half=cfg.half_precision)
        logger.info("ONNX exportado.")

    # 2. Exportar a TensorRT (solo si device=cuda)
    if cfg.export_trt and cfg.device == "cuda":
        trt_path = Path(cfg.output_dir) / f"{Path(cfg.model).stem}.engine"
        logger.info("Exportando a TensorRT: %s", trt_path)
        model.export(format="engine", imgsz=cfg.img_size, half=True)
        logger.info("TensorRT exportado.")

    # 3. Benchmark de latencia
    dummy_img = np.random.randint(0, 255, (cfg.img_size, cfg.img_size, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(5):
        _ = model.predict(dummy_img, verbose=False)

    # Medimos
    times = []
    for _ in range(cfg.benchmark_iters):
        start = time.perf_counter()
        _ = model.predict(dummy_img, verbose=False)
        times.append(time.perf_counter() - start)

    latency_ms = np.mean(times) * 1000
    p99_ms = np.percentile(times, 99) * 1000

    logger.info("Latencia: mediana=%.2f ms, p99=%.2f ms", latency_ms, p99_ms)

    # Guardamos resultados
    results = {
        "model": cfg.model,
        "img_size": cfg.img_size,
        "device": cfg.device,
        "latency_ms": latency_ms,
        "p99_ms": p99_ms,
        "iterations": cfg.benchmark_iters,
    }

    with open(Path(cfg.output_dir) / "benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark guardado en %s", cfg.output_dir)


if __name__ == "__main__":
    export()