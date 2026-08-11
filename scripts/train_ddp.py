"""Entrenamiento distribuido con DDP (Data Distributed Data Parallel).

Usado en Kaggle (2xT4) o en máquinas con múltiples GPUs.

Uso:
    python -m torch.distributed.run --nproc_per_node=2 scripts/train_ddp.py \
        data=/path/to/dataset.yaml \
        model=yolov11n.pt \
        epochs=50
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train_ddp")
def train_ddp(cfg: DictConfig) -> None:
    """Entrenamiento DDP con sincronización de métricas en MLflow."""
    # 1. Determinamos el rank del proceso actual
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Solo el proceso 0 (rank 0) inicializa MLflow y loggea
    if local_rank == 0:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(cfg.experiment_name)
        mlflow.start_run()
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    # 2. Configuramos el device
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # 3. Cargamos el modelo y lo movemos al device
    model = YOLO(cfg.model)
    model.to(device)

    # 4. Entrenamos con DDP
    logger.info("Rank %d: entrenando en %s", local_rank, device)
    results = model.train(
        data=cfg.data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch_size,
        patience=cfg.patience,
        device=device,
        project=cfg.output_dir,
        name=cfg.run_name,
        exist_ok=True,
        # DDP flags
        workers=0,  # evita conflictos con DataLoader multiproceso
        amp=True,   # mixed precision
    )

    # 5. Sincronizamos métricas (solo rank 0 loggea)
    if local_rank == 0:
        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50": metrics["metrics/mAP50"],
            "mAP50-95": metrics["metrics/mAP50-95"],
        })

        best_weights = Path(cfg.output_dir) / cfg.run_name / "weights" / "best.pt"
        mlflow.pytorch.log_model(model.model, artifact_path="yolo_model_ddp")
        mlflow.end_run()

        logger.info("DDP entrenamiento completado. Rank 0 loggeó métricas.")


if __name__ == "__main__":
    train_ddp()