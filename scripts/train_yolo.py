"""Entrenamiento de YOLOv11 (Ultralytics) con registro en MLflow.

Uso:
    python scripts/train_yolo.py data=/path/to/dataset.yaml epochs=50 imgsz=640

El script:
1. Lee la configuración vía Hydra (o CLI).
2. Inicia un run de MLflow.
3. Entrena el modelo con los hiperparámetros dados.
4. Loguea métricas (mAP50, precision, recall) y el modelo entrenado.
5. Sube el modelo a HuggingFace Hub (opcional).
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


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig) -> None:
    """Entrena un modelo YOLO con los hiperparámetros de configuración."""
    # 1. Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        # Loggeamos la configuración completa
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # 2. Cargamos el modelo base
        model = YOLO(cfg.model)  # p.ej. "yolov11n.pt" o "yolov11s.pt"

        # 3. Entrenamos
        logger.info("Iniciando entrenamiento: %s epochs", cfg.epochs)
        results = model.train(
            data=cfg.data,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch_size,
            patience=cfg.patience,
            device=cfg.device,
            project=cfg.output_dir,
            name=cfg.run_name,
            exist_ok=True,
        )

        # 4. Loggeamos métricas
        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50": metrics["metrics/mAP50"],
            "mAP50-95": metrics["metrics/mAP50-95"],
            "precision": metrics["metrics/precision"],
            "recall": metrics["metrics/recall"],
            "val_loss": metrics["metrics.val_loss"],
        })

        # 5. Loggeamos el modelo entrenado
        best_weights = Path(cfg.output_dir) / cfg.run_name / "weights" / "best.pt"
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            artifact_path="yolo_model",
            conda_env="environment.yml",  # opcional: definir conda env
        )

        # 6. (Opcional) Push a HuggingFace Hub
        if cfg.get("hf_repo_id"):
            model.push_to_hub(
                repo_id=cfg.hf_repo_id,
                token=os.getenv("HF_TOKEN"),
                commit_message=f"Fine-tune: {cfg.run_name}",
            )
            logger.info("Modelo publicado en HuggingFace Hub: %s", cfg.hf_repo_id)

        logger.info("Entrenamiento completado. Métricas: %s", metrics)


if __name__ == "__main__":
    train()