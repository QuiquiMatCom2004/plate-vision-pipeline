import logging
from typing import Any

import numpy as np
from instructor import patch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

from plate_vision_pipeline.state import Detection

logger = logging.getLogger(__name__)


class DetectModel:
    def __init__(self, modelpath: str, device: str, min_yolo_conf: float) -> None:
        self.model = YOLO(modelpath)
        self.min_conf = min_yolo_conf
        self.model.to(device=device)

    def predict(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(image, conf=self.min_conf)[0]
        solve: list[Detection] = []
        if result.boxes is None:
            return solve
        for b in result.boxes:
            score = float(b.conf[0])
            bbox = b.xyxy[0].tolist()
            idx = int(b.cls[0])
            name_cls = result.names[idx]
            detection = Detection(bbox=bbox, conf=score, cls=name_cls)
            solve.append(detection)
        return solve


class SegmentModel:
    """Modelo de segmentación basado en SAM2 de Meta.

    Decisión de diseño (AGENTS.md): `set_image()` se llama UNA vez por imagen
    (paso caro), `predict()` una vez POR bbox (barato). El `config_file` se
    inyecta — no es un setting de entorno — para respetar Open/Closed: si
    mañana mejora una arquitectura de SAM2, cambiamos el config inyectado sin
    tocar el modelo.
    """

    predictor: SAM2ImagePredictor

    def __init__(self, config_file, checkpoint_path, device) -> None:
        model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device,
        )
        self.predictor = SAM2ImagePredictor(model)

    def predict(self, image: np.ndarray, boxes: list[list[float]]) -> list[np.ndarray]:
        # Edge case: si no hay boxes, ahorrar el set_image (paso caro).
        if not boxes:
            return []
        self.predictor.set_image(image)
        result: list[np.ndarray] = []
        for box in boxes:
            masks, _scores, _logits = self.predictor.predict(
                box=box, multimask_output=False
            )
            result.append(masks[0])
        return result


class DescribeModel:
    """VLM (Qwen2-VL-7B local / Claude API) con fallback ordenado de candidatos.

    `vlm_name[vlm_mode]` es una lista ordenada de candidatos. El __init__
    prueba cada uno vía `_load_candidate(name)` hasta que uno no falle.
    Si todos fallan, NO lanza — deja `self.model = None` y deja que el
    nodo decida el fallback en runtime (patrón "start gracefully"). Esto
    evita que el container crashee en boot si la GPU no está disponible.
    """

    def __init__(self, vlm_mode, vlm_name, device, apikey=None) -> None:
        self.vlm_mode = vlm_mode
        self.vlm_name = vlm_name
        self.device = device
        self.apikey = apikey
        self.model: Any = None
        for name in vlm_name[vlm_mode]:
            try:
                self.model = self._load_candidate(name)
                break
            except Exception as e:
                logger.warning("VLM candidate %s failed: %s", name, e)
                continue

    def predict(self, image: np.ndarray, detections: list[Detection]) -> str:
        raise NotImplementedError

    def _load_candidate(self, name: str) -> Any:
        # Implementación real (Qwen2-VL cuantizado NF4 / Anthropic client)
        # se definirá cuando se implemente el nodo describe.
        raise NotImplementedError


class MeasureModel:
    """Modelo de medición con structured output via `instructor`.

    Recibe el client crudo (Anthropic/OpenAI/etc.) y lo patchea con
    `instructor.patch` dentro del __init__. El client ya construido se
    inyecta (DI) — así el modelo es testeable sin tocar red.
    """

    def __init__(self, raw_client) -> None:
        self.client = patch(raw_client)

    def predict(self, description: str) -> Any:
        raise NotImplementedError
