import base64
import io
import logging
from typing import Any, Protocol

import anthropic
import numpy as np
import torch
from instructor import from_anthropic
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from ultralytics import YOLO

from plate_vision_pipeline.schema import PlateAnalysis
from plate_vision_pipeline.state import Detection

logger = logging.getLogger(__name__)


def _build_describe_prompt(detections: list[Detection]) -> str:
    if not detections:
        return "Describe los alimentos visibles en el plato de esta imagen."
    classes = ", ".join(sorted({d["cls"] for d in detections}))
    return (
        "Describe el plato de comida en esta imagen. "
        f"El detector identificó estos elementos: {classes}. "
        "Usa esa lista como referencia — no inventes alimentos que no estén en la imagen."
    )


class VLMBackend(Protocol):
    """Interfaz uniforme para los dos backends del VLM (Strategy pattern).

    `predict()` en DescribeModel delega acá sin saber si el backend es local
    (Qwen2-VL en memoria) o remoto (API de Anthropic) — esa diferencia queda
    encapsulada en cada implementación de `generate()`.
    """

    def generate(self, image: np.ndarray, detections: list[Detection]) -> str: ...


class LocalVLMBackend:
    """Backend Qwen2-VL cargado en memoria (mismo proceso, sin serialización)."""

    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor

    def generate(self, image: np.ndarray, detections: list[Detection]) -> str:
        prompt = _build_describe_prompt(detections)
        pil_image = Image.fromarray(image)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt").to(
            self.model.device
        )
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


class ApiVLMBackend:
    """Backend Claude API — la imagen viaja como base64 dentro de un mensaje JSON."""

    def __init__(self, client, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    def generate(self, image: np.ndarray, detections: list[Detection]) -> str:
        prompt = _build_describe_prompt(detections)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return message.content[0].text


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
        if self.model is None:
            return ""
        return self.model.generate(image, detections)

    def _load_candidate(self, name: str) -> VLMBackend:
        if self.vlm_mode == "local":
            return self._load_local_candidate(name)
        return self._load_api_candidate(name)

    def _load_local_candidate(self, name: str) -> LocalVLMBackend:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            name, device_map=self.device, quantization_config=bnb_config
        )
        processor = AutoProcessor.from_pretrained(name)
        return LocalVLMBackend(model=model, processor=processor)

    def _load_api_candidate(self, name: str) -> ApiVLMBackend:
        client = anthropic.Anthropic(api_key=self.apikey)
        return ApiVLMBackend(client=client, model_name=name)


class MeasureModel:
    """Modelo de medición con structured output via `instructor`.

    Recibe el client Anthropic crudo y lo envuelve con
    `instructor.from_anthropic` dentro del __init__ (el `patch()` genérico
    de instructor asume forma OpenAI — `client.chat.completions.create` —
    y no sirve para Anthropic). El client ya construido se inyecta (DI) —
    así el modelo es testeable sin tocar red.
    """

    DEFAULT_MODEL_NAME = "claude-3-5-sonnet-20241022"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        raw_client,
        model_name: str = DEFAULT_MODEL_NAME,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = from_anthropic(raw_client)
        self.model_name = model_name
        self.max_tokens = max_tokens

    def predict(self, description: str) -> Any:
        return self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            response_model=PlateAnalysis,
            messages=[{"role": "user", "content": description}],
        )
