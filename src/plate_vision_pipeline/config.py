from typing import Literal

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    vlm_mode: Literal['local', 'api'] = 'api'
    device: Literal['cpu','cuda'] = 'cpu'
    max_measure_attempts: int = 3
    apikey: str | None = None
    trt_engine_path: str | None = None
    yolo_weights_path: str
    sam2_checkpoint_path: str
    min_yolo_conf: float = 0.3
    vlm_name: dict[Literal['local','api'],list[str]]
    measure_model_name: str = 'google/gemma-4-31b-it:free'
    measure_max_tokens: int = 1024
