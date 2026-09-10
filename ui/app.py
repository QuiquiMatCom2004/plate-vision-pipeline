"""Demo visual del pipeline — Gradio, solo red local (sin share=True).

Los modelos (YOLO, SAM2, VLM, cliente de measure) se cargan UNA sola vez al
importar este módulo — mismo motivo que el lifespan de FastAPI en
`api/main.py`: reconstruirlos en cada request sería inviable en latencia.

Uso (desde el servidor, con acceso solo por túnel SSH):
    python ui/app.py
    # en tu máquina local:
    ssh -L 7860:127.0.0.1:7860 server-casa
    # abrir http://127.0.0.1:7860
"""

import numpy as np

import gradio as gr

from plate_vision_pipeline.api.deps import build_pipeline
from plate_vision_pipeline.config import Settings
from plate_vision_pipeline.state import create_initial_pipelinestate

settings = Settings(_env_file=".env")
graph = build_pipeline(settings)


def analyze(image: np.ndarray):
    state = create_initial_pipelinestate(image)
    result = graph.invoke(state)

    segmentations = result.get("segmentations", [])
    annotations = [
        (seg["mask"], f"{seg['cls']} ({seg['conf']:.2f})") for seg in segmentations
    ]

    description = result.get("description", "")
    structure = result.get("structure", {})

    return (image, annotations), description, structure


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", label="Foto del plato"),
    outputs=[
        gr.AnnotatedImage(label="Detección + segmentación (SAM2)"),
        gr.Textbox(label="Descripción (VLM)"),
        gr.JSON(label="Análisis estructurado (PlateAnalysis)"),
    ],
    title="Plate Vision Pipeline",
    description="detect → segment → describe → measure",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
