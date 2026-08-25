# Roadmap — Plate Vision Pipeline

Mapa mental de trabajo: qué aprender, cómo trabajar, y en qué orden construir el proyecto.
Este documento responde al **por qué** y al **qué**, no al **cómo implementar**.

---

## Qué es este proyecto y por qué existe

Un agente de visión que recibe una imagen o documento y devuelve un JSON estructurado
pasando por cuatro etapas: detección → segmentación → descripción → estructuración.

**Por qué este proyecto específico:**
- Demuestra el ciclo completo de CV Engineer: datos → entrenamiento → optimización → deploy
- Combina tres áreas de alta demanda: Computer Vision, LLMs, y MLOps
- Cada componente tiene justificación técnica real, no es un collage de buzzwords
- Es demostrable en una entrevista técnica línea a línea

**Criterio de éxito del portfolio:**
> Si no puedes explicar cualquier parte en una pizarra en 30 minutos, no está lista.

---

## Mapa de conocimiento — qué necesitas saber y por qué

### Capa 1: Fundamentos (prerequisito para todo lo demás)

```
PyTorch
├── Por qué: todos los modelos del pipeline nacen en PyTorch
├── Qué necesitas saber:
│   ├── Tensores: shape, dtype, device — el lenguaje base
│   ├── nn.Module: cómo se define y carga un modelo
│   ├── Forward pass: tú lo defines, backward es automático
│   ├── DataLoader + Dataset: cómo alimentas datos al entrenamiento
│   ├── Loop de entrenamiento: optimizer, loss, backward, step
│   ├── Conexiones residuales: patrón fundamental en CNN modernas
│   ├── Transformer desde cero: MultiheadAttention, LayerNorm, positional encoding
│   └── Guardar/cargar modelos: state_dict, checkpoints
└── Cuándo lo necesitas: antes de tocar Ultralytics o los modelos del pipeline
```

```
Python avanzado (específico para este proyecto)
├── TypedDict: por qué LangGraph lo usa para el State
├── Decoradores: cómo se crean y para qué sirven
├── Context managers: __enter__/__exit__, el patrón with
├── Async/await: FastAPI y los nodos async de LangGraph
└── dataclasses/Pydantic: modelar datos de salida del pipeline
```

---

### Capa 2: El pipeline de visión (núcleo técnico del proyecto)

```
Ultralytics / YOLOv11
├── Por qué: es el nodo detect — el primer paso del pipeline
├── Qué necesitas saber:
│   ├── Arquitectura: backbone → neck → head (para explicar en entrevista)
│   ├── Formato de dataset YOLO: estructura de carpetas, formato .txt
│   ├── dataset.yaml: cómo lo defines para tu caso
│   ├── Transfer learning: partir de preentrenado, no desde cero
│   ├── Hiperparámetros clave: lr, batch, epochs, patience, augmentation
│   ├── Métricas: mAP50, mAP50-95, precision, recall, IoU — qué significa cada una
│   ├── Qué devuelve predict(): xyxy, conf, cls — el contrato con SAM2
│   └── Export: dynamic=True, opset=12 — por qué importa para TRT
└── Cuándo lo necesitas: semana 1, antes de ONNX/TRT
```

```
ONNX Runtime
├── Por qué: formato intermedio portable, paso obligatorio hacia TRT
├── Qué necesitas saber:
│   ├── Qué es ONNX: grafo computacional estándar, independiente de framework
│   ├── InferenceSession: providers, CUDAExecutionProvider vs CPU
│   ├── Inspeccionar inputs/outputs: nombres y shapes
│   ├── Construir input_feed: dict nombre → numpy array
│   └── IOBinding: eliminar copias CPU↔GPU, cuándo usarlo
└── Cuándo lo necesitas: después de entrenar YOLO, antes de TRT
```

```
TensorRT
├── Por qué: 2-4x speedup sobre ONNX, necesario para latencia < 400ms
├── Qué necesitas saber:
│   ├── Qué hace TRT: layer fusion, kernel tuning, precisión reducida
│   ├── fp16 vs int8: cuándo usar cada uno y por qué fp16 no necesita calibración
│   ├── trtexec: compilar desde línea de comandos (el camino más directo)
│   ├── Dynamic shapes: minShapes/optShapes/maxShapes, por qué importa para batch variable
│   ├── El engine es específico de la GPU: no es portable entre hardware
│   └── execute_async_v3: inferencia asíncrona con CUDA streams
└── Cuándo lo necesitas: después de ONNX validado
```

```
SAM2
├── Por qué: es el nodo segment — toma boxes de YOLO y produce máscaras precisas
├── Qué necesitas saber:
│   ├── SAM2ImagePredictor: el modo que usas (no el de video)
│   ├── set_image(): se llama UNA vez por imagen, es el paso caro
│   ├── predict(box=...): se llama por cada detección de YOLO, es barato
│   ├── Qué devuelve: masks (bool array H×W), scores, logits — cuál usar
│   └── El contrato de datos: bbox en xyxy píxeles (lo que YOLO devuelve)
└── Cuándo lo necesitas: semana 3, al construir el nodo segment
```

```
Transformers + VLM (Qwen2-VL-7B)
├── Por qué: es el nodo describe — genera texto sobre la imagen
├── Qué necesitas saber:
│   ├── AutoProcessor + AutoModelForVision2Seq: el par que carga el modelo
│   ├── from_pretrained con quantization_config: cómo cargarlo en 4GB en vez de 15GB
│   ├── bitsandbytes BitsAndBytesConfig: load_in_4bit, nf4, double_quant
│   ├── apply_chat_template: cómo construir el prompt multimodal
│   ├── model.generate: max_new_tokens, do_sample
│   └── TextIteratorStreamer: streaming token a token para FastAPI
└── Cuándo lo necesitas: semana 5, al construir el nodo describe
```

```
instructor + Pydantic
├── Por qué: es el nodo measure — convierte texto libre en JSON estructurado
├── Qué necesitas saber:
│   ├── BaseModel: cómo definir el schema de salida (InvoiceSchema)
│   ├── Field(description=...): por qué las descripciones importan para el LLM
│   ├── Modelos anidados: InvoiceItem dentro de InvoiceSchema
│   ├── instructor.from_anthropic() / patch(): wrappear el cliente LLM
│   ├── response_model=TuSchema: cómo pedirle al LLM que devuelva tu tipo
│   └── Reintentos automáticos: qué hace instructor cuando el LLM alucina JSON
└── Cuándo lo necesitas: semana 5, junto con el VLM
```

---

### Capa 3: Orquestación y API

```
LangGraph
├── Por qué: orquesta los 4 nodos del pipeline con estado compartido
├── Qué necesitas saber:
│   ├── StateGraph + TypedDict: el State es el contrato entre nodos
│   ├── add_node: cualquier callable (función, clase con __call__)
│   ├── add_edge: transición fija entre nodos
│   ├── add_conditional_edges: routing dinámico basado en el State
│   ├── Fan-out / fan-in: paralelismo declarativo
│   ├── Reducers con Annotated: cómo mergear cuando dos nodos escriben el mismo campo
│   ├── compile() + invoke() / stream(): ejecutar el grafo
│   └── checkpointer: memoria entre llamadas (SqliteSaver)
└── Cuándo lo necesitas: semana 3, es el corazón del proyecto
```

```
FastAPI
├── Por qué: la interfaz pública del pipeline
├── Qué necesitas saber:
│   ├── lifespan: cargar modelos UNA vez al arrancar, no por request
│   ├── UploadFile + File(...): recibir imágenes en multipart
│   ├── StreamingResponse: devolver resultados nodo a nodo
│   ├── HTTPException y status codes: manejo de errores
│   └── Pydantic response models: tipar la respuesta del endpoint
└── Cuándo lo necesitas: semana 3, junto con LangGraph
```

---

### Capa 4: MLOps

```
MLflow
├── Por qué: trackear experimentos y comparar PyTorch vs ONNX vs TRT
├── Qué necesitas saber:
│   ├── start_run(): context manager para un experimento
│   ├── log_param / log_metric / log_artifact: qué loguear de cada run
│   ├── set_tracking_uri(): apuntar a DagsHub desde Kaggle/Colab
│   └── Model Registry: registrar el mejor modelo como producción
└── Cuándo lo necesitas: desde semana 1, en el benchmark
```

```
DVC
├── Por qué: versionar datasets y modelos junto al código
├── Qué necesitas saber:
│   ├── dvc add: trackear archivos grandes sin subirlos a git
│   ├── dvc.yaml: definir stages con deps y outs
│   ├── dvc repro: ejecutar solo lo que cambió
│   └── dvc push/pull: sincronizar con DagsHub
└── Cuándo lo necesitas: desde semana 1, para el dataset de YOLO
```

```
LangSmith
├── Por qué: observabilidad del agente LangGraph — latencia por nodo
├── Qué necesitas saber:
│   ├── Variables de entorno para activar el tracing automático
│   └── Leer el árbol de ejecución para identificar cuellos de botella
└── Cuándo lo necesitas: semana 7, al hacer el polish final
```

---

## Metodología de trabajo

### Dónde vive cada cosa

```
GitHub          → todo el código, scripts, configs, dockerfiles
DagsHub         → MLflow runs (métricas, modelos), DVC (datasets)
Local (sin GPU) → escribir código, tests en CPU, arquitectura
Kaggle/Colab    → ejecutar entrenamiento y benchmarks
Railway/Render  → deploy público del pipeline (CPU + Claude API)
vast.ai/RunPod  → GPU on-demand para benchmarks reales (pagas por hora)
```

### Flujo de trabajo diario

```
1. En local:
   ├── Escribir el script
   ├── Testear con imagen pequeña en CPU (valida lógica, no rendimiento)
   ├── git commit + push a GitHub
   └── Abrir Kaggle/Colab

2. En Kaggle/Colab:
   ├── git clone / git pull desde GitHub
   ├── pip install dependencias
   ├── Configurar secrets (DAGSHUB_TOKEN, etc.) desde la UI de Kaggle
   ├── Ejecutar el script
   ├── Verificar que los runs aparecen en DagsHub MLflow
   └── Descargar el mejor modelo si es necesario

3. De vuelta en local:
   ├── mlflow ui apuntando a DagsHub → ver métricas
   └── dvc pull → descargar artefactos si los necesitas localmente
```

### Git flow del proyecto

```
main          → código estable, deploy de Railway apunta aquí
development   → integración de features
feature/xxx   → una feature a la vez

Cada semana = una rama + un PR con:
  - El código de esa semana
  - Resultados del benchmark en la descripción del PR
  - Screenshots de MLflow
```

### Cómo manejar que el VLM es muy grande para CPU

```
Variable de entorno VLM_MODE:
  "local"  → Qwen2-VL-7B (solo en vast.ai/Kaggle con GPU)
  "api"    → Claude API  (en Railway/Render, local, cualquier sitio)

El nodo describe del pipeline detecta VLM_MODE y usa la implementación correcta.
Esto permite demostrar el pipeline en Railway (gratis, CPU) sin el VLM local,
y los benchmarks reales con el VLM local en GPU cuando hace falta.
```

---

## Arquitectura decidida

### El State del pipeline

```python
class PipelineState(TypedDict):
    # Input
    image_bytes: bytes
    image_np: np.ndarray        # imagen en numpy RGB

    # Nodo detect
    detections: list            # [{bbox, clase, confianza}]

    # Nodo segment
    segments: list              # [{bbox, clase, confianza, mascara}]

    # Nodo describe
    description: str            # texto libre del VLM

    # Nodo measure
    structured: dict            # InvoiceSchema o schema según caso de uso

    # Metadata
    latency_ms: dict            # {detect: X, segment: Y, describe: Z, measure: W}
    errors: list                # errores no fatales acumulados
    intentos_measure: int       # para el ciclo de reintento
```

### El grafo de nodos

```
imagen
  │
  ▼
[detect]      YOLOv11 TensorRT fp16
  │
  ▼
¿hay detecciones?
  │ no → [END] con detections=[]
  │ sí
  ▼
[segment]     SAM2 — set_image una vez, predict por cada bbox
  │
  ▼
[describe]    Qwen2-VL-7B 4-bit / Claude API
  │
  ▼
[measure]     instructor + InvoiceSchema
  │
  ▼
¿JSON válido?
  │ no y intentos < 3 → vuelve a [describe] con más contexto
  │ sí
  ▼
[END]         JSON estructurado completo
```

### Contratos de datos entre nodos

```
detect  devuelve: [{bbox: [x1,y1,x2,y2], clase: str, confianza: float}]
segment devuelve: [{bbox, clase, confianza, mascara: np.array(H,W,bool)}]
describe devuelve: str (texto libre)
measure devuelve: InvoiceSchema (objeto Pydantic validado)
```

### Configuración por entorno

```
DEVICE=cpu|cuda           → qué hardware usar para inferencia
VLM_MODE=local|api        → VLM local (GPU) o Claude API (CPU)
MLFLOW_TRACKING_URI       → URL de DagsHub
TRT_ENGINE_PATH           → ruta al .engine compilado
```

---

## Fases de construcción

### Fase 0 — Setup (antes de escribir código)
```
□ Cuenta DagsHub + repo conectado a GitHub
□ Cuenta Kaggle con GPU verificada (T4 gratis)
□ Secrets configurados en Kaggle: DAGSHUB_USER, DAGSHUB_TOKEN
□ Estructura de carpetas creada en local
□ requirements.txt base
□ .gitignore para modelos grandes, datasets, mlruns local
□ dvc init + remote configurado a DagsHub
```

### Fase 1 — Dataset y entrenamiento YOLO
```
□ Decidir qué detectar (caso de uso: facturas, documentos, objetos...)
□ Recolectar imágenes (mínimo 300 para empezar)
□ Anotar con Roboflow en formato YOLO
□ dataset.yaml definido
□ Script train_yolo.py que loguea a MLflow (escribir en local, ejecutar en Kaggle)
□ Primer run en Kaggle con mAP50 logueado en DagsHub
□ Experimentar con hiperparámetros hasta mAP50 > 0.70
□ Mejor modelo (.pt) subido como artefacto en MLflow
```

### Fase 2 — Cadena de optimización
```
□ Script export_and_benchmark.py (escribir en local, ejecutar en Kaggle)
□ Export YOLO → ONNX validado (onnx.checker.check_model)
□ Benchmark PyTorch vs ONNX logueado en MLflow
□ Compilar ONNX → TRT fp16 con trtexec
□ Benchmark PyTorch vs ONNX vs TRT logueado en MLflow
□ Speedup TRT vs PyTorch > 2x confirmado
□ Engine .fp16.engine subido como artefacto en MLflow
```

### Fase 3 — Pipeline LangGraph v1
```
□ PipelineState definido con todos los campos
□ Nodo detect: YOLO ONNX en CPU (para poder testear en local)
□ Nodo segment: SAM2 integrado, contrato de datos validado
□ Nodo describe: modo API primero (Claude), modo local después
□ Nodo measure: InvoiceSchema Pydantic + instructor
□ Grafo compilado con conditional edges para reintentos
□ Test del pipeline completo con imagen de prueba en local (CPU, lento pero correcto)
```

### Fase 4 — FastAPI + CI/CD
```
□ Endpoint POST /analyze con UploadFile
□ lifespan cargando modelos al arrancar
□ StreamingResponse devolviendo resultados nodo a nodo
□ GET /health para Railway health checks
□ Dockerfile.agent funcional
□ GitHub Actions: lint + test + build Docker + push GHCR
□ Deploy en Railway apuntando a la imagen de GHCR
□ curl -X POST /analyze -F "image=@test.jpg" funciona en producción
```

### Fase 5 — VLM local + DDP
```
□ Nodo describe con Qwen2-VL-7B 4-bit (ejecutar en Kaggle con GPU)
□ Streaming con TextIteratorStreamer integrado en FastAPI
□ Script train_ddp.py para ResNet50 en 2 GPUs (Kaggle tiene 2xT4)
□ Speedup DDP 2x GPU vs 1x GPU > 1.8x logueado en MLflow
□ Benchmark suite: 50 imágenes, latencia p50/p99, costo por request
```

### Fase 6 — Polish y portfolio
```
□ README con arquitectura Mermaid, métricas reales, GIF de demo
□ LangSmith activado: captura de trazas con latencia por nodo
□ Badges: CI verde, coverage, deploy status
□ ARCHITECTURE.md con diagramas del pipeline
□ STAR_STORIES.md: 5 historias en formato STAR para entrevistas
□ SYSTEM_DESIGN_PREP.md: "diseña detección para 10 cámaras 24/7"
```

---

## Criterios de completitud por fase

Cada fase está completa cuando puedes responder estas preguntas sin mirar el código:

**Fase 1:** ¿Por qué tu mAP50 es X y no más alto? ¿Qué cambiarías para mejorar?
**Fase 2:** ¿Por qué TRT es más rápido que ONNX? ¿Qué hace layer fusion exactamente?
**Fase 3:** ¿Qué pasa en el State entre detect y segment? ¿Qué forma tiene ese dict?
**Fase 4:** ¿Por qué usas lifespan y no cargas el modelo en cada request?
**Fase 5:** ¿Por qué NF4 es mejor que INT4 para cuantizar pesos de un LLM?
**Fase 6:** ¿Cuál es el cuello de botella de latencia de tu pipeline y cómo lo resolverías?

---

## Señales de que necesitas aprender algo antes de implementarlo

```
"No sé qué forma tiene el output de este modelo"
→ Lee la doc / corre el modelo y haz print(output.shape) antes de integrarlo

"No entiendo por qué falla este error de shape"
→ Vuelve a PyTorch: entiende broadcasting y dimensiones antes de continuar

"No sé cómo conectar este nodo con el siguiente"
→ Define el contrato de datos en papel antes de escribir código

"El benchmark no tiene sentido, TRT es más lento que PyTorch"
→ Falta warmup o torch.cuda.synchronize() — lee la sección de benchmark

"No sé si este modelo cabe en la GPU con los demás"
→ Calcula: YOLO ~100MB + SAM2 ~180MB + Qwen4bit ~4GB — suma y compara con VRAM
```

---

## Lo que este proyecto demuestra en entrevista

```
Computer Vision:    entrenamiento custom, métricas de detección, optimización de inferencia
MLOps:              tracking de experimentos, versionado de modelos y datos, CI/CD
Systems:            pipeline event-driven con estado, streaming, manejo de errores
LLMs:               VLM multimodal, quantización, structured output
Infraestructura:    containerización, deploy en cloud, trabajo con/sin GPU
```
