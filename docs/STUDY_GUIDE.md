# Study Guide — Plate Vision Pipeline

Qué estudiar de cada doc, en qué fase del proyecto se usa, y por qué funciona así.

---

## Semana 1-2 — Foundation: PyTorch → ONNX → TensorRT

### PyTorch

**Qué estudiar:**
- `torch.Tensor`: dtypes, device (cpu/cuda), memory layout — para entender por qué fp16 ocupa la mitad
- `torch.nn.Module`: forward pass, parámetros, state_dict — para cargar/guardar pesos de YOLO
- `torch.cuda`: `is_available()`, `to(device)`, `torch.cuda.synchronize()` — para medir latencia real en GPU
- `torch.onnx.export()`: `opset_version`, `dynamic_axes`, `input_names/output_names` — esto es el puente YOLO → ONNX
- `torch.utils.data`: `Dataset`, `DataLoader`, `collate_fn` — para el script de benchmark con batches

**Por qué importa aquí:** PyTorch es el formato nativo de YOLOv11. Antes de exportar a ONNX/TRT necesitas entender cómo el modelo representa sus tensores y por qué `dynamic_axes` es obligatorio para batch variable.

---

### torchvision

**Qué estudiar:**
- `transforms.v2`: `Resize`, `Normalize`, `ToTensor` — preprocesamiento estándar antes de inferencia
- `models.resnet50(weights=...)` — para el script DDP de semana 5-6
- `datasets.ImageFolder` — para cargar dataset custom de YOLO

**Por qué importa aquí:** El pipeline necesita preprocesar imágenes de entrada antes de pasarlas al detector. Las transforms deben ser idénticas en PyTorch, ONNX y TRT para que el benchmark compare lo mismo.

---

### Ultralytics (YOLOv11)

**Qué estudiar:**
- Arquitectura: backbone (CSPDarknet) → neck (PANet) → head — para poder explicarlo en pizarra
- `YOLO('yolo11n.pt')`: cómo carga pesos, qué devuelve `Results`
- `model.predict(source, conf, iou, device)`: parámetros clave del inference
- `model.export(format='onnx', opset=12, dynamic=True, half=False)` — el comando central de semana 1
- `model.val(data='coco.yaml')`: cómo se calcula mAP50, mAP50-95
- Training: `model.train(data=..., epochs=..., imgsz=..., batch=...)` — para dataset custom

**Por qué importa aquí:** YOLOv11 es el primer nodo del pipeline (`detect`). Necesitas entender qué devuelven los bounding boxes (xyxy vs xywh, coordenadas normalizadas vs pixel) para pasarlo correctamente al siguiente nodo (SAM2).

---

### ONNX Runtime

**Qué estudiar:**
- `onnxruntime.InferenceSession(path, providers=[...])`: `CUDAExecutionProvider` vs `CPUExecutionProvider`
- `session.get_inputs() / get_outputs()`: nombres y shapes de tensores
- `session.run(output_names, input_feed)`: cómo construir el `input_feed` dict
- `IOBinding`: bind inputs/outputs directamente en GPU memory para evitar copias host↔device
- `SessionOptions`: `intra_op_num_threads`, `graph_optimization_level`

**Por qué importa aquí:** ONNX Runtime es el runtime intermedio entre PyTorch y TensorRT. El benchmark PyTorch vs ONNX vs TRT mide exactamente cuánto gana cada capa de optimización. IOBinding es lo que elimina el cuello de botella de transferencia CPU-GPU.

---

### TensorRT

**Qué estudiar:**
- Flujo de conversión: ONNX → `trtexec` → `.engine` — el comando de semana 1
  ```bash
  trtexec --onnx=yolo11.onnx --saveEngine=yolo11_fp16.engine --fp16
  ```
- `Builder`, `INetworkDefinition`, `BuilderConfig`: qué hace cada uno internamente
- `fp16` vs `int8`: por qué fp16 da ~2x speedup sin calibración
- `IRuntime`, `ICudaEngine`, `IExecutionContext`: el flujo de inferencia en Python
- `context.execute_async_v3()`: inferencia asíncrona con CUDA streams
- Profiles para dynamic shapes (batch variable)

**Por qué importa aquí:** TensorRT funde capas del grafo (layer fusion), elimina operaciones redundantes y usa kernels CUDA optimizados para la GPU específica. Entender esto te permite explicar en entrevista *por qué* es 2-4x más rápido, no solo que lo es.

---

### MLflow

**Qué estudiar:**
- `mlflow.start_run()` como context manager
- `mlflow.log_param()`, `log_metric()`, `log_artifact()`: qué loguear del benchmark
- `mlflow.pytorch.log_model()` / `mlflow.onnx.log_model()`
- Model Registry: `mlflow.register_model()`, stages (Staging → Production)
- `mlflow ui`: cómo navegar experimentos y comparar runs

**Por qué importa aquí:** Cada run del benchmark (PyTorch/ONNX/TRT) es un experimento MLflow. Sin tracking no puedes demostrar el 2x speedup con reproducibilidad — que es exactamente lo que piden en entrevistas de MLOps.

---

### DVC

**Qué estudiar:**
- `dvc init`, `dvc add <dataset>`: versionar datos sin subirlos a git
- `dvc.yaml`: definir stages con `cmd`, `deps`, `outs`
- `dvc repro`: ejecutar solo los stages cuyas dependencias cambiaron
- `dvc remote add`: configurar storage (S3, GDrive, local)
- `dvc push / pull`: sincronizar datos

**Por qué importa aquí:** El dataset de YOLO y los artefactos del benchmark deben estar versionados junto al código. DVC es lo que hace que `git checkout v1.2` también restaure los datos y modelos de esa versión.

---

## Semana 3-4 — Agent v1: LangGraph + FastAPI

### LangGraph

**Qué estudiar (en este orden):**
1. `StateGraph` y `TypedDict` como State — la estructura central del agente
   ```python
   class PipelineState(TypedDict):
       image: np.ndarray
       detections: list
       segments: list
       description: str
   ```
2. `graph.add_node(name, fn)`: cada tool es un nodo con signatura `fn(state) -> dict`
3. `graph.add_edge()` y `graph.add_conditional_edges()`: cuándo ir a cada nodo
4. `graph.compile()`: genera el grafo ejecutable
5. `graph.stream(input)`: streaming de resultados por nodo
6. `MemorySaver` / `SqliteSaver`: checkpointing entre llamadas
7. Retry con `RunnableConfig` y `max_retries`
8. Subgraphs: anidar grafos (útil si segment o describe se vuelven complejos)

**Por qué importa aquí:** LangGraph es el corazón del proyecto. Las 4 tools (`detect`, `segment`, `describe`, `measure`) son nodos del grafo. El State tipado es lo que garantiza que cada nodo recibe exactamente lo que necesita y devuelve exactamente lo que el siguiente espera.

---

### FastAPI

**Qué estudiar:**
- `UploadFile` y `File(...)`: recibir imágenes en multipart/form-data
  ```python
  @app.post("/analyze")
  async def analyze(image: UploadFile = File(...)):
  ```
- `StreamingResponse`: devolver resultados del agente conforme se generan nodo a nodo
- `lifespan`: cargar modelos al arrancar (no en cada request)
  ```python
  @asynccontextmanager
  async def lifespan(app):
      app.state.yolo = YOLO("yolo11_fp16.engine")
      yield
  ```
- `BackgroundTasks`: jobs asíncronos post-respuesta
- `HTTPException`, status codes, response models con Pydantic

**Por qué importa aquí:** El endpoint `/analyze` es la interfaz pública del pipeline. `lifespan` es crítico — cargar YOLOv11 + SAM2 + el VLM en cada request destruiría la latencia.

---

### Pydantic

**Qué estudiar:**
- `BaseModel` con tipos anotados: el schema `InvoiceSchema`
- `Field(description=...)`: importante para que `instructor` sepa qué extraer
- `model_validator` y `field_validator`: validación cruzada entre campos
- `model_json_schema()`: ver qué schema se le pasa al LLM
- Modelos anidados: `InvoiceItem` dentro de `InvoiceSchema`

**Por qué importa aquí:** El JSON de salida del pipeline es un modelo Pydantic. `instructor` usa el schema de Pydantic para instruir al VLM sobre qué estructura devolver — si el schema está mal definido, la extracción falla.

---

## Semana 5-6 — Agent v2: VLM + Quantización + DDP

### Transformers (HuggingFace)

**Qué estudiar:**
- `AutoModelForVision2Seq` y `AutoProcessor`: carga de Qwen2-VL-7B
- `from_pretrained(model_id, quantization_config=..., device_map="auto")`: carga con 4-bit
- `processor(images, text, return_tensors="pt")`: preparar input multimodal
- `model.generate(inputs, max_new_tokens=..., do_sample=False)`: inferencia
- `processor.decode(output_ids)`: decodificar output a texto
- Streaming con `TextIteratorStreamer`

**Por qué importa aquí:** El nodo `describe` del pipeline usa el VLM para generar la descripción de la imagen. Entender el flujo `processor → model → decode` es obligatorio para implementar el tool sin IA.

---

### bitsandbytes

**Qué estudiar:**
- `BitsAndBytesConfig`:
  ```python
  BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
  )
  ```
- Por qué NF4 (Normal Float 4) es mejor que INT4 para pesos de LLM
- Double quantization: cuantizar el factor de escala también
- Qué capas se cuantizan y cuáles se dejan en fp16 (lm_head)

**Por qué importa aquí:** Qwen2-VL-7B pesa ~15GB en fp16. Con 4-bit NF4 baja a ~4GB, lo que cabe en una RTX 3090 junto con YOLOv11 y SAM2. Sin entender esto, no puedes ajustar la config cuando la GPU se queda sin memoria.

---

### instructor

**Qué estudiar:**
- `instructor.from_anthropic()` / `instructor.patch()`: wrappear el cliente LLM
- `client.chat.completions.create(response_model=InvoiceSchema, ...)`: extracción estructurada
- `Partial[InvoiceSchema]`: streaming de structured output parcial
- Retry automático cuando el modelo devuelve JSON inválido
- `ValidationContext`: pasar contexto adicional al validador

**Por qué importa aquí:** El nodo `measure` / `analyze_document` debe devolver un `InvoiceSchema` Pydantic válido, no texto libre. `instructor` es el pegamento entre el VLM y Pydantic — maneja los reintentos cuando el modelo alucina campos.

---

### PyTorch (DDP)

**Qué estudiar (específico para semana 5-6):**
- `torch.distributed.init_process_group(backend="nccl")`
- `DistributedDataParallel(model, device_ids=[local_rank])`
- `DistributedSampler`: garantizar que cada GPU ve datos distintos
- `torchrun --nproc_per_node=2 train_ddp.py`: el launcher
- Loguear solo desde `rank == 0` para evitar outputs duplicados
- Métricas: speedup = `tiempo_1gpu / tiempo_2gpu`, debe ser > 1.8x

**Por qué importa aquí:** El script `train_ddp.py` demuestra que sabes escalar entrenamiento. El benchmark de speedup es una de las métricas objetivo del proyecto y un tema recurrente en entrevistas de CV Engineer.

---

### SAM2

**Qué estudiar (desde notebooks del repo):**
- `SAM2ImagePredictor`: predicción de máscaras dado bounding box o punto
- `set_image(image)`: encodear imagen una vez, predecir múltiples máscaras
- `predict(box=xyxy_bbox)`: usar detecciones de YOLO como prompt a SAM2
- `SAM2VideoPredictor`: tracking de máscaras en video
- Qué devuelve: `masks`, `scores`, `logits` — cuál usar y cuándo

**Por qué importa aquí:** SAM2 recibe los bounding boxes de YOLO y genera máscaras precisas. El flujo YOLO → SAM2 (box prompt) es el patrón estándar: YOLO detecta qué, SAM2 segmenta dónde exactamente.

---

## Semana 7-8 — Polish + Interview Ready

### LangSmith

**Qué estudiar:**
- Variables de entorno: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`
- Qué se traza automáticamente al usar LangGraph
- Cómo ver el árbol de ejecución de cada run del agente
- Datasets y evaluación: `client.create_dataset()`, `evaluate()`

**Por qué importa aquí:** LangSmith registra cada ejecución del StateGraph con latencia por nodo. Para el README final necesitas captura de trazas reales que demuestren la latencia p50/p99 del pipeline completo.

---

## Mapa doc → fase

| Documentación | Sem 1-2 | Sem 3-4 | Sem 5-6 | Sem 7-8 |
|---|:---:|:---:|:---:|:---:|
| PyTorch (base) | ✅ | | | |
| PyTorch (DDP) | | | ✅ | |
| torchvision | ✅ | | ✅ | |
| Ultralytics | ✅ | | | |
| ONNX Runtime | ✅ | | | |
| TensorRT | ✅ | | | |
| MLflow | ✅ | ✅ | | |
| DVC | ✅ | ✅ | | |
| LangGraph | | ✅ | ✅ | |
| FastAPI | | ✅ | | |
| Pydantic | | ✅ | ✅ | |
| Transformers | | | ✅ | |
| bitsandbytes | | | ✅ | |
| instructor | | | ✅ | |
| SAM2 | | ✅ | | |
| LangSmith | | | | ✅ |
