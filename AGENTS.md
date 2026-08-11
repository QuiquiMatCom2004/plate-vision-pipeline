# AGENTS.md — CV Agent Pipeline

## Qué es este proyecto

API REST que recibe una imagen y devuelve un JSON estructurado con toda la información visual:
objeto detectado, segmentación (máscara), descripción textual y datos medibles.
La API es el core — la interfaz (Telegram, Streamlit, etc.) es intercambiable y secundaria.

## Stack técnico

**Lenguaje:** Python  
**Pipeline de visión:** YOLOv11 (Ultralytics), SAM2, Qwen2-VL-7B / Claude API  
**Optimización:** ONNX Runtime, TensorRT (fp16)  
**Orquestación:** LangGraph (StateGraph + TypedDict)  
**API:** FastAPI  
**Structured output:** Pydantic + instructor  
**MLOps:** MLflow, DVC, LangSmith  

## Arquitectura

Pipeline event-driven con estado compartido (`PipelineState`). Definida en `docs/ROADMAP.md`:

```
imagen → [detect] → ¿hay detecciones? → [segment] → [describe] → [measure] → JSON
                         │ no
                         └→ END (detections=[])
```

Cuatro nodos LangGraph: `detect → segment → describe → measure`.  
`PipelineState` es el contrato de datos entre nodos — ningún nodo escribe fuera de su campo asignado.

**Decisiones no obvias:**
- `VLM_MODE=local|api` — conmuta entre Qwen2-VL-7B (GPU) y Claude API (CPU/deploy)
- El engine TRT es específico por GPU — no portable entre máquinas
- SAM2: `set_image()` se llama UNA vez por imagen (paso caro); `predict()` por cada bbox (barato)
- Quantización 4-bit NF4 para el VLM — necesaria para caber en GPU de 6-8GB

## Comandos

```bash
# Tests
pytest

# Arrancar API — pendiente de definir (depende de arquitectura final)
# Deploy — pendiente de definir
```

## Criterios de completitud ("done")

Una tarea está completa cuando:
- Los tests de pytest pasan
- Review aprobado (o auto-review si es solo)
- Métricas de modelo cumplen threshold: mAP50 > 0.70, TRT speedup > 2x vs PyTorch
- Teach-back hecho (ver sección "Teach-back") — sin esto, no cuenta como aprendido

## NUNCA (restricciones para agentes IA)

- **Un agente IA NO debe escribir ni una sola línea de código del src y/o scripts en este proyecto sin explicar antes todo lo necesario para que el usuario lo haga y verificar que lo entiende y seria capaz de replicarlo**
- El rol del agente ahí es explicar cómo implementarlo: conceptos, decisiones, trade-offs, ejemplos.
- El código de `src/` y `scripts/` lo escribe siempre el usuario o el agente despues de verificar que el usuario puede hacerlo.
- No asumas que el usuario tiene GPU disponible en local — usar CPU para tests de lógica.
- Nunca debe commitear y poner que fue hecho por un agente. 

## Depuración asistida (excepción puntual a NUNCA)

La regla de "no escribir código" existe para proteger la lucha productiva
(decisiones de diseño, lógica de los nodos) — no para forzar al usuario a
perder horas en fricción que no enseña nada (un typo, una firma de función
mal recordada, una versión de librería). Cuando el usuario esté trabado
depurando:

1. El agente puede señalar la línea y la causa raíz del bug.
2. El usuario sigue escribiendo el fix — el agente no lo escribe por él.
3. Si el bug es de lógica/diseño (no de sintaxis), el agente explica el
   concepto detrás del error en vez de dar la solución directa.

## Teach-back (parte del criterio de completitud)

Una función o nodo no cuenta como aprendido solo porque los tests pasan.
Al terminar cada función/nodo, el usuario debe explicarla en voz alta sin
mirar el código: qué hace, por qué esa decisión de diseño y no otra. Si no
puede explicarlo en ese momento, la tarea sigue abierta aunque el código
funcione.

## Infra / deploy (excepción a la regla anterior)

Config declarativa y de despliegue — `pyproject.toml`, `Dockerfile.agent`,
`.github/workflows/*.yml`, `dvc.yaml`, `configs/*.yaml`, `.env.example` —
el agente SÍ puede escribirla, con este flujo obligatorio:
1. Explicar el concepto y las alternativas antes de escribir el archivo.
2. Escribir un borrador.
3. Recorrerlo línea por línea con el usuario: qué hace cada directiva y por qué.
4. El usuario lo ejecuta de verdad (build, run, CI) antes de darlo por aprendido.

Criterio de completitud igual al de `src/scripts`: si no podés explicarlo en
pizarra en 30 min, no está aprendido — aunque el agente lo haya escrito.

## Restricciones duras

- **Latencia:** pipeline completo < 400ms (sin VLM local) en GPU
- **Hardware:** VLM requiere 4-bit quantization para caber en GPU de 6-8GB
- **Portabilidad:** TRT engine compilado es específico por GPU — debe recompilarse por entorno
- **Entornos:** local (CPU, pruebas de lógica), Kaggle/Colab (GPU, entrenamiento), Railway (CPU, deploy público)
