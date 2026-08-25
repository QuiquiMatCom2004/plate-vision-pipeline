# Plate Vision Pipeline

**Agente que recibe imagen/video → detecta → segmenta → describe → estructura → devuelve JSON**

Proyecto killer del roadmap de 8 semanas. Combina Computer Vision real-time con agent engineering para crear un pipeline completo que demuestra dominio de MLOps, modelos de visión y orquestación de agentes — el diferenciador para roles de CV Engineer junior-mid.

---

## Pipeline

```
Entrada: imagen / video / documento escaneado
         │
         ▼
┌────────────────────────────────────────────┐
│           LangGraph StateGraph             │
│  detect → segment → describe → measure     │
│  (tools registradas, streaming, retries)   │
└────────────────────────────────────────────┘
         │
         ▼
Salida JSON:
{
  "detections": [...],
  "segments": [...],
  "description": "Factura de Proveedor X, total $1,234.56",
  "structured": InvoiceSchema(total=1234.56, items=[...], date="2026-01-15"),
  "latency_ms": 342
}
```

---

## Stack

| Componente | Tecnología |
|---|---|
| Detección | YOLOv11 → ONNX → TensorRT (fp16) |
| Segmentación | SAM2 |
| Descripción / VLM | Qwen2-VL-7B / LLaVA-Next (4-bit via bitsandbytes) |
| Structured output | `instructor` + Pydantic |
| Orquestación | LangGraph StateGraph |
| API | FastAPI |
| MLOps | MLflow (tracking + registry) + DVC (data pipeline) |
| CI/CD | GitHub Actions (lint + test + build Docker + deploy staging) |
| Deploy | Railway / Fly.io (GPU) |
| Observabilidad | LangSmith / custom logs, Evidently (drift) |

---

## Objetivos por semana

### Semana 1-2 — Foundation
- [ ] Setup GPU en vast.ai (2x RTX 3090/4090 o 1x A100)
- [ ] Exportar YOLOv11 → ONNX → TensorRT engine
- [ ] Benchmark PyTorch vs ONNX vs TRT en T4: latencia, throughput, mAP drop
- [ ] MLflow tracking local: primer experimento YOLO logueado
- [ ] DVC data pipeline base
- [ ] Arquitectura documentada en `docs/ARCHITECTURE.md`
- [ ] PR #1 abierto en langgraph (CV tool calling pattern)

**Checkpoint:** benchmark ONNX listo, `docs/ARCHITECTURE.md` completo, PR #1 abierto

### Semana 3-4 — Agent v1 + MLOps pipeline
- [ ] Implementar 4 tools en LangGraph: `detect`, `segment`, `describe`, `measure`
- [ ] FastAPI endpoint `/analyze` funcional
- [ ] CI/CD verde: lint + test + Docker en GHCR + deploy staging
- [ ] Modelo registrado en MLflow registry
- [ ] Agent v1 deployed en Railway/Render/Fly.io
- [ ] README v1: arquitectura, métricas, GIF demo
- [ ] PR #1 merged

**Checkpoint:** `curl -X POST /analyze -F "image=@test.jpg"` → JSON con detections, segments, description

### Semana 5-6 — Agent v2 + VLM + Distributed
- [ ] Integrar Qwen2-VL-7B / LLaVA-Next (4-bit)
- [ ] Tool `analyze_document` con structured output via `instructor`: `InvoiceSchema(total, items, date)`
- [ ] Script DDP en 2x GPU: entrena YOLOv11-custom o ResNet50, loguea speedup y memory
- [ ] Streaming responses + error handling + retries
- [ ] Benchmark suite: 50 imágenes, mAP, latencia p50/p99, costo/request
- [ ] PR #2 (Transformers: VLM structured output) y PR #3 (torchvision) abiertos

**Checkpoint:** VLM tool funcionando, DDP benchmark con speedup logueado, 2 PRs abiertos

### Semana 7-8 — Portfolio polish + Interview ready
- [ ] README pro final: arquitectura Mermaid, métricas, GIF/demo, post-mortem
- [ ] PR #2 y PR #3 merged
- [ ] Badges en README (CI verde, coverage, deploy status)
- [ ] System design preparado: "Diseña detección 10 cámaras 24/7"

---

## Archivos a crear

```
Plate Vision Pipeline/
├── PROJECT.md                      ← este archivo
├── docs/
│   ├── ARCHITECTURE.md             ← Mermaid diagrams del pipeline
│   ├── YOLO_INTERNALS_30MIN.md     ← para entrevista técnica
│   ├── STAR_STORIES.md             ← 10 historias en formato STAR
│   └── SYSTEM_DESIGN_PREP.md      ← diagramas + talking points
├── scripts/
│   ├── benchmark_onnx_trt.py       ← benchmark reproducible PyTorch vs ONNX vs TRT
│   └── train_ddp.py                ← DDP training script
├── .github/
│   └── workflows/
│       └── ml.yml                  ← CI/CD pipeline
└── docker/
    └── Dockerfile.agent            ← deploy config
```

---

## Métricas objetivo

| Métrica | Target |
|---|---|
| Latencia p50 pipeline completo | < 400ms |
| mAP YOLOv11 custom | > 0.70 |
| TRT speedup vs PyTorch | > 2x |
| DDP speedup 2x GPU vs 1x | > 1.8x |
| Costo por request (staging GPU) | < $0.001 |

---

## Regla de uso de IA

Las 4 tools core (`detect`, `segment`, `describe`, `measure`) se implementan y se entienden línea a línea. Criterio: si no puedes explicarlo en pizarra en 30 min sin IA, no lo usas sin entenderlo primero.
