# Documentación offline — repos para buildear

## mkdocs (misma metodología que Pillow/OpenCV)

```bash
git clone https://github.com/langchain-ai/langgraph
git clone https://github.com/langchain-ai/langchain
git clone https://github.com/instructor-ai/instructor
git clone https://github.com/huggingface/transformers   # cubre transformers + bitsandbytes + accelerate
git clone https://github.com/huggingface/course         # NLP Course en español
```

Build de cada uno:
```bash
cd <repo>/docs && mkdocs build
```

## Sphinx (mismo proceso que PyTorch)

```bash
git clone https://github.com/d2l-ai/d2l-en
```

```bash
cd d2l-en && pip install -r requirements.txt && make html
```

## PDFs directos

- **TensorRT Developer Guide** — developer.nvidia.com/tensorrt → "Documentation" → PDF por versión
- **SAM2 paper** — arxiv.org/abs/2408.00714
- **Deep Learning (Goodfellow) en español** — deeplearningbook.org → traducción fan en PDF

## Repos sin doc buildeada (README + notebooks son la doc)

```bash
git clone https://github.com/facebookresearch/sam2      # notebooks/ son la doc principal
```
