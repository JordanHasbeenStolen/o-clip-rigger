# o-clip-rigger

Search personal photo collections with natural language queries —
from terminal, Telegram bot, or web UI.
Powered by OpenCLIP embeddings and FAISS vector search.

<!-- screenshot placeholder: replace with actual preview -->
<!-- ![demo](docs/screenshot.png) -->

## What it does

Indexes a local folder of images and enables search by text —
e.g. "cat on the sofa", "sunset over mountains", "document with signature".

## How it works

1. Each image is encoded into a 512-dim vector with OpenCLIP (ViT-B-32).
2. Vectors are stored in a FAISS index for fast cosine similarity search.
3. A text query is encoded with the same model and matched against the index.

## Stack

- OpenCLIP ViT-B-32 (`laion2b_s34b_b79k` weights)
- FAISS (CPU)
- PyTorch (CPU)
- FastAPI — backend API
- Telegram bot — alternative interface (planned)
- Streamlit — web UI (planned)
- uv for dependency management

## Roadmap

- [x] CLI prototype with FAISS
- [ ] FastAPI backend
- [ ] Telegram bot frontend
- [ ] Streamlit UI with image previews
- [ ] Docker packaging
- [ ] Qdrant migration