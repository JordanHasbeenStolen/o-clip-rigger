# o-clip-rigger

Search and tag personal photo collections with natural language queries —
from terminal, Telegram bot, or web UI (Streamlit).
Powered by OpenCLIP and SigLIP2 embeddings with FAISS vector search.

<!-- screenshot placeholder: replace with actual preview -->
<!-- ![demo](docs/screenshot.png) -->
<img width="846" height="376" alt="image" src="https://github.com/user-attachments/assets/df9c9dd8-f22a-47bd-89ab-c87f96bc0c2d" />


## What it does

Indexes a local folder of images and enables search by text —
e.g. "cat on the sofa", "sunset over mountains", "document with signature".
Can also take a single uploaded photo and suggest tags for it from a fixed
vocabulary.

## How it works

1. Each image is encoded into a vector with OpenCLIP (ViT-B-32) or SigLIP2 (ViT-B-16).
2. Vectors are stored in a FAISS index for fast cosine similarity search - a
   separate index per model, since their embeddings aren't comparable.
3. A text query is encoded with the same model and matched against its index.
4. Tagging matches an uploaded photo against a fixed vocabulary of candidate
   tags (not free-form captioning), with per-model calibrated confidence scores.

## Stack

- OpenCLIP (ViT-B-32) and SigLIP2 (ViT-B-16), selectable per request
- FAISS (CPU) — separate index per model
- PyTorch (CPU)
- FastAPI — backend API (`/search`, `/tag`, `/image`)
- Telegram bot (aiogram) — text search, allowlisted by Telegram user ID
- Streamlit — web UI with model/task selection and photo upload
- uv for dependency management

## Roadmap

- [x] CLI prototype with FAISS
- [x] FastAPI backend
- [x] Telegram bot frontend (raw, user-allowlisted)
- [x] Streamlit UI with image previews
- [x] Selectable model (OpenCLIP vs SigLIP2) for search and tagging
- [x] Zero-shot image tagging (vocabulary-based, no captioning)
- [ ] Low-confidence cutoff ("not found" instead of a weak top-1 guess)
- [ ] Model/task selection as buttons in the Telegram bot (currently text-only)
- [ ] Index the full photo collection (currently a small test subset)
- [ ] Docker packaging
- [ ] Qdrant migration
- [ ] Postgres + pgvector, Redis, Alembic (longer-term infra ideas)
