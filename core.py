import pickle
from pathlib import Path

import faiss
import torch
import open_clip

INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("metadata.pkl")
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None
_index = None
_metadata = None


def _load():
    global _model, _tokenizer, _index, _metadata
    if _model is not None:
        return
    _model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    _model = _model.to(DEVICE).eval()
    _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    _index = faiss.read_index(str(INDEX_PATH))
    _metadata = pickle.loads(META_PATH.read_bytes())


def search(query: str, k: int = 5):
    _load()
    tokens = _tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        tf = _model.encode_text(tokens)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    scores, ids = _index.search(tf.cpu().numpy().astype("float32"), k)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            results.append({
                "path": _metadata[i],
                "index": int(i),
                "score": float(s),
            })
    return results


def total_vectors() -> int:
    _load()
    return _index.ntotal


def get_path(idx: int) -> str | None:
    _load()
    if 0 <= idx < len(_metadata):
        return _metadata[idx]
    return None

def warmup():
    _load()