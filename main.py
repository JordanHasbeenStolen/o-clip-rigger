import pickle
from pathlib import Path

import faiss
import torch
import open_clip

# --- Config ---
INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("metadata.pkl")
TOP_K = 5

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model ---
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# --- Load index ---
if not INDEX_PATH.exists() or not META_PATH.exists():
    raise SystemExit("Index not found. Run index.py first.")

index = faiss.read_index(str(INDEX_PATH))
metadata = pickle.loads(META_PATH.read_bytes())
print(f"Loaded {index.ntotal} vectors")


def search(q, k=TOP_K):
    tokens = tokenizer([q]).to(device)
    with torch.no_grad():
        tf = model.encode_text(tokens)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    scores, ids = index.search(tf.cpu().numpy().astype("float32"), k)
    return [(metadata[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


if __name__ == "__main__":
    print("Type a query, or 'exit' to quit.")
    while True:
        try:
            q = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() == "exit":
            break
        for r, (p, s) in enumerate(search(q), 1):
            print(f"{r}. {s:.4f}  {p}")