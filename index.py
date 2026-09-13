import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

# --- Config ---
# DATA_DIR = Path("data") 
DATA_DIR = Path("data/test") 
INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("metadata.pkl")
BATCH = 32
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
TEST_LIMIT = 20  # set to None for full run

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# --- Model ---
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()

# --- Collect paths ---
paths = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]
print(f"Found {len(paths)} images")
if not paths:
    raise SystemExit("No images in data/")

if TEST_LIMIT is not None:
    paths = paths[:TEST_LIMIT]
    print(f"TEST_LIMIT active: using first {len(paths)} images")

# --- Encode in batches ---
vecs, kept = [], []
t0 = time.time()

for i in tqdm(range(0, len(paths), BATCH), desc="Encoding"):
    batch = paths[i:i + BATCH]
    tensors, valid = [], []
    for p in batch:
        try:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
            valid.append(p)
        except Exception as e:
            print(f"skip {p}: {e}")
    if not tensors:
        continue
    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        f = model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    vecs.append(f.cpu().numpy().astype(np.float32))
    kept.extend(valid)

emb = np.vstack(vecs)
print(f"Embeddings shape: {emb.shape}")

# --- Build FAISS index ---
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(INDEX_PATH))

# --- Save metadata ---
with open(META_PATH, "wb") as f:
    pickle.dump([str(p) for p in kept], f)

print(f"Done in {time.time() - t0:.1f}s")
print(f"Saved: {INDEX_PATH} ({index.ntotal} vectors)")
print(f"Saved: {META_PATH} ({len(kept)} paths)")