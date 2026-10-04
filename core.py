import io
import pickle
from pathlib import Path

import faiss
import torch
import open_clip
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Each model has its own embedding space, so each needs its own FAISS index -
# a OpenCLIP vector and a SigLIP2 vector for the same photo are not comparable.
# "temperature" is fitted per model by calibrate_tags.py (temperature scaling,
# Guo et al. 2017: https://arxiv.org/abs/1706.04599) - the single scalar that
# minimizes negative log-likelihood on our own labeled tag examples. It only
# sharpens/softens the resulting softmax, never changes which tag wins.
SEARCH_MODELS = {
    "openclip": {
        "model_name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "index_path": Path("faiss_index.bin"),
        "meta_path": Path("metadata.pkl"),
        "temperature": 1.5,
    },
    "siglip2": {
        "model_name": "ViT-B-16-SigLIP2",
        "pretrained": "webli",
        "index_path": Path("faiss_index_siglip2.bin"),
        "meta_path": Path("metadata_siglip2.pkl"),
        "temperature": 1.0,
    },
}

# Tagging has no index to match, so it reuses the same model configs.
TAG_MODELS = {name: (cfg["model_name"], cfg["pretrained"]) for name, cfg in SEARCH_MODELS.items()}

TAGS = {
    "natural_stone": "natural stone surface",
    "ceramic_tile": "ceramic tile",
    "laminate": "laminate flooring",
    "lacquered_wood": "lacquered wood",
    "oiled_wood": "oiled or waxed wood",
    "vinyl_pvc": "vinyl or linoleum floor",
    "stainless_steel": "stainless steel",
    "chrome_sanitary": "chrome faucet or shower fitting",
    "glass_mirror": "glass or mirror",
    "plastic": "plastic surface",
    "fabric_upholstery": "fabric upholstery",
    "leather": "leather",
}

# One cache for the model itself (shared by search and tagging - loading
# OpenCLIP or SigLIP2 twice for two different purposes would waste memory
# and startup time for no reason, since the weights are identical either way).
_model_cache: dict = {}
# Separate cache for the FAISS index/metadata, since only search needs those.
_index_cache: dict = {}


def _load_model(model: str):
    if model not in SEARCH_MODELS:
        raise ValueError(f"Unknown model: {model!r}, expected one of {list(SEARCH_MODELS)}")
    if model in _model_cache:
        return _model_cache[model]
    cfg = SEARCH_MODELS[model]
    m, _, preprocess = open_clip.create_model_and_transforms(cfg["model_name"], pretrained=cfg["pretrained"])
    m = m.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer(cfg["model_name"])
    _model_cache[model] = (m, tokenizer, preprocess)
    return _model_cache[model]


def _load_index(model: str):
    if model in _index_cache:
        return _index_cache[model]
    cfg = SEARCH_MODELS[model]
    index = faiss.read_index(str(cfg["index_path"]))
    metadata = pickle.loads(cfg["meta_path"].read_bytes())
    _index_cache[model] = (index, metadata)
    return _index_cache[model]


def search(query: str, k: int = 5, model: str = "openclip"):
    m, tokenizer, _ = _load_model(model)
    index, metadata = _load_index(model)
    tokens = tokenizer([query]).to(DEVICE)
    with torch.no_grad():
        tf = m.encode_text(tokens)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    scores, ids = index.search(tf.cpu().numpy().astype("float32"), k)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            results.append({
                "path": metadata[i],
                "index": int(i),
                "score": float(s),
            })
    return results


def total_vectors(model: str = "openclip") -> int:
    index, _ = _load_index(model)
    return index.ntotal


def get_path(idx: int, model: str = "openclip") -> str | None:
    _, metadata = _load_index(model)
    if 0 <= idx < len(metadata):
        return metadata[idx]
    return None


def warmup():
    for model in SEARCH_MODELS:
        _load_model(model)
        _load_index(model)


def tag_image(image_bytes: bytes, model: str = "openclip", k: int = 5):
    m, tokenizer, preprocess = _load_model(model)
    image = preprocess(Image.open(io.BytesIO(image_bytes)).convert("RGB")).unsqueeze(0).to(DEVICE)
    keys = list(TAGS.keys())
    tokens = tokenizer([TAGS[key] for key in keys]).to(DEVICE)

    with torch.no_grad():
        img_feat = m.encode_image(image)
        txt_feat = m.encode_text(tokens)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    # Same formula for every model: cosine similarity scaled by the model's
    # own learned logit_scale, then temperature-scaled softmax (see
    # SEARCH_MODELS comment) - always in [0, 1], always sums to 1 across
    # the 12 tags, and comparable across models since both go through the
    # identical, calibrated procedure.
    cos = (img_feat @ txt_feat.T).squeeze(0)
    logits = cos * m.logit_scale.exp()
    temperature = SEARCH_MODELS[model]["temperature"]
    probs = torch.softmax(logits / temperature, dim=-1)
    ranked = sorted(zip(keys, probs.tolist()), key=lambda x: -x[1])
    return [{"tag": tag, "score": float(score)} for tag, score in ranked[:k]]
