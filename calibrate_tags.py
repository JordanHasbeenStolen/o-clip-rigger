"""Fit a per-model temperature T for tag_image(), via Guo et al. 2017
(temperature scaling): the T that minimizes negative log-likelihood on our
own labeled ground truth. Doesn't change which tag wins, only how sharp
the resulting softmax distribution looks.

Run once after changing TAGS or the ground-truth set, then copy the
printed "temperature" values into core.SEARCH_MODELS.

Run: uv run python calibrate_tags.py
"""

import json
import math
from pathlib import Path

import torch
from PIL import Image

import core

DATA_DIR = Path("data/test_tags")
# Private, gitignored - built from data/3v  ИИ модуль - тесты фотографий.md.
# Shared with test_siglip2.py so there's one source of truth, not two.
GROUND_TRUTH = json.loads((DATA_DIR / "ground_truth.json").read_text())

T_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]


def fit_temperature(model: str):
    m, tokenizer, preprocess = core._load_model(model)
    keys = list(core.TAGS.keys())
    tokens = tokenizer([core.TAGS[k] for k in keys])
    with torch.no_grad():
        txt_feat = m.encode_text(tokens)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    scale = m.logit_scale.exp()

    all_logits, all_labels = [], []
    for filename, true_key in GROUND_TRUTH.items():
        image = preprocess(Image.open(DATA_DIR / filename).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            img_feat = m.encode_image(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        cos = (img_feat @ txt_feat.T).squeeze(0)
        all_logits.append(cos * scale)
        all_labels.append(keys.index(true_key))

    logits = torch.stack(all_logits)
    labels = torch.tensor(all_labels)

    best_t, best_nll = None, math.inf
    for t in T_GRID:
        nll = torch.nn.functional.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t, best_nll


if __name__ == "__main__":
    for model in core.SEARCH_MODELS:
        t, nll = fit_temperature(model)
        print(f"{model:10s} temperature={t}  NLL={nll:.4f}")
