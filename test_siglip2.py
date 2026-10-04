"""Accuracy check: OpenCLIP vs SigLIP2 on the real Clinny surface-tagging dataset.

Not a captioning test - just zero-shot top-1 matching against the fixed class
vocabulary, using data/test_tags images and the labels in
data/test_tags/ground_truth.json (private, gitignored - built from
data/3v  ИИ модуль - тесты фотографий.md).

Run: uv run python test_siglip2.py
"""

import json
from pathlib import Path

import torch
import open_clip
from PIL import Image

from core import TAGS as CLASSES

DATA_DIR = Path("data/test_tags")
GROUND_TRUTH = json.loads((DATA_DIR / "ground_truth.json").read_text())


def run(model_name: str, pretrained: str):
    print(f"\n--- {model_name} ({pretrained}) ---")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    keys = list(CLASSES.keys())
    tokens = tokenizer([CLASSES[k] for k in keys])
    with torch.no_grad():
        txt_feat = model.encode_text(tokens)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    correct = 0
    for filename, true_key in GROUND_TRUTH.items():
        image = preprocess(Image.open(DATA_DIR / filename).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            img_feat = model.encode_image(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ txt_feat.T).squeeze(0)
        pred_key = keys[sims.argmax().item()]
        ok = pred_key == true_key
        correct += ok
        mark = "OK " if ok else "XX "
        print(f"  {mark} {filename[:45]:45s} true={true_key:20s} pred={pred_key}")

    total = len(GROUND_TRUTH)
    print(f"Accuracy: {correct}/{total} ({100 * correct / total:.0f}%)")


if __name__ == "__main__":
    run("ViT-B-32", "laion2b_s34b_b79k")       # current OpenCLIP model
    run("ViT-B-16-SigLIP2", "webli")            # candidate replacement
