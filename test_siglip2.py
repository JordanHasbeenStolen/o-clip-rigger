"""Accuracy check: OpenCLIP vs SigLIP2 on the real Clinny surface-tagging dataset.

Not a captioning test - just zero-shot top-1 matching against the fixed class
vocabulary, using data/test_tags images and the ground truth from
data/3v  ИИ модуль - тесты фотографий.md.

Run: uv run python test_siglip2.py
"""

from pathlib import Path

import torch
import open_clip
from PIL import Image

DATA_DIR = Path("data/test_tags")

# key -> short English phrase fed to the text encoder
CLASSES = {
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

# filename (in data/test_tags) -> ground-truth class key, from the table
GROUND_TRUTH = {
    "photo_2026-09-30_20-09-019123.jpg": "natural_stone",      # 01
    "photo_2026-09-30_20-09-09123.jpg": "ceramic_tile",        # 02
    "a38hqeo6wqb36khyivla4pgd9frj0wrg 1.jpeg": "laminate",     # 03
    "Pasted image 20260930203235.png": "lacquered_wood",       # 04
    "photo_2026-09-30_20-09-0812.jpg": "oiled_wood",           # 05
    "Pasted image 20260930203417.png": "vinyl_pvc",            # 06
    "photo_2026-09-30_20-09-083 2.jpg": "stainless_steel",     # 07
    "photo_2026-09-30_20-09-08.jpg": "chrome_sanitary",        # 08
    "photo_2026-09-30_20-09-08 (2).jpg": "glass_mirror",       # 09
    "photo_2026-09-30_20-09-081.jpg": "plastic",                # 10
    "photo_2026-09-30_20-09-017.jpg": "fabric_upholstery",     # 11
    "photo_2026-09-30_20-09-07 (2).jpg": "leather",            # 12
    "Pasted image 20260930205009.png": "ceramic_tile",          # 13 (tile-that-looks-like-stone)
    "Pasted image 20260930204933.png": "vinyl_pvc",             # 15 (vinyl-that-looks-like-wood)
}


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
