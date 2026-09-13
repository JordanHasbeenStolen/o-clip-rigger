import torch
import open_clip
from PIL import Image

print("torch:", torch.__version__)
print("open_clip:", open_clip.__version__)

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

img = Image.new("RGB", (224, 224), color="red")
with torch.no_grad():
    img_feat = model.encode_image(preprocess(img).unsqueeze(0))
    txt_feat = model.encode_text(tokenizer(["test"]))

print("image features:", img_feat.shape)
print("text features:", txt_feat.shape)
print("OK")