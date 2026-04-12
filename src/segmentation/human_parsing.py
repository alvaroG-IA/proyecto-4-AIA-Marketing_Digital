import torch
import numpy as np
import cv2
import os
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_IMAGE = "../RyanGosling.jpg"
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n🔥 Human Parsing con HuggingFace (REAL)")
print("Device:", DEVICE)

# --------------------------------------------------
# 1️⃣ LOAD MODEL (PUBLIC HF)
# --------------------------------------------------
model_name = "mattmdjaga/segformer_b2_clothes"

processor = AutoImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name, use_safetensors=True).to(DEVICE)
model.eval()

# --------------------------------------------------
# 2️⃣ LOAD IMAGE
# --------------------------------------------------
image_pil = Image.open(INPUT_IMAGE).convert("RGB")
width, height = image_pil.size

inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)

# --------------------------------------------------
# 3️⃣ INFERENCE
# --------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)

logits = torch.nn.functional.interpolate(
    outputs.logits,
    size=(height, width),
    mode="bilinear",
    align_corners=False
)

parsing = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

cv2.imwrite(f"{OUTPUT_DIR}/parsing_classes.png", parsing)
print("✔ parsing_classes.png guardado")

# --------------------------------------------------
# 4️⃣ LABELS CIHP (MUY IMPORTANTE)
# --------------------------------------------------
"""
CIHP classes:

  0: "background",
  1: "hat",
  2: "hair",
  3: "sunglasses",
  4: "upper clothes",
  5: "dress",
  6: "pants",
  7: "coat",
  8: "skirt",
  9: "shorts",
  10: "face",
  11: "left arm",
  12: "right arm",
  13: "left leg",
  14: "right leg",
  15: "left shoe",
  16: "right shoe"
"""

UPPER_CLOTHES = [4, 7]
# ARMS = [14, 15]

# --------------------------------------------------
# 5️⃣ CREATE VTON MASK
# --------------------------------------------------
mask = np.zeros_like(parsing)

for cls in UPPER_CLOTHES:
    mask[parsing == cls] = 255

# --------------------------------------------------
# 6️⃣ IMPROVE MASK FOR DIFFUSION
# --------------------------------------------------
kernel = np.ones((15,15), np.uint8)

mask = cv2.dilate(mask, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

mask_pil = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=6))
mask_pil.save(f"{OUTPUT_DIR}/mask_torso.png")

print("🎉 mask_torso.png lista para diffusion")