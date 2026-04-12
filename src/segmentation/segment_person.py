import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# -------------------------
# CONFIG
# -------------------------
IMAGE_PATH = "../RyanGosling.jpg"
OUTPUT_MASK = "outputs/person_mask.png"
OUTPUT_OVERLAY = "outputs/person_overlay.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# -------------------------
# 1️⃣ Cargar modelo
# -------------------------
print("Loading SegFormer model...")

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    use_safetensors=True
)

model.to(DEVICE)
model.eval()

print("Model loaded ✓")


# -------------------------
# 2️⃣ Cargar imagen
# -------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)


# -------------------------
# 3️⃣ Preprocesar
# -------------------------
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# -------------------------
# 4️⃣ Inferencia
# -------------------------
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits  # (1, clases, h, w)

# Resize al tamaño original
logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

segmentation = logits.argmax(dim=1)[0].cpu().numpy()


# -------------------------
# 5️⃣ Obtener clase "person" correctamente
# -------------------------
person_class_id = None

for idx, label in model.config.id2label.items():
    if label.lower() == "person":
        person_class_id = int(idx)
        break

if person_class_id is None:
    raise ValueError("No se encontró la clase 'person' en el modelo")

print("Person class ID:", person_class_id)


# -------------------------
# 6️⃣ Crear máscara
# -------------------------
person_mask = (segmentation == person_class_id).astype(np.uint8) * 255


# -------------------------
# 7️⃣ Limpieza de máscara (MUY IMPORTANTE para VTON)
# -------------------------
kernel = np.ones((5, 5), np.uint8)

person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)


# -------------------------
# 8️⃣ Guardar máscara
# -------------------------
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(OUTPUT_MASK, person_mask)
print("Mask saved:", OUTPUT_MASK)


# -------------------------
# 9️⃣ Overlay para debug
# -------------------------
overlay = image_np.copy()
overlay[person_mask == 0] = (overlay[person_mask == 0] * 0.3).astype(np.uint8)

cv2.imwrite(OUTPUT_OVERLAY, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("Overlay saved:", OUTPUT_OVERLAY)


# -------------------------
# 🔟 Visualización
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Person Mask")
plt.imshow(person_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.show()

print("\nDONE 🎉 Mask ready for VTON pipeline")