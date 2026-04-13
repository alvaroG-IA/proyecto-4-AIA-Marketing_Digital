import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request
from segment_anything import sam_model_registry, SamPredictor

# =========================
# CONFIGURACIÓN
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# 1. Descargar checkpoint si no existe
if not os.path.exists(SAM_CHECKPOINT):
    print("Descargando checkpoint de SAM (esto puede tardar)...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)

# 2. Cargar modelo SAM
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# 3. Cargar imagen
# Cambia 'img.png' por 'card_deck.png' o cualquier otra imagen
image_path = "img.png"
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"No se pudo cargar la imagen en {image_path}. Revisa la ruta.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]
predictor.set_image(image)

print(f"Imagen cargada: {w}x{h}. Iniciando segmentación inteligente...")

# ==========================================
# 4. LÓGICA DE SEGMENTACIÓN ITERATIVA
# ==========================================
# Paso 1: Punto inicial en el centro geométrico
input_points = np.array([[w // 2, h // 2]])
input_labels = np.array([1])

num_iterations = 2
final_mask = None

for i in range(num_iterations):
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    # Seleccionamos la máscara con mejor score
    best_idx = np.argmax(scores)
    current_mask = masks[best_idx]
    current_scores = scores[best_idx]

    if i < num_iterations - 1:
        # Refinamiento: buscamos el punto de máxima confianza dentro de lo detectado
        mask_coords = np.argwhere(current_mask)
        if len(mask_coords) > 0:
            masked_scores = current_scores * current_mask
            max_score_pos = np.unravel_index(np.argmax(masked_scores), masked_scores.shape)
            new_point = [max_score_pos[1], max_score_pos[0]]  # x, y

            input_points = np.concatenate([input_points, [new_point]], axis=0)
            input_labels = np.concatenate([input_labels, [1]], axis=0)

    final_mask = current_mask

# ==========================================
# 5. POSTPROCESADO (LIMPIEZA DE FONDO)
# ==========================================
mask_uint8 = final_mask.astype(np.uint8)

# --- FILTRO DE COMPONENTES CONECTADAS ---
# Esto identifica manchas aisladas y elimina las que no son el objeto central
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)

if num_labels > 1:
    # Estrategia: Nos quedamos con la mancha que esté más cerca del centro de la imagen
    center_y, center_x = h // 2, w // 2

    # Verificamos si el centro cae en alguna etiqueta (label > 0 es objeto)
    label_central = labels[center_y, center_x]

    if label_central > 0:
        mask_refined = (labels == label_central).astype(np.uint8)
    else:
        # Si el centro exacto está vacío, tomamos la mancha más grande (excluyendo el fondo label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_refined = (labels == largest_label).astype(np.uint8)
else:
    mask_refined = mask_uint8

# --- MEJORA MORFOLÓGICA ---
kernel = np.ones((5, 5), np.uint8)
# Rellenar huecos internos (brillos metálicos)
mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel)
# Suavizar bordes dentados
mask_refined = cv2.medianBlur(mask_refined, 5)

# =========================
# 6. VISUALIZACIÓN
# =========================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original y Puntos de Semilla")
plt.imshow(image)
plt.scatter(input_points[:, 0], input_points[:, 1], color='blue', marker='o', s=40)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Máscara Limpia (Sin Fondo)")
plt.imshow(mask_refined, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Resultado Final")
overlay = image.copy()
overlay[mask_refined == 1] = [255, 0, 0]  # Pintamos el objeto de rojo
plt.imshow(cv2.addWeighted(image, 0.7, overlay, 0.3, 0))
plt.axis("off")

plt.tight_layout()
plt.show()

# Guardar la máscara final
mask_final = 1 - mask_refined
cv2.imwrite("mask.png", mask_final * 255)
print("Proceso finalizado. Máscara guardada como 'mask_perfecta.png'")