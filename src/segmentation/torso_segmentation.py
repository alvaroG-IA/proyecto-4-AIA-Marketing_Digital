import cv2
import numpy as np

mask = cv2.imread("outputs/person_mask.png", 0)
h, w = mask.shape

# Contar píxeles blancos por fila
white_pixels = np.sum(mask > 0, axis=1)

# -------------------------
# 1️⃣ Encontrar inicio del torso (hombros)
# -------------------------
shoulder_threshold = w * 0.25  # ancho mínimo hombros
torso_start = 0

for i, count in enumerate(white_pixels):
    if count > shoulder_threshold:
        torso_start = i
        break

# -------------------------
# 2️⃣ Encontrar inicio de piernas
# (cuando vuelve a estrecharse mucho)
# -------------------------
leg_threshold = w * 0.3
torso_end = h

for i in range(torso_start + 50, h):  # saltamos pecho para evitar ruido
    if white_pixels[i] < leg_threshold:
        torso_end = i
        break

# -------------------------
# 3️⃣ Crear máscara torso limpia
# -------------------------
torso_mask = np.zeros_like(mask)
torso_mask[torso_start:torso_end, :] = mask[torso_start:torso_end, :]

cv2.imwrite("outputs/mask_torso.png", torso_mask)

print("Torso mask created ✓")
print("Start row:", torso_start)
print("End row:", torso_end)