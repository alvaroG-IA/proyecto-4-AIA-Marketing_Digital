# tools/sam_tool.py
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import urllib.request
import os

# 1. Inicialización de SAM (Se mantiene igual, cargando el modelo una sola vez)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
if not os.path.exists(SAM_CHECKPOINT):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        SAM_CHECKPOINT
    )

sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
predictor = SamPredictor(sam)


def sam_tool(state):
    """
    Tool que segmenta el objeto principal usando SAM con un punto central
    y aplica limpieza avanzada de islas flotantes y bordes.
    """
    print("\n✂️ Segmenting product with advanced logic...")

    image_path = state.get("image_path")
    if not image_path or not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return {}

    # 2. Cargar imagen y generar embeddings
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # 3. Lógica del Punto Central Inteligente
    h, w = image_rgb.shape[:2]
    centro_x, centro_y = w // 2, h // 2

    point = np.array([[centro_x, centro_y]])
    label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True
    )

    best_idx = np.argmax(scores)
    final_mask = masks[best_idx]

    # ==========================================
    # 4. POSTPROCESADO (Limpieza de higiene)
    # ==========================================
    mask_uint8 = final_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

    if num_labels > 1:
        # Comprobamos a qué "isla" pertenece el centro exacto
        label_central = labels[centro_y, centro_x]

        if label_central > 0:
            mask_refined = (labels == label_central).astype(np.uint8)
        else:
            # Si el centro está vacío, cogemos la isla más grande
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_refined = (labels == largest_label).astype(np.uint8)
    else:
        mask_refined = mask_uint8

    # Suavizado de bordes
    kernel = np.ones((5, 5), np.uint8)
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel, iterations=5)
    mask_refined = cv2.medianBlur(mask_refined, 5)

    # 5. Inversión y Guardado (Invertimos colores para FLUX)
    # Nota: Tu código antiguo tenía un error lógico en la inversión que aquí está corregido.
    mask_final = 1 - mask_refined
    mask_final_img = mask_final * 255  # Escalamos a 0-255

    path = "mask.png"
    cv2.imwrite(path, mask_final_img)

    print(f"✅ Mask successfully saved at: {path}")

    return {"mask_path": path}