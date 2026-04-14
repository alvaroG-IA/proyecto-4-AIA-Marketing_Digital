import os
import urllib.request
from typing import Dict, Any
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

# ==========================================
# 1. CONFIGURACIÓN Y CARGA GLOBAL DEL MODELO
# ==========================================
# Selección de device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# Descargar checkpoint si no existe en local
if not os.path.exists(SAM_CHECKPOINT):
    print("📥 Descargando checkpoint de SAM (esto puede tardar un poco)...")
    urllib.request.urlretrieve(SAM_URL, SAM_CHECKPOINT)

print(f"👁️ Cargando modelo SAM en memoria usando dispositivo: {DEVICE}...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)

predictor = SamPredictor(sam)
print("✅ SAM listo y a la espera.")

# ==========================================
# 2. Función Nodo
# ==========================================


def nodo_segmentador(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo que recibe la ruta de la imagen, aplica un Punto Central Inteligente
    para segmentar el objeto principal y guarda la máscara en disco temporalmente.
    """
    print("\n[Nodo 2] ✂️  Iniciando Segment Anything Model (SAM) con Punto Central...")
    
    original_path = state.get("original_img_path")
    
    if not original_path or not os.path.exists(original_path):
        print(f"[Nodo 2] ❌ Error: No se encontró la imagen original en {original_path}.")
        return {}

    try:

        image = cv2.imread(original_path)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen con cv2: {original_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        predictor.set_image(image_rgb)

        # Punto central donde se encuentra nuestro objeto
        centro_x = w // 2
        centro_y = h // 2
        input_point = np.array([[centro_x, centro_y]])
        input_label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # Seleccionamos la máscara con mayor valor de confianza
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx]

        # Limpiamos la máscara mediante operaciones morfolóficas
        mask_uint8 = final_mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

        if num_labels > 1:
            label_central = labels[centro_y, centro_x]
            if label_central > 0:
                mask_refined = (labels == label_central).astype(np.uint8)
            else:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask_refined = (labels == largest_label).astype(np.uint8)
        else:
            mask_refined = mask_uint8

        # Suavizado de bordes
        kernel = np.ones((5, 5), np.uint8)
        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_refined = cv2.medianBlur(mask_refined, 5)

        # 4. Guardado de la máscara (Invirtiendo colores para FLUX)
        mask_final = 1 - mask_refined
        ruta_mascara_salida = "mascara_sam_temporal.png"
        cv2.imwrite(ruta_mascara_salida, mask_final * 255)
        
        print(f"[Nodo 2] ✅ Máscara limpia generada en 1 solo paso.")

        return {
            "ruta_mascara_sam": ruta_mascara_salida
        }

    except Exception as e:
        print(f"[Nodo 2] ❌ Error crítico procesando la imagen con SAM: {e}")
        return {}
