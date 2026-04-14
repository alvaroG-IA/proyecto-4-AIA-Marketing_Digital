# src/agent/nodes/node_api.py
import os
import requests
from typing import Dict, Any
import replicate
from dotenv import load_dotenv
import numpy as np
import cv2

# Cargamos el entorno para acceder a la key de la API
load_dotenv()


def nodo_api_replicate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo que recibe la imagen, la máscara y el prompt orientado a Flux, y genera la
    imagen final usando el modelo Flux mediante el uso de la API Replicate.
    """
    print("\n[Nodo 3] 🚀 Iniciando Generación en la nube (Replicate + FLUX)...")

    image_path = state.get("original_img_path")
    mask_path = state.get("sam_mask_path")
    prompt_flux = state.get("flux_prompt")

    if not image_path or not os.path.exists(image_path):
        print(f"❌ Error Crítico: No se encontró la imagen original en: '{image_path}'")
        return {}
    if not mask_path or not os.path.exists(mask_path):
        print(f"❌ Error Crítico: No se encontró la máscara en: '{mask_path}'")
        return {}

    try:
        print(f"[Nodo 3] ⚡ Procesando máscara localmente (Blur + Dilate)...")

        # Procesamos la máscara para darle más realismo a la imagen generada
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_blurred = cv2.GaussianBlur(mask, (13, 13), 0)
        new_mask_path = "mascara_procesada.png"
        cv2.imwrite(new_mask_path, mask_blurred)

        print(f"[Nodo 3] ⚡ Enviando petición a Replicate...")
        # 5. Llamada a Replicate usando los archivos abiertos
        output = replicate.run(
            "black-forest-labs/flux-fill-dev",
            input={
                "prompt": prompt_flux,
                "image": open(image_path, "rb"),
                "mask": open(new_mask_path, "rb"),
                "guidance": 35.0,
                "steps": 50,
                "output_format": "png",
                "output_quality": 100,
                "prompt_upsampling": False
            }
        )

        image_url_resultado = str(output[0])
        ruta_resultado = "output/resultado_final_replicate.png"
        print(f"[Nodo 3] 📥 Descargando imagen final...")
        response = requests.get(image_url_resultado)

        if response.status_code == 200:
            with open(ruta_resultado, 'wb') as f:
                f.write(response.content)
            print(f"[Nodo 3] ✅ ¡ÉXITO! Imagen guardada en: '{ruta_resultado}'")
            return {"final_img_path": ruta_resultado}
        else:
            print(f"[Nodo 3] ❌ Error HTTP: {response.status_code}")
            return {}

    except Exception as e:
        print(f"[Nodo 3] ❌ Error Crítico: {e}")
        return {}