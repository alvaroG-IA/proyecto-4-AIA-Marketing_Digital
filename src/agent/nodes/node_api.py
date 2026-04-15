# src/agent/nodes/node_api.py
import os
import requests
from typing import Dict, Any
import replicate
from dotenv import load_dotenv
import numpy as np
import cv2

# ==========================================
# 1. CONFIGURACIÓN Y CARGA GLOBAL DEL MODELO
# ==========================================
# Cargamos el entorno para acceder a la key de la API
load_dotenv()


# ==========================================
# 2. FUNCIÓN NODO
# ==========================================
def nodo_generador_online(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo que recibe la imagen, la máscara y el prompt orientado a Flux, y genera la
    imagen final usando el modelo Flux mediante el uso de la API Replicate.
    """
    print("\n[Nodo 3] 🚀 Iniciando Generación en la nube (Replicate + Nano-Banana)...")

    image_path = state.get("original_img_path")
    mask_path = state.get("sam_mask_path")
    descriptive_prompt = state.get("descriptive_prompt")

    if not image_path or not os.path.exists(image_path):
        print(f"❌ Error Crítico: No se encontró la imagen original en: '{image_path}'")
        return {}
    if not mask_path or not os.path.exists(mask_path):
        print(f"❌ Error Crítico: No se encontró la máscara en: '{mask_path}'")
        return {}

    try:
        print(f"[Nodo 3] ⚡ Enviando petición a Replicate...")
        # 5. Llamada a Replicate usando los archivos abiertos
        output = replicate.run(
            "google/nano-banana",
            input={
                "prompt": descriptive_prompt,
                "image_input": [
                    open(image_path, "rb")

                ],
                "output_format": "jpg"
            }
        )

        image_url_resultado = str(output)
        ruta_resultado = "output/resultado_final_online.png"
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
