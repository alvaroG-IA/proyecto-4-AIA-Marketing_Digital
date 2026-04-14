# src/agent/nodes/node_api.py
import os
import requests
from typing import Dict, Any
import replicate
from dotenv import load_dotenv
import numpy as np
import cv2

# 1. Forzamos la carga del .env buscando en la carpeta raíz
load_dotenv()

# 2. VERIFICACIÓN BLINDADA DEL TOKEN
token_replicate = os.getenv("REPLICATE_API_TOKEN")

if not token_replicate:
    print("🚨 ERROR FATAL: No se ha encontrado la variable de entorno REPLICATE_API_TOKEN.")
else:
    os.environ["REPLICATE_API_TOKEN"] = token_replicate


def nodo_api_replicate(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n[Nodo 3] 🚀 Iniciando Generación en la nube (Replicate + FLUX)...")

    ruta_imagen = state.get("original_img_path")
    ruta_mascara = state.get("sam_mask_path")
    prompt_flux = state.get("flux_prompt")

    if not ruta_imagen or not os.path.exists(ruta_imagen):
        print(f"❌ Error Crítico: No se encontró la imagen original en: '{ruta_imagen}'")
        return {}
    if not ruta_mascara or not os.path.exists(ruta_mascara):
        print(f"❌ Error Crítico: No se encontró la máscara en: '{ruta_mascara}'")
        return {}

    try:
        print(f"[Nodo 3] ⚡ Procesando máscara localmente (Blur + Dilate)...")
        # 1. Leer máscara
        mask = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 3. Aplicar Blur Gaussiano
        mask_blurred = cv2.GaussianBlur(mask, (13, 13), 0)

        # 4. GUARDAR LA MÁSCARA PROCESADA (Paso crítico para solucionar el error 422)
        ruta_mascara_temp = "mascara_procesada.png"
        cv2.imwrite(ruta_mascara_temp, mask_blurred)

        print(f"[Nodo 3] ⚡ Enviando petición a Replicate...")

        # 5. Llamada a Replicate usando los archivos abiertos
        output = replicate.run(
            "black-forest-labs/flux-fill-dev",
            input={
                "prompt": prompt_flux,
                "image": open(ruta_imagen, "rb"),
                "mask": open(ruta_mascara_temp, "rb"),
                "guidance": 35.0,
                "steps": 50,
                "output_format": "png",
                "output_quality": 100,
                "prompt_upsampling": False
            }
        )

        # Extracción robusta de la URL
        image_url_resultado = None
        if isinstance(output, list) and len(output) > 0:
            image_url_resultado = str(output[0])
        elif isinstance(output, str):
            image_url_resultado = output
        else:
            raise ValueError(f"Tipo de resultado inesperado: {type(output)}")

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