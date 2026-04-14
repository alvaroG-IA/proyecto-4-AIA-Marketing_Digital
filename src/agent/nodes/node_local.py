import os
from typing import Dict, Any
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from controlnet_aux import CannyDetector

# ==========================================
# 1. CONFIGURACIÓN Y CARGA GLOBAL DEL MODELO
# ==========================================
# Detección inteligente de hardware para evitar errores de float16 en CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32  # Usamos float32 en Mac para evitar bugs de compatibilidad
else:
    DEVICE = "cpu"
    DTYPE = torch.float32  # CPU no soporta float16 nativamente

print(f"👁️ Cargando modelos de Difusión en memoria usando dispositivo: {DEVICE}...")

# 1. Cargamos el modelo Canny
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=DTYPE
)

# 2. Cargamos el Pipeline Inpaint
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Solo aplicamos CPU offload si estamos usando CUDA (en Mac o CPU puede causar fallos)
if DEVICE == "cuda":
    pipe.enable_model_cpu_offload()

canny = CannyDetector()
print("✅ Modelos de Difusión listos y a la espera.")


# ==========================================
# 2. LA FUNCIÓN DEL NODO (Para LangGraph)
# ==========================================

def nodo_generador(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo que recibe la imagen, la máscara y los prompts, y genera la 
    imagen final usando Stable Diffusion + ControlNet con resize dinámico.
    """
    print("\n[Nodo 3] 🎨 Iniciando Generación de Imagen (Diffusion + ControlNet)...")

    # 1. Extraemos variables del estado (Soportamos tu nomenclatura antigua y la nueva)
    ruta_imagen = state.get("image_path") or state.get("ruta_imagen_original")
    ruta_mascara = state.get("mask_path") or state.get("ruta_mascara_sam")
    prompt_positivo = state.get("prompt_optimizado")

    print(prompt_positivo)

    prompt_negativo = state.get("prompt_negativo")

    # Validaciones de seguridad
    if not ruta_imagen or not os.path.exists(ruta_imagen):
        print(f"[Nodo 3] ❌ Error: No se encontró la imagen original en {ruta_imagen}.")
        return {}
    if not ruta_mascara or not os.path.exists(ruta_mascara):
        print(f"[Nodo 3] ❌ Error: No se encontró la máscara en {ruta_mascara}.")
        return {}
    if not prompt_positivo:
        print("[Nodo 3] ⚠️ Advertencia: No se recibió prompt optimizado. Usando valores base.")
        prompt_positivo = "high quality, detailed masterpiece"

    try:
        # ==========================================
        # 2. CÁLCULO DE RESIZE DINÁMICO
        # ==========================================
        img_orig = Image.open(ruta_imagen).convert("RGB")
        mask_orig = Image.open(ruta_mascara).convert("L")
        w_orig, h_orig = img_orig.size

        # Para SD 1.5, el límite ideal es 512px (o 768px si tienes buena GPU)
        MAX_DIM = 512

        if w_orig > h_orig:
            new_w = MAX_DIM
            new_h = int(MAX_DIM * (h_orig / w_orig))
        else:
            new_h = MAX_DIM
            new_w = int(MAX_DIM * (w_orig / h_orig))

        # Importante: Las dimensiones DEBEN ser múltiplos de 64 para el VAE
        new_w = (new_w // 64) * 64
        new_h = (new_h // 64) * 64

        print(f"[Nodo 3] 📏 Dynamic Resize: {w_orig}x{h_orig} -> {new_w}x{new_h}")

        image = img_orig.resize((new_w, new_h), Image.LANCZOS)
        mask = mask_orig.resize((new_w, new_h), Image.LANCZOS)

        image_np = np.array(image)
        mask_np = np.array(mask)

        # ==========================================
        # 3. ELIMINACIÓN DEL HALO Y PROCESAMIENTO
        # ==========================================
        # Suavizado de bordes (Feathering)
        mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0)
        mask_pil = Image.fromarray(mask_np)

        # ==========================================
        # 4. CONTROLNET CANNY (Ajustado a la nueva resolución)
        # ==========================================
        # Forzamos a Canny a trabajar con el tamaño exacto del resize
        control = canny(image_np, 100, 200, detect_resolution=new_h, image_resolution=new_h)

        # Aseguramos coincidencia de dimensiones
        if control.shape[:2] != mask_np.shape[:2]:
            control = cv2.resize(control, (new_w, new_h))

        # Limpiamos las líneas del fondo
        control[mask_np > 127] = 0
        control_pil = Image.fromarray(control)

        # ==========================================
        # 5. GENERACIÓN
        # ==========================================
        print(f"[Nodo 3] ⏳ Generando píxeles... (60 Pasos)")
        result = pipe(
            prompt=prompt_positivo,
            negative_prompt=prompt_negativo,
            image=image,
            mask_image=mask_pil,
            control_image=control_pil,
            num_inference_steps=60,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.9
        ).images[0]

        # ==========================================
        # 6. VOLVER AL TAMAÑO ORIGINAL
        # ==========================================
        print(f"[Nodo 3] 🔝 Upscaling back to original size: {w_orig}x{h_orig}")
        result_final = result.resize((w_orig, h_orig), Image.LANCZOS)

        ruta_resultado_salida = "resultado_final.png"
        result_final.save(ruta_resultado_salida)

        print(f"[Nodo 3] ✅ Imagen final guardada con éxito.")

        return {
            "ruta_resultado": ruta_resultado_salida
        }

    except Exception as e:
        print(f"[Nodo 3] ❌ Error crítico durante la difusión: {e}")
        return {}