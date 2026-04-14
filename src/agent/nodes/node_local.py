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
# Selección de device y su tipo recomendado de usar
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"👁️ Cargando modelos de Difusión en memoria usando dispositivo: {DEVICE}...")

# 1. Cargamos el modelo Control-Net basado en Canny
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=DTYPE
)

# 2. Cargamos el Pipeline de Inpaint
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

if DEVICE == "cuda":
    pipe.enable_model_cpu_offload()

canny = CannyDetector()
print("✅ Modelos de Difusión listos y a la espera.")

NUM_INFERENCE_STEPS = 60


# ==========================================
# 2. FUNCIÓN NODO
# ==========================================
def nodo_generador(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo que recibe la imagen, la máscara y los prompts, y genera la 
    imagen final usando Stable Diffusion + ControlNet con resize dinámico.
    """
    print("\n[Nodo 3] 🎨 Iniciando Generación de Imagen (Diffusion + ControlNet)...")

    # 1. Extraemos variables del estado (Soportamos tu nomenclatura antigua y la nueva)
    img_path = state.get("original_img_path")
    mask_path = state.get("sam_mask_path")
    positive_prompt = state.get("flux_prompt")
    negative_prompt = state.get("negative_prompt")

    # Validaciones de seguridad de las variables del estado
    if not img_path or not os.path.exists(img_path):
        print(f"[Nodo 3] ❌ Error: No se encontró la imagen original en {img_path}.")
        return {}
    if not mask_path or not os.path.exists(mask_path):
        print(f"[Nodo 3] ❌ Error: No se encontró la máscara en {mask_path}.")
        return {}
    if not positive_prompt or not negative_prompt:
        print(f"[Nodo 3] ❌ Error: No se encontró ningún prompt positivo o ningún prompt negativo.")
        return {}
    try:

        img_orig = Image.open(img_path).convert("RGB")
        mask_orig = Image.open(mask_path).convert("L")
        w_orig, h_orig = img_orig.size

        MAX_DIM = 512 # Fijamos un valor máximo para el modelo de difusión (se recomienda este o como máximo 768px)

        if w_orig > h_orig:
            new_w = MAX_DIM
            new_h = int(MAX_DIM * (h_orig / w_orig))
        else:
            new_h = MAX_DIM
            new_w = int(MAX_DIM * (w_orig / h_orig))

        # Nos aseguramos que las nuevas dimensiones sigan siendo múltiplo de 64
        new_w = (new_w // 64) * 64
        new_h = (new_h // 64) * 64

        image = img_orig.resize((new_w, new_h), Image.LANCZOS)
        mask = mask_orig.resize((new_w, new_h), Image.LANCZOS)

        image_np = np.array(image)
        mask_np = np.array(mask)

        # Se realiza un suavizado a la máscara para evitar que el objeto se vea como una pegatina
        mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)
        mask_pil = Image.fromarray(mask_np)

        # Obtenemos la imagen de control
        control = canny(image_np, 100, 200, detect_resolution=new_h, image_resolution=new_h)
        if control.shape[:2] != mask_np.shape[:2]:
            control = cv2.resize(control, (new_w, new_h))
        control[mask_np > 127] = 0
        control_pil = Image.fromarray(control)

        print(f"[Nodo 3] ⏳ Generando píxeles... ({NUM_INFERENCE_STEPS} Pasos)")
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_pil,
            control_image=control_pil,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.9
        ).images[0]

        final_img = result.resize((w_orig, h_orig), Image.LANCZOS)

        final_img_path = "output/resultado_final_local.png"
        final_img.save(final_img_path)

        print(f"[Nodo 3] ✅ Imagen final guardada con éxito.")

        return {
            "final_img_path": final_img_path
        }

    except Exception as e:
        print(f"[Nodo 3] ❌ Error crítico durante la difusión: {e}")
        return {}