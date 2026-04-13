import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from controlnet_aux import CannyDetector
from prompt_optimizer import optimize_prompt

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODELOS (Realistic Vision V5.1 + ControlNet 1.5)
# =====================================================

# ControlNet específico para SD 1.5
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Pipeline de SD 1.5 con Realistic Vision
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None  # Desactivado para evitar falsos positivos
).to(device)

# Usamos DDIM para mejores resultados en inpainting
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# =====================================================
# CONTROLNET (CANNY)
# =====================================================
canny = CannyDetector()


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def generate_scene(image_path, mask_path, prompt):
    # SD 1.5 funciona nativamente a 512x512 (es más rápido y estable)
    # Si necesitas más calidad, luego se puede escalar (Upscale)
    res = (512, 512)
    image = Image.open(image_path).convert("RGB").resize(res)
    mask = Image.open(mask_path).convert("L").resize(res)

    image_np = np.array(image)
    mask_np = np.array(mask)

    # --- LIMPIEZA DE BORDES (Clave para el realismo) ---
    control_image_np = canny(image_np, low_threshold=100, high_threshold=200)

    # Borramos los bordes del fondo (donde la máscara es blanca)
    # Esto evita que la IA intente dibujar la pared vieja
    control_image_np[mask_np > 127] = 0
    control_image = Image.fromarray(control_image_np)

    # --- PROMPTS ---
    # Realistic Vision funciona mejor con prompts en inglés
    negative_prompt = (
        "bad art, cartoon, lowres, blurry, error, cropped, worst quality, "
        "low quality, jpeg artifacts, duplicate, out of frame, watermark, "
        "signature, text, flat colors, abstract"
    )

    # --- INPAINTING ---
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        control_image=control_image,
        num_inference_steps=40,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.9,  # Dejamos que la IA ilumine la botella
        eta=0.0
    ).images[0]

    return result


# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":

    prompt = optimize_prompt("Pon el producto en un bosque")

    output = generate_scene(
        "img.png",
        "mask.png",
        prompt
    )

    output.save("result.png")
    print("Image saved!")