import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from controlnet_aux import CannyDetector

from prompt_optimizer import optimize_prompt

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODELOS
# =====================================================

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# Fix the VAE tiling warning while we are at it
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

# =====================================================
# CONTROLNET (CANNY)
# =====================================================
canny = CannyDetector()

# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def generate_scene(image_path, mask_path, prompt):
    # SDXL es estricto con los múltiplos de 8 o 64. 1024 es ideal.
    res = (1024, 1024)
    image = Image.open(image_path).convert("RGB").resize(res)
    mask = Image.open(mask_path).convert("L").resize(res)

    image_np = np.array(image)

    # Procesamos Canny
    control_image_np = canny(image_np, low_threshold=100, high_threshold=200)
    control_image = Image.fromarray(control_image_np)

    negative_prompt = (
        "low quality, bad anatomy, text, watermark, deformed, blurry, "
        "abstract, messy, overexposed, cartoon, painting, unrealistic, "
        "extra legs, distorted background, floating object"
    )

    # EJECUCIÓN CON CORRECCIÓN DE DIMENSIONES
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        control_image=control_image,
        num_inference_steps=40,
        guidance_scale=8.5,
        controlnet_conditioning_scale=0.7,
        # Añadimos estos para forzar la coherencia de dimensiones en SDXL
        height=1024,
        width=1024,
    ).images[0]

    return result


# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":

    prompt = optimize_prompt("Pon el producto en una playa tropical al atardecer")

    output = generate_scene(
        "img.png",
        "mask.png",
        prompt
    )

    output.save("result.png")
    print("Image saved!")