# tools/diffusion_tool.py
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from controlnet_aux import CannyDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. FIX CRÍTICO: Cargamos el modelo Canny, no el Depth
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

canny = CannyDetector()


def diffusion_tool(state):
    print("\n🎨 Generating final image with Anti-Halo processing...")

    # Redimensionar suele ser necesario, pero ojo con deformar objetos que no sean cuadrados perfectos.
    image = Image.open(state["image_path"]).convert("RGB").resize((512, 512))
    mask = Image.open(state["mask_path"]).convert("L").resize((512, 512))

    image_np = np.array(image)
    mask_np = np.array(mask)

    # ==========================================
    # 2. ELIMINACIÓN DEL HALO (Anti-Halo Magic)
    # ==========================================
    # Hacemos que el fondo blanco "crezca" y se coma el borde del objeto original.
    # iterations=2 significa que morderá unos 2-3 píxeles hacia adentro.
    kernel = np.ones((5, 5), np.uint8)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    # Después de dilatar, aplicamos el Feathering (suavizado) para fundir la luz.
    mask_np = cv2.GaussianBlur(mask_np, (11, 11), 0)

    # Volvemos a convertir la máscara a formato PIL para el pipeline
    mask_pil = Image.fromarray(mask_np)

    # ==========================================
    # 3. EXTRACCIÓN DE LÍNEAS ESTRUCTURALES
    # ==========================================
    control = canny(image_np, 100, 200)

    # Borramos las líneas de Canny que caigan en la zona del fondo a repintar (zona blanca)
    # Así la IA tiene libertad total para crear un fondo nuevo sin basarse en la geometría antigua.
    control[mask_np > 127] = 0

    control_pil = Image.fromarray(control)

    # 4. GENERACIÓN
    result = pipe(
        prompt=state["optimized_prompt"],
        negative_prompt=state.get("negative_prompt", ""),
        image=image,
        mask_image=mask_pil,
        control_image=control_pil,
        num_inference_steps=40,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8
    ).images[0]

    path = "result.png"
    result.save(path)

    print(f"✅ Final image successfully saved at: {path}")

    return {"result_path": path}