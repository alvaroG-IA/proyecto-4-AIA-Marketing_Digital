# tools/diffusion_tool.py
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from controlnet_aux import CannyDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Cargamos el modelo Canny
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
    print("\n🎨 Generating final image with Dynamic Resizing and Anti-Halo...")

    # 1. CARGA DE IMÁGENES ORIGINALES
    img_orig = Image.open(state["image_path"]).convert("RGB")
    mask_orig = Image.open(state["mask_path"]).convert("L")
    w_orig, h_orig = img_orig.size

    # 2. CÁLCULO DE RESIZE DINÁMICO
    # Para SD 1.5, el límite ideal es 768px para evitar duplicidades de objetos
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

    print(f"📏 Dynamic Resize: {w_orig}x{h_orig} -> {new_w}x{new_h}")

    # Aplicar el resize
    image = img_orig.resize((new_w, new_h), Image.LANCZOS)
    mask = mask_orig.resize((new_w, new_h), Image.LANCZOS)

    image_np = np.array(image)
    mask_np = np.array(mask)

    # ==========================================
    # 2. ELIMINACIÓN DEL HALO Y PROCESAMIENTO
    # ==========================================
    # Dilatamos un poco la máscara para que la IA "muerda" el borde original

    # Suavizado de bordes (Feathering)
    mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0)
    mask_pil = Image.fromarray(mask_np)

    # ==========================================
    # 3. CONTROLNET CANNY (Ajustado a la nueva resolución)
    # ==========================================
    # Forzamos a Canny a trabajar con el tamaño exacto del resize
    control = canny(image_np, 100, 200, detect_resolution=new_h, image_resolution=new_h)

    # Aseguramos coincidencia de dimensiones por si Canny redondea distinto
    if control.shape[:2] != mask_np.shape[:2]:
        control = cv2.resize(control, (new_w, new_h))

    # Limpiamos las líneas del fondo para dar libertad a la IA
    control[mask_np > 127] = 0
    control_pil = Image.fromarray(control)

    # 4. GENERACIÓN
    result = pipe(
        prompt=state["optimized_prompt"],
        negative_prompt=state.get("negative_prompt", ""),
        image=image,
        mask_image=mask_pil,
        control_image=control_pil,
        num_inference_steps=60,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.9
    ).images[0]

    # 5. VOLVER AL TAMAÑO ORIGINAL (Opcional pero recomendado para entrega)
    # Re-escalamos el resultado de 768px de vuelta a los 2048px originales
    print(f"🔝 Upscaling back to original size: {w_orig}x{h_orig}")
    result_final = result.resize((w_orig, h_orig), Image.LANCZOS)

    path = "result.png"
    result_final.save(path)

    print(f"✅ Final image saved at: {path}")

    return {"result_path": path}