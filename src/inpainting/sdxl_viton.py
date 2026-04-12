import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector

# =====================================================
# 0️⃣ CONFIG
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SIZE = (512, 768)

print("Device:", DEVICE)

# =====================================================
# 1️⃣ OPENPOSE + CONTROLNET
# =====================================================

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

controlnet_pose = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=DTYPE
)

# =====================================================
# 2️⃣ PIPELINE SDXL INPAINTING
# =====================================================

print("Loading diffusion model...")

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet_pose,
    torch_dtype=DTYPE,
    variant="fp16"
).to(DEVICE)

# IP-Adapter (solo guía visual, SIN hacks raros)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

pipe.set_ip_adapter_scale(2.0)

# =====================================================
# 3️⃣ CARGAR INPUTS (IMPORTANTE: usamos WARPED IMAGE)
# =====================================================

size = (512, 768)

# 🔥 CAMBIO IMPORTANTE: usamos la imagen ya deformada
person  = Image.open("init_tryon.png").convert("RGB").resize(size)

mask    = Image.open("../segmentation/outputs/mask_torso.png").convert("L").resize(size)
garment = Image.open("../../garments/tshirt_1.png").convert("RGB").resize(size)

# Pose (puedes reutilizar la del warp o recalcular)
pose_image = openpose(person).resize(size)
pose_image.save("pose.png")

# =====================================================
# 4️⃣ PROMPTS (SIMPLIFICADOS = MEJOR RESULTADO)
# =====================================================

prompt = "a realistic photo of a person wearing the provided garment"

negative_prompt = """
different clothes, different shirt, changed design,
logos changed, text altered, blurry, deformed body,
extra arms, bad proportions
"""

# =====================================================
# 5️⃣ INPAINTING
# =====================================================

print("Generating try-on...")

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,

    image=person,          # 🔥 init_tryon (NO avatar limpio)
    mask_image=mask,

    control_image=pose_image,

    ip_adapter_image=garment,

    num_inference_steps=30,
    guidance_scale=6.0,
    controlnet_conditioning_scale=0.9,
    strength=0.95
).images[0]

# =====================================================
# 6️⃣ OUTPUT
# =====================================================

result.save("tryon_result.png")

print("DONE ✅")
print("Saved: tryon_result.png")