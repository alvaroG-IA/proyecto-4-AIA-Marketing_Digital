import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
import cv2
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

controlnet_pose = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16
)

# -------------------------
# 1️⃣ Cargar modelo Inpainting
# -------------------------
print("Loading diffusion model...")

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet_pose,
    torch_dtype=torch.float16,
    variant="fp16"
).to(DEVICE)

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

pipe.set_ip_adapter_scale(1.5)


# -------------------------
# 2️⃣ Cargar imágenes
# -------------------------
size = (640, 1030)
person = Image.open("../RyanGosling.jpg").convert("RGB").resize(size)
mask   = Image.open("../segmentation/outputs/mask_torso.png").convert("L").resize(size)
garment = Image.open("../../garments/tshirt_1.png").convert("RGB").resize(size)

pose_image = openpose(person)
pose_image = pose_image.resize(size)

pose_image.save("pose.png")

# -------------------------
# 3️⃣ Crear prompt guiado por la prenda
# -------------------------
prompt = """
a photorealistic person wearing a t-shirt,
realistic fabric, natural lighting
"""

negative_prompt = """
different shirt, altered clothing, changed design, logos, text changes,
blurry, extra arms, deformed body
"""


# -------------------------
# 4️⃣ Inpainting (MAGIA)
# -------------------------
print("Generating try-on...")

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=person,
    mask_image=mask,
    ip_adapter_image=garment,
    control_image=pose_image,
    num_inference_steps=30,
    guidance_scale=8.0,
    controlnet_conditioning_scale=0.9,
    strength=0.99
).images[0]


# -------------------------
# 5️⃣ Guardar resultado
# -------------------------
result.save("tryon_result.png")
print("Saved: tryon_result.png")