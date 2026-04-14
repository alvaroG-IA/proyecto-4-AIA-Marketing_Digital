# tools/diffusion_tool.py
import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DDIMScheduler
from controlnet_aux import CannyDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    print("🎨 Generating final image...")

    image = Image.open(state["image_path"]).convert("RGB").resize((512, 512))
    mask = Image.open(state["mask_path"]).convert("L").resize((512, 512))

    image_np = np.array(image)
    mask_np = np.array(mask)

    control = canny(image_np, 100, 200)
    control[mask_np > 127] = 0

    control = Image.fromarray(control)

    negative = "low quality, blurry, watermark, text, bad anatomy"

    result = pipe(
        prompt=state["optimized_prompt"],
        negative_prompt=negative,
        image=image,
        mask_image=mask,
        control_image=control,
        num_inference_steps=40,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.9
    ).images[0]

    path = "result.png"
    result.save(path)

    return {"result_path": path}