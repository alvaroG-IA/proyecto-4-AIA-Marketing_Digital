# tools/sam_tool.py
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import urllib.request
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
if not os.path.exists(SAM_CHECKPOINT):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        SAM_CHECKPOINT
    )

sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
predictor = SamPredictor(sam)


def sam_tool(state):
    print("✂️ Segmenting product...")

    image = cv2.imread(state["image_path"])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    h, w = image_rgb.shape[:2]
    point = np.array([[w//2, h//2]])
    label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True
    )

    best = masks[np.argmax(scores)]
    mask = 1 - best
    mask = (mask * 255).astype(np.uint8)
    path = "mask.png"
    cv2.imwrite(path, mask)

    return {"mask_path": path}