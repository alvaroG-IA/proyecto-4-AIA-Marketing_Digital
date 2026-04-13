import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class HumanParser:

    def __init__(self, device="cuda"):
        self.device = device

        model_name = "mattmdjaga/segformer_b2_clothes"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(device)

        self.model.eval()

    # --------------------------------------------------
    # 1. PARSE IMAGEN (ropa real)
    # --------------------------------------------------
    def parse(self, image_pil):

        w, h = image_pil.size

        inputs = self.processor(
            images=image_pil,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        parsing = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        return parsing

    # --------------------------------------------------
    # 2. MÁSCARAS (ROPA REAL)
    # --------------------------------------------------
    def get_clothing_masks(self, parsing):

        masks = {}

        # 🧥 torso ropa
        masks["torso_clothes"] = np.isin(parsing, [4, 7]).astype(np.uint8) * 255

        # 👖 pantalón / falda
        masks["bottom_clothes"] = np.isin(parsing, [5, 6]).astype(np.uint8) * 255

        return masks

    # --------------------------------------------------
    # 3. MÁSCARAS CORPORALES (SMPL-X -> ESTO ES LO IMPORTANTE)
    # --------------------------------------------------
    def get_body_masks_from_smpl(self, vertex_labels):

        """
        vertex_labels viene de SMPL-X (NO de SegFormer)
        """

        masks = {}

        masks["torso"] = (vertex_labels == "torso").astype(np.uint8) * 255

        masks["arms"] = np.isin(vertex_labels, ["left_arm", "right_arm"]).astype(np.uint8) * 255

        masks["legs"] = np.isin(vertex_labels, ["left_leg", "right_leg"]).astype(np.uint8) * 255

        masks["feet"] = np.isin(vertex_labels, ["left_foot", "right_foot"]).astype(np.uint8) * 255

        return masks

    # --------------------------------------------------
    # 4. REFINE (para diffusion)
    # --------------------------------------------------
    def refine_mask(self, mask):

        kernel = np.ones((15, 15), np.uint8)

        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = Image.fromarray(mask).filter(
            ImageFilter.GaussianBlur(radius=6)
        )

        return np.array(mask)