import cv2
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector

# =====================================================
# CONFIG
# =====================================================

SIZE = (512, 768)

print("Loading OpenPose detector...")
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# =====================================================
# 1️⃣ REMOVE WHITE BACKGROUND
# =====================================================

def remove_white_background(img):
    img = np.array(img).astype(np.uint8)

    mask = np.all(img > 240, axis=-1)  # blancos casi puros
    img[mask] = [0, 0, 0]

    return img

# =====================================================
# 2️⃣ TORSO APPROXIMATION (SOLO REFERENCIA)
# =====================================================

def get_torso_bbox(person_img):
    h, w = person_img.shape[:2]

    # bbox aproximado del torso (NO máscara de blending)
    x1 = int(w * 0.33)
    x2 = int(w * 0.67)
    y1 = int(h * 0.28)
    y2 = int(h * 0.55)

    return x1, y1, x2, y2

# =====================================================
# 3️⃣ SIMPLE OVERLAY (CORRECTO)
# =====================================================

def overlay_garment(person_img, garment_img):

    person = np.array(person_img).astype(np.uint8)
    garment = np.array(garment_img).astype(np.uint8)

    h, w = person.shape[:2]

    x1, y1, x2, y2 = get_torso_bbox(person)

    target_w = x2 - x1
    target_h = y2 - y1

    # resize prenda al torso
    garment_resized = cv2.resize(garment, (target_w, target_h))

    # overlay
    result = person.copy()

    result[y1:y2, x1:x2] = blend_images(
        result[y1:y2, x1:x2],
        garment_resized
    )

    return Image.fromarray(result)

# =====================================================
# 4️⃣ BLENDING SUAVE (SIN MASCARES GLOBALES)
# =====================================================

def blend_images(bg, fg):

    bg = bg.astype(np.float32)
    fg = fg.astype(np.float32)

    # máscara automática: quitar negros (fondo eliminado)
    mask = np.any(fg > 10, axis=-1).astype(np.float32)
    mask = np.stack([mask]*3, axis=-1)

    result = bg * (1 - mask) + fg * mask

    return result.astype(np.uint8)

# =====================================================
# 5️⃣ PIPELINE
# =====================================================

print("Loading images...")

person  = Image.open("../RyanGosling.jpg").convert("RGB").resize(SIZE)
garment = Image.open("../../garments/tshirt_1.png").convert("RGB").resize(SIZE)

# =====================================================
# 6️⃣ CLEAN GARMENT
# =====================================================

print("Removing background from garment...")
garment_clean = remove_white_background(garment)
garment_clean = Image.fromarray(garment_clean)

# =====================================================
# 7️⃣ OPTIONAL POSE (SOLO PARA DEBUG)
# =====================================================

print("Detecting pose...")
pose_image = openpose(person).resize(SIZE)
pose_image.save("pose.png")

# =====================================================
# 8️⃣ OVERLAY FINAL
# =====================================================

print("Creating try-on (overlay)...")

init_tryon = overlay_garment(person, garment_clean)
init_tryon.save("init_tryon.png")

print("DONE ✅")
print("Generated files:")
print(" - pose.png")
print(" - init_tryon.png")