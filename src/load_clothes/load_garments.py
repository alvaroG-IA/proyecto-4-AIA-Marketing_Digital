from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import os

DATA_PATH = "../../garments"
os.makedirs("../../garments", exist_ok=True)


# -------------------------
# 1️⃣ CARGAR DATASETS
# -------------------------
print("Loading datasets...")

tshirts = load_dataset("gilek19/tshirts", split="train")
jeans = load_dataset("ndr01/jeans-captioning-dataset", split="train")
shoes = load_dataset("Seekysense/IT-Running-Shoes-Descriptions", split="train")

print("Datasets loaded ✓")


# -------------------------
# 2️⃣ FUNCIÓN DE GUARDADO LIMPIO
# -------------------------
def save_image(img, path):
    img = img.convert("RGB")
    img.save(path)


# -------------------------
# 3️⃣ EXTRAER ALGUNAS PRENDAS
# -------------------------
print("Extracting garments...")

num_samples = 10  # puedes subir esto luego

# 👕 T-shirts
for i in range(num_samples):
    item = tshirts[i]

    # algunos datasets usan 'image' o 'img'
    img = item.get("image", None)
    if img is None:
        continue

    path = f"{DATA_PATH}/tshirt_{i}.png"
    save_image(img, path)


# 👖 Jeans
for i in range(num_samples):
    item = jeans[i]

    img = item.get("image", None)
    if img is None:
        continue

    path = f"{DATA_PATH}/jeans_{i}.png"
    save_image(img, path)


# 👟 Shoes
for i in range(num_samples):
    item = shoes[i]

    img = item.get("image", None)
    if img is None:
        continue

    path = f"{DATA_PATH}/shoes_{i}.png"
    save_image(img, path)


print("Done extracting garments ✓")


# -------------------------
# 4️⃣ VISUALIZACIÓN RÁPIDA
# -------------------------
def show_samples(folder):
    imgs = os.listdir(folder)[:5]

    plt.figure(figsize=(12, 5))

    for i, img_name in enumerate(imgs):
        img = Image.open(os.path.join(folder, img_name))

        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(img_name)

    plt.show()


show_samples(DATA_PATH)