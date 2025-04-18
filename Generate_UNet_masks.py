import os
import cv2
import numpy as np

# === CONFIGURATION ===
source_dir = "Images/"           # Dossier contenant les images X-ray
mask_dir = "Masks/"              # Dossier où enregistrer les masques

os.makedirs(mask_dir, exist_ok=True)

# === PARAMÈTRES DE DÉTECTION ===
clipLimit = 0.6
tileGridSize = (5, 5)
blockSize = 25
C = 10
kernel_size = (3, 3)

image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png'))]

total = 0

for fname in image_files:
    path = os.path.join(source_dir, fname)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Prétraitement
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_eq = clahe.apply(img_gray)

    # Seuillage
    thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize, C)

    # Morphologie pour remplir les trous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sauvegarde du masque
    mask_path = os.path.join(mask_dir, fname)
    cv2.imwrite(mask_path, morph)
    total += 1

    print(f"✅ Masque généré : {fname}")

print(f"\n🟢 {total} masques générés dans '{mask_dir}'")
