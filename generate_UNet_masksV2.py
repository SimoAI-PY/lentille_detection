import os
import cv2
import numpy as np

# === CONFIGURATION ===
source_dir = "Images_a_predire/"           # Dossier contenant les images X-ray
mask_dir = "masks_predits/"              # Dossier où enregistrer les masques

os.makedirs(mask_dir, exist_ok=True)

# === PARAMÈTRES OPTIMISÉS ===
clipLimit = 0.5
tileGridSize = (7, 7)
blockSize = 25
C = 10
kernel_size = (5, 5)

image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png'))]
total = 0

for fname in image_files:
    path = os.path.join(source_dir, fname)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Prétraitement
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_eq = clahe.apply(img_gray)

    # Seuillage inversé
    thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize, C)

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Filtrage des contours pour générer le masque final
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img_gray)

    valid_contours = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if area > 1200 and circularity > 0.1:
            cv2.drawContours(mask, [cnt], -1, 255, -1)  # Remplit la graine en blanc
            valid_contours += 1

    # Sauvegarde du masque
    mask_path = os.path.join(mask_dir, fname)
    cv2.imwrite(mask_path, mask)
    total += 1

    print(f"✅ Masque généré : {fname} - {valid_contours} graines détectées")

print(f"\n🟢 {total} masques générés dans '{mask_dir}' avec filtrage optimisé")
