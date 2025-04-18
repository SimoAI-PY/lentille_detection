import cv2
import numpy as np
import os
from tqdm import tqdm

# === CONFIGURATION ===
images_dir = "images_a_predire"        # Dossier avec les images originales à analyser
masks_dir = "masks_predits"            # Dossier où seront enregistrés les masques optimisés
os.makedirs(masks_dir, exist_ok=True)

objectif_graines = 100                 # Objectif de graines détectées par image
tolerance = 10                         # Marge d'erreur acceptée autour de 100 (entre 90 et 110)

# === PARAMÈTRES À TESTER AUTOMATIQUEMENT ===
clipLimits = [0.5, 1.0, 2.0]
tileGridSizes = [(8, 8), (9, 9)]
blockSizes = [19, 21, 25]
constants = [5, 7, 10]
kernel_sizes = [(3, 3), (5, 5)]
min_area = 300
min_circularity = 0.2

# === FONCTION D'OPTIMISATION D'UNE IMAGE ===
def generate_best_mask(image_gray):
    best_mask = None
    best_score = float('inf')

    for clipLimit in clipLimits:
        for tileGridSize in tileGridSizes:
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            img_eq = clahe.apply(image_gray)

            for blockSize in blockSizes:
                for C in constants:
                    thresh = cv2.adaptiveThreshold(img_eq, 255,
                                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV,
                                                   blockSize, C)
                    for kernel_size in kernel_sizes:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
                        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

                        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        valid_contours = []
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area < min_area:
                                continue
                            perimeter = cv2.arcLength(cnt, True)
                            if perimeter == 0:
                                continue
                            circularity = 4 * np.pi * (area / (perimeter ** 2))
                            if circularity > min_circularity:
                                valid_contours.append(cnt)

                        count = len(valid_contours)
                        score = abs(objectif_graines - count)

                        if score < best_score:
                            best_score = score
                            best_mask = np.zeros_like(image_gray)
                            cv2.drawContours(best_mask, valid_contours, -1, 255, -1)

                        if best_score <= tolerance:
                            return best_mask  # early stop si très bon résultat

    return best_mask

# === TRAITEMENT DE CHAQUE IMAGE ===
print("🔍 Génération des masques optimisés...")
for filename in tqdm(os.listdir(images_dir), desc="Traitement des images"):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    path = os.path.join(images_dir, filename)
    image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"❌ Image illisible : {filename}")
        continue

    best_mask = generate_best_mask(image_gray)
    if best_mask is not None:
        save_path = os.path.join(masks_dir, filename)
        cv2.imwrite(save_path, best_mask)
        print(f"✅ Masque généré : {filename}")
    else:
        print(f"⚠️ Aucun masque trouvé pour : {filename}")

print("\n✅ Tous les masques ont été générés dans :", masks_dir)
