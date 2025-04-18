import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
model_path = "unet_model/unet_graines.h5"
image_dir = "images_a_predire"
output_dir = "resultats_unet"
img_size = 128
os.makedirs(output_dir, exist_ok=True)

# === CHARGEMENT DU MODÈLE ===
print("📦 Chargement du modèle U-Net...")
model = load_model(model_path)

# === TRAITEMENT DE CHAQUE IMAGE ===
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # Chargement & prétraitement
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(original, (img_size, img_size))
    input_img = resized.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=(0, -1))

    # Prédiction masque
    pred_mask = model.predict(input_img)[0, :, :, 0]
    mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_bin = cv2.resize(mask_bin, original.shape[::-1])

    # Trouver les contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius*0.8)
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            count += 1

    #count = 0
    #for cnt in contours:
    #    area = cv2.contourArea(cnt)
    #    if area > 100:  # Ignore petits bruits
    #        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
    #        count += 1

    print(f"✅ {img_name} → {count} graines détectées")

    # Sauvegarde de l'image annotée
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, output)

print("\n🎯 Prédictions terminées. Résultats dans :", output_dir)
