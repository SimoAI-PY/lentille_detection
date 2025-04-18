import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
img_dir = "images_a_predire"
mask_dir = "masks_predits"
model_path = "graines_CNN_model.h5"
output_dir = "resultats_detection_cnn"
img_size = 64

os.makedirs(output_dir, exist_ok=True)

# Charger le modèle CNN
model = load_model(model_path)

# === TRAITEMENT DES IMAGES ===
for filename in os.listdir(img_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bruch_count = 0
    saine_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        margin = 5
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)

        roi = image[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (img_size, img_size))
        roi_input = roi_resized.astype(np.float32) / 255.0
        roi_input = np.expand_dims(roi_input, axis=-1)
        roi_input = np.repeat(roi_input, 3, axis=-1)
        roi_input = np.expand_dims(roi_input, axis=0)

        pred = model.predict(roi_input)[0][0]
        if pred > 0.5:
            label = "saine"
            color = (0, 255, 0)
            saine_count += 1
        else:
            label = "bruchée"
            color = (0, 0, 255)
            bruch_count += 1

        # Dessin d’un cercle (réduit)
        radius = int(0.4 * max(w, h))
        center = (x + w//2, y + h//2)
        cv2.circle(output, center, radius, color, 2)
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    total = bruch_count + saine_count
    taux = (bruch_count / total) * 100 if total > 0 else 0

    # Résumé sur l'image
    cv2.putText(output, f"Saines: {saine_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(output, f"Bruchées: {bruch_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(output, f"Taux Bruchage: {taux:.1f}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imwrite(os.path.join(output_dir, filename), output)
    print(f"✅ Image traitée : {filename} | Saines: {saine_count} | Bruchées: {bruch_count} | Taux: {taux:.1f}%")
