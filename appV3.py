import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
MODEL_PATH = "graines_CNN_model.h5"  # Modèle entraîné
IMG_SIZE = 64

# Charger le modèle
model = load_model(MODEL_PATH)

# Détection automatique
def detect_graines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Traitement de l'image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)
    thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if 2000 < cv2.contourArea(cnt) < 50000]

    sain_count, bruche_count = 0, 0

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        margin = 5
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(img.shape[1], x+w+margin), min(img.shape[0], y+h+margin)

        roi = img[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi_array = np.expand_dims(roi_resized, axis=-1) / 255.0
        roi_array = np.expand_dims(roi_array, axis=0)
        roi_array = np.repeat(roi_array, 3, axis=-1)

        pred = model.predict(roi_array, verbose=0)[0][0]

        if pred > 0.5:
            label = "saine"
            color = (0,255,0)
            sain_count += 1
        else:
            label = "bruchee"
            color = (0,0,255)
            bruche_count += 1

        center = (int(x + w/2), int(y + h/2))
        radius = int(max(w,h)/2 * 0.8)
        cv2.circle(original, center, radius, color, 2)
        cv2.putText(original, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    total = sain_count + bruche_count
    taux_bruchees = (bruche_count / total) * 100 if total > 0 else 0

    return original, sain_count, bruche_count, taux_bruchees

# Interface Streamlit
def main():
    st.title("🌾 Détection de Graines Saines & Bruchées - U-Net + CNN")

    uploaded_files = st.file_uploader("🚀 Sélectionne plusieurs images à analyser", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        results = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_image.jpg")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            output_img, sain, bruche, taux = detect_graines(file_path)

            st.image(output_img, caption=f"Résultat : {uploaded_file.name}", use_column_width=True)
            st.write(f"🌱 Graines Saines détectées : {sain}")
            st.write(f"🐛 Graines Bruchées détectées : {bruche}")
            st.write(f"📊 Taux de Bruchage détecté : {taux:.2f}%")

            # --- Ajustement manuel ---
            sain_corrige = st.number_input(f"🛠️ Correction graines saines ({uploaded_file.name}) :", min_value=0, value=sain)
            bruche_corrige = st.number_input(f"🛠️ Correction graines bruchées ({uploaded_file.name}) :", min_value=0, value=bruche)

            total_corrige = sain_corrige + bruche_corrige
            taux_corrige = (bruche_corrige / total_corrige) * 100 if total_corrige > 0 else 0

            st.success(f"✅ Nouveau Taux de Bruchage : {taux_corrige:.2f}%")

            # Enregistrement
            results.append({
                "Image": uploaded_file.name,
                "Saines": sain_corrige,
                "Bruchées": bruche_corrige,
                "Taux (%)": taux_corrige
            })

        # Téléchargement CSV
        if results:
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Télécharger le CSV corrigé",
                data=csv,
                file_name="corrections_graines.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
