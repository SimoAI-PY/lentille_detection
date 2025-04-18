import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# === CONFIGURATION ===
model_path = "graines_CNN_model.h5"
img_size = 64

# === CHARGEMENT DU MODÈLE ===
@st.cache_resource
def load_cnn_model():
    return load_model(model_path)

model = load_cnn_model()

# === ANALYSE D'UNE IMAGE ===
def analyze_image(image):
    img = np.array(image.convert("L"))
    original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)
    thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if 1000 < cv2.contourArea(cnt) < 50000]

    sain_count, bruche_count = 0, 0

    for cnt in valid_contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        margin = 5
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)

        roi = img[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (img_size, img_size))
        roi_array = np.expand_dims(roi_resized, axis=-1)
        roi_array = np.repeat(roi_array, 3, axis=-1)
        roi_array = roi_array / 255.0
        roi_array = np.expand_dims(roi_array, axis=0)

        pred = model.predict(roi_array, verbose=0)[0][0]

        if pred > 0.5:
            label = "saine"
            color = (0, 255, 0)
            sain_count += 1
        else:
            label = "bruchee"
            color = (0, 0, 255)
            bruche_count += 1

        radius = int(0.5 * max(w, h) * 0.8)
        center = (x + w//2, y + h//2)
        cv2.circle(original, center, radius, color, 2)
        cv2.putText(original, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    total = sain_count + bruche_count
    taux = (bruche_count / total) * 100 if total else 0

    cv2.putText(original, f"Saines: {sain_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(original, f"Bruchees: {bruche_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(original, f"Taux de bruchage: {taux:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return original

# === STREAMLIT UI ===
st.title("Détection de graines bruchées 🌱")
st.write("Analyse automatique d'images radiographiques de graines.")

option = st.radio("Choisissez une option :", ("Analyser une image", "Analyser un dossier d'images"))

if option == "Analyser une image":
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_column_width=True)
        if st.button("Analyser"):
            result = analyze_image(image)
            st.image(result, caption="Résultat de l'analyse", use_column_width=True)

elif option == "Analyser un dossier d'images":
    images_dir = st.text_input("Chemin vers le dossier contenant les images :")
    if images_dir and st.button("Analyser le dossier"):
        result_images = []
        for filename in os.listdir(images_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(images_dir, filename)
                image = Image.open(image_path)
                result = analyze_image(image)
                st.image(result, caption=filename, use_column_width=True)
        st.success("✅ Analyse du dossier terminée.")
