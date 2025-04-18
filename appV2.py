import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from io import BytesIO

# === Chargement du modèle ===
MODEL_PATH = "graines_CNN_model.h5"
model = load_model(MODEL_PATH)

# === Fonctions ===

def segment_grains(img):
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(9, 9))
    img_eq = clahe.apply(img)

    thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 7)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if area > 100 and circularity > 0.2:
            valid_contours.append(cnt)
    return valid_contours

def classify_and_draw(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours = segment_grains(img)

    sain, bruche = 0, 0
    data = []

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), int(radius * 0.8)

        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(img.shape[1], x + radius), min(img.shape[0], y + radius)

        roi = img[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (64, 64))
        roi_array = roi_resized.astype(np.float32) / 255.0

        if len(roi_array.shape) == 2:
            roi_array = np.expand_dims(roi_array, axis=-1)
            roi_array = np.repeat(roi_array, 3, axis=-1)

        roi_array = np.expand_dims(roi_array, axis=0)

        pred = model.predict(roi_array, verbose=0)[0][0]

        if pred > 0.5:
            label = "saine"
            color = (0, 255, 0)
            sain += 1
        else:
            label = "bruchee"
            color = (0, 0, 255)
            bruche += 1

        cv2.circle(original, (x, y), radius, color, 2)
        cv2.putText(original, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        data.append((label, x, y, radius))

    total = sain + bruche
    taux = (bruche / total) * 100 if total > 0 else 0
    return original, sain, bruche, taux, data

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# === Interface Streamlit ===
st.set_page_config(layout="wide")
st.title("Détection de graines saines et bruchées - CNN")

uploaded_files = st.file_uploader("Choisissez des images à analyser", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(bytes_data)

        output_img, sain, bruche, taux, data = classify_and_draw(temp_path)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(output_img, channels="BGR", caption=f"Résultat - {uploaded_file.name}", use_column_width=True)

        with col2:
            st.markdown(f"**Graines saines :** {sain}")
            st.markdown(f"**Graines bruchées :** {bruche}")
            st.markdown(f"**Taux de bruchage :** {taux:.2f}%")

        # Export CSV
        df = pd.DataFrame(data, columns=["Classe", "X", "Y", "Rayon"])
        csv = convert_df_to_csv(df)
        st.download_button(
            label="📂 Télécharger les données CSV",
            data=csv,
            file_name=f"donnees_{uploaded_file.name}.csv",
            mime='text/csv'
        )
