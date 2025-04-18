import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# === CONFIGURATION ===
img_size = 128
image_dir = "Images/"
mask_dir = "Masks/"

# === CHARGEMENT DES IMAGES ET MASQUES ===
def load_data(image_dir, mask_dir, img_size):
    images = []
    masks = []
    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        images.append(np.expand_dims(img, axis=-1))
        masks.append(np.expand_dims(mask, axis=-1))

    return np.array(images), np.array(masks)

print("🔄 Chargement des données...")
X, y = load_data(image_dir, mask_dir, img_size)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Données chargées : {X.shape[0]} images")

# === ARCHITECTURE U-NET SIMPLIFIÉE ===
def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = layers.UpSampling2D()(c3)
    concat1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat1)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D()(c4)
    concat2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(concat2)
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = models.Model(inputs, outputs)
    return model

# === COMPILATION & ENTRAÎNEMENT ===
model = build_unet((img_size, img_size, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n🚀 Entraînement du modèle U-Net...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8
)

# === SAUVEGARDE DU MODÈLE ===
os.makedirs("unet_model", exist_ok=True)
model.save("unet_model/unet_graines.h5")
print("\n✅ Modèle U-Net sauvegardé dans 'unet_model/unet_graines.h5'")
