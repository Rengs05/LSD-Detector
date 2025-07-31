import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# LOAD MODEL
model_path = r'C:/Tugas/Penulisan Ilmiah/LSD/skin/model_lsd_mobilenetv2.h5'
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully.")

# CLASS LABELS SESUAI URUTAN TRAINING
class_list = ['NormalSkin', 'LumpySkin']

# Mendapatkan image PATH
file_path = input("Drag & drop gambar LSD untuk prediksi, lalu Enter: ").strip('"')

if not os.path.isfile(file_path):
    print("❌ File tidak ditemukan. Program berhenti.")
    exit()

# LOAD AND PREPROCESS GAMBAR
img = image.load_img(file_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediksi
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions, axis=1)[0]
predicted_label = class_list[predicted_class_idx]
confidence = np.max(predictions) * 100

# Hasil Display
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
plt.show()

print(f"✅ Gambar '{os.path.basename(file_path)}' terdeteksi sebagai: {predicted_label} ({confidence:.2f}%)")
