import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load Model
@st.cache_resource
def load_lsd_model():
    model = load_model("skin/model_lsd_mobilenetv2.h5")
    return model

model = load_lsd_model()
class_list = ['LumpySkin', 'NormalSkin']

# Judul Halaman
st.set_page_config(page_title="LSD Detector", layout="centered")
st.title("🩺 LSD Detector")
st.subheader("Pendeteksi Lumpy Skin Disease pada Sapi dan Kerbau")

# Upload Gambar
uploaded_file = st.file_uploader("📷 Upload gambar kulit ternak", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Tampilkan gambar
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)

        # Pra-pemrosesan gambar
        img_resized = img.resize((256, 256))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_label = class_list[predicted_class_idx]
        confidence = np.max(predictions) * 100

        # Hasil
        if confidence < 60:
            st.warning("❌ Gambar tidak dikenali sebagai kulit sapi.\nGunakan gambar yang lebih jelas.")
        else:
            emoji = "✅" if predicted_label == "NormalSkin" else "⚠️"
            st.success(f"{emoji} **{predicted_label}** ({confidence:.2f}%)")

    except Exception as e:
        st.error(f"Gagal memproses gambar. Error: {e}")

# Info Pencegahan LSD
with st.expander("ℹ️ Info Penanggulangan LSD"):
    st.markdown("""
    **🐄 Pencegahan Lumpy Skin Disease (LSD):**
    - Vaksinasi ternak secara rutin.
    - Kontrol serangga (nyamuk, lalat) di sekitar kandang.
    - Karantina sapi baru sebelum dicampur dengan sapi lain.
    - Bersihkan dan desinfeksi kandang secara rutin.
    - Hindari lalu lintas ternak antar wilayah tanpa pemeriksaan.

    **💊 Jika Terinfeksi:**
    - Pisahkan sapi sakit dari yang sehat.
    - Hubungi dokter hewan.
    - Berikan perawatan suportif & vitamin.

    **📍 Wabah:**
    - Lakukan karantina wilayah.
    - Vaksinasi ring di sekitar kasus.

    **💡 Tips Deteksi:**
    Gunakan gambar kulit sapi yang jelas & terang untuk akurasi maksimal.
    """)

