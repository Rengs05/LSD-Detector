import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox, font
from PIL import Image, ImageTk, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import sys

# Untuk memastikan sys.stdout dan sys.stderr valid (supaya tidak error saat menggunakan PyInstaller)
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Fungsi untuk Resource Path (PyInstaller compatibility)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load Model
model_path = resource_path(os.path.join('skin', 'model_lsd_mobilenetv2.h5'))
model = load_model(model_path)
class_list = ['LumpySkin', 'NormalSkin']

# Info Penanggulangan LSD
def show_prevention_info():
    prevention_text = (
        "🐄 Penanggulangan Lumpy Skin Disease (LSD)\n\n"
        "✅ Pencegahan:\n"
        "- Vaksinasi ternak secara rutin.\n"
        "- Kontrol serangga (nyamuk, lalat) di sekitar kandang.\n"
        "- Karantina sapi baru sebelum dicampur dengan sapi lain.\n"
        "- Bersihkan dan desinfeksi kandang serta peralatan secara rutin.\n"
        "- Hindari lalu lintas ternak antar wilayah tanpa pemeriksaan.\n\n"
        "✅ Jika Terinfeksi:\n"
        "- Pisahkan sapi sakit dari sapi sehat.\n"
        "- Hubungi dokter hewan atau dinas peternakan setempat.\n"
        "- Berikan pakan cukup, vitamin, dan perawatan suportif.\n\n"
        "✅ Jika Terjadi Wabah:\n"
        "- Vaksinasi sapi di sekitar kasus (ring vaccination).\n"
        "- Karantina wilayah dan kontrol serangga secara intensif.\n"
        "- Pantau kesehatan ternak secara berkala.\n\n"
        "💡 Tips: Gunakan gambar kulit sapi yang jelas dan pencahayaan baik untuk akurasi deteksi aplikasi."
    )
    messagebox.showinfo("ℹ️ Info Penanggulangan LSD", prevention_text)

# Fungsi Prediksi
def predict_image():
    file_path = filedialog.askopenfilename(
        title='Pilih gambar LSD untuk prediksi',
        filetypes=[('Image Files', '*.png *.jpg *.jpeg')]
    )

    if not file_path:
        return  # User membatalkan pilihan file

    print("File gambar dipilih:", file_path)

    try:
        # Validasi gambar
        img = Image.open(file_path)
        img.verify()  # Pastikan ini file gambar valid

        # Buka ulang untuk diproses
        img = Image.open(file_path).convert('RGB')
        img = img.resize((256, 256))
        print("Ukuran gambar setelah resize:", img.size)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        print("Model input shape:", model.input_shape)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_label = class_list[predicted_class_idx]
        confidence = np.max(predictions) * 100

        # Menampilkan hasil
        if confidence < 60:
            emoji = "❌"
            display_text = f"{emoji} Gambar tidak dikenali sebagai kulit sapi.\nGunakan gambar yang lebih jelas."
            color = "#8B0000"
            root.bell()
        else:
            emoji = "✅" if predicted_label == "NormalSkin" else "⚠️"
            color = "#28a745" if predicted_label == "NormalSkin" else "#dc3545"
            display_text = f"{emoji} {predicted_label} ({confidence:.2f}%)"

        result_label.config(text=display_text, fg=color)

        # Tampilkan gambar di GUI
        img_display = Image.open(file_path).convert('RGB')
        img_display = img_display.resize((display_img_size, display_img_size))
        img_display = ImageOps.expand(img_display, border=4, fill='#333')

        img_tk = ImageTk.PhotoImage(img_display)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Simpan referensi agar tidak dihapus GC

    except Exception as e:
        messagebox.showerror("Error", f"Gagal memproses gambar atau melakukan prediksi.\n{e}")

# Hover Style Buttons
def on_enter(e):
    predict_button['background'] = '#0056b3'
def on_leave(e):
    predict_button['background'] = '#007bff'

def on_info_enter(e):
    info_button['background'] = '#117a8b'
def on_info_leave(e):
    info_button['background'] = '#17a2b8'

# GUI Layout
root = tk.Tk()
root.title("🩺 Aplikasi Pendeteksi Lumpy Skin Disease")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = min(480, screen_width - 50)
window_height = min(700, screen_height - 100)
root.geometry(f"{window_width}x{window_height}")
root.configure(bg="#f0f2f5")
root.resizable(True, True)

# Font
base_font = font.nametofont("TkDefaultFont")
base_font.configure(size=int(window_height * 0.02))
title_font = ("Helvetica", int(window_height * 0.035), "bold")
subtitle_font = ("Helvetica", int(window_height * 0.02))
button_font = ("Helvetica", int(window_height * 0.025), "bold")
result_font = ("Helvetica", int(window_height * 0.025), "bold")
footer_font = ("Helvetica", int(window_height * 0.015))

# Gambar
display_img_size = int(window_width * 0.6)

# Judul dan Subtitle
Label(root, text="🩺 LSD Detector", font=title_font, bg="#f0f2f5", fg="#333").pack(pady=5)
Label(root, text="Pendeteksi Lumpy Skin Disease pada Sapi dan Kerbau", font=subtitle_font, bg="#f0f2f5", fg="#555").pack()

# Frame Gambar
image_frame = Frame(root, bg="#ffffff", width=display_img_size, height=display_img_size, highlightbackground="#888", highlightthickness=2)
image_frame.pack(pady=10)
image_label = Label(image_frame, bg="#ffffff")
image_label.pack()

# Label Hasil
result_label = Label(root, text="Belum ada prediksi\nSilakan upload gambar.", font=result_font, bg="#f0f2f5", fg="#555", justify="center")
result_label.pack(pady=10)

# Button Prediksi
predict_button = Button(
    root,
    text="📷 Pilih Gambar & Prediksi",
    command=predict_image,
    bg="#007bff",
    fg="white",
    activebackground="#0056b3",
    font=button_font,
    relief="flat",
    padx=10,
    pady=6,
    cursor="hand2"
)
predict_button.pack(pady=10)
predict_button.bind("<Enter>", on_enter)
predict_button.bind("<Leave>", on_leave)

# Button Info
info_button = Button(
    root,
    text="ℹ️ Info Penanggulangan LSD",
    command=show_prevention_info,
    bg="#17a2b8",
    fg="white",
    activebackground="#117a8b",
    font=button_font,
    relief="flat",
    padx=10,
    pady=6,
    cursor="hand2"
)
info_button.pack(pady=5)
info_button.bind("<Enter>", on_info_enter)
info_button.bind("<Leave>", on_info_leave)

# Tips
Label(root, text="💡 Tips:\nGunakan gambar area kulit sapi yang jelas\ndan pencahayaan baik untuk akurasi maksimal.", font=subtitle_font, bg="#f0f2f5", fg="#666", justify="center").pack(pady=5)

root.mainloop()
