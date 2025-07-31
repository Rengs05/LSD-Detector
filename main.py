import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# ==== Menentukan Path ====
base_dir = r'C:\Tugas\Penulisan Ilmiah\LSD\skin'
bahan_dir = os.path.join(base_dir, 'bahan')
latih_dir = os.path.join(base_dir, 'latih')
validasi_dir = os.path.join(base_dir, 'validasi')
test_dir = os.path.join(base_dir, 'test')

# ==== Membagi DATASET ====
def split_dataset(class_name):
    source_dir = os.path.join(bahan_dir, class_name)
    train_dir = os.path.join(latih_dir, class_name)
    valid_dir = os.path.join(validasi_dir, class_name)
    test_dir_ = os.path.join(test_dir, class_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir_, exist_ok=True)

    images = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_valid, test = train_test_split(images, test_size=0.2, random_state=42)
    train, valid = train_test_split(train_valid, test_size=0.2, random_state=42)

    def copy_files(files, target_dir):
        for f in files:
            shutil.copy(f, os.path.join(target_dir, os.path.basename(f)))

    copy_files(train, train_dir)
    copy_files(valid, valid_dir)
    copy_files(test, test_dir_)

    print(f"{class_name} - Train: {len(os.listdir(train_dir))}, Valid: {len(os.listdir(valid_dir))}, Test: {len(os.listdir(test_dir_))}")

split_dataset('NormalSkin')
split_dataset('LumpySkin')

# ==== DATA GENERATOR ====
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

tain_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = tain_datagen.flow_from_directory(
    latih_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
valid_generator = valid_test_datagen.flow_from_directory(
    validasi_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
test_generator = valid_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

print("Detected class indices:", train_generator.class_indices)

# ==== MODEL MOBILENETV2 ====
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==== Melatih Data ====
EPOCHS = 200  #Banyak Data yang dilatih
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)

# OPTIONAL FINE-TUNING
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20
)

# Menyimpan MODEL
model_save_path = os.path.join(base_dir, 'model_lsd_mobilenetv2.h5')
model.save(model_save_path)
print(f"✅ Model saved successfully at {model_save_path}")

# ==== Visualisasi ====
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy per Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss per Epoch')

plt.tight_layout()
plt.show()

# ==== Evaluasi & Confusion Matrix ====
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.2f}")

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
