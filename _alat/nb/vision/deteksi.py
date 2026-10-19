# %% [markdown]
# # Vision — Deteksi Objek (YOLOv8n)
# **Kapan dipakai:** soal meminta kotak (bounding box) + kelas objek di gambar.
#
# **Langkah:** buat `data.yaml` → ukur mAP model pretrained di data val → (opsional) fine-tune →
# prediksi → jumlah objek per gambar.
#
# **Data demo:** coco8 (8 gambar COCO, format YOLO). Folder data harus berisi `images/train`,
# `images/val`, `labels/train`, `labels/val`. Bobot: `pretrained/yolov8n/yolov8n.pt`
# (COCO 80 kelas). Untuk fine-tune: hapus `#` di sel **Fine-tune**, lalu jalankan ulang evaluasi.

# %%
import glob
import os

import pandas as pd
import torch
from ultralytics import YOLO

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/detection/coco8"   # GANTI saat lomba
WEIGHTS = "Z:/GEMASTIK-DATABASE/pretrained/yolov8n/yolov8n.pt"
IMAGE_PATTERN = "*.jpg"     # GANTI bila gambar .png

EPOCHS = 3
IMAGE_SIZE = 640
MIN_CONFIDENCE = 0.25
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = 0

# %% [markdown]
# ## Baca data

# %%
train_images = sorted(glob.glob(f"{DATA_FOLDER}/images/train/{IMAGE_PATTERN}"))
val_images = sorted(glob.glob(f"{DATA_FOLDER}/images/val/{IMAGE_PATTERN}"))
print("gambar train:", len(train_images), "| gambar val:", len(val_images))

# %% [markdown]
# ## Lihat satu file label
# Format YOLO: satu baris per objek = `kelas x_tengah y_tengah lebar tinggi` (skala 0–1).

# %%
first_name = os.path.splitext(os.path.basename(val_images[0]))[0]
label_file = f"{DATA_FOLDER}/labels/val/{first_name}.txt"
print(open(label_file).read()[:200])

# %% [markdown]
# ## Model (YOLOv8n pretrained, offline)

# %%
model = YOLO(WEIGHTS)
print("jumlah kelas:", len(model.names), "| contoh:", model.names[0], model.names[15])

# %% [markdown]
# ## File data.yaml untuk ultralytics
# Nama kelas diambil dari model. Saat lomba dengan kelas sendiri, tulis daftar kelas soal.

# %%
yaml_text = f"path: {DATA_FOLDER}\ntrain: images/train\nval: images/val\nnames:\n"
for class_number in model.names:
    yaml_text = yaml_text + f"  {class_number}: {model.names[class_number]}\n"
with open("data.yaml", "w") as file:
    file.write(yaml_text)
print(yaml_text[:200])

# %% [markdown]
# ## Evaluasi (mAP di data val)

# %%
metrics = model.val(data="data.yaml", imgsz=IMAGE_SIZE, batch=8, workers=0, device=DEVICE,
                    project="runs", name="val", exist_ok=True, plots=False, verbose=False)
print("mAP50:", round(metrics.box.map50, 4), "| mAP50-95:", round(metrics.box.map, 4))

# %% [markdown]
# ## Fine-tune (opsional, hapus `#` untuk memakai)

# %%
# model.train(data="data.yaml", epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=8, workers=0,
#             device=DEVICE, project="runs", name="train", exist_ok=True, plots=False)

# %% [markdown]
# ## Prediksi + hasil (jumlah objek terdeteksi per gambar)

# %%
results = model.predict(val_images, imgsz=IMAGE_SIZE, conf=MIN_CONFIDENCE, verbose=False)

file_names = []
box_counts = []
for image_path, image_result in zip(val_images, results):
    file_names.append(os.path.basename(image_path))
    box_counts.append(len(image_result.boxes))

result = pd.DataFrame()
result["file"] = file_names
result["jumlah_deteksi"] = box_counts
result.to_csv("hasil.csv", index=False)
result.head()
