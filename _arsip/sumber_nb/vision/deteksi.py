# %% [markdown]
# # Vision — Deteksi Objek dengan Kelas Soal (fine-tune YOLOv8n)
# **Kapan dipakai:** soal memberi gambar + kotak berlabel (bounding box), dan meminta kotak di
# gambar test. Contoh: cari strip tanggal di label obat, plat nomor, cacat produk.
#
# **Langkah:** baca anotasi COCO → susun folder YOLO (`images/`, `labels/`, `data.yaml`) dengan
# split train/val → fine-tune YOLOv8n → mAP di val → prediksi test → lihat kasus paling ragu →
# simpan kotak + potongan gambar untuk tahap berikutnya (misalnya OCR).
#
# **Input yang diharapkan:** satu folder gambar train + satu file COCO JSON. Soal memberi YOLO
# `.txt` atau VOC `.xml`? Ubah dulu jadi tabel dengan `vision/konversi_anotasi`, atau langsung
# pakai folder YOLO yang sudah ada (lihat sel "Susun folder YOLO").
#
# **Data demo:** `vignette` (label harga obat). Train = 750 foto rata dari ekspor Roboflow,
# kelas `entete` (strip LOT/FAB/EXP) dan `vin` (seluruh stiker). Test = 674 foto **tanpa kotak**,
# banyak yang miring/terbalik/dipegang tangan. Jadi mAP val (dari train) terlalu optimistis untuk
# test; selalu lihat grid prediksi test.

# %%
import glob
import json
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# %% [markdown]
# ## Konfigurasi
# `CLASS_NAMES` = kelas soal, urutannya menjadi nomor kelas YOLO. Kelas COCO dicocokkan lewat nama.

# %%
DATA_FOLDER = "../../data_latihan/vision/vignette"   # GANTI saat lomba
TRAIN_FOLDER = f"{DATA_FOLDER}/train"
COCO_FILE = f"{TRAIN_FOLDER}/_annotations.coco.json"
TEST_FOLDER = f"{DATA_FOLDER}/test"
IMAGE_EXTENSION = ".jpg"
CLASS_NAMES = ["entete", "vin"]       # GANTI: nama kelas persis seperti di anotasi
TARGET_CLASS = "entete"               # kelas yang dipotong untuk tahap berikutnya
WEIGHTS = "../../pretrained/yolov8n/yolov8n.pt"

# semua hasil ke luar folder repo (JANGAN diarahkan ke folder data soal)
OUTPUT_FOLDER = "../../outputs/vision/deteksi"
DATASET_FOLDER = f"{OUTPUT_FOLDER}/dataset_yolo"     # dibuat ulang setiap Run All
CROP_FOLDER = f"{OUTPUT_FOLDER}/potongan"
RUNS_FOLDER = os.path.abspath(f"{OUTPUT_FOLDER}/runs")   # ultralytics butuh path lengkap
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VAL_FRACTION = 0.2
EPOCHS = 30
IMAGE_SIZE = 416       # samakan dengan ukuran gambar soal (lebih besar = lebih lambat)
BATCH_SIZE = 16        # kehabisan memori GPU? turunkan ke 8 atau 4
MIN_CONFIDENCE = 0.1
MARGIN = 0.05          # tepi tambahan di sekitar potongan (5% lebar/tinggi kotak)
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = 0

# %% [markdown]
# ## Baca anotasi COCO → tabel piksel
# `bbox` COCO = `[kiri, atas, lebar, tinggi]` dalam piksel.

# %%
with open(COCO_FILE) as file:
    coco = json.load(file)
print("categories:", coco["categories"])

category_names = {}
for category in coco["categories"]:
    category_names[category["id"]] = category["name"]
image_info = {}
for item in coco["images"]:
    image_info[item["id"]] = item

rows = []
for annotation in coco["annotations"]:
    info = image_info[annotation["image_id"]]
    left, top, box_width, box_height = annotation["bbox"]
    rows.append([info["file_name"], info["width"], info["height"],
                 category_names[annotation["category_id"]], left, top, box_width, box_height])
boxes = pd.DataFrame(rows, columns=["image", "width", "height", "class_name",
                                    "left", "top", "box_width", "box_height"])
print(boxes.shape)
print(boxes["class_name"].value_counts())

# %% [markdown]
# ## EDA singkat: ukuran dan bentuk kotak per kelas
# Rasio lebar/tinggi < 1 = kotak tegak. Di demo, semua `entete` di train tegak, padahal di test
# ada yang mendatar. Detektor tidak akan bisa menemukan bentuk yang tidak pernah dilihatnya.

# %%
boxes["ratio_w_h"] = boxes["box_width"] / boxes["box_height"]
image_area = boxes["width"] * boxes["height"]
boxes["area_fraction"] = boxes["box_width"] * boxes["box_height"] / image_area
summary = boxes.groupby("class_name")[["ratio_w_h", "area_fraction"]].describe()
print(summary.round(2).T)
boxes_per_image = boxes.groupby("image").size()
print("kotak per gambar:", boxes_per_image.value_counts().to_dict())

# %% [markdown]
# ## Susun folder YOLO: `images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`
# Split per **gambar** (bukan per kotak), supaya kotak dari satu gambar tidak ada di train dan val.
# Soal sudah memberi folder YOLO lengkap? Lewati sel ini, isi `DATASET_FOLDER` dengan folder itu,
# dan pastikan `data.yaml` berisi `path` lengkap, `train`, `val`, dan `names`.

# %%
shutil.rmtree(DATASET_FOLDER, ignore_errors=True)       # mulai dari folder kosong
image_names = []
for item in coco["images"]:
    image_names.append(item["file_name"])
train_names, val_names = train_test_split(image_names, test_size=VAL_FRACTION, random_state=SEED)

for part, names in [("train", train_names), ("val", val_names)]:
    os.makedirs(f"{DATASET_FOLDER}/images/{part}")
    os.makedirs(f"{DATASET_FOLDER}/labels/{part}")
    for name in names:
        shutil.copy(f"{TRAIN_FOLDER}/{name}", f"{DATASET_FOLDER}/images/{part}")
        lines = []
        for box in boxes[boxes["image"] == name].itertuples():
            cx = (box.left + box.box_width / 2) / box.width
            cy = (box.top + box.box_height / 2) / box.height
            class_id = CLASS_NAMES.index(box.class_name)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {box.box_width / box.width:.6f} "
                         f"{box.box_height / box.height:.6f}")
        stem = os.path.splitext(name)[0]
        with open(f"{DATASET_FOLDER}/labels/{part}/{stem}.txt", "w") as file:
            file.write("\n".join(lines))

dataset_path = os.path.abspath(DATASET_FOLDER).replace("\\", "/")   # YOLO butuh path lengkap
yaml_text = f"path: {dataset_path}\ntrain: images/train\nval: images/val\nnames:\n"
for class_id in range(len(CLASS_NAMES)):
    yaml_text = yaml_text + f"  {class_id}: {CLASS_NAMES[class_id]}\n"
with open(f"{DATASET_FOLDER}/data.yaml", "w") as file:
    file.write(yaml_text)
print(yaml_text)
print("train:", len(train_names), "| val:", len(val_names))

# %% [markdown]
# ## Fine-tune YOLOv8n
# Bobot pretrained COCO hanya titik awal; kelas soal dipelajari dari nol di lapisan akhir.
# Train vignette selalu tegak, test banyak yang miring/terbalik. Karena itu dipakai augmentasi
# rotasi: `degrees` = putar acak ±derajat, `flipud` = peluang dibalik atas-bawah.
# Hasil di data demo: tanpa augmentasi 50 gambar test tanpa entete, `degrees=90` 12,
# `degrees=180, flipud=0.5` 5. mAP val sedikit turun (val tegak semua), tetapi test jauh lebih baik.
# Test setegak train? Pakai baris alternatif tanpa augmentasi rotasi.

# %%
model = YOLO(WEIGHTS)
model.train(data=f"{DATASET_FOLDER}/data.yaml", epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
            workers=0, device=DEVICE, project=RUNS_FOLDER, name="train", exist_ok=True, plots=False,
            seed=SEED, verbose=False, degrees=180, flipud=0.5)
# model.train(data=f"{DATASET_FOLDER}/data.yaml", epochs=EPOCHS, imgsz=IMAGE_SIZE,
#             batch=BATCH_SIZE, workers=0, device=DEVICE, project=RUNS_FOLDER, name="train",
#             exist_ok=True, plots=False, seed=SEED, verbose=False)   # tanpa rotasi

# %% [markdown]
# ## Evaluasi di val
# mAP50 = ketepatan kotak bila "kena" cukup dengan IoU ≥ 0.5. mAP50-95 lebih ketat.

# %%
model = YOLO(model.trainer.best)        # bobot terbaik hasil latih
metrics = model.val(data=f"{DATASET_FOLDER}/data.yaml", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                    workers=0, device=DEVICE, project=RUNS_FOLDER, name="val", exist_ok=True,
                    plots=False, verbose=False)
print("mAP50:", round(metrics.box.map50, 4), "| mAP50-95:", round(metrics.box.map, 4))
for class_id in range(len(CLASS_NAMES)):
    print(CLASS_NAMES[class_id], "mAP50:", round(metrics.box.ap50[class_id], 4))

# %% [markdown]
# ## Prediksi test: satu kotak `TARGET_CLASS` dengan keyakinan tertinggi per gambar
# `stream=True` = gambar diproses satu per satu, jadi ribuan gambar tidak memenuhi memori GPU.

# %%
test_paths = sorted(glob.glob(f"{TEST_FOLDER}/*{IMAGE_EXTENSION}"))
target_id = CLASS_NAMES.index(TARGET_CLASS)
results = model.predict(test_paths, imgsz=IMAGE_SIZE, conf=MIN_CONFIDENCE, stream=True,
                        verbose=False, device=DEVICE)
rows = []
for image_path, result in zip(test_paths, results):
    best = [os.path.basename(image_path), 0.0, 0, 0, 0, 0]
    for number in range(len(result.boxes)):
        confidence = float(result.boxes.conf[number])
        if int(result.boxes.cls[number]) == target_id and confidence > best[1]:
            left, top, right, bottom = result.boxes.xyxy[number].tolist()
            best = [best[0], confidence, left, top, right, bottom]
    rows.append(best)
predictions = pd.DataFrame(rows, columns=["id", "confidence", "x1", "y1", "x2", "y2"])
print("gambar test:", len(predictions))
print("tanpa", TARGET_CLASS, "terdeteksi:", (predictions["confidence"] == 0).sum())
predictions["confidence"].describe()

# %% [markdown]
# ## Lihat 8 prediksi test paling ragu
# Di sinilah kelemahan detektor terlihat (gambar miring, terbalik, kotak salah tempat).

# %%
doubtful = predictions.sort_values("confidence").head(8)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, row in zip(axes.ravel(), doubtful.itertuples()):
    ax.imshow(Image.open(f"{TEST_FOLDER}/{row.id}"))
    ax.add_patch(Rectangle((row.x1, row.y1), row.x2 - row.x1, row.y2 - row.y1,
                           fill=False, color="red", lw=2))
    ax.set_title(f"{row.confidence:.2f}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
fig.savefig(f"{OUTPUT_FOLDER}/prediksi_ragu.png", dpi=80)
plt.show()

# %% [markdown]
# ## Simpan kotak + potongan gambar
# Potongan diberi sedikit tepi (`MARGIN`) supaya huruf di pinggir tidak terpotong.
# `<OUTPUT_FOLDER>/potongan/<id>` siap dipakai tahap berikutnya, misalnya OCR.

# %%
shutil.rmtree(CROP_FOLDER, ignore_errors=True)
os.makedirs(CROP_FOLDER)
for row in predictions[predictions["confidence"] > 0].itertuples():
    image = Image.open(f"{TEST_FOLDER}/{row.id}")
    margin_x = (row.x2 - row.x1) * MARGIN
    margin_y = (row.y2 - row.y1) * MARGIN
    crop = image.crop((max(row.x1 - margin_x, 0), max(row.y1 - margin_y, 0),
                       min(row.x2 + margin_x, image.width), min(row.y2 + margin_y, image.height)))
    crop.save(f"{CROP_FOLDER}/{row.id}")
print("potongan disimpan:", len(os.listdir(CROP_FOLDER)))

predictions.to_csv(f"{OUTPUT_FOLDER}/hasil_deteksi.csv", index=False)
predictions.head()

# %% [markdown]
# ## Masalah umum
# - **`CUDA out of memory`:** turunkan `BATCH_SIZE` (8 → 4) atau `IMAGE_SIZE`.
# - **mAP val tinggi tapi test buruk:** train dan test berbeda (sudut, pencahayaan). Lihat grid
#   prediksi ragu, lalu tambah augmentasi (`degrees`, `flipud`, `hsv_v`) atau data yang mirip test.
# - **Kelas tertukar:** urutan `CLASS_NAMES` harus sama dengan `names` di `data.yaml`.
# - **`No labels found`:** folder `labels/` harus sejajar `images/`, nama `.txt` = nama gambar.
# - **Format label 9 angka per baris:** itu OBB/polygon; ekspor ulang sebagai deteksi.
