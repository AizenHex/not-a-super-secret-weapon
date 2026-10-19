# %% [markdown]
# # Vision — Anotasi Sendiri untuk Deteksi Objek (auto-label YOLO)
# **Kapan dipakai:** gambar **belum punya kotak** dan harus dilabeli sendiri.
#
# **Langkah:** YOLO pretrained menebak kotak (auto-label) → lihat grid hasilnya → perbaiki yang
# salah → susun folder YOLO → fine-tune → cek mAP. Tanpa GUI, tanpa internet.
#
# **Data demo:** PennFudanPed (170 foto pejalan kaki). Kotak aslinya **tidak dipakai** sampai sel
# kunci jawaban terakhir, hanya untuk mengukur mutu label kita.

# %%
import glob
import os
import re
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

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/detection/PennFudanPed"   # GANTI
IMAGE_FOLDER = f"{DATA_FOLDER}/PNGImages"
KEY_FOLDER = f"{DATA_FOLDER}/Annotation"     # kotak asli, hanya ada di data latihan
IMAGE_EXTENSION = ".png"                      # ganti ".jpg" bila perlu
YOLO_WEIGHTS = "Z:/GEMASTIK-DATABASE/pretrained/yolov8n/yolov8n.pt"
LABEL_FOLDER = "labels_auto"     # file .txt hasil auto-label (di folder notebook)
DATASET_FOLDER = "dataset"       # folder YOLO images/ + labels/ (di folder notebook)

IMAGE_SIZE = 640
EPOCHS = 5
BATCH_SIZE = 8
VAL_FRACTION = 0.2    # bagian gambar yang jadi data val
N_GRID = 8            # jumlah gambar per grid tinjauan (genap)
MIN_CONFIDENCE = 0.25   # kotak dengan keyakinan di bawah ini dibuang
MIN_AREA = 0.01       # kotak lebih kecil dari 1% gambar dibuang
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = 0

CLASS_MAP = {0: 0}               # id kelas COCO -> id kelas kita (COCO 0 = person)
CLASS_NAMES = {0: "pejalan"}     # id kelas kita -> nama
# CLASS_MAP = {2: 0, 5: 1, 7: 1}    # contoh kelas sendiri: car -> 0; bus dan truck -> 1
# CLASS_NAMES = {0: "mobil", 1: "kendaraan_besar"}

# Koreksi manual: isi SETELAH melihat grid, lalu jalankan ulang dari sel koreksi
DELETE_ROWS = {}      # contoh: {"FudanPed00007": [1]} -> hapus baris ke-1 (mulai dari 0)
CHANGE_CLASS = {}     # contoh: {"FudanPed00003": {0: 1}} -> baris ke-0 jadi kelas 1
ADD_BOXES = {}        # contoh: {"FudanPed00010": [(0, 100, 50, 220, 400)]}
#                       = (kelas, kiri, atas, kanan, bawah) dalam piksel

# %% [markdown]
# ## Baca data (gambar tanpa label)

# %%
image_paths = sorted(glob.glob(f"{IMAGE_FOLDER}/*{IMAGE_EXTENSION}"))
image_names = []
for image_path in image_paths:
    image_names.append(os.path.splitext(os.path.basename(image_path))[0])
print("jumlah gambar:", len(image_paths), "| ukuran contoh:", Image.open(image_paths[0]).size)

# %% [markdown]
# ## Auto-label: YOLO pretrained menebak kotak, ditulis sebagai file `.txt` format YOLO
# Satu baris = `kelas cx cy w h` (titik tengah, lebar, tinggi; semua 0–1 relatif ukuran gambar).

# %%
os.makedirs(LABEL_FOLDER, exist_ok=True)
model = YOLO(YOLO_WEIGHTS)
results = model.predict(image_paths, conf=MIN_CONFIDENCE, imgsz=IMAGE_SIZE, device=DEVICE,
                        verbose=False)

box_counts = []
lowest_confidence = []
for name, result in zip(image_names, results):
    lines = []
    scores = []
    for box_number in range(len(result.boxes)):
        coco_class = int(result.boxes.cls[box_number])
        cx, cy, width, height = result.boxes.xywhn[box_number].tolist()
        if coco_class in CLASS_MAP and width * height >= MIN_AREA:
            lines.append(f"{CLASS_MAP[coco_class]} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
            scores.append(float(result.boxes.conf[box_number]))
    with open(f"{LABEL_FOLDER}/{name}.txt", "w") as file:
        file.write("\n".join(lines))
    box_counts.append(len(lines))
    lowest_confidence.append(min(scores, default=0))

summary = pd.DataFrame()
summary["image"] = image_names
summary["n_boxes"] = box_counts
summary["min_conf"] = lowest_confidence
print("total kotak:", summary["n_boxes"].sum())
print("gambar tanpa deteksi:", (summary["n_boxes"] == 0).sum())

# %% [markdown]
# ## Tinjau: gambar + kotak dari file label (sumber kebenaran = file `.txt`)
# Dua grid: 8 gambar acak, dan 8 gambar yang keyakinannya paling rendah (cek ini lebih dulu).

# %%
def show_boxes(names, file_name):
    fig, axes = plt.subplots(2, N_GRID // 2, figsize=(14, 8))
    for ax, name in zip(axes.ravel(), names):
        image = Image.open(f"{IMAGE_FOLDER}/{name}{IMAGE_EXTENSION}")
        ax.imshow(image)
        for line in open(f"{LABEL_FOLDER}/{name}.txt").read().splitlines():
            class_id, cx, cy, width, height = line.split()
            box_width = float(width) * image.width
            box_height = float(height) * image.height
            left = float(cx) * image.width - box_width / 2
            top = float(cy) * image.height - box_height / 2
            ax.add_patch(Rectangle((left, top), box_width, box_height, fill=False, color="red"))
            ax.text(left, top, CLASS_NAMES[int(class_id)], color="white", backgroundcolor="red",
                    fontsize=7)
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(file_name, dpi=80)
    plt.show()


check_first = summary.sort_values("min_conf").head(N_GRID)
print(check_first)
show_boxes(summary.sample(N_GRID, random_state=SEED)["image"], "grid_acak.png")
show_boxes(check_first["image"], "grid_perlu_dicek.png")

# %% [markdown]
# ## Koreksi: hapus baris, ubah kelas
# Cara tercepat tetap membuka file `labels_auto/<nama>.txt` dan mengedit barisnya langsung.

# %%
for name in DELETE_ROWS:
    label_file = f"{LABEL_FOLDER}/{name}.txt"
    lines = open(label_file).read().splitlines()
    kept = []
    for line_number in range(len(lines)):
        if line_number not in DELETE_ROWS[name]:
            kept.append(lines[line_number])
    with open(label_file, "w") as file:
        file.write("\n".join(kept))

for name in CHANGE_CLASS:
    label_file = f"{LABEL_FOLDER}/{name}.txt"
    lines = open(label_file).read().splitlines()
    for line_number in CHANGE_CLASS[name]:
        box_numbers = lines[line_number].split(" ", 1)[1]     # "cx cy w h" tanpa kelas
        lines[line_number] = f"{CHANGE_CLASS[name][line_number]} {box_numbers}"
    with open(label_file, "w") as file:
        file.write("\n".join(lines))

# %% [markdown]
# ## Koreksi: tambah kotak manual (koordinat piksel → format YOLO)

# %%
for name in ADD_BOXES:
    image = Image.open(f"{IMAGE_FOLDER}/{name}{IMAGE_EXTENSION}")
    new_lines = ""
    for class_id, left, top, right, bottom in ADD_BOXES[name]:
        cx = (left + right) / 2 / image.width
        cy = (top + bottom) / 2 / image.height
        width = (right - left) / image.width
        height = (bottom - top) / image.height
        new_lines = new_lines + f"\n{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
    with open(f"{LABEL_FOLDER}/{name}.txt", "a") as file:
        file.write(new_lines)

# %% [markdown]
# ## Susun folder YOLO: `images/train`, `images/val`, `labels/train`, `labels/val`, `data.yaml`

# %%
shutil.rmtree(DATASET_FOLDER, ignore_errors=True)     # mulai dari folder kosong
train_names, val_names = train_test_split(image_names, test_size=VAL_FRACTION,
                                          random_state=SEED)
for part, names in [("train", train_names), ("val", val_names)]:
    os.makedirs(f"{DATASET_FOLDER}/images/{part}")
    os.makedirs(f"{DATASET_FOLDER}/labels/{part}")
    for name in names:
        shutil.copy(f"{IMAGE_FOLDER}/{name}{IMAGE_EXTENSION}", f"{DATASET_FOLDER}/images/{part}")
        shutil.copy(f"{LABEL_FOLDER}/{name}.txt", f"{DATASET_FOLDER}/labels/{part}")

dataset_path = os.path.abspath(DATASET_FOLDER).replace("\\", "/")   # YOLO butuh path lengkap
yaml_text = f"path: {dataset_path}\ntrain: images/train\nval: images/val\nnames:\n"
for class_id in CLASS_NAMES:
    yaml_text = yaml_text + f"  {class_id}: {CLASS_NAMES[class_id]}\n"
with open(f"{DATASET_FOLDER}/data.yaml", "w") as file:
    file.write(yaml_text)
print(yaml_text)
print("train:", len(train_names), "| val:", len(val_names))

# %% [markdown]
# ## Fine-tune singkat dari bobot pretrained, lalu mAP di data val
# mAP di sini diukur terhadap label hasil anotasi kita sendiri.

# %%
model = YOLO(YOLO_WEIGHTS)
model.train(data=f"{DATASET_FOLDER}/data.yaml", epochs=EPOCHS, imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE, workers=0, device=DEVICE, project="runs", name="train",
            exist_ok=True, plots=False, seed=SEED)

model = YOLO(model.trainer.best)        # file bobot terbaik hasil latih
metrics = model.val(data=f"{DATASET_FOLDER}/data.yaml", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                    workers=0, device=DEVICE, project="runs", name="val", exist_ok=True,
                    plots=False, verbose=False)
print("mAP50:", round(metrics.box.map50, 4), "| mAP50-95:", round(metrics.box.map, 4))

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# Kotak asli PennFudan (`Annotation/*.txt`, berisi "(kiri, atas) - (kanan, bawah)") diubah ke
# format YOLO untuk gambar val. Hasilnya: seberapa bagus model dari label kita bila diukur dengan
# label manusia.

# %%
KEY_DATASET = "dataset_kunci"
shutil.rmtree(KEY_DATASET, ignore_errors=True)
os.makedirs(f"{KEY_DATASET}/images/val")
os.makedirs(f"{KEY_DATASET}/labels/val")
for name in val_names:
    image_file = f"{IMAGE_FOLDER}/{name}{IMAGE_EXTENSION}"
    shutil.copy(image_file, f"{KEY_DATASET}/images/val")
    image = Image.open(image_file)
    key_text = open(f"{KEY_FOLDER}/{name}.txt").read()
    lines = []
    for left, top, right, bottom in re.findall(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", key_text):
        cx = (int(left) + int(right)) / 2 / image.width
        cy = (int(top) + int(bottom)) / 2 / image.height
        width = (int(right) - int(left)) / image.width
        height = (int(bottom) - int(top)) / image.height
        lines.append(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    with open(f"{KEY_DATASET}/labels/val/{name}.txt", "w") as file:
        file.write("\n".join(lines))

key_path = os.path.abspath(KEY_DATASET).replace("\\", "/")
with open(f"{KEY_DATASET}/data.yaml", "w") as file:
    file.write(f"path: {key_path}\ntrain: images/val\nval: images/val\nnames:\n  0: pejalan\n")
metrics = model.val(data=f"{KEY_DATASET}/data.yaml", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                    workers=0, device=DEVICE, project="runs", name="val_kunci", exist_ok=True,
                    plots=False, verbose=False)
print("mAP50 terhadap label manusia:", round(metrics.box.map50, 4))
