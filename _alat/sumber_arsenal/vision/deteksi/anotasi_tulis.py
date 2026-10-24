# %% [markdown]
# # Tulis tabel kotak → format lain (YOLO untuk ultralytics, COCO json)
# Masuk: `images` (path, width, height) dan `boxes` (path, label, x1, y1, x2, y2) dari
# `anotasi_baca.ipynb`. Keluar: folder dataset di `OUTPUT_FOLDER` + `data.yaml` → dipakai
`deteksi_yolo.ipynb`.
#
# ## Cara menempel ke notebook-mu
# Salin **berurutan**, di bawah sel baca anotasi:
#
# | urutan | sel | wajib? |
# |---|---|---|
# | 1 | Pilih / ganti nama kelas | opsional |
# | 2 | Dataset YOLO: konfigurasi + bagi train/val | wajib (membuat `yolo_line`, `train_paths`) |
# | 3 | Dataset YOLO: salin gambar + tulis label | wajib, **sesudah** sel 2 |
# | 4 | Augmentasi putar 90° | opsional, **sesudah** sel 3 |
# | - | COCO json | hanya bila butuh format COCO |
#
# Urutan penting: sel 3 dan 4 memakai `OUTPUT_FOLDER`, `yolo_line`, `train_paths` dari sel 2.
# Dataset cukup dibuat **sekali**: sudah ada di disk? Lewati semua sel ini, langsung ke
# `deteksi_yolo.ipynb`.

# %% [markdown]
# ## Pilih / ganti nama kelas (opsional)
# Ubah: `KEEP_CLASSES` (kelas yang dipakai), `RENAME` (gabung/ganti nama, mis. {"car": "vehicle"}).

# %%
KEEP_CLASSES = ["entete", "vin"]   # UBAH: lihat boxes["label"].value_counts()
RENAME = {}                        # UBAH: {"nama_lama": "nama_baru"}

boxes = boxes[boxes["label"].isin(KEEP_CLASSES)].copy()
boxes["label"] = boxes["label"].replace(RENAME)
print(boxes["label"].value_counts())

# %% [markdown]
# ## Dataset YOLO: bagi train/val per gambar, salin gambar, tulis label .txt + data.yaml
# Ubah: `OUTPUT_FOLDER` (folder baru, jangan di dalam folder soal), `VAL_FRACTION`.
# Gambar tanpa kotak tetap disalin dengan label kosong (jadi contoh latar belakang).

# %%
import os
import shutil

from sklearn.model_selection import train_test_split

OUTPUT_FOLDER = "outputs/yolo_dataset"         # UBAH
CLASS_NAMES = sorted(boxes["label"].unique())  # UBAH: atau manual, urutan = nomor kelas
VAL_FRACTION = 0.2                             # UBAH


def yolo_line(label, corners, width, height):
    # piksel (x1, y1, x2, y2) → "kelas x_tengah y_tengah lebar tinggi" (0–1), dijepit ke gambar
    x1, y1, x2, y2 = corners
    x1, x2 = max(x1, 0), min(x2, width)
    y1, y2 = max(y1, 0), min(y2, height)
    x_center, y_center = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    box_width, box_height = (x2 - x1) / width, (y2 - y1) / height
    number = CLASS_NAMES.index(label)
    return f"{number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


train_paths, val_paths = train_test_split(list(images["path"]), test_size=VAL_FRACTION,
                                          random_state=42)
print("train:", len(train_paths), "| val:", len(val_paths), "| kelas:", CLASS_NAMES)

# %%
for split, paths in [("train", train_paths), ("val", val_paths)]:
    os.makedirs(f"{OUTPUT_FOLDER}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/labels/{split}", exist_ok=True)
    for path in paths:
        name = os.path.basename(path)
        shutil.copy(path, f"{OUTPUT_FOLDER}/images/{split}/{name}")
        size = images[images["path"] == path].iloc[0]
        lines = []
        for box in boxes[boxes["path"] == path].itertuples():
            corners = (box.x1, box.y1, box.x2, box.y2)
            lines.append(yolo_line(box.label, corners, size["width"], size["height"]))
        label_file = f"{OUTPUT_FOLDER}/labels/{split}/{os.path.splitext(name)[0]}.txt"
        with open(label_file, "w") as file:
            file.write("\n".join(lines))

DATA_YAML = f"{OUTPUT_FOLDER}/data.yaml"
with open(DATA_YAML, "w") as file:
    file.write(f"path: {os.path.abspath(OUTPUT_FOLDER)}\n".replace("\\", "/"))
    file.write("train: images/train\nval: images/val\n")
    file.write(f"names: {CLASS_NAMES}\n")
print(open(DATA_YAML).read())
print("label train:", len(os.listdir(f"{OUTPUT_FOLDER}/labels/train")))

# %% [markdown]
# ## (Opsional) Augmentasi putar 90°/180°/270° untuk train saja
# Pakai bila foto test bisa terbalik/menyamping tapi train selalu tegak. Ultralytics `degrees`
# hanya memutar sedikit; sel ini menambah salinan gambar train yang diputar beserta kotaknya.
# Jalankan SETELAH sel dataset YOLO. Ubah: `ROTATIONS`.

# %%
from PIL import Image

ROTATIONS = [1, 2, 3]   # UBAH: 1 = 90°, 2 = 180°, 3 = 270° (berlawanan jarum jam)


def rotate_corners(corners, width, height, times):
    # sama dengan Image.rotate(90 * times, expand=True): titik (x, y) → (y, lebar - x) per 90°
    x1, y1, x2, y2 = corners
    for _ in range(times % 4):
        x1, y1, x2, y2 = y1, width - x2, y2, width - x1
        width, height = height, width
    return (x1, y1, x2, y2)


for path in train_paths:
    stem, extension = os.path.splitext(os.path.basename(path))
    image = Image.open(path)
    for times in ROTATIONS:
        rotated = image.rotate(90 * times, expand=True)
        rotated.save(f"{OUTPUT_FOLDER}/images/train/{stem}_rot{times}{extension}")
        lines = []
        for box in boxes[boxes["path"] == path].itertuples():
            corners = rotate_corners((box.x1, box.y1, box.x2, box.y2), *image.size, times)
            lines.append(yolo_line(box.label, corners, *rotated.size))
        with open(f"{OUTPUT_FOLDER}/labels/train/{stem}_rot{times}.txt", "w") as file:
            file.write("\n".join(lines))
print("gambar train sekarang:", len(os.listdir(f"{OUTPUT_FOLDER}/images/train")))

# %% [markdown]
# ## COCO json (untuk alat lain, mis. detectron/mmdet, atau unggah balik ke CVAT/Roboflow)
# Ubah: `OUTPUT_JSON`. `file_name` = nama file saja, jadi letakkan json di folder gambarnya.

# %%
import json
import os

OUTPUT_JSON = "outputs/annotations.coco.json"   # UBAH

class_list = sorted(boxes["label"].unique())
coco = {"images": [], "annotations": [], "categories": []}
for number, name in enumerate(class_list):
    coco["categories"].append({"id": number, "name": name})
image_id = {}
for number, row in enumerate(images.itertuples()):
    image_id[row.path] = number
    coco["images"].append({"id": number, "file_name": os.path.basename(row.path),
                           "width": int(row.width), "height": int(row.height)})
for number, box in enumerate(boxes.itertuples()):
    width, height = float(box.x2 - box.x1), float(box.y2 - box.y1)
    coco["annotations"].append({"id": number, "image_id": image_id[box.path],
                                "category_id": class_list.index(box.label),
                                "bbox": [float(box.x1), float(box.y1), width, height],
                                "area": width * height, "iscrowd": 0})
os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(coco, file)
print("gambar:", len(coco["images"]), "| kotak:", len(coco["annotations"]))
