# %% [markdown]
# # Mask dan kontur: hitung objek, luas, poligon ↔ mask, YOLO-seg (bentuk per objek)
#
# **Butuh:** gambar dan/atau mask. **Menghasilkan:** tabel objek (luas, kotak, pusat), file
# label YOLO-seg, model YOLO-seg, tabel prediksi per objek.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | kapan |
# |---|---|---|
# | 1 | **tanpa training**: Otsu → bersihkan → hitung objek + luas | latar polos (koin, sel) |
# | 2 | mask instance (1, 2, 3 = objek ke-n) → tabel objek | sudah punya mask |
# | 3 | poligon (COCO/CVAT) → mask, mask → poligon | konversi anotasi |
# | 4 | mask instance → dataset YOLO-seg + `data.yaml` | mau latih YOLO-seg |
# | 5 | latih YOLO-seg | |
# | 6 | prediksi YOLO-seg → tabel per objek (kelas, skor, luas, poligon) | |
#
# Segmentasi per piksel tanpa memisahkan objek (latar/jalan/bangunan): `semantik.ipynb`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Foto koin (bawaan scikit-image) + foto pejalan kaki dengan mask instance.

# %%
import glob

import pandas as pd
from skimage import data

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

coins = data.coins()           # array abu-abu (tinggi, lebar)
folder = f"{DEMO_FOLDER}/vision/detection/PennFudanPed"
pairs = pd.DataFrame()
pairs["image_path"] = sorted(glob.glob(f"{folder}/PNGImages/*.png"))
pairs["mask_path"] = pairs["image_path"].str.replace("PNGImages", "PedMasks")
pairs["mask_path"] = pairs["mask_path"].str.replace(".png", "_mask.png")
print(coins.shape, len(pairs))

# %% [markdown]
# ## 1. Tanpa training: threshold → bersihkan → hitung objek → `objects`
# Otsu memilih batas terang/gelap otomatis. `THRESH_BINARY_INV` kalau objek lebih GELAP dari
# latar. `MORPH_OPEN` membuang bintik kecil, `MORPH_CLOSE` menutup lubang. `MIN_AREA` membuang
# sisa noise. Objek menempel jadi satu? Perbesar `KERNEL_SIZE` open, atau pakai watershed.

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gray = coins                    # UBAH: cv2.imread(path, cv2.IMREAD_GRAYSCALE)
KERNEL_SIZE = 5                 # UBAH
MIN_AREA = 100                  # UBAH: luas minimum (piksel)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
objects = pd.DataFrame(stats[1:], columns=["x", "y", "width", "height", "area"])  # [0]=latar
objects["center_x"], objects["center_y"] = centers[1:, 0], centers[1:, 1]
objects = objects[objects["area"] >= MIN_AREA].reset_index(drop=True)
print("jumlah objek:", len(objects))
print(objects.describe().round(1))

# %% [markdown]
# Lihat: kotak merah = objek terhitung. Kanan = hasil threshold (putih = objek).

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(gray, cmap="gray")
axes[1].imshow(binary, cmap="gray")
for _, row in objects.iterrows():
    axes[0].add_patch(plt.Rectangle((row["x"], row["y"]), row["width"], row["height"],
                                    fill=False, color="red"))
plt.show()

# %% [markdown]
# ## 2. Mask instance → tabel objek per gambar → `instances`
# **Butuh:** mask dengan nilai 0 = latar, 1, 2, 3, ... = objek ke-n.

# %%
import numpy as np
import pandas as pd
from PIL import Image


def mask_objects(mask_path):
    mask = np.array(Image.open(mask_path))
    rows = []
    for number in np.unique(mask):
        if number == 0:
            continue
        ys, xs = np.where(mask == number)
        rows.append({"mask_path": mask_path, "object": int(number), "area": len(xs),
                     "x1": xs.min(), "y1": ys.min(), "x2": xs.max(), "y2": ys.max()})
    return rows


rows = []
for mask_path in pairs["mask_path"]:
    rows.extend(mask_objects(mask_path))
instances = pd.DataFrame(rows)
print(instances.head())
objects_per_image = instances.groupby("mask_path").size()
print("objek per gambar:", objects_per_image.describe().round(1))

# %% [markdown]
# ## 3. Poligon ↔ mask
# Poligon = daftar titik `[x1, y1, x2, y2, ...]` (format COCO `segmentation`, CVAT polygon).
# `mask_to_polygons` mengambil tepi luar tiap objek; `epsilon` besar = titik lebih sedikit.

# %%
import cv2
import numpy as np
from PIL import Image, ImageDraw


def polygon_to_mask(points, width, height):
    canvas = Image.new("L", (width, height), 0)
    pairs_xy = list(zip(points[0::2], points[1::2]))
    ImageDraw.Draw(canvas).polygon(pairs_xy, fill=1)
    return np.array(canvas)


def mask_to_polygons(binary_mask, epsilon=1.0):
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) >= 3:
            polygons.append(contour.reshape(-1).tolist())
    return polygons


square = polygon_to_mask([10, 10, 40, 10, 40, 30, 10, 30], 50, 40)
print("luas persegi:", square.sum(), "| poligon balik:", mask_to_polygons(square))

# %% [markdown]
# ## 4. Mask instance → dataset YOLO-seg → `outputs/yolo_seg/data.yaml`
# Satu baris label per objek: `kelas x1 y1 x2 y2 ...` dengan koordinat dibagi lebar/tinggi
# (0..1). Di sini semua objek kelas 0 (`orang`); banyak kelas → tulis nomor kelasnya.

# %%
import os

import numpy as np
from PIL import Image

DATASET_FOLDER = os.path.abspath("outputs/yolo_seg")   # UBAH
CLASS_NAMES = ["orang"]                                # UBAH


def write_yolo_seg_label(mask_path, label_path):
    mask = np.array(Image.open(mask_path))
    height, width = mask.shape[:2]
    lines = []
    for number in np.unique(mask)[1:]:                 # [1:] = lewati latar (0)
        for polygon in mask_to_polygons(mask == number, epsilon=2.0):
            coordinates = np.array(polygon, dtype=float)
            coordinates[0::2] /= width
            coordinates[1::2] /= height
            lines.append("0 " + " ".join(coordinates.round(5).astype(str)))   # UBAH: kelas
    with open(label_path, "w") as file:
        file.write("\n".join(lines))


os.makedirs(DATASET_FOLDER, exist_ok=True)
write_yolo_seg_label(pairs["mask_path"][0], f"{DATASET_FOLDER}/coba.txt")
print(open(f"{DATASET_FOLDER}/coba.txt").read()[:120])

# %% [markdown]
# Terapkan ke SEMUA: split train/val, salin gambar, tulis label.

# %%
import os
import shutil

from sklearn.model_selection import train_test_split

part_train, part_val = train_test_split(pairs, test_size=0.2, random_state=42)
for part, table in [("train", part_train), ("val", part_val)]:
    os.makedirs(f"{DATASET_FOLDER}/images/{part}", exist_ok=True)
    os.makedirs(f"{DATASET_FOLDER}/labels/{part}", exist_ok=True)
    for image_path, mask_path in zip(table["image_path"], table["mask_path"]):
        name = os.path.splitext(os.path.basename(image_path))[0]
        shutil.copy(image_path, f"{DATASET_FOLDER}/images/{part}/{name}.png")
        write_yolo_seg_label(mask_path, f"{DATASET_FOLDER}/labels/{part}/{name}.txt")

# %% [markdown]
# Tulis `data.yaml` (daftar folder + nama kelas).

# %%
with open(f"{DATASET_FOLDER}/data.yaml", "w") as file:
    file.write(f"path: {DATASET_FOLDER}\ntrain: images/train\nval: images/val\nnames:\n")
    for i in range(len(CLASS_NAMES)):
        file.write(f"  {i}: {CLASS_NAMES[i]}\n")
print(open(f"{DATASET_FOLDER}/data.yaml").read())

# %% [markdown]
# ## 5. Latih YOLO-seg
# Sama dengan `vision/deteksi/deteksi_yolo.ipynb`, hanya bobot awal `-seg.pt`. Lanjutkan
# yang terputus, evaluasi, 4 rotasi: lihat notebook itu.

# %%
import os

from ultralytics import YOLO

PRETRAINED_FOLDER = "pretrained"   # UBAH
EPOCHS = 30                        # UBAH: 3 dulu untuk cek jalan
RUNS_FOLDER = os.path.abspath("outputs/yolo_runs")
BEST_WEIGHTS = f"{RUNS_FOLDER}/seg/weights/best.pt"

model = YOLO(f"{PRETRAINED_FOLDER}/yolov8n_seg/yolov8n-seg.pt")
model.train(data=f"{DATASET_FOLDER}/data.yaml", epochs=EPOCHS, imgsz=640, batch=8, workers=0,
            project=RUNS_FOLDER, name="seg", exist_ok=True, seed=42, verbose=False, plots=False)
metrics = model.val(data=f"{DATASET_FOLDER}/data.yaml", workers=0, project=RUNS_FOLDER,
                    name="seg_val", exist_ok=True, verbose=False, plots=False)

print("mask mAP50:", round(metrics.seg.map50, 4), "| mask mAP50-95:", round(metrics.seg.map, 4))

# %% [markdown]
# ## 6. Prediksi → tabel per objek `pred_objects` (+ poligon dalam piksel)
# `result.masks.xy` = poligon tiap objek (sudah dalam piksel gambar asli).

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

CONF = 0.25   # UBAH
model = YOLO(BEST_WEIGHTS)
rows = []
for path in part_val["image_path"]:
    result = model.predict(path, conf=CONF, verbose=False)[0]
    if result.masks is None:
        continue
    for i in range(len(result.boxes)):
        polygon = result.masks.xy[i]
        rows.append({"path": path, "label": CLASS_NAMES[int(result.boxes.cls[i])],
                     "score": float(result.boxes.conf[i]),
                     "area": cv2.contourArea(polygon.astype(np.float32)),   # piksel asli
                     "polygon": polygon.tolist()})

pred_objects = pd.DataFrame(rows, columns=["path", "label", "score", "area", "polygon"])
print("objek ditemukan:", len(pred_objects))
print(pred_objects[["path", "label", "score", "area"]].head())
plt.imshow(result.plot()[:, :, ::-1])   # gambar terakhir + mask tebakan (BGR → RGB)
plt.show()
