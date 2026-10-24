# %% [markdown]
# # Satelit/Udara — Deteksi Objek Kecil pada Citra Besar (tiling + YOLO + NMS)
# **Kapan dipakai:** citra satelit/drone sangat besar (ribuan piksel) dan objeknya kecil
# (mobil, kapal, pesawat). Kalau citra utuh dikecilkan ke 640 px, objek kecil hilang.
#
# **Idenya:** citra dipotong jadi kotak (tile) yang saling tumpang-tindih → YOLO tiap tile →
# kotak digeser ke koordinat citra utuh → kotak ganda di area tumpang-tindih dibuang dengan
# NMS (non-maximum suppression).
#
# **Langkah:**
# 1. Deteksi pada satu tile (fungsi `detect_tile`).
# 2. Potong citra jadi tile, deteksi semua, gabungkan dengan NMS.
# 3. Simpan semua kotak. Cek recall/precision di sel terakhir (hanya data demo).
#
# **Data demo:** 19 foto udara VisDrone 1920×1080 (mobil kecil) + anotasinya (lisensi
# non-komersial). YOLOv8n COCO dipakai apa adanya, kelas "car". Saat lomba bobot hasil
# fine-tune biasanya jauh lebih baik.

# %%
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import box_iou, nms
from ultralytics import YOLO

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/udara_visdrone"   # GANTI saat lomba
IMAGE_PATTERN = f"{DATA_FOLDER}/gambar/*.jpg"
ANNOTATION_FOLDER = f"{DATA_FOLDER}/anotasi"     # hanya untuk cek
WEIGHTS = "../../../pretrained/yolov8n/yolov8n.pt"

TILE_SIZE = 640           # sama dengan ukuran masukan YOLO, 320 untuk objek lebih kecil
OVERLAP = 0.25            # tile bersebelahan berbagi 25% supaya objek di tepi tidak terpotong
CONFIDENCE = 0.15         # objek kecil keyakinannya rendah, jadi ambangnya rendah
NMS_IOU = 0.5             # dua kotak dengan IoU > 0.5 dianggap objek yang sama
YOLO_CLASS = 2            # id kelas COCO: 2 = car
KEY_CLASS = "4"           # id kelas VisDrone: 4 = car (kolom ke-6 file anotasi)
MATCH_IOU = 0.5           # deteksi dianggap benar bila IoU dengan kunci >= 0.5

# %% [markdown]
# ## Baca daftar citra

# %%
image_paths = sorted(glob.glob(IMAGE_PATTERN))
image_names = []
for path in image_paths:
    file_name = os.path.basename(path)
    image_names.append(os.path.splitext(file_name)[0])
model = YOLO(WEIGHTS)
print(len(image_paths), "citra, ukuran", Image.open(image_paths[0]).size)

# %% [markdown]
# ## Langkah 1 — deteksi pada satu tile
# Kotak hasil YOLO digeser sebesar posisi tile, jadi koordinatnya ikut citra utuh.

# %%
def detect_tile(image, shift_x, shift_y):
    results = model.predict(image, imgsz=TILE_SIZE, conf=CONFIDENCE, classes=[YOLO_CLASS],
                            verbose=False)
    boxes = results[0].boxes.xyxy.cpu()
    boxes[:, [0, 2]] += shift_x
    boxes[:, [1, 3]] += shift_y
    return boxes, results[0].boxes.conf.cpu()


# coba pada citra pertama utuh (tanpa tiling)
first_image = Image.open(image_paths[0]).convert("RGB")
whole_boxes, whole_scores = detect_tile(first_image, 0, 0)
print("citra utuh:", len(whole_boxes), "kotak")

# %% [markdown]
# ## Langkah 2 — tiling + NMS
# Tile di tepi citra digeser mundur supaya semua tile berukuran penuh.

# %%
def detect_tiled(image):
    width, height = image.size
    step = int(TILE_SIZE * (1 - OVERLAP))
    all_boxes = []
    all_scores = []
    for top in range(0, height, step):
        bottom = min(top + TILE_SIZE, height)
        top = max(bottom - TILE_SIZE, 0)
        for left in range(0, width, step):
            right = min(left + TILE_SIZE, width)
            left = max(right - TILE_SIZE, 0)
            boxes, scores = detect_tile(image.crop((left, top, right, bottom)), left, top)
            all_boxes.append(boxes)
            all_scores.append(scores)
    all_boxes = torch.cat(all_boxes)
    all_scores = torch.cat(all_scores)
    keep = nms(all_boxes, all_scores, NMS_IOU)       # buang kotak ganda
    return all_boxes[keep], all_scores[keep]


# coba pada citra pertama
tiled_boxes, tiled_scores = detect_tiled(first_image)
print("tiling + NMS:", len(tiled_boxes), "kotak")

# %% [markdown]
# ## Deteksi semua citra (pilih satu cara)

# %%
detections = {}
for path, name in zip(image_paths, image_names):
    image = Image.open(path).convert("RGB")
    detections[name] = detect_tiled(image)
    # detections[name] = detect_tile(image, 0, 0)     # citra utuh, untuk pembanding
print("selesai", len(detections), "citra")

plt.figure(figsize=(14, 8))
plt.imshow(first_image)
for x1, y1, x2, y2 in detections[image_names[0]][0].tolist():
    rectangle = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", lw=1)
    plt.gca().add_patch(rectangle)
plt.axis("off")
plt.savefig("contoh_deteksi.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Langkah 3 — simpan semua kotak
# Satu baris per kotak: nama citra, koordinat, skor.

# %%
rows = []
for name in image_names:
    boxes, scores = detections[name]
    for corner, score in zip(boxes.tolist(), scores.tolist()):
        rows.append([name] + corner + [score])
table = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score"])

submission = pd.DataFrame()
submission["image_id"] = table["image_id"]
submission["x1"] = table["x1"]
submission["y1"] = table["y1"]
submission["x2"] = table["x2"]
submission["y2"] = table["y2"]
submission["score"] = table["score"]
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# Format VisDrone per baris: x, y, lebar, tinggi, skor, kelas, ... (skor 0 = area diabaikan).
# Recall = porsi mobil asli yang terdeteksi. Precision = porsi deteksi yang benar.

# %%
found_truth = 0
correct_detections = 0
total_truth = 0
total_detections = 0
for name in image_names:
    truth_boxes = []
    for line in open(f"{ANNOTATION_FOLDER}/{name}.txt"):
        parts = line.strip().split(",")
        if parts[5] == KEY_CLASS and parts[4] == "1":
            left = float(parts[0])
            top = float(parts[1])
            truth_boxes.append([left, top, left + float(parts[2]), top + float(parts[3])])
    truth_boxes = torch.tensor(truth_boxes)
    found_boxes = detections[name][0]
    total_truth += len(truth_boxes)
    total_detections += len(found_boxes)
    if len(truth_boxes) > 0 and len(found_boxes) > 0:
        iou = box_iou(truth_boxes, found_boxes)
        found_truth += int((iou.max(dim=1).values >= MATCH_IOU).sum())
        correct_detections += int((iou.max(dim=0).values >= MATCH_IOU).sum())
print("mobil asli:", total_truth, "| deteksi:", total_detections)
print("recall   :", round(found_truth / total_truth, 3))
print("precision:", round(correct_detections / max(total_detections, 1), 3))
