# %% [markdown]
# # Pipeline: deteksi kotak → potong → luruskan → OCR → submission
# Soal tipe: "baca teks di bagian tertentu gambar" (vignette obat, plat nomor, label, struk).
# Train hanya punya kotak (tanpa teks) → yang dilatih hanya detektor; OCR memakai pretrained.
# Ubah sel konfigurasi saja, lalu Run All. Ganti bagian mana pun dengan potongan `arsenal/vision/`.

# %% [markdown]
# ## Konfigurasi

# %%
import os

TRAIN_COCO = "data/train/_annotations.coco.json"   # UBAH: anotasi COCO (format lain: anotasi_baca)
TRAIN_IMAGES = "data/train"                        # UBAH: folder file_name di json
TEST_IMAGES = "data/test"                          # UBAH
SAMPLE_SUBMISSION = "data/sample_submission.csv"   # UBAH
ID_COLUMN = "id"                                   # UBAH: kolom submission
TEXT_COLUMN = "text"                               # UBAH
TEXT_CLASS = "entete"                              # UBAH: kelas kotak yang teksnya dibaca
PRETRAINED_YOLO = "pretrained/yolov8n/yolov8n.pt"  # UBAH
EASYOCR_FOLDER = "pretrained/easyocr"              # UBAH
WORK_FOLDER = os.path.abspath("outputs/pipeline")  # UBAH: semua hasil antara di sini
EPOCHS = 30                                        # UBAH: 3 dulu untuk cek jalan
IMGSZ = 640                                        # UBAH
BATCH = 8                                          # UBAH: kecil = GPU aman
TRAIN_ROTATIONS = [1, 2, 3]                        # UBAH: augmentasi putar 90/180/270, [] = tidak
os.makedirs(WORK_FOLDER, exist_ok=True)

# %% [markdown]
# ## Baca COCO → `images`, `boxes`

# %%
import json

import pandas as pd

with open(TRAIN_COCO, encoding="utf-8") as file:
    coco = json.load(file)
category_name, image_path, image_rows, box_rows = {}, {}, [], []
for category in coco["categories"]:
    category_name[category["id"]] = category["name"]
for image in coco["images"]:
    image_path[image["id"]] = f"{TRAIN_IMAGES}/{image['file_name']}"
    image_rows.append([image_path[image["id"]], image["width"], image["height"]])
for annotation in coco["annotations"]:
    left, top, width, height = annotation["bbox"]
    box_rows.append([image_path[annotation["image_id"]], category_name[annotation["category_id"]],
                     left, top, left + width, top + height])
images = pd.DataFrame(image_rows, columns=["path", "width", "height"])
boxes = pd.DataFrame(box_rows, columns=["path", "label", "x1", "y1", "x2", "y2"])
CLASS_NAMES = sorted(boxes["label"].unique())
print("gambar:", len(images), "| kotak:", len(boxes), "| kelas:", CLASS_NAMES)

# %% [markdown]
# ## Tulis dataset YOLO (train/val per gambar) + augmentasi putar 90° untuk train

# %%
def yolo_line(label, corners, width, height):
    # piksel (x1, y1, x2, y2) → "kelas x_tengah y_tengah lebar tinggi" (0–1), dijepit ke gambar
    x1, y1, x2, y2 = corners
    x1, x2 = max(x1, 0), min(x2, width)
    y1, y2 = max(y1, 0), min(y2, height)
    x_center, y_center = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    box_width, box_height = (x2 - x1) / width, (y2 - y1) / height
    number = CLASS_NAMES.index(label)
    return f"{number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def rotate_corners(corners, width, height, times):
    # sama dengan Image.rotate(90 * times, expand=True): titik (x, y) → (y, lebar - x) per 90°
    x1, y1, x2, y2 = corners
    for _ in range(times % 4):
        x1, y1, x2, y2 = y1, width - x2, y2, width - x1
        width, height = height, width
    return (x1, y1, x2, y2)


print(rotate_corners((10, 20, 30, 40), 100, 50, 1))   # → (20, 70, 40, 90)

# %%
import os

from PIL import Image
from sklearn.model_selection import train_test_split

DATASET = f"{WORK_FOLDER}/yolo_dataset"
train_paths, val_paths = train_test_split(list(images["path"]), test_size=0.2, random_state=42)
for split, paths in [("train", train_paths), ("val", val_paths)]:
    os.makedirs(f"{DATASET}/images/{split}", exist_ok=True)
    os.makedirs(f"{DATASET}/labels/{split}", exist_ok=True)
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(path).convert("RGB")
        times_list = [0]
        if split == "train":
            times_list = [0] + TRAIN_ROTATIONS
        for times in times_list:
            rotated = image.rotate(90 * times, expand=True)
            rotated.save(f"{DATASET}/images/{split}/{stem}_rot{times}.jpg")
            lines = []
            for box in boxes[boxes["path"] == path].itertuples():
                corners = rotate_corners((box.x1, box.y1, box.x2, box.y2), *image.size, times)
                lines.append(yolo_line(box.label, corners, *rotated.size))
            with open(f"{DATASET}/labels/{split}/{stem}_rot{times}.txt", "w") as file:
                file.write("\n".join(lines))
with open(f"{DATASET}/data.yaml", "w") as file:
    file.write(f"path: {DATASET}\ntrain: images/train\nval: images/val\nnames: {CLASS_NAMES}\n")
print("gambar train:", len(os.listdir(f"{DATASET}/images/train")))

# %% [markdown]
# ## Latih YOLO + mAP val

# %%
from ultralytics import YOLO

model = YOLO(PRETRAINED_YOLO)
model.train(data=f"{DATASET}/data.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, degrees=10,
            patience=10, workers=0, project=f"{WORK_FOLDER}/yolo_runs", name="latih",
            exist_ok=True, seed=42, verbose=False)
BEST_WEIGHTS = f"{model.trainer.save_dir}/weights/best.pt"
detector = YOLO(BEST_WEIGHTS)
metrics = detector.val(data=f"{DATASET}/data.yaml", imgsz=IMGSZ, batch=BATCH, workers=0,
                       project=f"{WORK_FOLDER}/yolo_runs", name="val", exist_ok=True)
print("mAP50:", round(metrics.box.map50, 4), "| mAP50-95:", round(metrics.box.map, 4))

# %% [markdown]
# ## Prediksi test di 4 rotasi → 1 kotak terbaik `TEXT_CLASS` per gambar (cadangan: seluruh gambar)

# %%
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

test_ids = pd.read_csv(SAMPLE_SUBMISSION)[ID_COLUMN]
best_rows = []
for image_id in tqdm(test_ids):
    path = f"{TEST_IMAGES}/{image_id}"    # UBAH bila id tanpa ekstensi: f"{...}/{image_id}.jpg"
    image = Image.open(path).convert("RGB")
    best = [path, 0, 0, image.width, image.height, 0.0]
    for times in [0, 1, 2, 3]:
        rotated = image.rotate(90 * times, expand=True)
        found = detector.predict(rotated, imgsz=IMGSZ, conf=0.1, verbose=False)[0].boxes
        for corner, number, score in zip(found.xyxy.tolist(), found.cls.tolist(),
                                         found.conf.tolist()):
            if detector.names[int(number)] == TEXT_CLASS and score > best[5]:
                best = [path, *rotate_corners(corner, *rotated.size, 4 - times), score]
    best_rows.append(best)
best_boxes = pd.DataFrame(best_rows, columns=["path", "x1", "y1", "x2", "y2", "score"])
print("tanpa kotak (pakai seluruh gambar):", (best_boxes["score"] == 0).sum())

# %% [markdown]
# ## Luruskan teks miring sedikit (`deskew`), dipakai `fix` di sel berikutnya
# Coba putar -15°…15°, pilih sudut di mana jumlah piksel gelap per baris paling naik-turun (baris
# teks rapat, celah kosong). Tidak terganggu barcode/latar gelap.

# %%
import cv2
import numpy as np


def rotate_gray(gray, angle):
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew(gray):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.resize(mask, None, fx=0.5, fy=0.5)   # kecilkan supaya cepat
    best_angle, best_score = 0, -1
    for angle in np.arange(-15, 15.5, 0.5):
        score = np.var(rotate_gray(mask, angle).sum(axis=1))
        if score > best_score:
            best_angle, best_score = angle, score
    return rotate_gray(gray, best_angle)


print(deskew(np.full((40, 120), 255, dtype=np.uint8)).shape)

# %% [markdown]
# ## Potong + perbaiki (mendatar, grayscale, perbesar, CLAHE, tajam, lurus)
# Teks berdiri diputar 90° searah jarum jam; kira-kira separuh jadi terbalik 180°. Itu normal:
# OCR di bawah mencoba 0° dan 180°.

# %%
import cv2
import numpy as np
from PIL import Image

PAD = 0.05   # UBAH


def crop_box(box):
    image = Image.open(box.path).convert("RGB")
    pad_x, pad_y = (box.x2 - box.x1) * PAD, (box.y2 - box.y1) * PAD
    left, top = max(0, box.x1 - pad_x), max(0, box.y1 - pad_y)
    right, bottom = min(image.width, box.x2 + pad_x), min(image.height, box.y2 + pad_y)
    return np.array(image.crop((int(left), int(top), int(right), int(bottom))))


def fix(image):
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)   # teks berdiri → mendatar
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return deskew(gray)


fixed_crops = []
for box in best_boxes.itertuples():
    fixed_crops.append(fix(crop_box(box)))
print("potongan:", len(fixed_crops))

# %% [markdown]
# ## OCR (EasyOCR) di rotasi 0° dan 180° → ambil yang skornya tertinggi

# %%
import easyocr
import numpy as np

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/.-, "   # UBAH
reader = easyocr.Reader(["fr", "en"], model_storage_directory=EASYOCR_FOLDER,
                        download_enabled=False, gpu=True, verbose=False)


def read_text(image):
    # → (teks urut baca: baris atas→bawah, kiri→kanan; skor rata-rata 0–1)
    results = reader.readtext(image, allowlist=ALLOWLIST)
    results = sorted(results, key=reading_order)
    texts, scores = [], [0.0]
    for _, text, score in results:
        texts.append(text)
        scores.append(score)
    return " ".join(texts), float(np.mean(scores[1:] or scores))


def reading_order(result):
    corners = np.array(result[0])
    return (round(corners[:, 1].mean() / 20), corners[:, 0].min())   # UBAH 20: tinggi baris


print(read_text(fixed_crops[0]))

# %%
import numpy as np


def read_best_rotation(image):
    best = ("", -1.0)
    for times in [0, 2]:
        text, score = read_text(np.ascontiguousarray(np.rot90(image, times)))
        if score * np.sqrt(len(text.replace(" ", ""))) > best[1]:
            best = (text, score * np.sqrt(len(text.replace(" ", ""))))
    return best[0]


print(read_best_rotation(fixed_crops[0]))

# %% [markdown]
# ## Baca semua → rapikan → submission
# Isi `TEXT_COLUMN` belum tentu seluruh teks: cek deskripsi soal (mis. hanya LOT/EXP), lalu ubah
# `final_text` (pakai `extract_field` dari `ocr.ipynb`).

# %%
import re

import pandas as pd
from tqdm.auto import tqdm


def final_text(text):
    text = re.sub(r"[^A-Z0-9:/.\- ]", "", text.upper())   # UBAH
    return re.sub(r"\s+", " ", text).strip()


texts = []
for image in tqdm(fixed_crops):
    texts.append(final_text(read_best_rotation(image)))
submission = pd.DataFrame()
submission[ID_COLUMN] = test_ids
submission[TEXT_COLUMN] = texts
submission.to_csv(f"{WORK_FOLDER}/submission.csv", index=False)
print(submission.head(10).to_string())
print("kosong:", (submission[TEXT_COLUMN] == "").sum(), "| file:", f"{WORK_FOLDER}/submission.csv")
