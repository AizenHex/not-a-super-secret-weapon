# %% [markdown]
# # Vision — Menghitung Objek dengan Detektor (YOLOv8n pretrained COCO)
# **Kapan dipakai:** jumlah orang di antrean/keramaian, kendaraan di jalan, buah di pohon,
# ternak di kandang.
#
# **Langkah:** YOLO mendeteksi kotak tiap objek → kotak kelas yang dicari dihitung per gambar.
# Dua ambang dicari dari data latih agar MAE terkecil: keyakinan minimum dan tinggi kotak minimum
# (orang kecil di latar belakang sering tidak ikut dihitung di label).
#
# **Data demo:** PennFudan (`jumlah` pejalan kaki per foto). Kelas COCO: 0 = person, 2 = car,
# dst. Mau fine-tune YOLO? Lihat `vision/deteksi`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from ultralytics import YOLO

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/regresi"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/pennfudan_train.csv"
TEST_CSV = f"{DATA_FOLDER}/pennfudan_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/pennfudan_test_labels.csv"   # hanya ada di data latihan
IMAGE_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/detection/PennFudanPed"
WEIGHTS = "Z:/GEMASTIK-DATABASE/pretrained/yolov8n/yolov8n.pt"

PATH_COL = "path"       # path gambar, relatif terhadap IMAGE_FOLDER
TARGET = "jumlah"
COCO_CLASSES = [0]      # kelas COCO yang dihitung (0 = person)
IMAGE_SIZE = 640        # 960 untuk objek kecil/jauh, 320 untuk lebih cepat
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = 0

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts().sort_index())
train.head()

# %% [markdown]
# ## Deteksi semua gambar sekali (ambang rendah 0.05), simpan tiap kotak
# Satu baris per kotak: path, keyakinan, tinggi kotak relatif terhadap tinggi gambar (0–1).

# %%
model = YOLO(WEIGHTS)


def detect_boxes(paths):
    rows = []
    for path in paths:
        image_results = model.predict(f"{IMAGE_FOLDER}/{path}", imgsz=IMAGE_SIZE, conf=0.05,
                                      classes=COCO_CLASSES, device=DEVICE, verbose=False)
        boxes = image_results[0].boxes
        image_height = image_results[0].orig_shape[0]
        for box, confidence in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
            box_height = (box[3] - box[1]) / image_height
            rows.append([path, confidence, box_height])
    return pd.DataFrame(rows, columns=["path", "conf", "tinggi_rel"])


train_boxes = detect_boxes(train[PATH_COL])
test_boxes = detect_boxes(test[PATH_COL])
print("kotak train:", len(train_boxes), "| kotak test:", len(test_boxes))
train_boxes.describe().round(3)

# %% [markdown]
# ## Hitung objek per gambar dengan ambang tertentu

# %%
def count_objects(boxes, paths, min_confidence, min_height):
    chosen = boxes[(boxes["conf"] >= min_confidence) & (boxes["tinggi_rel"] >= min_height)]
    per_image = chosen["path"].value_counts()
    counts = []
    for path in paths:
        counts.append(per_image.get(path, 0))     # gambar tanpa kotak = 0
    return np.array(counts)


default_counts = count_objects(train_boxes, train[PATH_COL], 0.25, 0.0)
print("MAE ambang bawaan (conf 0.25):", round(mean_absolute_error(train[TARGET],
                                                                   default_counts), 4))

# %% [markdown]
# ## Cari ambang keyakinan dan tinggi minimum dari data latih (MAE terkecil)

# %%
best_mae = 1e9
for min_confidence in np.arange(0.1, 0.81, 0.05):
    for min_height in [0.0, 0.1, 0.2, 0.3, 0.4]:
        counts = count_objects(train_boxes, train[PATH_COL], min_confidence, min_height)
        mae = mean_absolute_error(train[TARGET], counts)
        if mae < best_mae:
            best_mae = mae
            BEST_CONFIDENCE = min_confidence
            BEST_HEIGHT = min_height
print("ambang terbaik: conf", round(BEST_CONFIDENCE, 2), "| tinggi min", BEST_HEIGHT,
      "| MAE", round(best_mae, 4))

# %% [markdown]
# ## Evaluasi: asli vs hitungan + contoh kotak

# %%
train_counts = count_objects(train_boxes, train[PATH_COL], BEST_CONFIDENCE, BEST_HEIGHT)
print(pd.crosstab(train[TARGET], train_counts, rownames=["asli"], colnames=["hitung"]))

first_path = train.loc[0, PATH_COL]
first_result = model.predict(f"{IMAGE_FOLDER}/{first_path}", imgsz=IMAGE_SIZE,
                             conf=BEST_CONFIDENCE, classes=COCO_CLASSES, device=DEVICE,
                             verbose=False)
plt.imshow(first_result[0].plot()[:, :, ::-1])     # plot() memberi BGR → dibalik ke RGB
plt.title(f"asli {train.loc[0, TARGET]} orang")
plt.axis("off")
plt.show()

# %% [markdown]
# ## Prediksi test + submission (ambang dari data latih)

# %%
predictions = count_objects(test_boxes, test[PATH_COL], BEST_CONFIDENCE, BEST_HEIGHT)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["jumlah"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["jumlah"]
print("MAE test (kunci latihan):", round(mean_absolute_error(test_key, predictions), 4))
