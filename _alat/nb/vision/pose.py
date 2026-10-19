# %% [markdown]
# # Vision — Estimasi Pose (YOLOv8n-pose): 17 titik tubuh → fitur tabular
# **Kapan dipakai:** soal tentang orang dan gerakannya: klasifikasi aktivitas, deteksi jatuh,
# postur kerja, olahraga.
#
# **Langkah:** tiap orang mendapat 17 titik (hidung, mata, bahu, siku, pergelangan, pinggul,
# lutut, mata kaki) → titik diubah jadi **fitur angka** (sudut lutut, lebar langkah, tangan di
# atas bahu) → dipakai model tabular biasa. Contoh di akhir: KMeans "melangkah" vs "berdiri".
#
# **Data demo:** Penn-Fudan (pejalan kaki) + kunci jumlah orang dari `vision/regresi`.
# Bobot: `pretrained/yolov8n_pose/yolov8n-pose.pt` (COCO keypoints, zero-shot).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from ultralytics import YOLO

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/detection/PennFudanPed"   # GANTI
TABLE_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/regresi"
TRAIN_CSV = f"{TABLE_FOLDER}/pennfudan_train.csv"   # kolom path + jumlah orang
WEIGHTS = "Z:/GEMASTIK-DATABASE/pretrained/yolov8n_pose/yolov8n-pose.pt"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
COUNT_COL = "jumlah"
POINT_NAMES = ["hidung", "mata_ki", "mata_ka", "telinga_ki", "telinga_ka", "bahu_ki", "bahu_ka",
               "siku_ki", "siku_ka", "tangan_ki", "tangan_ka", "pinggul_ki", "pinggul_ka",
               "lutut_ki", "lutut_ka", "kaki_ki", "kaki_ka"]
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = 0

# %% [markdown]
# ## Baca data

# %%
images = pd.read_csv(TRAIN_CSV)
print(images.shape)
images.head()

# %% [markdown]
# ## Model + prediksi semua gambar
# Pilih SATU ambang keyakinan: hapus `#` di baris yang ingin dipakai.

# %%
model = YOLO(WEIGHTS)
MIN_CONFIDENCE = 0.5
# MIN_CONFIDENCE = 0.25     # lebih banyak orang terdeteksi, lebih banyak salah
IMAGE_SIZE = 640
# IMAGE_SIZE = 960          # orang kecil/jauh

results = []
for path in images[PATH_COL]:
    image_results = model.predict(f"{DATA_FOLDER}/{path}", conf=MIN_CONFIDENCE, imgsz=IMAGE_SIZE,
                                  device=DEVICE, verbose=False)
    results.append(image_results[0])

plt.imshow(results[0].plot()[:, :, ::-1])     # plot() memberi BGR → dibalik ke RGB
plt.axis("off")
plt.show()

# %% [markdown]
# ## Evaluasi jumlah orang

# %%
person_counts = []
for image_result in results:
    person_counts.append(len(image_result.boxes))
images["jumlah_pose"] = person_counts
print("MAE jumlah orang:", round(mean_absolute_error(images[COUNT_COL], images["jumlah_pose"]), 4))

# %% [markdown]
# ## Titik tubuh → tabel (satu baris per orang)
# Koordinat dinormalkan ke kotak orang (0–1), jadi tidak bergantung jauh/dekatnya orang.
# `angle(a, b, c)` = sudut di titik b (derajat), misal pinggul–lutut–kaki = sudut lutut.

# %%
def angle(point_a, point_b, point_c):
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
    cosine = (vector_ba * vector_bc).sum()
    cosine = cosine / (np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


print(angle(np.array([0, 0]), np.array([0, 1]), np.array([1, 1])))   # siku-siku = 90

# %% [markdown]
# Ubah semua orang di semua gambar jadi baris fitur.

# %%
rows = []
for path, image_result in zip(images[PATH_COL], results):
    boxes = image_result.boxes.xyxy.cpu().numpy()
    keypoints = image_result.keypoints.data.cpu().numpy()   # (orang, 17, [x, y, yakin])
    for box, points in zip(boxes, keypoints):
        left, top, right, bottom = box
        xy = (points[:, :2] - [left, top]) / [right - left, bottom - top]
        features = {"path": path, "tinggi_px": bottom - top, "yakin_titik": points[:, 2].mean()}
        features["sudut_lutut_ki"] = angle(xy[11], xy[13], xy[15])
        features["sudut_lutut_ka"] = angle(xy[12], xy[14], xy[16])
        features["lebar_langkah"] = abs(xy[15, 0] - xy[16, 0])
        hand_y = min(xy[9, 1], xy[10, 1])
        shoulder_y = min(xy[5, 1], xy[6, 1])
        features["tangan_di_atas_bahu"] = int(hand_y < shoulder_y)   # y kecil = lebih atas
        for i in range(len(POINT_NAMES)):
            features[POINT_NAMES[i] + "_x"] = xy[i, 0]
            features[POINT_NAMES[i] + "_y"] = xy[i, 1]
        rows.append(features)
people = pd.DataFrame(rows)
print(people.shape)
people.iloc[:, :7].describe().round(3)

# %% [markdown]
# ## Pemakaian fitur: kelompokkan postur tanpa label (KMeans 2 kelompok)

# %%
POSTURE_FEATURES = ["sudut_lutut_ki", "sudut_lutut_ka", "lebar_langkah"]
clear = people[people["yakin_titik"] > 0.6].copy()     # hanya orang yang titiknya jelas
kmeans = KMeans(n_clusters=2, n_init=10, random_state=SEED)
clear["kelompok"] = kmeans.fit_predict(clear[POSTURE_FEATURES])
group_means = clear.groupby("kelompok")[POSTURE_FEATURES].mean()
print(group_means.round(2))
print("→ sudut lutut kecil (menekuk) = sedang melangkah, mendekati 180 = berdiri tegak")

# %% [markdown]
# ## Hasil: tabel titik tubuh per orang (bahan model tabular)

# %%
people.to_csv("titik_tubuh.csv", index=False)

result = pd.DataFrame()
result["path"] = images[PATH_COL]
result["jumlah_orang"] = images["jumlah_pose"]
result.to_csv("hasil.csv", index=False)
result.head()
