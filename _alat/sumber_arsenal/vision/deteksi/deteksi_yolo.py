# %% [markdown]
# # Deteksi kotak dengan YOLO (ultralytics): latih → evaluasi → prediksi → kotak terbaik
#
# **Butuh:** `data.yaml` dari `anotasi_tulis.ipynb` (sel "Dataset YOLO"), file .pt pretrained.
#
# **Menghasilkan (tersimpan di disk):**
# - `outputs/yolo_runs/latih/weights/best.pt`: model hasil latih
# - `outputs/pred_boxes.csv`: semua kotak prediksi (path, label, x1, y1, x2, y2, score)
# - `outputs/best_boxes.csv`: satu kotak terbaik per gambar per kelas → dipakai `potong_perbaiki`
#
# ## Cara menempel ke notebook-mu
# Salin **berurutan**, di bawah sel yang membuat `data.yaml`:
#
# | urutan | sel | wajib? | catatan |
# |---|---|---|---|
# | 1 | Konfigurasi | wajib | semua angka yang boleh diubah ada di sini |
# | 2 | Latih | sekali saja | ±30 menit. Sudah pernah? **Lewati**, `best.pt` sudah di disk |
# | 2b | Lanjutkan latih yang terputus | kalau perlu | laptop mati di tengah latih |
# | 3 | Evaluasi | dianjurkan | mAP di val |
# | 4 / 4a | Prediksi / Prediksi 4 rotasi | pilih satu | 4a: foto test bisa menyamping |
# | 5 | Kotak terbaik + simpan | wajib | menghasilkan `best_boxes` untuk dipotong |
# | 6 | Lihat hasil | dianjurkan | |
#
# **Kernel mati / buka notebook lagi besok:** jalankan sel 1, **lewati sel 2**, lanjut sel 3.
# Model dibaca dari `BEST_WEIGHTS` (path tetap di disk), jadi tidak perlu latih ulang.

# %% [markdown]
# ## 1. Konfigurasi
# `DEGREES`/`FLIPUD`: foto test terputar → naikkan. `BATCH` kecil = GPU aman (laptop tidak terlalu
# panas). `IMGSZ` naik bila objek kecil. `PATIENCE` = berhenti bila val tidak membaik sekian epoch.

# %%
import os

DATA_YAML = "outputs/yolo_dataset/data.yaml"   # UBAH
PRETRAINED = "pretrained/yolov8n/yolov8n.pt"   # UBAH: yolo26n / yolov8n / yolo11n .pt
EPOCHS = 30                                    # UBAH: 3 dulu untuk cek jalan
IMGSZ = 640                                    # UBAH
BATCH = 8                                      # UBAH
DEGREES = 15                                   # UBAH: rotasi acak ± derajat
FLIPUD = 0.0                                   # UBAH: peluang flip atas-bawah
PATIENCE = 10                                  # UBAH
PREDICT_FOLDER = "data/test"                   # UBAH: folder gambar yang diprediksi
CONF = 0.25                                    # UBAH: ambang skor, rendah = lebih banyak kotak
# absolut, supaya ultralytics tidak menaruh hasil di runs/detect/...
RUNS_FOLDER = os.path.abspath("outputs/yolo_runs")
BEST_WEIGHTS = f"{RUNS_FOLDER}/latih/weights/best.pt"
LAST_WEIGHTS = f"{RUNS_FOLDER}/latih/weights/last.pt"
print("model sudah ada:", os.path.exists(BEST_WEIGHTS))

# %% [markdown]
# ## 2. Latih (sekali saja)
# **Butuh:** sel 1. **Menghasilkan:** `BEST_WEIGHTS`, `LAST_WEIGHTS`.
# Menjalankan ulang sel ini = **menimpa** model lama. Colok charger + mode daya performa.

# %%
from ultralytics import YOLO

model = YOLO(PRETRAINED)
model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, degrees=DEGREES,
            flipud=FLIPUD, patience=PATIENCE, workers=0, project=RUNS_FOLDER,
            name="latih", exist_ok=True, seed=42, verbose=False)
print("bobot terbaik:", BEST_WEIGHTS)

# %% [markdown]
# ## 2b. (Kalau perlu) Lanjutkan latih yang terputus
# Laptop/kernel mati di tengah sel 2? Jangan jalankan sel 2 lagi (mulai dari epoch 1). Jalankan sel
# ini: lanjut dari epoch terakhir yang tersimpan di `last.pt`.

# %%
from ultralytics import YOLO

model = YOLO(LAST_WEIGHTS)
model.train(resume=True)

# %% [markdown]
# ## 3. Evaluasi di val (mAP per kelas)
# **Butuh:** sel 1 + `best.pt` di disk. mAP50 = kotak benar bila IoU ≥ 0.5.
# mAP50-95 = rata-rata IoU 0.5 … 0.95 (lebih ketat).

# %%
from ultralytics import YOLO

model = YOLO(BEST_WEIGHTS)
metrics = model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH, workers=0, verbose=False,
                    project=RUNS_FOLDER, name="val", exist_ok=True)
print("mAP50:", round(metrics.box.map50, 4), "| mAP50-95:", round(metrics.box.map, 4))
for number, name in model.names.items():
    print(f"  {name}: mAP50-95 = {metrics.box.maps[number]:.4f}")

# %% [markdown]
# ## 4. Prediksi satu folder → tabel `pred_boxes`
# **Butuh:** sel 1 + `best.pt`. **Menghasilkan:** `paths` (daftar gambar), `pred_boxes`.
# `AUGMENT = True`: TTA (flip + skala), lebih lambat, kadang lebih akurat.

# %%
import glob

import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

AUGMENT = False   # UBAH

model = YOLO(BEST_WEIGHTS)
paths = sorted(glob.glob(f"{PREDICT_FOLDER}/*"))
rows = []
for path in tqdm(paths):
    found = model.predict(path, imgsz=IMGSZ, conf=CONF, augment=AUGMENT, verbose=False)[0].boxes
    for corner, number, score in zip(found.xyxy.tolist(), found.cls.tolist(),
                                     found.conf.tolist()):
        rows.append([path, model.names[int(number)]] + corner + [score])
pred_boxes = pd.DataFrame(rows, columns=["path", "label", "x1", "y1", "x2", "y2", "score"])
print("gambar:", len(paths), "| kotak:", len(pred_boxes))
print(pred_boxes["label"].value_counts())

# %% [markdown]
# ## 4a. (Pengganti sel 4) Prediksi di 4 rotasi → kotak dikembalikan ke koordinat gambar asli
# Untuk foto test yang bisa menyamping/terbalik. Kolom `rotation` = putaran (×90° berlawanan jarum
# jam) yang memberi kotak itu; sel 5 memilih skor tertinggi dari semua rotasi. 4× lebih lambat.

# %%
def rotate_corners(corners, width, height, times):
    # sama dengan Image.rotate(90 * times, expand=True): titik (x, y) → (y, lebar - x) per 90°
    x1, y1, x2, y2 = corners
    for _ in range(times % 4):
        x1, y1, x2, y2 = y1, width - x2, y2, width - x1
        width, height = height, width
    return (x1, y1, x2, y2)


print(rotate_corners((10, 20, 30, 40), 100, 50, 1))   # → (20, 70, 40, 90)

# %%
import glob

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO

model = YOLO(BEST_WEIGHTS)
paths = sorted(glob.glob(f"{PREDICT_FOLDER}/*"))
rows = []
for path in tqdm(paths):
    image = Image.open(path).convert("RGB")
    for times in [0, 1, 2, 3]:
        rotated = image.rotate(90 * times, expand=True)
        found = model.predict(rotated, imgsz=IMGSZ, conf=CONF, verbose=False)[0].boxes
        for corner, number, score in zip(found.xyxy.tolist(), found.cls.tolist(),
                                         found.conf.tolist()):
            corners = rotate_corners(corner, *rotated.size, 4 - times)
            rows.append([path, model.names[int(number)], *corners, score, times])
pred_boxes = pd.DataFrame(rows, columns=["path", "label", "x1", "y1", "x2", "y2", "score",
                                         "rotation"])
print("gambar:", len(paths), "| kotak:", len(pred_boxes))

# %% [markdown]
# ## 5. Satu kotak terbaik per gambar per kelas + cadangan untuk gambar tanpa deteksi → simpan
# **Butuh:** `paths`, `pred_boxes` (sel 4 atau 4a). **Menghasilkan:** `best_boxes`,
# `outputs/best_boxes.csv`, `outputs/pred_boxes.csv`.
# Gambar yang tidak punya kotak kelas wajib → pakai seluruh gambar sebagai kotak (score 0).

# %%
import pandas as pd
from PIL import Image

REQUIRED_CLASSES = ["entete", "vin"]   # UBAH: kelas yang wajib ada di tiap gambar

best_boxes = pred_boxes.sort_values("score", ascending=False)
best_boxes = best_boxes.drop_duplicates(["path", "label"])
fallback_rows = []
for label in REQUIRED_CLASSES:
    found = set(best_boxes[best_boxes["label"] == label]["path"])
    for path in paths:
        if path not in found:
            width, height = Image.open(path).size
            fallback_rows.append({"path": path, "label": label, "x1": 0, "y1": 0,
                                  "x2": width, "y2": height, "score": 0.0})
print("kotak cadangan (seluruh gambar):", len(fallback_rows))
best_boxes = pd.concat([best_boxes, pd.DataFrame(fallback_rows)], ignore_index=True)
best_boxes = best_boxes.sort_values("path")
best_boxes.to_csv("outputs/best_boxes.csv", index=False)
pred_boxes.to_csv("outputs/pred_boxes.csv", index=False)
score_summary = best_boxes.groupby("label")["score"].describe()
print(score_summary.round(3))

# %% [markdown]
# **Kernel mati setelah sel 5?** Tidak perlu prediksi ulang:
# `best_boxes = pd.read_csv("outputs/best_boxes.csv")` lalu lanjut ke `potong_perbaiki.ipynb`.

# %% [markdown]
# ## 6. Lihat hasil prediksi (kotak + skor)
# **Butuh:** `best_boxes`.

# %%
import matplotlib.pyplot as plt
from PIL import Image

N_SHOW = 8   # UBAH

image_paths = best_boxes["path"].drop_duplicates()
preview_paths = image_paths.sample(N_SHOW, random_state=0)   # UBAH: random_state
fig, axes = plt.subplots(2, N_SHOW // 2, figsize=(4 * N_SHOW // 2, 8))
for ax, path in zip(axes.ravel(), preview_paths):
    ax.imshow(Image.open(path).convert("RGB"))
    for box in best_boxes[best_boxes["path"] == path].itertuples():
        ax.add_patch(plt.Rectangle((box.x1, box.y1), box.x2 - box.x1, box.y2 - box.y1,
                                   fill=False, color="red", linewidth=2))
        ax.text(box.x1, box.y1, f"{box.label} {box.score:.2f}", color="yellow", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
