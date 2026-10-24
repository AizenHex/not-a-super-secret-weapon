# %% [markdown]
# # Anomali / cacat gambar: belajar dari gambar NORMAL saja, lalu beri skor "seberapa aneh"
#
# **Butuh:** `normal` (kolom `path`, gambar normal saja), `test` (kolom `path`). Opsional:
# beberapa gambar cacat berlabel untuk memilih ambang. **Menghasilkan:** `test["score"]`,
# `test["pred"]` (0 normal / 1 cacat), peta panas lokasi cacat, `submission.csv`.
#
# Kapan: contoh cacat sangat sedikit/tidak ada (inspeksi produk, kain, PCB, jalan retak).
# Kalau contoh cacat banyak dan jenisnya tetap → klasifikasi biasa (`vision/klasifikasi`).
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + backbone ResNet18 (ambil peta fitur tengah) | **ya** |
# | 2 | `patch_features`: gambar → fitur per potongan kecil (patch) | **ya** |
# | 3 | bank memori dari gambar normal | **ya** |
# | 4 | `anomaly_map`: skor per patch = jarak ke patch normal terdekat | **ya** |
# | 5 | skor SEMUA val/test (skor gambar = patch paling aneh) | **ya** |
# | 6 | ambang: dari contoh berlabel (F1 terbaik) atau persentil normal | **ya** |
# | 7 | `submission.csv` + lihat peta panas | **ya** |
#
# Metode = versi sederhana PatchCore. Cacat kecil (goresan, titik) tetap terdeteksi karena skor
# dihitung per patch, bukan per gambar utuh.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Botol MVTec: train = normal + sedikit cacat (dipakai untuk memilih ambang), test campuran.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

folder = f"{DEMO_FOLDER}/vision/defect_mvtec_bottle"
train = pd.read_csv(f"{folder}/train.csv")
train["path"] = f"{folder}/" + train["path"]
test = pd.read_csv(f"{folder}/test.csv")
test["path"] = f"{folder}/" + test["path"]
normal = train[train["label"] == "normal"].sample(frac=0.8, random_state=42)
val = train.drop(normal.index)            # sisa normal + semua cacat → untuk ambang
val["target"] = (val["label"] == "cacat").astype(int)
print(len(normal), "normal |", val["target"].value_counts().to_dict(), "val |", len(test), "test")

# %% [markdown]
# ## 1. Konfigurasi + backbone ResNet18 → `extractor`
# Lapisan tengah (`layer2`, `layer3`) menyimpan tekstur dan bentuk lokal, cocok untuk cacat.

# %%
import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

PRETRAINED_FOLDER = "pretrained"   # UBAH
IMAGE_SIZE = 224                   # UBAH: cacat sangat kecil → 320
MEMORY_SIZE = 20000                # UBAH: jumlah patch normal yang disimpan (besar = lambat)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

resnet = models.resnet18(weights=None)
resnet.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/resnet18/model.pth"))
extractor = create_feature_extractor(resnet, {"layer2": "mid", "layer3": "deep"})
extractor = extractor.to(DEVICE)
extractor.eval()
anomaly_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# %% [markdown]
# ## 2. `patch_features(path)` → (jumlah patch, panjang fitur) + ukuran grid patch
# Fitur `layer3` diperbesar ke ukuran grid `layer2`, digabung, lalu dirata-rata dengan
# tetangga 3×3 (patch jadi sedikit lebih "tahu konteks").

# %%
import torch
from PIL import Image
from torch.nn import functional


def patch_features(path):
    image = anomaly_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = extractor(image)
    mid = functional.avg_pool2d(output["mid"], 3, stride=1, padding=1)
    deep = functional.avg_pool2d(output["deep"], 3, stride=1, padding=1)
    deep = functional.interpolate(deep, size=mid.shape[-2:], mode="bilinear")
    features = torch.cat([mid, deep], dim=1)[0]            # (fitur, tinggi grid, lebar grid)
    grid = features.shape[1:]
    return features.reshape(features.shape[0], -1).T, grid  # (patch, fitur)


example, grid = patch_features(normal["path"].iloc[0])
print(example.shape, "grid:", tuple(grid))

# %% [markdown]
# ## 3. Bank memori dari gambar normal → `memory`
# Semua patch normal dikumpulkan lalu diambil acak `MEMORY_SIZE` (supaya cepat).

# %%
import torch
from tqdm import tqdm

all_patches = []
for path in tqdm(normal["path"]):
    patches, _ = patch_features(path)
    all_patches.append(patches)
all_patches = torch.cat(all_patches)
generator = torch.Generator().manual_seed(42)
chosen = torch.randperm(len(all_patches), generator=generator)[:MEMORY_SIZE]
memory = all_patches[chosen.to(all_patches.device)]
print("patch normal:", len(all_patches), "→ disimpan:", memory.shape)

# %% [markdown]
# ## 4. `anomaly_map(path)` → skor per patch (array tinggi × lebar grid)
# Skor patch = jarak ke patch normal paling dekat. Normal → dekat (skor kecil).

# %%
import torch


def anomaly_map(path):
    patches, grid = patch_features(path)
    distances = torch.cdist(patches, memory)               # (patch, MEMORY_SIZE)
    nearest = distances.min(dim=1).values
    nearest = nearest.reshape(grid[0], grid[1])
    return nearest.cpu().numpy()


example_map = anomaly_map(val["path"].iloc[0])
print(example_map.shape, "skor gambar:", round(float(example_map.max()), 3))

# %% [markdown]
# ## 5. Skor SEMUA val dan test → kolom `score` (= patch paling aneh)

# %%
from tqdm import tqdm


def image_scores(paths):
    scores = []
    for path in tqdm(paths):
        scores.append(float(anomaly_map(path).max()))
    return scores


val["score"] = image_scores(val["path"])
test["score"] = image_scores(test["path"])
summary = val.groupby("label")["score"].describe()
print(summary.round(3))   # skor cacat harus jelas lebih tinggi dari normal

# %% [markdown]
# ## 6. Ambang → `threshold`
# (a) Ada contoh cacat berlabel: pilih ambang dengan F1 terbaik (aktif).
# (b) Tidak ada: persentil skor gambar normal, mis. 99 → ±1% normal ikut dianggap cacat.

# %%
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

print("AUC val:", round(roc_auc_score(val["target"], val["score"]), 4))
threshold, best_f1 = 0.0, -1.0
for candidate in np.unique(val["score"]):
    score = f1_score(val["target"], val["score"] >= candidate)
    if score > best_f1:
        threshold, best_f1 = candidate, score
print("ambang (a):", round(threshold, 3), "| F1 val:", round(best_f1, 4))
# threshold = np.percentile(image_scores(normal["path"].head(50)), 99)   # (b), UBAH

# %% [markdown]
# ## 7. `submission.csv` + peta panas (merah = bagian yang dianggap cacat)

# %%
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

test["pred"] = (test["score"] >= threshold).astype(int)
submission = pd.DataFrame()
submission["id"] = test["id"]                                              # UBAH
submission["label"] = test["pred"].map({0: "normal", 1: "cacat"})          # UBAH
submission.to_csv("submission.csv", index=False)
print(submission["label"].value_counts())

worst = test.sort_values("score", ascending=False).head(4)
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, path, score in zip(axes, worst["path"], worst["score"]):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    heat = Image.fromarray(anomaly_map(path)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    ax.imshow(image)
    ax.imshow(np.array(heat), cmap="jet", alpha=0.4)
    ax.set_title(f"skor {score:.2f}")
    ax.axis("off")
plt.show()
