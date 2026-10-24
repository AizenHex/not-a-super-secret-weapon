# %% [markdown]
# # Vision — Anotasi Sendiri untuk Klasifikasi (embedding → klaster → beri nama)
# **Kapan dipakai:** ada banyak gambar **tanpa label** dan kita harus melabeli sendiri. Sudah ada
# label sebagian? Pakai `semi_supervised`. Sudah ada label lengkap? Pakai `klasifikasi_embedding`.
#
# **Langkah:** gambar → embedding CLIP → KMeans → CLIP menyarankan nama tiap klaster dari daftar
# `CANDIDATES` (bila soal menyebut daftar kelas) → lihat grid tiap klaster → betulkan nama di
# `CLUSTER_NAMES` → periksa gambar paling jauh dari pusat klaster, koreksi di `CORRECTIONS` →
# cek kualitas label dengan LogisticRegression → simpan CSV `id,label`.
#
# **Urutan kerja:** jalankan semua → lihat grid → isi `CLUSTER_NAMES` / `CORRECTIONS` di
# Konfigurasi → jalankan ulang. Nomor klaster tetap sama selama `SEED` dan gambar sama.
#
# **Data demo:** gambar `pets/train` (10 ras kucing/anjing, ada GIF/grayscale/RGBA). Label di
# CSV **tidak dipakai**; baru dipakai di sel kunci jawaban terakhir. Tanpa koreksi apa pun,
# akurasi label otomatis 0.845: anjing terpisah rapi, tetapi tiga ras kucing belang (Abyssinian,
# Bengal, Egyptian_Mau) bercampur di tiga klaster yang semuanya disarankan "Abyssinian". Kelas
# yang mirip seperti ini harus dilabeli manual (`CORRECTIONS`) atau pakai `N_CLUSTERS` lebih besar.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from torch.nn import functional
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi
# - `CANDIDATES`: daftar kelas dari soal → teks bahasa Inggris untuk CLIP. Tidak ada daftar
#   kelas? Isi `{}`; klaster diberi nama `klaster_0`, `klaster_1`, … lalu beri nama sendiri.
# - `CLUSTER_NAMES`: nama pilihanmu per nomor klaster, MENGALAHKAN saran CLIP. Dua klaster
#   boleh bernama sama (satu kelas terpecah dua).

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
IMAGE_FOLDER = f"{DATA_FOLDER}/train"             # semua subfolder ikut dicari
CLIP_FOLDER = "../../pretrained/clip_b32"
OUTPUT_FOLDER = "../../outputs/vision/anotasi_klasifikasi"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

N_CLUSTERS = 12       # boleh lebih banyak dari jumlah kelas (klaster kecil lebih murni)
PER_ROW = 8           # jumlah gambar per baris grid
BATCH_SIZE = 64
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

CANDIDATES = {"Abyssinian": "Abyssinian cat", "Bengal": "Bengal cat", "Persian": "Persian cat",
              "Siamese": "Siamese cat", "Egyptian_Mau": "Egyptian Mau cat", "beagle": "beagle",
              "pug": "pug", "shiba_inu": "shiba inu", "yorkshire_terrier": "yorkshire terrier",
              "staffordshire_bull_terrier": "staffordshire bull terrier"}
CLUSTER_NAMES = {}    # contoh setelah melihat grid: {3: "Bengal", 7: "Bengal"}
CORRECTIONS = {}      # koreksi per gambar: {"Bengal_12.jpg": "Abyssinian"}

# %% [markdown]
# ## Baca data (hanya daftar gambar, tanpa label)
# `**` + `recursive=True` = cari juga di dalam subfolder. Saring ekstensi gambar yang umum.

# %%
image_paths = []
for path in sorted(glob.glob(f"{IMAGE_FOLDER}/**/*", recursive=True)):
    if path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
        image_paths.append(path.replace("\\", "/"))
image_ids = []
for path in image_paths:
    image_ids.append(os.path.relpath(path, IMAGE_FOLDER).replace("\\", "/"))
print("jumlah gambar:", len(image_paths), "| contoh:", image_ids[0])

# %% [markdown]
# ## Embedding gambar (CLIP ViT-B/32 lokal)
# `convert("RGB")` menyamakan GIF/grayscale/RGBA. Tiap vektor dibuat panjang 1, jadi jarak
# KMeans = beda arah (cosine).

# %%
clip = CLIPModel.from_pretrained(CLIP_FOLDER, local_files_only=True)
processor = CLIPProcessor.from_pretrained(CLIP_FOLDER, local_files_only=True)
clip = clip.to(DEVICE)
clip.eval()


def image_vectors(paths):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for path in paths[start:start + BATCH_SIZE]:
            images.append(Image.open(path).convert("RGB"))
        encoded = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            batches.append(clip.get_image_features(**encoded).pooler_output.cpu())
    return functional.normalize(torch.cat(batches), dim=-1).numpy()


embeddings = image_vectors(image_paths)
print("ukuran embedding:", embeddings.shape)

# %% [markdown]
# ## KMeans
# `distances` = jarak tiap gambar ke pusat klasternya; jauh = kurang khas, mungkin salah klaster.

# %%
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
clusters = kmeans.fit_predict(embeddings)
distances = np.linalg.norm(embeddings - kmeans.cluster_centers_[clusters], axis=1)
print("ukuran klaster:", np.bincount(clusters))

sorted_rows = []     # nomor baris gambar tiap klaster, dari paling dekat ke paling jauh
for cluster in range(N_CLUSTERS):
    rows = np.where(clusters == cluster)[0]
    sorted_rows.append(rows[np.argsort(distances[rows])])

# %% [markdown]
# ## Saran nama klaster dari CLIP (pusat klaster vs teks kandidat)
# Ini hanya SARAN; tetap cek grid. `kemiripan` rendah (< 0.25) = saran meragukan.

# %%
names = {}
if len(CANDIDATES) > 0:
    prompts = []
    for text in CANDIDATES.values():
        prompts.append(f"a photo of a {text}.")
    encoded = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_vectors = clip.get_text_features(**encoded).pooler_output.cpu()
    text_vectors = functional.normalize(text_vectors, dim=-1).numpy()
    centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, axis=1,
                                                       keepdims=True)
    similarity = centers @ text_vectors.T
    for cluster in range(N_CLUSTERS):
        names[cluster] = list(CANDIDATES)[similarity[cluster].argmax()]
        print("klaster", cluster, "→", names[cluster], "| kemiripan",
              round(similarity[cluster].max(), 3), "| jumlah", (clusters == cluster).sum())
for cluster in CLUSTER_NAMES:
    names[cluster] = CLUSTER_NAMES[cluster]       # pilihan manual menang

# %% [markdown]
# ## Grid per klaster
# Tiap baris = satu klaster, urut dari gambar paling khas. Cek: apakah semua gambar satu baris
# memang kelas yang sama dengan namanya?

# %%
def show_grid(rows_per_cluster, file_name):
    fig, axes = plt.subplots(N_CLUSTERS, PER_ROW, figsize=(PER_ROW * 1.2, N_CLUSTERS * 1.4))
    for ax in axes.ravel():
        ax.axis("off")
    for cluster in range(N_CLUSTERS):
        name = names.get(cluster, f"klaster_{cluster}")
        axes[cluster, 0].set_title(f"klaster {cluster}: {name}", loc="left", fontsize=8)
        for column, row in enumerate(rows_per_cluster[cluster][:PER_ROW]):
            axes[cluster, column].imshow(Image.open(image_paths[row]).convert("RGB"))
    fig.savefig(f"{OUTPUT_FOLDER}/{file_name}", dpi=70, bbox_inches="tight")
    plt.show()


show_grid(sorted_rows, "grid_khas.png")

# %% [markdown]
# ## Beri label dari nama klaster, lalu terapkan koreksi manual

# %%
labels = []
for image_id, cluster in zip(image_ids, clusters):
    label = names.get(cluster, f"klaster_{cluster}")
    if image_id in CORRECTIONS:
        label = CORRECTIONS[image_id]
    labels.append(label)

table = pd.DataFrame()
table["id"] = image_ids
table["cluster"] = clusters
table["distance"] = distances
table["label"] = labels
print(table["label"].value_counts())

# %% [markdown]
# ## Periksa gambar paling JAUH dari pusat klaster (paling mungkin salah klaster)
# Salah? Tulis nama filenya di `CORRECTIONS` di sel Konfigurasi.

# %%
farthest_rows = []
for rows in sorted_rows:
    farthest_rows.append(rows[::-1])
show_grid(farthest_rows, "grid_terjauh.png")
table.sort_values("distance", ascending=False).head(10)

# %% [markdown]
# ## Cek kualitas label: LogisticRegression di embedding (cross-validation)
# Gambar yang label kita kurang diyakini model = kandidat label salah, periksa manual.

# %%
checker = LogisticRegression(max_iter=2000)
probabilities = cross_val_predict(checker, embeddings, table["label"], cv=5,
                                  method="predict_proba")
class_names = list(np.sort(table["label"].unique()))
checker_labels = []
own_label_confidence = []
for row, label in enumerate(table["label"]):
    checker_labels.append(class_names[probabilities[row].argmax()])
    own_label_confidence.append(probabilities[row, class_names.index(label)])
table["confidence"] = own_label_confidence
score = f1_score(table["label"], checker_labels, average="macro")
print("F1 macro (label kita vs LogisticRegression):", round(score, 3))
table.sort_values("confidence").head(10)

# %% [markdown]
# ## Simpan CSV label `id,label` (siap dipakai `klasifikasi_embedding` / `klasifikasi_cnn`)

# %%
table[["id", "label"]].to_csv(f"{OUTPUT_FOLDER}/label_awal.csv", index=False)
print("tersimpan:", f"{OUTPUT_FOLDER}/label_awal.csv")

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# Kelas asli di data demo ada di `train.csv`.

# %%
key = pd.read_csv(f"{DATA_FOLDER}/train.csv")
true_label = dict(zip(key["id"], key["label"]))
true_classes = []
for image_id in image_ids:
    true_classes.append(true_label[image_id])
print("akurasi label kita vs kelas asli:", round(accuracy_score(true_classes, table["label"]), 3))
pd.crosstab(pd.Series(true_classes, name="asli"), table["cluster"])

# %% [markdown]
# ## Masalah umum
# - **Satu klaster berisi dua kelas:** naikkan `N_CLUSTERS` (klaster lebih kecil, lebih murni),
#   lalu beri nama sama ke klaster yang satu kelas.
# - **Satu kelas terpecah ke beberapa klaster:** wajar (pose/warna berbeda); beri nama sama.
# - **Saran CLIP salah:** kelasnya istilah teknis/bahasa Indonesia. Terjemahkan di
#   `CANDIDATES`, atau abaikan saran dan isi `CLUSTER_NAMES` manual.
# - **Nomor klaster berubah setelah jalan ulang:** gambar atau `SEED`/`N_CLUSTERS` berubah;
#   isi ulang `CLUSTER_NAMES` dari grid baru.
