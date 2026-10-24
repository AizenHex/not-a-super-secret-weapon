# %% [markdown]
# # Kemiripan gambar: cari yang paling mirip, pasangan mirip/tidak, gambar kembar
#
# **Butuh:** tabel gambar dengan kolom `path`. **Menghasilkan:** vektor ternormalisasi,
# tabel top-k, tebakan pasangan, kelompok gambar kembar.
#
# Semua bagian memakai ide yang sama: gambar → vektor (norma 1) → kemiripan cosine = perkalian
# titik (1 = identik, 0 = tidak berhubungan).
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | soal |
# |---|---|---|
# | 1 | `embed_images` (vektor norma 1) | **wajib** untuk 2–3 |
# | 2 | cari k gambar termirip untuk tiap query + recall@k | pencarian gambar / retrieval |
# | 3 | pasangan (A, B) → skor → ambang terbaik → mirip 0/1 | verifikasi wajah/produk sama |
# | 4 | gambar kembar persis / hampir kembar (hash) | bersihkan data, cek bocor train↔test |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

folder = f"{DEMO_FOLDER}/vision/pets"
gallery = pd.read_csv(f"{folder}/train.csv")
gallery["path"] = f"{folder}/train/" + gallery["id"]
pairs_train = pd.read_csv(f"{folder}/pasangan_train.csv")
pairs_test = pd.read_csv(f"{folder}/pasangan_test.csv")
for table in [pairs_train, pairs_test]:
    table["path_a"] = f"{folder}/" + table["gambar_a"]
    table["path_b"] = f"{folder}/" + table["gambar_b"]
print(pairs_train.head(3))

# %% [markdown]
# ## 1. `embed_images(paths)` → vektor norma 1
# Ganti backbone: `jembatan/gambar_ke_vektor.ipynb` sel 2. Untuk wajah/produk, CLIP (sel 6 di
# sana) sering lebih baik.

# %%
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

PRETRAINED_FOLDER = "pretrained"   # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backbone = models.convnext_tiny(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/convnext_tiny/model.pth"))
backbone.classifier[2] = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()
embed_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# %% [markdown]
# Fungsi `embed_images`: daftar path → array (jumlah gambar, 768), tiap baris norma 1.

# %%
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm


def embed_images(paths, batch_size=64):

    vectors = []
    for start in tqdm(range(0, len(paths), batch_size)):
        batch = []
        for path in paths[start:start + batch_size]:
            image = ImageOps.exif_transpose(Image.open(path))
            batch.append(embed_transform(image.convert("RGB")))
        with torch.no_grad():
            vectors.append(backbone(torch.stack(batch).to(DEVICE)).cpu().numpy())
    vectors = np.concatenate(vectors)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


print(embed_images(gallery["path"].tolist()[:2]).shape)

# %% [markdown]
# ## 2. k gambar termirip → `neighbors` (query, peringkat, path, skor)
# Di sini query = galeri itu sendiri, jadi peringkat 0 (dirinya sendiri) dibuang.
# Query dari folder lain: `query_vectors = embed_images(query["path"].tolist())`.
# recall@k = persen query yang punya minimal satu gambar sekelas di k teratas.

# %%
import numpy as np
import pandas as pd

TOP_K = 5   # UBAH

gallery_vectors = embed_images(gallery["path"].tolist())
query_vectors = gallery_vectors                      # UBAH
similarity = query_vectors @ gallery_vectors.T       # (jumlah query, jumlah galeri)
order = np.argsort(-similarity, axis=1)[:, 1:TOP_K + 1]   # UBAH: [:, :TOP_K] kalau query beda
rows = []
for query in range(len(order)):
    for rank in range(TOP_K):
        found = order[query, rank]
        rows.append({"query": query, "rank": rank + 1, "path": gallery["path"][found],
                     "score": similarity[query, found], "same_label":
                     gallery["label"][query] == gallery["label"][found]})
neighbors = pd.DataFrame(rows)
for top in [1, TOP_K]:
    hit = neighbors[neighbors["rank"] <= top].groupby("query")["same_label"].max()
    print(f"recall@{top}:", round(hit.mean(), 4))

# %% [markdown]
# Lihat: query (kiri) + hasil termirip.

# %%
import matplotlib.pyplot as plt

QUERY = 0   # UBAH
fig, axes = plt.subplots(1, TOP_K + 1, figsize=(2.5 * (TOP_K + 1), 3))
axes[0].imshow(Image.open(gallery["path"][QUERY]).convert("RGB"))
axes[0].set_title("query")
results = neighbors[neighbors["query"] == QUERY]
for ax, (_, row) in zip(axes[1:], results.iterrows()):
    ax.imshow(Image.open(row["path"]).convert("RGB"))
    ax.set_title(f"{row['score']:.2f}")
for ax in axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# ## 3. Pasangan (A, B) → skor kemiripan → ambang terbaik → `submission.csv`
# Ambang dicari di pasangan train yang berlabel. Fungsi `pair_scores` menghitung vektor sekali
# per gambar unik (gambar yang muncul di banyak pasangan tidak dihitung ulang).

# %%
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def pair_scores(table):
    unique_paths = sorted(set(table["path_a"]) | set(table["path_b"]))
    vectors = embed_images(unique_paths)
    position = dict(zip(unique_paths, range(len(unique_paths))))
    vectors_a = vectors[table["path_a"].map(position)]
    vectors_b = vectors[table["path_b"].map(position)]
    return (vectors_a * vectors_b).sum(axis=1)       # cosine per baris


train_scores = pair_scores(pairs_train)
print("AUC train:", round(roc_auc_score(pairs_train["mirip"], train_scores), 4))
best_threshold, best_f1 = 0.5, -1.0
for threshold in np.arange(0.1, 0.96, 0.01):
    score = f1_score(pairs_train["mirip"], train_scores >= threshold)
    if score > best_f1:
        best_threshold, best_f1 = threshold, score
print("ambang:", round(best_threshold, 2), "| F1:", round(best_f1, 4), "| akurasi:",
      round(accuracy_score(pairs_train["mirip"], train_scores >= best_threshold), 4))

# %% [markdown]
# Terapkan ke pasangan test. Skor mentah (`score`) kadang diminta langsung (metrik AUC).

# %%
import pandas as pd

test_scores = pair_scores(pairs_test)
submission = pd.DataFrame()
submission["id"] = pairs_test["id"]                                     # UBAH
submission["mirip"] = (test_scores >= best_threshold).astype(int)      # UBAH
submission.to_csv("submission.csv", index=False)
print(submission["mirip"].value_counts())

# %% [markdown]
# ## 4. Gambar kembar: persis (hash file) dan hampir kembar (perceptual hash) → `duplicates`
# `phash` tahan resize/kompresi ringan. Jarak hash 0 = sama, ≤ `MAX_DISTANCE` = hampir kembar.
# Pakai untuk: buang kembar di train, cek gambar test yang juga ada di train (bocor → salin
# labelnya), cek satu gambar berlabel beda (label berisik).

# %%
import hashlib

import imagehash
import pandas as pd
from PIL import Image

MAX_DISTANCE = 4   # UBAH

files = gallery[["path", "label"]].copy()
md5_list, phash_list = [], []
for path in tqdm(files["path"]):
    with open(path, "rb") as file:
        md5_list.append(hashlib.md5(file.read()).hexdigest())
    phash_list.append(imagehash.phash(Image.open(path)))
files["md5"] = md5_list
print("kembar persis:", files["md5"].duplicated().sum())
rows = []
for i in range(len(phash_list)):
    for other in range(i + 1, len(phash_list)):
        distance = phash_list[i] - phash_list[other]          # jumlah bit berbeda
        if distance <= MAX_DISTANCE:
            rows.append({"path_a": files["path"][i], "path_b": files["path"][other],
                         "distance": distance, "label_a": files["label"][i],
                         "label_b": files["label"][other]})
duplicates = pd.DataFrame(rows)
print("pasangan hampir kembar:", len(duplicates))
print(duplicates.head())
