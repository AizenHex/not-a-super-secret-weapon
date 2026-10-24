# %% [markdown]
# # Klasifikasi gambar TANPA data latih: CLIP zero-shot (gambar dibandingkan dengan kalimat)
#
# **Butuh:** `images` (kolom `path`), daftar nama kelas. **Menghasilkan:** kolom `clip_label`,
# `clip_score`, dan `submission.csv`.
#
# Kapan: label train tidak ada / sangat sedikit, atau untuk **label awal otomatis** yang lalu
# dicek mata (lebih cepat daripada melabeli dari nol). Kelas harus berupa benda/pemandangan umum
# yang bisa dijelaskan dengan kalimat bahasa Inggris. Tidak cocok untuk aksara/kode/tekstur khusus.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | muat CLIP + `clip_images` (vektor gambar) | **ya** |
# | 2 | kalimat per kelas → vektor teks | **ya** |
# | 3 | tebak SEMUA gambar (kelas dengan kemiripan tertinggi) | **ya** |
# | 4 | cek akurasi kalau sebagian gambar berlabel | disarankan |
# | 5 | `submission.csv` / simpan label awal untuk dicek | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

images = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/train.csv")
images["path"] = f"{DEMO_FOLDER}/vision/pets/train/" + images["id"]
images = images.head(300)

# %% [markdown]
# ## 1. Muat CLIP + fungsi `clip_images(paths)` → vektor gambar (norma 1)

# %%
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PRETRAINED_FOLDER = "pretrained"   # UBAH
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained(f"{PRETRAINED_FOLDER}/clip_b32").to(DEVICE)
processor = CLIPProcessor.from_pretrained(f"{PRETRAINED_FOLDER}/clip_b32")
clip.eval()


def clip_images(paths):
    vectors = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch = []
        for path in paths[start:start + BATCH_SIZE]:
            batch.append(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(DEVICE)
        with torch.no_grad():
            output = clip.get_image_features(pixel_values=pixels).pooler_output
        vectors.append(output.cpu().numpy())
    vectors = np.concatenate(vectors)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


print(clip_images(images["path"].tolist()[:2]).shape)

# %% [markdown]
# ## 2. Kalimat per kelas → `class_vectors`
# Beberapa kalimat per kelas lalu dirata-rata = lebih stabil. Nama kelas aneh/singkatan → tulis
# deskripsi bahasa Inggris yang jelas. `TEMPLATES` berisi `{}` = tempat nama kelas.

# %%
import numpy as np
import torch

CLASS_NAMES = sorted(images["label"].unique())   # UBAH: daftar kelas soal
DESCRIPTIONS = {}                                # UBAH: {"kelas": "deskripsi inggris"} opsional
TEMPLATES = ["a photo of a {}.", "a photo of a {}, a type of pet.", "a close-up photo of a {}."]

class_vectors = []
for class_name in CLASS_NAMES:
    readable = DESCRIPTIONS.get(class_name, class_name.replace("_", " "))
    sentences = []
    for template in TEMPLATES:
        sentences.append(template.format(readable))
    tokens = processor(text=sentences, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_vectors = clip.get_text_features(**tokens).pooler_output
    mean_vector = text_vectors.mean(dim=0)
    mean_vector = mean_vector.cpu().numpy()

    class_vectors.append(mean_vector / np.linalg.norm(mean_vector))
class_vectors = np.stack(class_vectors)
print(class_vectors.shape)

# %% [markdown]
# ## 3. Tebak SEMUA gambar → `images["clip_label"]`, `images["clip_score"]`
# Kemiripan = perkalian titik vektor gambar × vektor kelas. `clip_score` rendah = CLIP ragu,
# prioritaskan baris itu untuk dicek mata.

# %%
import numpy as np

image_vectors = clip_images(images["path"].tolist())
similarity = image_vectors @ class_vectors.T            # (jumlah gambar, jumlah kelas)
images["clip_label"] = np.array(CLASS_NAMES)[similarity.argmax(axis=1)]
images["clip_score"] = similarity.max(axis=1)
print(images[["path", "clip_label", "clip_score"]].head())

# %% [markdown]
# ## 4. Cek akurasi (hanya kalau sebagian gambar sudah berlabel)

# %%
from sklearn.metrics import accuracy_score, f1_score

print("akurasi:", round(accuracy_score(images["label"], images["clip_label"]), 4),
      "| F1 macro:", round(f1_score(images["label"], images["clip_label"], average="macro"), 4))
print(pd.crosstab(images["label"], images["clip_label"]))

# %% [markdown]
# ## 5. Simpan: `submission.csv` atau label awal untuk dicek mata
# Label awal → urutkan dari `clip_score` terendah, perbaiki kolom `clip_label` di Excel, lalu
# pakai sebagai label train untuk `embedding.ipynb` / `finetune.ipynb`.

# %%
import os

os.makedirs("outputs", exist_ok=True)
to_check = images.sort_values("clip_score")
to_check.to_csv("outputs/label_awal_clip.csv", index=False)

submission = pd.DataFrame()
submission["id"] = images["id"]               # UBAH
submission["label"] = images["clip_label"]    # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head())
