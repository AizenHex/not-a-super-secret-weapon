# %% [markdown]
# # Teks → vektor angka, siap dipakai model tabular / kemiripan / clustering
#
# **Butuh:** tabel `train` dan `test` dengan satu kolom teks. **Menghasilkan:** tiga pilihan
# vektor + file di `outputs/`:
#
# | cara | hasil | kapan |
# |---|---|---|
# | TF-IDF (kata + huruf) | matriks sparse ±50.000 kolom | **coba duluan**: cepat, CPU |
# | MiniLM (sentence-transformers) | 384 angka per teks | makna kalimat, kemiripan |
# | IndoBERT (rata-rata token) | 768 angka per teks | teks Indonesia, sebelum fine-tune |
#
# Vektor ini hanya **dibaca** dari model pretrained (tanpa melatih). Melatih ulang BERT (lebih
# akurat, lebih lama): `teks/klasifikasi/finetune_bert.ipynb`.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi | **ya** |
# | 2 | TF-IDF kata + huruf → `train_tfidf`, `test_tfidf` | pilih ≥1 |
# | 3 | MiniLM → `train_vectors`, `test_vectors` + simpan `.npy` | pilih ≥1 |
# | 4 | IndoBERT rata-rata token → `train_bert`, `test_bert` + simpan | pilih ≥1 |
# | 5 | muat `.npy` (setelah kernel restart) | opsional |
# | 6 | vektor → kolom tabel `txt_0`, `txt_1`, ... | opsional |
#
# Lanjutan: `teks/klasifikasi/*`, `teks/kemiripan`, `teks/pengelompokan`,
# `jembatan/vektor_ke_model.ipynb`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Ulasan berlabel sentimen (positive, neutral, negative).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/sentiment/smsa_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/text/sentiment/smsa_test.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi
# Teks kosong (NaN) diganti `""` supaya tidak error.

# %%
import os

import torch

PRETRAINED_FOLDER = "pretrained"   # UBAH
TEXT = "text"                      # UBAH: nama kolom teks
BATCH_SIZE = 64                    # UBAH: "CUDA out of memory" → 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

train[TEXT] = train[TEXT].fillna("").astype(str)
test[TEXT] = test[TEXT].fillna("").astype(str)
print(DEVICE, train[TEXT].str.len().median(), "huruf per teks (median)")

# %% [markdown]
# ## 2. TF-IDF kata + potongan huruf → `train_tfidf`, `test_tfidf`
# Kata (1–2 kata berurutan) + potongan 2–5 huruf (tahan salah ketik & imbuhan: `makanannya`
# tetap mirip `makanan`). `fit` HANYA di train. Hasil sparse: pakai LogisticRegression /
# LinearSVC / Ridge, jangan diubah ke DataFrame (memori meledak).

# %%
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

word_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                             sublinear_tf=True)
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                             max_features=100000, sublinear_tf=True)
train_tfidf = hstack([word_tfidf.fit_transform(train[TEXT]),
                      char_tfidf.fit_transform(train[TEXT])]).tocsr()
test_tfidf = hstack([word_tfidf.transform(test[TEXT]),
                     char_tfidf.transform(test[TEXT])]).tocsr()
print(train_tfidf.shape, test_tfidf.shape)

# %% [markdown]
# ## 3. MiniLM (sentence-transformers) → `train_vectors`, `test_vectors` (384 angka)
# Model multibahasa (Indonesia + Inggris). `normalize_embeddings=True` → panjang 1, jadi
# perkalian titik = cosine similarity. Teks Inggris saja: `minilm_en` sedikit lebih baik.

# %%
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(f"{PRETRAINED_FOLDER}/minilm", device=DEVICE)   # UBAH: minilm_en
train_vectors = encoder.encode(train[TEXT].tolist(), batch_size=BATCH_SIZE,
                               normalize_embeddings=True, show_progress_bar=True)
test_vectors = encoder.encode(test[TEXT].tolist(), batch_size=BATCH_SIZE,
                              normalize_embeddings=True, show_progress_bar=True)
np.save("outputs/train_vectors.npy", train_vectors)
np.save("outputs/test_vectors.npy", test_vectors)
print(train_vectors.shape, test_vectors.shape)

# %% [markdown]
# ## 4. IndoBERT, rata-rata semua token → `train_bert`, `test_bert` (768 angka)
# `MAX_LENGTH`: token terpanjang (teks lebih panjang dipotong). Token padding tidak ikut dirata.

# %%
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MAX_LENGTH = 128   # UBAH: teks panjang → 256 / 512 (lebih lambat)
tokenizer = AutoTokenizer.from_pretrained(f"{PRETRAINED_FOLDER}/indobert")
bert = AutoModel.from_pretrained(f"{PRETRAINED_FOLDER}/indobert").to(DEVICE)
bert.eval()


def embed_bert(texts):
    vectors = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        tokens = tokenizer(texts[start:start + BATCH_SIZE], padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            hidden = bert(**tokens).last_hidden_state              # (teks, token, 768)
        mask = tokens["attention_mask"].unsqueeze(-1).float()      # 1 = token asli
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        vectors.append(mean.cpu().numpy())
    return np.concatenate(vectors)


print(embed_bert(train[TEXT].tolist()[:3]).shape)

# %% [markdown]
# Terapkan ke SEMUA train dan test + simpan.

# %%
import numpy as np

train_bert = embed_bert(train[TEXT].tolist())
test_bert = embed_bert(test[TEXT].tolist())
np.save("outputs/train_bert.npy", train_bert)
np.save("outputs/test_bert.npy", test_bert)
print(train_bert.shape, test_bert.shape)

# %% [markdown]
# ## 5. Muat vektor tersimpan (setelah kernel restart)
# Jalankan sel baca data + sel ini saja; sel 3–4 dilewati.

# %%
import numpy as np

train_vectors = np.load("outputs/train_vectors.npy")
test_vectors = np.load("outputs/test_vectors.npy")
print(train_vectors.shape, test_vectors.shape)

# %% [markdown]
# ## 6. Vektor → kolom tabel `txt_0`, `txt_1`, ... (digabung dengan kolom lain)
# **Menghasilkan:** `train_with_vectors`. Urutan baris harus sama dengan `train`.
# Lanjut: `jembatan/vektor_ke_model.ipynb`.

# %%
import pandas as pd

column_names = []
for i in range(train_vectors.shape[1]):
    column_names.append(f"txt_{i}")
vector_table = pd.DataFrame(train_vectors, columns=column_names)
train_with_vectors = pd.concat([train.reset_index(drop=True), vector_table], axis=1)
print(train_with_vectors.shape)
print(train_with_vectors.iloc[:3, :5])
