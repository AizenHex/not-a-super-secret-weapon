# %% [markdown]
# # Audio — Retrieval (cari klip yang paling mirip)
# **Kapan dipakai:** soal "untuk tiap rekaman query, cari rekaman paling mirip di galeri"
# (cari lagu/ucapan/suara serupa), atau sebagai klasifikasi tanpa latih model.
#
# **Idenya:** tiap klip diubah menjadi embedding wav2vec2. Dua klip mirip = embedding-nya
# searah (cosine similarity tinggi). Hasil = klip galeri dengan skor tertinggi.
#
# **Langkah:**
# 1. Embedding galeri dan query, lalu dinormalkan (panjang vektor = 1).
# 2. Cosine similarity = perkalian matriks; urutkan dari yang paling mirip.
# 3. Simpan hasil teratas. Evaluasi recall@k di sel terakhir (hanya data demo).
#
# **Data demo:** Speech Commands mini. Galeri = `train.csv`, query = `test.csv`.

# %%
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from transformers import AutoFeatureExtractor, Wav2Vec2Model

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/audio/classification"   # GANTI saat lomba
DATA_FOLDER = f"{DATA_FOLDER}/speech_commands_mini"
# Data speaker (label = ID speaker, kolom tetap `path` dan `label`):
# DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/audio/speaker/speech_commands_speakers"
GALLERY_CSV = f"{DATA_FOLDER}/train.csv"
QUERY_CSV = f"{DATA_FOLDER}/test.csv"
QUERY_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"    # kunci jawaban, hanya ada di data demo
PRETRAINED_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/wav2vec2_base"

ID_COL = "id"
PATH_COL = "path"
LABEL_COL = "label"
SAMPLE_RATE = 16000
CLIP_LENGTH = 16000         # 1 detik per klip
BATCH_SIZE = 32
TOP_K = 10                  # simpan 10 hasil teratas per query
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
gallery = pd.read_csv(GALLERY_CSV)
query = pd.read_csv(QUERY_CSV)
print(gallery.shape, query.shape)
gallery.head()

# %% [markdown]
# ## Langkah 1 — embedding wav2vec2

# %%
feature_extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_FOLDER, local_files_only=True)
encoder = Wav2Vec2Model.from_pretrained(PRETRAINED_FOLDER, local_files_only=True)
encoder.eval()
encoder = encoder.to(DEVICE)


def load_clip(path):
    audio, _ = librosa.load(f"{DATA_FOLDER}/{path}", sr=SAMPLE_RATE)
    audio = np.pad(audio, (0, CLIP_LENGTH))     # tambah nol di belakang
    return audio[:CLIP_LENGTH]                  # lalu potong tepat 1 detik


def extract_embeddings(paths):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        clips = []
        for path in paths[start:start + BATCH_SIZE]:
            clips.append(load_clip(path))
        inputs = feature_extractor(clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            output = encoder(inputs.input_values.to(DEVICE))
        mean_over_time = output.last_hidden_state.mean(dim=1)
        batches.append(mean_over_time.cpu().numpy())
    return np.concatenate(batches)


# coba pada 2 klip
example = extract_embeddings(list(gallery[PATH_COL][:2]))
print("ukuran embedding 2 klip:", example.shape)

# %%
gallery_embeddings = normalize(extract_embeddings(list(gallery[PATH_COL])))   # panjang = 1
query_embeddings = normalize(extract_embeddings(list(query[PATH_COL])))
print(gallery_embeddings.shape, query_embeddings.shape)

# %% [markdown]
# ## Langkah 2 — pencarian
# Karena panjang vektor = 1, perkalian matriks langsung = cosine similarity.
# `top_rows[i]` = nomor baris galeri yang paling mirip dengan query ke-i, urut dari termirip.

# %%
similarity = query_embeddings @ gallery_embeddings.T    # ukuran (jumlah query, jumlah galeri)
order = np.argsort(-similarity, axis=1)                 # minus = urut dari skor terbesar
top_rows = order[:, :TOP_K]
gallery_labels = gallery[LABEL_COL].to_numpy()
gallery_paths = gallery[PATH_COL].to_numpy()
top_labels = gallery_labels[top_rows]                   # label hasil, per query
print("query pertama, 5 label teratas:", top_labels[0, :5])

# %% [markdown]
# ## Langkah 3 — simpan hasil (satu baris per query, hasil teratas)

# %%
result = pd.DataFrame()
result["id"] = query[ID_COL]
result["top1_label"] = top_labels[:, 0]
result["top1_path"] = gallery_paths[top_rows[:, 0]]
result.to_csv("hasil.csv", index=False)
result.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# recall@k = persen query yang punya minimal satu hasil berlabel sama di k hasil teratas.
# Pembanding "acak" = peluang yang sama bila hasil dipilih acak.

# %%
query_key = pd.read_csv(QUERY_KEY_CSV)          # urutan baris sama dengan test.csv
query_labels = query_key[LABEL_COL].to_numpy()
chance_one = 1 / gallery[LABEL_COL].nunique()   # peluang satu hasil acak berlabel benar
for top_k in [1, 5, 10]:
    hits = 0
    for i in range(len(query_labels)):
        if query_labels[i] in top_labels[i, :top_k]:
            hits += 1
    recall = hits / len(query_labels)
    chance = 1 - (1 - chance_one) ** top_k
    print(f"recall@{top_k}: {recall:.4f} (acak: {chance:.4f})")
