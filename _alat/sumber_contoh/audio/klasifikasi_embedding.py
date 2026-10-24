# %% [markdown]
# # Audio — Klasifikasi dengan Embedding wav2vec2
# **Kapan dipakai:** klasifikasi klip suara yang lebih sulit dari MFCC (ucapan, nada, aksen).
# Model wav2vec2 sudah belajar dari ribuan jam suara, jadi feature-nya jauh lebih kaya.
#
# **Idenya:** wav2vec2 (tanpa dilatih ulang) mengubah klip menjadi 768 angka (embedding),
# lalu model sklearn biasa belajar dari angka-angka itu.
#
# **Langkah:**
# 1. Ubah tiap klip menjadi embedding (rata-rata hidden state di sumbu waktu).
# 2. Latih model, evaluasi di data validasi.
# 3. Latih ulang di semua data train, simpan submission.
#
# **Data demo:** Speech Commands mini (8 kata).

# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoFeatureExtractor, Wav2Vec2Model

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/audio/classification"   # GANTI saat lomba
DATA_FOLDER = f"{DATA_FOLDER}/speech_commands_mini"
# Data speaker (label = ID speaker, kolom tetap `path` dan `label`):
# DATA_FOLDER = "../../../data_latihan/audio/speaker/speech_commands_speakers"
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
PRETRAINED_FOLDER = "../../../pretrained/wav2vec2_base"

ID_COL = "id"
PATH_COL = "path"           # kolom berisi nama file audio, relatif terhadap DATA_FOLDER
LABEL_COL = "label"
SAMPLE_RATE = 16000         # wav2vec2 dilatih dengan 16000 sampel per detik
CLIP_LENGTH = 16000         # 1 detik per klip
BATCH_SIZE = 32             # kecilkan bila memori GPU penuh
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print(train.shape, test.shape)
print(train.isna().sum())
train.head()

# %% [markdown]
# ## EDA: jumlah klip per kelas

# %%
class_counts = train[LABEL_COL].value_counts()
print(class_counts)
class_counts.plot.bar(title="Jumlah klip per kelas")
plt.show()

# %% [markdown]
# ## Langkah 1 — embedding wav2vec2
# `load_clip` membaca satu file dan menyamakan panjangnya. `extract_embeddings` memproses klip
# per batch: hidden state wav2vec2 (satu vektor per 20 ms) dirata-rata menjadi satu vektor.

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
example = extract_embeddings(list(train[PATH_COL][:2]))
print("ukuran embedding 2 klip:", example.shape)

# %%
X = extract_embeddings(list(train[PATH_COL]))
X_test = extract_embeddings(list(test[PATH_COL]))
y = train[LABEL_COL]
print(X.shape, X_test.shape)

# %% [markdown]
# ## Split validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                  random_state=SEED)

# %% [markdown]
# ## Langkah 2 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))   # kuat untuk embedding
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = make_pipeline(StandardScaler(), LGBMClassifier(verbose=-1, random_state=SEED))
# model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
# model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500,
#                                                                random_state=SEED))
# model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,),
#                                                       max_iter=1000, random_state=SEED))

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi (validasi dari data train)

# %%
pred_val = model.predict(X_val)
print("F1 macro:", f1_score(y_val, pred_val, average="macro"))
# soal 2 kelas: print("F1 biner:", f1_score(y_val, pred_val, pos_label="nama_kelas_positif"))
print(classification_report(y_val, pred_val))
ConfusionMatrixDisplay.from_predictions(y_val, pred_val, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Langkah 3 — latih ulang di semua data + submission

# %%
model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = test[ID_COL]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
