# %% [markdown]
# # Audio — Klasifikasi dengan Feature MFCC
# **Kapan dipakai:** klip suara pendek (kata, perintah, suara hewan/mesin) dan tiap klip punya
# satu label. Cara paling cepat dan ringan, tanpa GPU.
#
# **Idenya:** MFCC = ringkasan "warna suara" per potongan pendek. Tiap klip diringkas menjadi
# mean dan std 20 MFCC (40 angka), lalu model tabular biasa belajar dari angka-angka itu.
#
# **Langkah:**
# 1. Ubah tiap klip menjadi satu baris feature MFCC.
# 2. Latih model, evaluasi di data validasi.
# 3. Latih ulang di semua data train, simpan submission.
#
# **Data demo:** Speech Commands mini (8 kata: down, go, left, no, right, stop, up, yes).

# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/audio/classification"   # GANTI saat lomba
DATA_FOLDER = f"{DATA_FOLDER}/speech_commands_mini"
# Data speaker (label = ID speaker, kolom tetap `path` dan `label`):
# DATA_FOLDER = "../../../data_latihan/audio/speaker/speech_commands_speakers"
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"

ID_COL = "id"
PATH_COL = "path"           # kolom berisi nama file audio, relatif terhadap DATA_FOLDER
LABEL_COL = "label"
SAMPLE_RATE = 16000         # jumlah sampel per detik
CLIP_LENGTH = 16000         # 1 detik per klip: lebih pendek diberi nol, lebih panjang dipotong
N_MFCC = 20
SEED = 42

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
# ## Langkah 1 — satu klip menjadi satu baris feature MFCC

# %%
def extract_mfcc(path):
    audio, _ = librosa.load(f"{DATA_FOLDER}/{path}", sr=SAMPLE_RATE)
    audio = np.pad(audio, (0, CLIP_LENGTH))     # tambah nol di belakang
    audio = audio[:CLIP_LENGTH]                 # lalu potong tepat 1 detik
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


# coba pada satu klip
example = extract_mfcc(train[PATH_COL][0])
print("jumlah feature per klip:", len(example))

# %%
train_rows = []
for path in train[PATH_COL]:
    train_rows.append(extract_mfcc(path))
X = np.stack(train_rows)

test_rows = []
for path in test[PATH_COL]:
    test_rows.append(extract_mfcc(path))
X_test = np.stack(test_rows)

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
# StandardScaler menyamakan skala feature (wajib untuk SVC, LogReg, KNN, MLP).

# %%
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500,
                                                               random_state=SEED))
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))    # sering terbaik di MFCC
# model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))   # baseline cepat
# model = make_pipeline(StandardScaler(), LGBMClassifier(random_state=SEED, verbose=-1))
# model = make_pipeline(StandardScaler(), ExtraTreesClassifier(n_estimators=500,
#                                                              random_state=SEED))
# model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
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
