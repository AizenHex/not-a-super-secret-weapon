# %% [markdown]
# # Audio — Identifikasi Pembicara (siapa yang bicara, bukan kata apa)
# **Kapan dipakai:** absensi/verifikasi suara, memisahkan rekaman per penelepon, tebak
# gender/usia dari suara.
#
# **Idenya:** lapisan tengah wav2vec2 lebih banyak menyimpan "warna suara" orang, lapisan akhir
# lebih menyimpan "isi kata". Jadi kita ambil satu lapisan (`LAYER`) sebagai feature.
#
# **Langkah:**
# 1. Ubah tiap klip menjadi embedding dari lapisan `LAYER` (atau MFCC).
# 2. Latih model, evaluasi; bonus: cek verifikasi "dua klip ini orang yang sama?".
# 3. Latih ulang di semua data train, simpan submission.
#
# **Data demo:** Speech Commands, 10 pembicara, 15 klip 1 detik per orang, kata beragam.

# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from transformers import AutoFeatureExtractor, Wav2Vec2Model

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/audio/speaker"   # GANTI saat lomba
DATA_FOLDER = f"{DATA_FOLDER}/speech_commands_speakers"
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"    # kunci jawaban, hanya ada di data demo
PRETRAINED_FOLDER = "../../pretrained/wav2vec2_base"

ID_COL = "id"
PATH_COL = "path"
LABEL_COL = "label"         # ID pembicara
SAMPLE_RATE = 16000
CLIP_LENGTH = 16000         # 1 detik per klip
LAYER = 6                   # lapisan wav2vec2 (0 = paling awal, 12 = terakhir), coba juga 3 atau 12
N_MFCC = 20
BATCH_SIZE = 32
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
# ## EDA: klip per pembicara dan kata yang diucapkan
# Nama file demo = "kata_nomor.wav", jadi kata diambil dari bagian sebelum "_".

# %%
print(train[LABEL_COL].value_counts())
words = []
for path in train[PATH_COL]:
    file_name = path.split("/")[-1]
    words.append(file_name.split("_")[0])
train["word"] = words
print(pd.crosstab(train[LABEL_COL], train["word"]))

# %% [markdown]
# ## Langkah 1a — feature embedding wav2vec2 (satu lapisan)

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
            output = encoder(inputs.input_values.to(DEVICE), output_hidden_states=True)
        mean_over_time = output.hidden_states[LAYER].mean(dim=1)
        batches.append(mean_over_time.cpu().numpy())
    return np.concatenate(batches)


# coba pada 2 klip
example = extract_embeddings(list(train[PATH_COL][:2]))
print("ukuran embedding 2 klip:", example.shape)

# %% [markdown]
# ## Langkah 1b — feature MFCC (alternatif tanpa GPU)

# %%
def extract_mfcc(paths):
    rows = []
    for path in paths:
        mfcc = librosa.feature.mfcc(y=load_clip(path), sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        rows.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))
    return np.stack(rows)


# coba pada 2 klip
example = extract_mfcc(list(train[PATH_COL][:2]))
print("ukuran feature MFCC 2 klip:", example.shape)

# %% [markdown]
# ## Hitung feature semua klip (pilih satu pasang baris)

# %%
X = extract_embeddings(list(train[PATH_COL]))
X_test = extract_embeddings(list(test[PATH_COL]))
# X = extract_mfcc(list(train[PATH_COL]))
# X_test = extract_mfcc(list(test[PATH_COL]))
y = train[LABEL_COL]
print(X.shape, X_test.shape)

# %% [markdown]
# ## Split validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y,
                                                  random_state=SEED)

# %% [markdown]
# ## Langkah 2 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = KNeighborsClassifier(n_neighbors=3, metric="cosine")   # tetangga dengan suara termirip
# model = make_pipeline(StandardScaler(), LGBMClassifier(verbose=-1, random_state=SEED))
# model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi (validasi dari data train)

# %%
pred_val = model.predict(X_val)
print("akurasi:", accuracy_score(y_val, pred_val))
print("F1 macro:", f1_score(y_val, pred_val, average="macro"))
ConfusionMatrixDisplay.from_predictions(y_val, pred_val, xticks_rotation=90)
plt.show()

# %% [markdown]
# ## Bonus verifikasi: "dua klip ini orang yang sama?" (kemiripan cosine)
# Tiap pasangan klip validasi diberi skor kemiripan. AUC tinggi = pasangan orang sama punya
# kemiripan lebih besar dari pasangan orang berbeda.

# %%
centered = X_val - X_train.mean(axis=0)        # geser ke tengah supaya cosine lebih adil
val_embeddings = normalize(centered)
similarity = val_embeddings @ val_embeddings.T
val_labels = y_val.to_numpy()
pair_scores = []
pair_same = []
for i in range(len(val_labels)):
    for other in range(i + 1, len(val_labels)):   # tiap pasangan dihitung sekali
        pair_scores.append(similarity[i, other])
        pair_same.append(val_labels[i] == val_labels[other])
print("AUC verifikasi (cosine):", roc_auc_score(pair_same, pair_scores))

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

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
print("F1 macro test (kunci demo):", f1_score(test_key[LABEL_COL], predictions, average="macro"))
