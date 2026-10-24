# %% [markdown]
# # Audio — Pengenalan Emosi dari Suara (Speech Emotion Recognition)
# **Kapan dipakai:** menebak emosi dari rekaman suara (penelepon call center, stres/depresi,
# kepuasan pelanggan).
#
# **Penting:** split harus **per pembicara**. Kalau aktor yang sama ada di train dan validasi,
# model menghafal suara orangnya dan skor validasi terlalu tinggi (dibuktikan di sel Evaluasi).
#
# **Langkah:**
# 1. Ubah tiap rekaman menjadi feature: embedding wav2vec2 satu lapisan (`LAYER`), atau MFCC.
# 2. Split per pembicara, latih model, evaluasi.
# 3. Latih ulang di semua data train, simpan submission.
#
# **Data demo:** RAVDESS (12 aktor), 8 emosi, kalimat yang diucapkan sama semua.

# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoFeatureExtractor, Wav2Vec2Model

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/audio/emosi"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/ravdess.csv"
PRETRAINED_FOLDER = "../../pretrained/wav2vec2_base"

ID_COL = "id"
PATH_COL = "path"           # kolom berisi nama file audio, relatif terhadap DATA_FOLDER
LABEL_COL = "emosi"
GROUP_COL = "speaker"       # validasi dipisah per kolom ini
TEST_SPEAKERS = ["aktor10", "aktor11", "aktor12"]   # data demo: aktor ini dijadikan test
SAMPLE_RATE = 16000
CLIP_LENGTH = 3 * SAMPLE_RATE   # 3 detik, setelah hening di awal dan akhir dibuang
LAYER = 8                   # lapisan wav2vec2 (0 = paling awal, 12 = terakhir), coba juga 4 atau 12
N_MFCC = 20
BATCH_SIZE = 16             # kecilkan bila memori GPU penuh
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data
# Data demo hanya satu file, jadi 3 aktor dipisah sebagai test.
# Saat lomba: ganti sel ini dengan `train = pd.read_csv(...)` dan `test = pd.read_csv(...)`.

# %%
data = pd.read_csv(DATA_CSV)
is_test = data[GROUP_COL].isin(TEST_SPEAKERS)
train = data[~is_test].reset_index(drop=True)
test = data[is_test].reset_index(drop=True)
test = test.drop(columns=LABEL_COL)
print(train.shape, test.shape)
print(train.isna().sum())
train.head()

# %% [markdown]
# ## EDA: jumlah per emosi dan contoh gelombang suara

# %%
print(train[LABEL_COL].value_counts())
print(pd.crosstab(train[LABEL_COL], train["intensitas"]))
example_audio, _ = librosa.load(f"{DATA_FOLDER}/{train[PATH_COL][0]}", sr=SAMPLE_RATE)
print("durasi contoh (detik):", len(example_audio) / SAMPLE_RATE)
seconds = np.arange(len(example_audio)) / SAMPLE_RATE
plt.plot(seconds, example_audio)
plt.title(train[LABEL_COL][0])
plt.xlabel("detik")
plt.show()

# %% [markdown]
# ## Langkah 1a — feature embedding wav2vec2 (satu lapisan)
# `output_hidden_states=True` membuat wav2vec2 mengembalikan semua lapisan; kita ambil lapisan
# `LAYER` lalu rata-rata di sumbu waktu.

# %%
feature_extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_FOLDER, local_files_only=True)
encoder = Wav2Vec2Model.from_pretrained(PRETRAINED_FOLDER, local_files_only=True)
encoder.eval()
encoder = encoder.to(DEVICE)


def load_clip(path):
    audio, _ = librosa.load(f"{DATA_FOLDER}/{path}", sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=30)   # buang hening di awal dan akhir
    audio = np.pad(audio, (0, CLIP_LENGTH))
    return audio[:CLIP_LENGTH]


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


# coba pada 2 rekaman
example = extract_embeddings(list(train[PATH_COL][:2]))
print("ukuran embedding 2 rekaman:", example.shape)

# %% [markdown]
# ## Langkah 1b — feature MFCC + energi + nada (alternatif tanpa GPU)
# Nada (pitch) dan energi (RMS) penting untuk emosi: marah = keras dan tinggi, sedih = pelan.

# %%
def extract_mfcc(paths):
    rows = []
    for path in paths:
        audio = load_clip(path)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        rms = librosa.feature.rms(y=audio)
        pitch = librosa.yin(audio, fmin=60, fmax=400, sr=SAMPLE_RATE)
        extra = [rms.mean(), rms.std(), np.median(pitch), pitch.std()]
        rows.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), extra]))
    return np.stack(rows)


# coba pada 2 rekaman
example = extract_mfcc(list(train[PATH_COL][:2]))
print("ukuran feature MFCC 2 rekaman:", example.shape)

# %% [markdown]
# ## Hitung feature semua rekaman (pilih satu pasang baris)

# %%
X = extract_embeddings(list(train[PATH_COL]))
X_test = extract_embeddings(list(test[PATH_COL]))
# X = extract_mfcc(list(train[PATH_COL]))          # tanpa GPU, lebih cepat, biasanya lebih lemah
# X_test = extract_mfcc(list(test[PATH_COL]))
y = train[LABEL_COL]
print(X.shape, X_test.shape)

# %% [markdown]
# ## Split per pembicara (aktor validasi tidak pernah dilihat saat train)

# %%
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
splits = splitter.split(X, y, groups=train[GROUP_COL])
train_rows, val_rows = next(splits)
print("aktor validasi:", sorted(train[GROUP_COL][val_rows].unique()))

# %% [markdown]
# ## Langkah 2 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
# model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = make_pipeline(StandardScaler(), LGBMClassifier(verbose=-1, random_state=SEED))

model.fit(X[train_rows], y[train_rows])

# %% [markdown]
# ## Evaluasi: split per aktor (jujur) vs split acak (bocor suara aktor)

# %%
pred_val = model.predict(X[val_rows])
print("F1 macro split per aktor:", f1_score(y[val_rows], pred_val, average="macro"))
print(classification_report(y[val_rows], pred_val))
ConfusionMatrixDisplay.from_predictions(y[val_rows], pred_val, normalize="true",
                                        values_format=".2f", xticks_rotation=45)
plt.show()

# pembanding: split acak, aktor yang sama bisa ada di train dan validasi
random_train, random_val = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y,
                                            random_state=SEED)
random_model = clone(model)
random_model.fit(X[random_train], y[random_train])
pred_random = random_model.predict(X[random_val])
print("F1 macro split acak (terlalu optimis):",
      f1_score(y[random_val], pred_random, average="macro"))

# %% [markdown]
# ## Langkah 3 — latih ulang di semua data + submission

# %%
model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = test[ID_COL]
submission["emosi"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# Label aktor test hanya ada di data demo.

# %%
test_key = data[is_test][LABEL_COL]
print("F1 macro test (kunci demo):", f1_score(test_key, predictions, average="macro"))
