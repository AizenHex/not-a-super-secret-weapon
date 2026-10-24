# %% [markdown]
# # Audio — Klasifikasi dengan CNN (log-mel spectrogram)
# **Kapan dipakai:** klasifikasi klip suara saat data cukup banyak (ribuan klip) dan ada GPU.
# Spectrogram diperlakukan seperti gambar, lalu CNN kecil belajar dari nol.
#
# **Langkah:**
# 1. Ubah tiap klip menjadi log-mel spectrogram (gambar 64 pita frekuensi x waktu).
# 2. Latih CNN kecil, evaluasi di data validasi.
# 3. Latih ulang di semua data train, simpan submission.
#
# **Data demo:** Speech Commands mini (8 kata). Epoch sengaja kecil supaya cepat.

# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn

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
PATH_COL = "path"
LABEL_COL = "label"
SAMPLE_RATE = 16000
CLIP_LENGTH = 16000         # 1 detik per klip
N_MELS = 64                 # jumlah pita frekuensi mel
EPOCHS = 20                 # naikkan bila loss masih turun di epoch terakhir
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
torch.manual_seed(SEED)

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
# ## Langkah 1 — log-mel spectrogram
# Nilai dinormalkan dengan mean dan std dari data train (dipakai juga untuk test).

# %%
def extract_mel(path):
    audio, _ = librosa.load(f"{DATA_FOLDER}/{path}", sr=SAMPLE_RATE)
    audio = np.pad(audio, (0, CLIP_LENGTH))     # tambah nol di belakang
    audio = audio[:CLIP_LENGTH]                 # lalu potong tepat 1 detik
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=512,
                                         hop_length=160)
    return librosa.power_to_db(mel)


# coba pada satu klip
print("ukuran spectrogram satu klip:", extract_mel(train[PATH_COL][0]).shape)

# %%
train_mels = []
for path in train[PATH_COL]:
    train_mels.append(extract_mel(path))
train_mels = np.stack(train_mels)

test_mels = []
for path in test[PATH_COL]:
    test_mels.append(extract_mel(path))
test_mels = np.stack(test_mels)

mel_mean = train_mels.mean()
mel_std = train_mels.std()
train_mels = (train_mels - mel_mean) / mel_std
test_mels = (test_mels - mel_mean) / mel_std
print(train_mels.shape, test_mels.shape)

# %% [markdown]
# ## Label teks → nomor kelas (torch butuh angka)

# %%
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train[LABEL_COL])
N_CLASSES = len(label_encoder.classes_)
print(label_encoder.classes_)

# %% [markdown]
# ## Split validasi dan ubah ke tensor
# `unsqueeze(1)` menambah sumbu channel: (jumlah klip, 1, pita mel, waktu), seperti gambar
# hitam-putih.

# %%
X_train, X_val, y_train, y_val = train_test_split(train_mels, y, test_size=0.2, stratify=y,
                                                  random_state=SEED)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_train_tensor = X_train_tensor.unsqueeze(1).to(DEVICE)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_val_tensor = X_val_tensor.unsqueeze(1).to(DEVICE)
X_all_tensor = torch.tensor(train_mels, dtype=torch.float32)
X_all_tensor = X_all_tensor.unsqueeze(1).to(DEVICE)
X_test_tensor = torch.tensor(test_mels, dtype=torch.float32)
X_test_tensor = X_test_tensor.unsqueeze(1).to(DEVICE)
y_train_tensor = torch.tensor(y_train).to(DEVICE)
y_all_tensor = torch.tensor(y).to(DEVICE)
print(X_train_tensor.shape)

# %% [markdown]
# ## Langkah 2a — bentuk CNN
# 3 lapis konvolusi (mencari pola kecil di spectrogram) → rata-rata global → lapisan akhir
# dengan satu angka per kelas. Fungsi dipakai dua kali: untuk validasi dan latih ulang.
# Alternatif tanpa melatih CNN: notebook `klasifikasi_embedding` (wav2vec2 + sklearn).

# %%
def build_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
        nn.Linear(64, N_CLASSES),
    )
    return model.to(DEVICE)


# coba: 2 klip masuk, keluar 2 baris x N_CLASSES angka
print(build_model()(X_train_tensor[:2]).shape)

# %% [markdown]
# ## Langkah 2b — latih
# Tiap epoch: data diacak, diproses per batch, bobot diperbarui supaya loss turun.

# %%
def train_model(X, y):
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(X), device=DEVICE)
        total_loss = 0.0
        for start in range(0, len(X), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            loss = loss_function(model(X[rows]), y[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(rows)
        print(f"epoch {epoch + 1}: loss {total_loss / len(X):.4f}")
    model.eval()
    return model


model = train_model(X_train_tensor, y_train_tensor)

# %% [markdown]
# ## Evaluasi (validasi dari data train)

# %%
with torch.no_grad():
    outputs = model(X_val_tensor)
pred_val = outputs.argmax(dim=1)          # kelas dengan skor tertinggi
pred_val = pred_val.cpu().numpy()
print("F1 macro:", f1_score(y_val, pred_val, average="macro"))
print(classification_report(y_val, pred_val, target_names=label_encoder.classes_))
ConfusionMatrixDisplay.from_predictions(y_val, pred_val, display_labels=label_encoder.classes_,
                                        xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Langkah 3 — latih ulang di semua data + submission

# %%
model = train_model(X_all_tensor, y_all_tensor)
with torch.no_grad():
    outputs = model(X_test_tensor)
class_numbers = outputs.argmax(dim=1)
class_numbers = class_numbers.cpu().numpy()
predictions = label_encoder.inverse_transform(class_numbers)   # nomor kelas -> nama kelas

submission = pd.DataFrame()
submission["id"] = test[ID_COL]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
