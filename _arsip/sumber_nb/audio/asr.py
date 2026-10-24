# %% [markdown]
# # Audio — ASR (ubah suara menjadi teks) dengan Whisper
# **Kapan dipakai:** soal transkripsi: tiap rekaman ucapan harus ditulis ulang sebagai teks.
# Tanpa melatih: model Whisper lokal dipakai apa adanya.
#
# **Langkah:**
# 1. Muat Whisper-tiny lokal, coba pada beberapa klip.
# 2. Transkripsi semua klip test, rapikan teks (huruf kecil, buang tanda baca).
# 3. Simpan submission. Evaluasi WER/CER di sel terakhir (hanya data demo).
#
# **Data demo:** Speech Commands mini, tiap klip satu kata (label asli ada di `test_labels.csv`).

# %%
import re

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/audio/classification"   # GANTI saat lomba
DATA_FOLDER = f"{DATA_FOLDER}/speech_commands_mini"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"    # kunci jawaban, hanya ada di data demo
WHISPER_FOLDER = "../../pretrained/whisper_tiny"

ID_COL = "id"
PATH_COL = "path"
SAMPLE_RATE = 16000         # Whisper dilatih dengan 16000 sampel per detik
CLIP_LENGTH = 16000         # 1 detik per klip
LANGUAGE = "en"             # "id" untuk bahasa Indonesia
BATCH_SIZE = 16
MAX_TOKENS = 10             # panjang teks maksimal per klip (naikkan untuk kalimat panjang)
KEEP_PATTERN = r"[^a-z ]"   # buang semua karakter selain huruf kecil dan spasi
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
test = pd.read_csv(TEST_CSV)
print(test.shape)
test.head()

# %% [markdown]
# ## Langkah 1 — Whisper lokal
# `processor` mengubah suara menjadi input Whisper dan token menjadi teks.
# `model.generate` menulis teks token demi token.

# %%
processor = WhisperProcessor.from_pretrained(WHISPER_FOLDER, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_FOLDER, local_files_only=True)
model.eval()
model = model.to(DEVICE)


def load_clip(path):
    audio, _ = librosa.load(f"{DATA_FOLDER}/{path}", sr=SAMPLE_RATE)
    audio = np.pad(audio, (0, CLIP_LENGTH))     # tambah nol di belakang
    return audio[:CLIP_LENGTH]                  # lalu potong tepat 1 detik


def transcribe(paths):
    texts = []
    for start in range(0, len(paths), BATCH_SIZE):
        clips = []
        for path in paths[start:start + BATCH_SIZE]:
            clips.append(load_clip(path))
        inputs = processor(clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = inputs.input_features.to(DEVICE)
        with torch.no_grad():
            token_ids = model.generate(features, language=LANGUAGE, task="transcribe",
                                       max_new_tokens=MAX_TOKENS)
        texts += processor.batch_decode(token_ids, skip_special_tokens=True)
    return texts


# coba pada 3 klip
print(transcribe(list(test[PATH_COL][:3])))

# %% [markdown]
# ## Langkah 2 — transkripsi semua klip, lalu rapikan teks
# Whisper menulis "Yes." atau " yes!", padahal jawaban "yes". Teks dibuat huruf kecil dan
# tanda baca dibuang supaya bisa dibandingkan.

# %%
raw_texts = transcribe(list(test[PATH_COL]))
clean_texts = []
for text in raw_texts:
    text = text.lower()
    text = re.sub(KEEP_PATTERN, "", text)
    clean_texts.append(text.strip())
print(raw_texts[:6])
print(clean_texts[:6])

# %% [markdown]
# ## Langkah 3 — submission
# Tidak ada latih ulang. Kolom `text` berisi transkripsi yang sudah dirapikan.

# %%
submission = pd.DataFrame()
submission["id"] = test[ID_COL]
submission["text"] = clean_texts
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# - **WER** (word error rate) = kata yang salah / jumlah kata jawaban. 0 = sempurna.
# - **CER** (character error rate) = sama, tapi per huruf.
# Tabel di bawah = contoh klip yang salah ditranskripsi.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)        # urutan baris sama dengan test.csv
truth = []
for text in test_key["label"]:
    truth.append(text.lower())
print("WER:", round(jiwer.wer(truth, clean_texts), 4))
print("CER:", round(jiwer.cer(truth, clean_texts), 4))

check = pd.DataFrame()
check["truth"] = truth
check["whisper"] = raw_texts
check["clean"] = clean_texts
check.to_csv("transkrip.csv", index=False)
check[check["truth"] != check["clean"]].head(10)
