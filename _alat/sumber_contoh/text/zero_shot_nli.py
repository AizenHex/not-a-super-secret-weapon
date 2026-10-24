# %% [markdown]
# # Teks — Klasifikasi Zero-shot (NLI, tanpa data latih)
# **Kapan dipakai:** label belum ada / sangat sedikit, atau kategorinya baru diumumkan saat
# lomba.
#
# **Cara kerja:** model NLI menilai "apakah teks ini mendukung kalimat *Sentimen teks ini
# positif*?" untuk tiap label, lalu label dengan skor tertinggi dipilih. Tidak ada pelatihan.
#
# **Data demo:** SmSA (sentimen). Model: `pretrained/nli_mdeberta` (mDeBERTa-v3 XNLI
# multibahasa, dari `unduh_model_tambahan.py`).

# %%
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline

# %% [markdown]
# ## Konfigurasi
# `LABEL_WORDS`: kata yang dibaca model (kiri) → label asli di data (kanan).

# %%
DATA_FOLDER = "../../../data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/smsa_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/smsa_test_labels.csv"   # hanya ada di data latihan
NLI_MODEL = "../../../pretrained/nli_mdeberta"

TEXT_COL = "text"
TARGET = "label"
LABEL_WORDS = {"positif": "positive", "negatif": "negative", "netral": "neutral"}
N_VALIDATION = 400     # zero-shot lambat di CPU, validasi cukup sebagian
SEED = 42
DEVICE = -1                    # -1 = CPU
if torch.cuda.is_available():
    DEVICE = 0                 # 0 = GPU pertama

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())
train.head()

# %% [markdown]
# ## Validasi kecil
# Zero-shot tidak dilatih, jadi train hanya dipakai untuk mengukur skor.

# %%
rest, validation = train_test_split(train, test_size=N_VALIDATION, stratify=train[TARGET],
                                    random_state=SEED)
print("validasi:", validation.shape)

# %% [markdown]
# ## Model
# `{}` diganti tiap kata di `LABEL_WORDS`. Kalimat hipotesis sangat memengaruhi skor.
# Pilih SATU kalimat: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
TEMPLATE = "Sentimen teks ini {}."
# TEMPLATE = "Penulis merasa {} tentang hal ini."
# TEMPLATE = "Ulasan ini bernada {}."
# TEMPLATE = "This text is {}."

nli = pipeline("zero-shot-classification", model=NLI_MODEL, device=DEVICE)


def zero_shot(texts):
    outputs = nli(texts, candidate_labels=list(LABEL_WORDS), hypothesis_template=TEMPLATE,
                  batch_size=16)
    labels = []
    for output in outputs:
        best_word = output["labels"][0]       # label urut dari skor tertinggi
        labels.append(LABEL_WORDS[best_word])
    return labels


# coba pada dua teks
print(zero_shot(["Makanannya enak sekali, pelayannya ramah.", "Pesanan datang terlambat."]))

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = zero_shot(list(validation[TEXT_COL]))
print("F1 macro:", round(f1_score(validation[TARGET], val_predictions, average="macro"), 4))
print(classification_report(validation[TARGET], val_predictions))
ConfusionMatrixDisplay.from_predictions(validation[TARGET], val_predictions)
plt.show()

# %% [markdown]
# ## Prediksi test + submission (tanpa latih ulang: zero-shot tidak dilatih)

# %%
predictions = zero_shot(list(test[TEXT_COL]))

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["label"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
