# %% [markdown]
# # Teks — Ujaran Kebencian & Bahasa Kasar Indonesia (multilabel + normalisasi alay)
# **Kapan dipakai:** tweet/komentar berbahasa Indonesia yang harus ditandai ujaran kebencian
# (hate speech), bahasa kasar (abusive), target (individu/kelompok), atau tingkat keparahan.
# Satu teks boleh punya banyak label sekaligus (multilabel).
#
# **Langkah:** normalisasi teks (huruf kecil, buang USER/URL, huruf berulang, kata alay → baku
# dengan kamus) → TF-IDF → satu LogisticRegression per label (OneVsRest) → ambang per label.
#
# **Data demo:** Ibrohim & Budi (2019), 13 ribu tweet Indonesia, 12 label (CC BY-NC-SA 4.0,
# non-komersial). Peringatan: teks asli berisi kata kasar dan SARA, hanya untuk riset deteksi.

# %%
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/ujaran_kebencian"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
SLANG_CSV = "Z:/GEMASTIK-DATABASE/data_latihan/preprocessing/kamus_alay_salsabila.csv"

TEXT_COL = "text"
TARGETS = ["HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion", "HS_Race",
           "HS_Physical", "HS_Gender", "HS_Other", "HS_Weak", "HS_Moderate", "HS_Strong"]
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGETS].sum())
labels_per_text = train[TARGETS].sum(axis=1)
print("rata-rata label per teks:", round(labels_per_text.mean(), 2))
train[[TEXT_COL, "HS", "Abusive"]].head()

# %% [markdown]
# ## Normalisasi teks
# Urutan: huruf kecil → buang token USER/URL/RT → sisa kode emoji → huruf berulang
# (`bangeeet` → `banget`) → buang simbol → kata alay/singkatan → baku (`gue`, `jgn`).
# Kamus: Salsabila dkk.

# %%
slang_table = pd.read_csv(SLANG_CSV)
slang_to_standard = dict(zip(slang_table["slang"], slang_table["baku"]))
print(len(slang_to_standard), "kata di kamus alay")


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\buser\b|\burl\b|\brt\b", " ", text)      # token anonim dataset
    text = re.sub(r"\\x[0-9a-f]{2}", " ", text)               # sisa kode emoji
    text = re.sub(r"(\w)\1{2,}", r"\1", text)                 # huruf berulang
    text = re.sub(r"[^a-z\s]", " ", text)                     # angka dan simbol
    words = []
    for word in text.split():
        words.append(slang_to_standard.get(word, word))      # kata alay -> baku
    return " ".join(words)


print(normalize_text("USER gue gak sukaaaa bgt sm dia!!! URL"))

train["bersih"] = train[TEXT_COL].apply(normalize_text)
test["bersih"] = test[TEXT_COL].apply(normalize_text)
train[[TEXT_COL, "bersih"]].head(4)

# %% [markdown]
# ## Split train / validasi

# %%
train_part, val_part = train_test_split(train, test_size=0.2, random_state=SEED,
                                        stratify=train["HS"])
print("train:", train_part.shape, "| validasi:", val_part.shape)

# %% [markdown]
# ## Model
# Satu LogisticRegression per label. `class_weight="balanced"` karena label seperti HS_Strong
# jarang. Pilih SATU vectorizer: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
# vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
#                              sublinear_tf=True)
model = OneVsRestClassifier(LogisticRegression(C=4, max_iter=2000, class_weight="balanced"))
pipe = make_pipeline(vectorizer, model)

TEXT_USED = "bersih"        # teks hasil normalisasi
# TEXT_USED = TEXT_COL      # teks mentah, untuk melihat sumbangan normalisasi
pipe.fit(train_part[TEXT_USED], train_part[TARGETS])

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(val_part[TEXT_USED])
print("F1 macro:", round(f1_score(val_part[TARGETS], val_predictions, average="macro"), 4))
print("F1 micro:", round(f1_score(val_part[TARGETS], val_predictions, average="micro"), 4))
print(classification_report(val_part[TARGETS], val_predictions, target_names=TARGETS,
                            zero_division=0))

# %% [markdown]
# ## Ambang per label
# Probabilitas > 0.5 tidak selalu terbaik. Cari ambang dengan F1 tertinggi tiap label di
# validasi.

# %%
val_proba = pipe.predict_proba(val_part[TEXT_USED])
thresholds = []
for label_number in range(len(TARGETS)):
    best_f1 = 0
    best_threshold = 0.5
    for candidate in np.arange(0.2, 0.8, 0.05):
        guess = val_proba[:, label_number] > candidate
        candidate_f1 = f1_score(val_part[TARGETS[label_number]], guess)
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_threshold = candidate
    thresholds.append(best_threshold)
thresholds = np.array(thresholds)

threshold_predictions = (val_proba > thresholds).astype(int)
print("ambang:", thresholds.round(2))
threshold_f1 = f1_score(val_part[TARGETS], threshold_predictions, average="macro")
print("F1 macro dengan ambang per label:", round(threshold_f1, 4))

# %% [markdown]
# ## Latih ulang di semua data + submission
# Satu kolom per label, ditambah satu per satu di dalam loop.

# %%
pipe.fit(train[TEXT_USED], train[TARGETS])
test_proba = pipe.predict_proba(test[TEXT_USED])
predictions = (test_proba > thresholds).astype(int)

submission = pd.DataFrame()
submission["id"] = test["id"]
for label_number in range(len(TARGETS)):
    submission[TARGETS[label_number]] = predictions[:, label_number]
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
print("F1 macro test:", round(f1_score(test_key[TARGETS], predictions, average="macro"), 4))
print("F1 HS test   :", round(f1_score(test_key["HS"], predictions[:, 0]), 4))
print("F1 Abusive test:", round(f1_score(test_key["Abusive"], predictions[:, 1]), 4))
