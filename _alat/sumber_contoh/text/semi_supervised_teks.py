# %% [markdown]
# # Teks — Semi-supervised: label sangat sedikit + banyak teks tanpa label
# **Kapan dipakai:** pola soal GEMASTIK 2012: 2000 dokumen, hanya ±100 yang berlabel.
#
# **Cara:**
# 1. TF-IDF di-fit pada SEMUA teks (berlabel + tanpa label + test) → kosakata lebih kaya.
# 2. Pseudo-label: tebakan yang paling yakin dijadikan label tambahan, diulang beberapa putaran.
#
# Dipakai "N teratas per kelas", bukan ambang probabilitas: dengan banyak kelas probabilitas
# kecil semua, dan ambang membuat pseudo-label menumpuk di satu kelas (skor malah turun).
#
# **Data demo:** abstrak arXiv (Inggris, 11 kategori), label kosong = NaN.

# %%
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/text/arxiv"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/arxiv_train.csv"
TEST_CSV = f"{DATA_FOLDER}/arxiv_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/arxiv_test_labels.csv"   # hanya ada di data latihan
EMBEDDING_MODEL = "../../../pretrained/minilm"

TEXT_COL = "text"
TARGET = "label"
PER_CLASS = 20      # tiap putaran, tiap kelas ditambah 20 teks tanpa label yang paling yakin
N_ROUNDS = 5
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train[TEXT_COL] = train[TEXT_COL].fillna("")
test[TEXT_COL] = test[TEXT_COL].fillna("")
print("train:", train.shape, "| test:", test.shape)
print("berlabel:", train[TARGET].notna().sum(), "| tanpa label:", train[TARGET].isna().sum())
print(train[TARGET].value_counts())

# %% [markdown]
# ## Feature dari SEMUA teks
# Baris 0..len(train)-1 = train, sisanya = test.
# Pilih SATU: TF-IDF (aktif) atau embedding (hapus `#` pada dua baris embedding).

# %%
all_texts = pd.concat([train[TEXT_COL], test[TEXT_COL]], ignore_index=True)
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5,
                             sublinear_tf=True)
features = vectorizer.fit_transform(all_texts)
# encoder = SentenceTransformer(EMBEDDING_MODEL)
# features = encoder.encode(list(all_texts), normalize_embeddings=True, batch_size=64)

X_test = features[len(train):]
print(features.shape)

# %% [markdown]
# ## Split: validasi hanya dari baris berlabel
# Semua baris tetap di `features`; yang dipindah-pindah hanya NOMOR barisnya.
# `pool_rows` = baris tanpa label (train tanpa label + semua test).

# %%
all_labels = train[TARGET].to_numpy()
labeled_rows = np.where(train[TARGET].notna())[0]
unlabeled_rows = np.where(train[TARGET].isna())[0]
test_rows = len(train) + np.arange(len(test))
pool_rows = np.concatenate([unlabeled_rows, test_rows])

train_rows, val_rows = train_test_split(labeled_rows, test_size=0.3,
                                        stratify=all_labels[labeled_rows], random_state=SEED)
print("latih:", len(train_rows), "| validasi:", len(val_rows), "| pool:", len(pool_rows))

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai (harus punya `predict_proba`).

# %%
model = LogisticRegression(C=10, max_iter=3000)
# model = LogisticRegression(C=1, max_iter=3000)
# model = CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)

# %% [markdown]
# ## Pseudo-label + evaluasi
# Tiap putaran: tebak pool → untuk tiap kelas ambil teks paling yakin → latih ulang dengan
# label asli + pseudo-label. Jumlah pseudo-label bertambah tiap putaran.

# %%
def pseudo_label(model, rows, labels):
    model.fit(features[rows], labels)
    for round_number in range(1, N_ROUNDS + 1):
        proba = model.predict_proba(features[pool_rows])
        n_take = PER_CLASS * round_number
        new_rows = [rows]
        new_labels = [labels]
        for class_index in range(len(model.classes_)):
            most_sure = np.argsort(proba[:, class_index])[::-1][:n_take]
            new_rows.append(pool_rows[most_sure])
            new_labels.append(np.full(n_take, model.classes_[class_index]))
        model.fit(features[np.concatenate(new_rows)], np.concatenate(new_labels))
        print("putaran", round_number, ":", n_take, "pseudo-label per kelas")
    return model


# coba: hanya data berlabel vs + pseudo-label, dinilai di validasi
y_val = all_labels[val_rows]
model.fit(features[train_rows], all_labels[train_rows])
base_predictions = model.predict(features[val_rows])
print("F1 macro hanya berlabel:", round(f1_score(y_val, base_predictions, average="macro"), 4))

model = pseudo_label(model, train_rows, all_labels[train_rows])
val_predictions = model.predict(features[val_rows])
print("F1 macro + pseudo-label:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions, zero_division=0))

# %% [markdown]
# ## Latih ulang (semua baris berlabel + pseudo-label) + submission

# %%
model = pseudo_label(model, labeled_rows, all_labels[labeled_rows])
predictions = model.predict(X_test)

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
