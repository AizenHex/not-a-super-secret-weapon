# %% [markdown]
# # Teks — Ensemble (gabung beberapa model teks)
# **Kapan dipakai:** beberapa model teks skornya mirip tetapi kesalahannya berbeda. Rata-rata
# probabilitasnya biasanya lebih stabil daripada satu model.
#
# **Anggota:** TF-IDF kata + LogReg, TF-IDF huruf + LogReg, TF-IDF kata + Naive Bayes.
#
# **Cara gabung:** rata-rata probabilitas berbobot (soft voting). Alternatif: stacking.
#
# **Data demo:** SmSA (sentimen 3 kelas).

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline

# %% [markdown]
# ## Konfigurasi
# `WEIGHTS` = bobot tiap anggota saat dirata-rata (anggota lemah diberi bobot kecil).

# %%
DATA_FOLDER = "../../../data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/smsa_test.csv"

TEXT_COL = "text"
TARGET = "label"
WEIGHTS = {"kata_logreg": 1.0, "huruf_logreg": 1.0, "kata_nb": 0.5}
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train[TEXT_COL] = train[TEXT_COL].fillna("")
test[TEXT_COL] = test[TEXT_COL].fillna("")
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[TEXT_COL], train[TARGET], test_size=0.2, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Anggota ensemble
# Tambah anggota: tambah satu entri di `members` dan satu bobot di `WEIGHTS`.
# Anggota lain yang bagus: embedding + LogReg (lihat `text/klasifikasi_embedding`).

# %%
members = {}
members["kata_logreg"] = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(C=10, max_iter=3000))
members["huruf_logreg"] = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    LogisticRegression(C=10, max_iter=3000))
members["kata_nb"] = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    ComplementNB(alpha=0.3))

# %% [markdown]
# ## Latih tiap anggota + skornya di validasi

# %%
val_proba = {}
for name in members:
    members[name].fit(X_train, y_train)
    val_proba[name] = members[name].predict_proba(X_val)
    member_predictions = members[name].predict(X_val)
    print(name, "F1 macro:", round(f1_score(y_val, member_predictions, average="macro"), 4))
classes = members["kata_logreg"].classes_

# %% [markdown]
# ## Gabung: rata-rata probabilitas berbobot
# Ensemble tidak selalu menang. Di demo: rata-rata 0.862 < anggota terbaik 0.867
# (stacking 0.869). Selalu bandingkan dengan anggota terbaik; kecilkan bobot anggota lemah.

# %%
combined = np.zeros((len(X_val), len(classes)))
for name in members:
    combined = combined + WEIGHTS[name] * val_proba[name]
combined = combined / sum(WEIGHTS.values())
val_predictions = classes[combined.argmax(axis=1)]
print("ENSEMBLE F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))

# %% [markdown]
# ## Alternatif: stacking (model kedua belajar cara menggabungkan)
# Lebih lambat (tiap anggota dilatih ulang 5 kali di dalam CV). Hapus `#` untuk mencoba.

# %%
# member_list = list(members.items())
# stacking = StackingClassifier(member_list, final_estimator=LogisticRegression(max_iter=2000),
#                               cv=5)
# stacking.fit(X_train, y_train)
# print("STACKING F1 macro:", f1_score(y_val, stacking.predict(X_val), average="macro"))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
combined_test = np.zeros((len(test), len(classes)))
for name in members:
    members[name].fit(train[TEXT_COL], train[TARGET])
    combined_test = combined_test + WEIGHTS[name] * members[name].predict_proba(test[TEXT_COL])
predictions = classes[combined_test.argmax(axis=1)]

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
