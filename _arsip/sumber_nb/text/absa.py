# %% [markdown]
# # Teks — Sentimen per Aspek (ABSA)
# **Kapan dipakai:** satu ulasan dinilai untuk banyak aspek sekaligus (kebersihan, AC, wifi,
# ...). Tiap aspek punya label sendiri: `neg`, `neut` (tidak dibahas/netral), `pos`, `neg_pos`
# (campur).
#
# **Cara:** satu model per aspek (loop). Feature TF-IDF sama untuk semua aspek.
#
# **Data demo:** HoASA (ulasan hotel, 10 aspek).

# %%
import re

import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/absa"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/hoasa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/hoasa_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/hoasa_test_labels.csv"   # hanya ada di data latihan

TEXT_COL = "review"
ASPECTS = ["ac", "air_panas", "bau", "general", "kebersihan", "linen", "service",
           "sunrise_meal", "tv", "wifi"]
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
missing_per_column = train.isna().sum()
print("sel kosong:", missing_per_column.sum())
train.head()

# %% [markdown]
# ## EDA: sebaran label tiap aspek
# Hampir semua aspek didominasi `neut` → pakai F1 macro per aspek, bukan akurasi.

# %%
label_spread = pd.DataFrame()
for aspect in ASPECTS:
    label_spread[aspect] = train[aspect].value_counts()
label_spread = label_spread.fillna(0)
label_spread.astype(int).T

# %% [markdown]
# ## Bersihkan teks

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


print(clean_text("Kamar BERSIH, tapi AC-nya rusak!!"))

train[TEXT_COL] = train[TEXT_COL].fillna("").apply(clean_text)
test[TEXT_COL] = test[TEXT_COL].fillna("").apply(clean_text)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[TEXT_COL], train[ASPECTS], test_size=0.2, random_state=SEED
)

# %% [markdown]
# ## Model
# `class_weight="balanced"` membantu kelas langka (`neg_pos`).
# Pilih SATU vectorizer dan SATU model: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2))
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

model = LogisticRegression(C=5, class_weight="balanced", max_iter=2000)
# model = LinearSVC(C=0.5, class_weight="balanced")
# model = RidgeClassifier(alpha=1.0, class_weight="balanced")
# model = ComplementNB(alpha=0.3)

pipe = make_pipeline(vectorizer, model)

# %% [markdown]
# ## Latih + evaluasi per aspek
# `clone(pipe)` = salinan pipeline yang belum dilatih, satu untuk tiap aspek.

# %%
val_predictions = pd.DataFrame()
aspect_scores = pd.Series(dtype=float)
for aspect in ASPECTS:
    aspect_pipe = clone(pipe)
    aspect_pipe.fit(X_train, y_train[aspect])
    val_predictions[aspect] = aspect_pipe.predict(X_val)
    aspect_scores[aspect] = f1_score(y_val[aspect], val_predictions[aspect], average="macro")

print(aspect_scores.round(4))
print("rata-rata F1 macro:", round(aspect_scores.mean(), 4))
print(classification_report(y_val["kebersihan"], val_predictions["kebersihan"]))

# %% [markdown]
# ## Latih ulang di semua data + submission
# Satu kolom per aspek, ditambah satu per satu di dalam loop.

# %%
submission = pd.DataFrame()
submission["id"] = test["id"]
for aspect in ASPECTS:
    aspect_pipe = clone(pipe)
    aspect_pipe.fit(train[TEXT_COL], train[aspect])
    submission[aspect] = aspect_pipe.predict(test[TEXT_COL])
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
test_scores = []
for aspect in ASPECTS:
    test_scores.append(f1_score(test_key[aspect], submission[aspect], average="macro"))
print("rata-rata F1 macro test (kunci latihan):", round(sum(test_scores) / len(test_scores), 4))
