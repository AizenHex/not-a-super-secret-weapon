# %% [markdown]
# # Teks — Klasifikasi dengan TF-IDF (baseline teks paling cepat)
# **Kapan dipakai:** klasifikasi teks (sentimen, kategori berita, spam, dll.). Selalu
# jalankan ini dulu sebelum model transformer: cepat, dan sering sudah kuat.
#
# **Langkah:** bersihkan teks → TF-IDF (kata 1–2 gram) → model linear → F1 macro.
#
# **Data demo:** SmSA (sentimen Bahasa Indonesia: positive / neutral / negative).

# %%
import re

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/smsa_test.csv"

TEXT_COL = "text"
TARGET = "label"
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train.isna().sum())
train.head()

# %% [markdown]
# ## EDA: jumlah per kelas dan panjang teks

# %%
print(train[TARGET].value_counts())
word_counts = train[TEXT_COL].str.split().str.len()
word_counts.hist(bins=50)
plt.title("Jumlah kata per teks")
plt.show()

# %% [markdown]
# ## Bersihkan teks
# Huruf kecil, buang link dan mention, sisakan huruf dan angka saja.

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


print(clean_text("Mantap!! Cek di https://contoh.com @admin, harganya 50rb :)"))

train[TEXT_COL] = train[TEXT_COL].fillna("").apply(clean_text)
test[TEXT_COL] = test[TEXT_COL].fillna("").apply(clean_text)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[TEXT_COL], train[TARGET], test_size=0.2, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Model
# Pilih SATU vectorizer dan SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada
# baris lain.

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
# vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
#                              sublinear_tf=True)                      # tahan salah ketik
# vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, binary=True)
# vectorizer = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
#                         TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2))

model = LinearSVC(C=0.5)                          # baseline teks yang kuat
# model = LogisticRegression(C=5, max_iter=2000)  # punya predict_proba
# model = ComplementNB(alpha=0.3)
# model = MultinomialNB(alpha=0.1)
# model = SGDClassifier(loss="modified_huber", random_state=SEED)
# model = RidgeClassifier(alpha=1.0)
# model = CalibratedClassifierCV(LinearSVC(C=0.5))   # LinearSVC + predict_proba
# model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)

pipe = make_pipeline(vectorizer, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
# soal 2 kelas: print("F1:", f1_score(y_val, val_predictions, pos_label="isi_kelas_positif"))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(train[TEXT_COL], train[TARGET])
predictions = pipe.predict(test[TEXT_COL])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
