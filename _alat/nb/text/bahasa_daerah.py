# %% [markdown]
# # Teks — Klasifikasi Teks Multibahasa / Bahasa Daerah
# **Kapan dipakai:** teks campuran bahasa Indonesia, Jawa, Sunda, Bali, Minang, dll.
# Kata dasar beda-beda tiap bahasa, jadi feature **n-gram huruf** lebih tahan daripada kata.
#
# **Bonus:** identifikasi bahasa (target `bahasa`), pola yang sama, hanya ganti target.
#
# **Awas data paralel:** di NusaX kalimat yang sama diterjemahkan ke 12 bahasa. Split acak
# membuat terjemahannya ada di train dan validasi sekaligus (skor validasi palsu tinggi). Jadi
# split per kalimat asal (`GroupShuffleSplit`).
#
# **Data demo:** NusaX-senti (12 bahasa, sentimen negative/neutral/positive).

# %%
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/nusax"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/nusax_train.csv"
TEST_CSV = f"{DATA_FOLDER}/nusax_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/nusax_test_labels.csv"   # hanya ada di data latihan

TEXT_COL = "text"
TARGET = "label"
LANGUAGE_COL = "bahasa"
SENTENCES_PER_LANGUAGE = 600   # baris ke-n tiap bahasa = terjemahan kalimat yang sama
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
# ## EDA: jumlah per bahasa dan label, contoh kalimat

# %%
print(pd.crosstab(train[LANGUAGE_COL], train[TARGET]))
for language in ["javanese", "sundanese", "balinese"]:
    first_text = train.loc[train[LANGUAGE_COL] == language, TEXT_COL].iloc[0]
    print(language, "→", first_text[:90])

# %% [markdown]
# ## Bersihkan teks
# Huruf beraksen (é, è) dipertahankan karena dipakai di beberapa bahasa daerah.

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\u00C0-\u024F]", " ", text)
    return text


print(clean_text("Kulo seneng sanget!! @admin http://x.id"))

train[TEXT_COL] = train[TEXT_COL].fillna("").apply(clean_text)
test[TEXT_COL] = test[TEXT_COL].fillna("").apply(clean_text)

# %% [markdown]
# ## Split per kalimat asal
# Semua terjemahan satu kalimat masuk ke sisi yang sama (train ATAU validasi).
# Data tidak paralel? Ganti dengan `train_test_split` biasa.

# %%
source_sentence = train["id"] % SENTENCES_PER_LANGUAGE
splitter = GroupShuffleSplit(test_size=0.2, random_state=SEED)
train_rows, val_rows = next(splitter.split(train, groups=source_sentence))
train_part = train.iloc[train_rows]
val_part = train.iloc[val_rows]
print("train:", train_part.shape, "| validasi:", val_part.shape)

# %% [markdown]
# ## Model
# Pilih SATU vectorizer dan SATU model: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                        sublinear_tf=True),
                        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True))
# vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
#                              sublinear_tf=True)
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

model = LogisticRegression(C=5, max_iter=3000)
# model = LinearSVC(C=0.3)
# model = SGDClassifier(loss="modified_huber", random_state=SEED)
# model = ComplementNB(alpha=0.3)

pipe = make_pipeline(vectorizer, model)
pipe.fit(train_part[TEXT_COL], train_part[TARGET])

# %% [markdown]
# ## Evaluasi: keseluruhan + per bahasa

# %%
val_predictions = pipe.predict(val_part[TEXT_COL])
print("F1 macro:", round(f1_score(val_part[TARGET], val_predictions, average="macro"), 4))
print(classification_report(val_part[TARGET], val_predictions))

per_language = pd.DataFrame()
per_language["asli"] = val_part[TARGET].to_numpy()
per_language["prediksi"] = val_predictions
per_language["bahasa"] = val_part[LANGUAGE_COL].to_numpy()
language_scores = pd.Series(dtype=float)
for language, group in per_language.groupby("bahasa"):
    language_scores[language] = f1_score(group["asli"], group["prediksi"], average="macro")
language_scores.sort_values().round(4)

# %% [markdown]
# ## Bonus: identifikasi bahasa (target = kolom `bahasa`)

# %%
language_pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2),
                              LogisticRegression(C=10, max_iter=3000))
language_pipe.fit(train_part[TEXT_COL], train_part[LANGUAGE_COL])
language_predictions = language_pipe.predict(val_part[TEXT_COL])
language_f1 = f1_score(val_part[LANGUAGE_COL], language_predictions, average="macro")
print("F1 macro identifikasi bahasa:", round(language_f1, 4))
ConfusionMatrixDisplay.from_predictions(val_part[LANGUAGE_COL], language_predictions,
                                        xticks_rotation=90)
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

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["label"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
