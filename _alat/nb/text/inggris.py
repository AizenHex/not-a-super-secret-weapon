# %% [markdown]
# # Teks — Klasifikasi Teks Bahasa Inggris (tweet sentimen)
# **Kapan dipakai:** teks berbahasa Inggris. Bedanya dengan teks Indonesia:
# - stopword Inggris (kata negasi `not`, `no` TIDAK dibuang, penting untuk sentimen);
# - kontraksi (`can't` → `can not`);
# - stemming Porter (`loved`, `loving` → `love`);
# - emoji diubah jadi kata (`😂` → `face_with_tears_of_joy`).
#
# **Data demo:** TweetEval sentiment (negative / neutral / positive), `@user` sudah dianonimkan.
# Model Inggris pretrained (all-MiniLM-L6-v2 / distilbert-base-uncased) ada di `pretrained/` →
# pakai di `text/klasifikasi_embedding` atau `text/klasifikasi_finetune`.

# %%
import re

import emoji
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/english"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/tweet_sentiment_train.csv"
TEST_CSV = f"{DATA_FOLDER}/tweet_sentiment_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/tweet_sentiment_test_labels.csv"   # hanya ada di data latihan

TEXT_COL = "text"
TARGET = "label"
CONTRACTIONS = {"can't": "can not", "won't": "will not", "n't": " not", "'re": " are",
                "'s": " is", "'ll": " will", "'ve": " have", "'m": " am", "'d": " would"}
STOP_WORDS = ENGLISH_STOP_WORDS - {"not", "no", "nor", "never"}
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train[TEXT_COL] = train[TEXT_COL].fillna("")
test[TEXT_COL] = test[TEXT_COL].fillna("")
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## EDA: kelas, emoji, hashtag, panjang tweet

# %%
print(train[TARGET].value_counts())
emoji_counts = train[TEXT_COL].apply(emoji.emoji_count)
print("ada emoji:", round((emoji_counts > 0).mean(), 3))
print("ada hashtag:", round(train[TEXT_COL].str.contains("#").mean(), 3))
word_counts = train[TEXT_COL].str.split().str.len()
word_counts.hist(bins=40)
plt.title("Jumlah kata per tweet")
plt.show()

# %% [markdown]
# ## Praproses Inggris
# Urutan: emoji → kata, huruf kecil, kontraksi, buang link/mention/tanda `#`, sisakan huruf,
# buang stopword, stemming.

# %%
stemmer = PorterStemmer()


def clean_text(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.lower()
    for short, long in CONTRACTIONS.items():
        text = text.replace(short, long)
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z_\s]", " ", text)
    words = []
    for word in text.split():
        if word not in STOP_WORDS:
            words.append(stemmer.stem(word))
    return " ".join(words)


print(train.loc[3, TEXT_COL])
print(clean_text(train.loc[3, TEXT_COL]))

train["bersih"] = train[TEXT_COL].apply(clean_text)
test["bersih"] = test[TEXT_COL].apply(clean_text)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train["bersih"], train[TARGET], test_size=0.2, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Model
# Pilih SATU vectorizer dan SATU model: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                        sublinear_tf=True))
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

model = LogisticRegression(C=2, class_weight="balanced", max_iter=3000)
# model = LinearSVC(C=0.3, class_weight="balanced")
# model = SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=SEED)
# model = ComplementNB(alpha=0.5)

pipe = make_pipeline(vectorizer, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, normalize="true",
                                        values_format=".2f")
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(train["bersih"], train[TARGET])
predictions = pipe.predict(test["bersih"])

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
