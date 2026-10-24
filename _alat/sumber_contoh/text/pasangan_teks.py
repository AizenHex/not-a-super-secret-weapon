# %% [markdown]
# # Teks — Pasangan Kalimat (entailment / parafrasa / kemiripan)
# **Kapan dipakai:** input berupa DUA teks (kalimat A dan B), tebak hubungannya: sama makna,
# saling mendukung, atau tidak. Contoh lain: pertanyaan duplikat, cocokkan judul–isi berita.
#
# **Cara:** ubah tiap pasangan menjadi feature angka → classifier biasa.
# - feature makna: embedding MiniLM `|a−b|`, `a*b`, cosine;
# - feature kata: jaccard kata, beda panjang, cosine TF-IDF.
#
# **Data demo:** WReTE (IndoNLU).

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/text/pair"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/wrete_train.csv"
TEST_CSV = f"{DATA_FOLDER}/wrete_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/wrete_test_labels.csv"   # hanya ada di data latihan
EMBEDDING_MODEL = "../../../pretrained/minilm"

TEXT_A = "sent_A"
TEXT_B = "sent_B"
TARGET = "label"
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())
train[[TEXT_A, TEXT_B, TARGET]].head()

# %% [markdown]
# ## EDA: seberapa mirip kata-kata A dan B per label?
# Jaccard = jumlah kata yang sama dibagi jumlah semua kata unik (0 = beda total, 1 = sama).

# %%
def jaccard(first_text, second_text):
    first_words = set(first_text.lower().split())
    second_words = set(second_text.lower().split())
    return len(first_words & second_words) / len(first_words | second_words)


def jaccard_column(table):
    values = []
    for first_text, second_text in zip(table[TEXT_A], table[TEXT_B]):
        values.append(jaccard(first_text, second_text))
    return values


print(jaccard("saya suka kopi", "saya suka teh"))
train["jaccard"] = jaccard_column(train)
jaccard_per_label = train.groupby(TARGET)["jaccard"].describe()
print(jaccard_per_label.round(3))

# %% [markdown]
# ## Feature pasangan
# Embedding A dan B → beda mutlak `|a−b|` dan perkalian `a*b`, ditambah 5 feature kata.

# %%
encoder = SentenceTransformer(EMBEDDING_MODEL)
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
tfidf.fit(pd.concat([train[TEXT_A], train[TEXT_B], test[TEXT_A], test[TEXT_B]]))


def pair_features(table):
    vectors_a = encoder.encode(list(table[TEXT_A]), normalize_embeddings=True)
    vectors_b = encoder.encode(list(table[TEXT_B]), normalize_embeddings=True)
    tfidf_a = tfidf.transform(table[TEXT_A])
    tfidf_b = tfidf.transform(table[TEXT_B])
    lexical = pd.DataFrame()
    lexical["cos_emb"] = (vectors_a * vectors_b).sum(axis=1)
    lexical["cos_tfidf"] = np.asarray(tfidf_a.multiply(tfidf_b).sum(axis=1)).ravel()
    lexical["jaccard"] = jaccard_column(table)
    length_a = table[TEXT_A].str.split().str.len()
    length_b = table[TEXT_B].str.split().str.len()
    lexical["beda_panjang"] = (length_a - length_b).abs().to_numpy()
    lexical["rasio_panjang"] = (length_b / length_a).to_numpy()
    return np.hstack([np.abs(vectors_a - vectors_b), vectors_a * vectors_b, lexical.to_numpy()])


X = pair_features(train)
X_test = pair_features(test)
print("train:", X.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split train / validasi

# %%
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

# %% [markdown]
# ## Model
# Mau hanya feature kata (tanpa embedding)? Pakai 5 kolom terakhir saja: `X[:, -5:]`.
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
# model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
# model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = model.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(X, y)
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
