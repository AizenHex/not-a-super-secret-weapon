# %% [markdown]
# # Dasar teks: kenali data, bersihkan, kata dasar (stemming), stopword, kata khas per kelas
#
# **Butuh:** tabel `train` dengan kolom teks (+ kolom label untuk bagian 4).
# **Menghasilkan:** kolom `clean`, `stem`, laporan panjang teks, kata khas per kelas.
#
# Bersihkan teks hanya untuk model berbasis kata (TF-IDF). Model BERT / sentence-transformers
# lebih baik diberi teks **asli** (hanya buang URL/username), karena tanda baca dan imbuhan
# juga informasi baginya.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | lihat data: panjang teks, label, teks kembar, teks kembar dengan label beda | **ya** |
# | 2 | `clean_text`: huruf kecil, buang URL/username/angka/simbol, kata gaul | TF-IDF |
# | 3 | stemming Sastrawi (`makanannya` → `makan`) + buang stopword | opsional, coba dua-duanya |
# | 4 | kata/frasa paling khas per kelas (chi²) | EDA, presentasi |
#
# Regex, ambil field, angka/tanggal dari teks: `umum/teks_regex.ipynb`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Tweet berlabel emosi (anger, fear, happy, love, sadness).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_train.csv")
print(train.head())

# %% [markdown]
# ## 1. Lihat data
# Teks sangat panjang (> 512 token) → BERT memotongnya; teks kembar dengan label beda = label
# berisik (pilih label terbanyak atau buang).

# %%
TEXT = "text"     # UBAH
TARGET = "label"  # UBAH

lengths = train[TEXT].str.split().str.len()
print("jumlah kata per teks:\n", lengths.describe().round(1))
print(train[TARGET].value_counts())
print("teks kosong:", train[TEXT].isna().sum(), "| teks kembar:", train[TEXT].duplicated().sum())
labels_per_text = train.groupby(TEXT)[TARGET].nunique()
print("teks kembar dengan label berbeda:", (labels_per_text > 1).sum())
for label in sorted(train[TARGET].unique()):
    print(f"[{label}]", train.loc[train[TARGET] == label, TEXT].iloc[0][:100])

# %% [markdown]
# ## 2. `clean_text` → kolom `clean`
# `SLANG`: kamus kata gaul → baku (tulis sendiri dari kata yang sering muncul di bagian 4).

# %%
import re

SLANG = {"gak": "tidak", "ga": "tidak", "nggak": "tidak", "tdk": "tidak", "yg": "yang",
         "dgn": "dengan", "bgt": "banget", "udh": "sudah", "udah": "sudah", "aja": "saja",
         "tp": "tapi", "krn": "karena", "karna": "karena", "sy": "saya", "gue": "saya",
         "gw": "saya", "lu": "kamu", "lo": "kamu", "dr": "dari", "jd": "jadi"}   # UBAH


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[username\]|@\w+", " ", text)          # username
    text = re.sub(r"https?://\S+|www\.\S+|\[url\]", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)                    # #hashtag → hashtag
    text = re.sub(r"(\w)\1{2,}", r"\1", text)                # bangeeettt → banget
    text = re.sub(r"[^a-z\s]", " ", text)                    # UBAH: angka dibuang
    words = []
    for word in text.split():
        words.append(SLANG.get(word, word))
    return " ".join(words)


print(clean_text("[USERNAME] Gak bangeeettt!!! cek https://x.id #kecewa 😡"))
train["clean"] = train[TEXT].apply(clean_text)
print(train[["clean"]].head(3))

# %% [markdown]
# ## 3. Stemming (kata dasar) + buang stopword → kolom `stem`
# Sastrawi lambat (±1 ms/kata) → setiap kata unik di-stem SEKALI lalu disimpan di dict.
# Stopword bawaan Sastrawi membuang `tidak`/`bukan` juga; untuk sentimen, keluarkan dari daftar.

# %%
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm

stemmer = StemmerFactory().create_stemmer()
STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
for keep in ["tidak", "bukan", "belum", "jangan", "kurang"]:   # UBAH: kata penting (sentimen)
    STOPWORDS.discard(keep)

unique_words = set()
for text in train["clean"]:
    unique_words.update(text.split())
stem_of = {}
for word in tqdm(unique_words):
    stem_of[word] = stemmer.stem(word)


def stem_text(text):
    words = []
    for word in text.split():
        if word not in STOPWORDS:
            words.append(stem_of.get(word, word))
    return " ".join(words)


train["stem"] = train["clean"].apply(stem_text)
print(train["clean"].iloc[1][:90], "\n→", train["stem"].iloc[1][:90])

# %% [markdown]
# ## 4. Kata/frasa paling khas per kelas (chi²)
# Chi² tinggi = kata itu muncul jauh lebih sering di satu kelas dibanding kelas lain.
# Berguna untuk: mengerti data, menambah kamus `SLANG`, bahan presentasi.

# %%
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

TOP_WORDS = 10   # UBAH

counter = CountVectorizer(ngram_range=(1, 2), min_df=3)
counts = counter.fit_transform(train["clean"])
words = np.array(counter.get_feature_names_out())
for label in sorted(train[TARGET].unique()):
    in_class = (train[TARGET] == label).to_numpy()
    scores, _ = chi2(counts, in_class)
    scores = np.nan_to_num(scores)
    rate_in = np.asarray(counts[in_class].mean(axis=0)).ravel()
    rate_out = np.asarray(counts[~in_class].mean(axis=0)).ravel()
    scores[rate_in <= rate_out] = 0        # buang kata yang justru JARANG di kelas ini
    print(f"{label}: {', '.join(words[np.argsort(-scores)[:TOP_WORDS]])}")
