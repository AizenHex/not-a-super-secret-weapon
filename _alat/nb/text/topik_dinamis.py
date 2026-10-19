# %% [markdown]
# # Teks — Topik Dinamis (topik berubah dari waktu ke waktu)
# **Kapan dipakai:** pola Satria Data 2022: kumpulan teks bertanggal → topik apa saja, kapan
# naik/turun, dan bagaimana kata kuncinya bergeser per periode.
#
# **Langkah:**
# 1. Satu model topik (NMF) untuk semua teks → tiap teks diberi topik.
# 2. Proporsi topik per bulan (tren).
# 3. Kata kunci tiap topik per kuartal (pergeseran isi topik).
#
# **Data demo:** TweetTopic (tweet Inggris 2019–2021). Label hanya untuk membandingkan (NMI).

# %%
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/topik_waktu"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/tweet_topic.csv"

TEXT_COL = "text"
DATE_COL = "date"
LABEL_COL = "label_name"   # hanya untuk membandingkan, boleh tidak ada saat lomba
N_TOPICS = 10
N_TOP_WORDS = 8
MIN_TEXTS = 5              # kuartal dengan < 5 teks untuk satu topik dilewati
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV, parse_dates=[DATE_COL])
print(data.shape)
print(data[DATE_COL].min(), "s.d.", data[DATE_COL].max())
data.head()

# %% [markdown]
# ## EDA: jumlah teks per bulan

# %%
data["bulan"] = data[DATE_COL].dt.to_period("M")
data["kuartal"] = data[DATE_COL].dt.to_period("Q")
texts_per_month = data.groupby("bulan").size()
texts_per_month.plot(figsize=(10, 3))
plt.title("Jumlah teks per bulan")
plt.show()

# %% [markdown]
# ## Bersihkan teks
# Penanda `{{USERNAME}}`, `{{URL}}`, link, dan mention dibuang; sisakan huruf saja.

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\{\{username\}\}|\{\{url\}\}|http\S+|@\w+", " ", text)
    text = re.sub(r"[{}@]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


print(clean_text("Great game by {{USERNAME}} tonight!! http://t.co/x"))
data["bersih"] = data[TEXT_COL].fillna("").apply(clean_text)

# %% [markdown]
# ## Model
# NMF memakai `TfidfVectorizer`, LDA memakai `CountVectorizer`.
# Pilih SATU pasangan: hapus `#` di dua baris yang ingin dipakai, beri `#` pada pasangan lain.

# %%
vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.4, ngram_range=(1, 2))
model = NMF(n_components=N_TOPICS, init="nndsvda", random_state=SEED, max_iter=400)
# vectorizer = CountVectorizer(stop_words="english", min_df=5, max_df=0.4)
# model = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="batch",
#                                   random_state=SEED)

X = vectorizer.fit_transform(data["bersih"])
doc_topics = model.fit_transform(X)
data["topik"] = doc_topics.argmax(axis=1)

vocabulary = vectorizer.get_feature_names_out()
topic_names = []
for topic_number in range(N_TOPICS):
    top_columns = np.argsort(model.components_[topic_number])[::-1][:N_TOP_WORDS]
    top_words = vocabulary[top_columns]
    topic_names.append(f"T{topic_number}: " + ", ".join(top_words[:3]))
    print("topik", topic_number, ":", ", ".join(top_words))

# %% [markdown]
# ## Evaluasi: cocok dengan label manusia? (hanya bila label ada)
# Satu tweet bisa punya beberapa label ("a|b"), ambil label pertama saja.

# %%
first_label = data[LABEL_COL].fillna("tanpa_label").str.split("|").str[0]
print("NMI topik vs label:", round(normalized_mutual_info_score(first_label, data["topik"]), 4))
pd.crosstab(data["topik"], first_label).T

# %% [markdown]
# ## Tren: proporsi topik per bulan

# %%
topic_per_month = pd.crosstab(data["bulan"], data["topik"], normalize="index")
topic_per_month.columns = topic_names
topic_per_month.to_csv("topik_per_bulan.csv")
topic_per_month.plot.area(figsize=(11, 5), cmap="tab10")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.title("Proporsi topik per bulan")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Topik yang naik / turun
# Rata-rata 6 bulan terakhir dikurangi rata-rata 6 bulan pertama.

# %%
change = topic_per_month.iloc[-6:].mean() - topic_per_month.iloc[:6].mean()
print(change.sort_values().round(3))

# %% [markdown]
# ## Pergeseran kata kunci: tiap topik per kuartal
# Kata kunci = rata-rata TF-IDF teks bertopik itu di kuartal itu.

# %%
quarter_words = pd.DataFrame()
for quarter in sorted(data["kuartal"].unique()):
    for topic_number in range(N_TOPICS):
        rows = (data["kuartal"] == quarter) & (data["topik"] == topic_number)
        rows = rows.to_numpy()
        if rows.sum() < MIN_TEXTS:
            continue
        mean_weight = np.asarray(X[rows].mean(axis=0)).ravel()
        top_columns = np.argsort(mean_weight)[::-1][:5]
        quarter_words.loc[str(quarter), f"T{topic_number}"] = ", ".join(vocabulary[top_columns])
quarter_words.to_csv("kata_kunci_per_kuartal.csv")
quarter_words[["T0", "T1", "T2"]]

# %% [markdown]
# ## Submission: topik tiap teks
# Format soal minta kolom lain? Tambah satu baris `submission["nama"] = ...`.

# %%
names_per_text = []
for topic_number in data["topik"]:
    names_per_text.append(topic_names[topic_number])

submission = pd.DataFrame()
submission["date"] = data[DATE_COL].dt.date
submission["text"] = data[TEXT_COL]
submission["topik"] = data["topik"]
submission["nama_topik"] = names_per_text
submission.to_csv("submission.csv", index=False)
submission.head()
