# %% [markdown]
# # Teks — Topic Modeling (LDA / NMF)
# **Kapan dipakai:** soal meminta menemukan topik-topik utama dari kumpulan dokumen, lalu
# menjelaskan tiap topik dengan kata-kata teratasnya.
#
# **Langkah:** hitung kata → LDA/NMF mencari K topik → 10 kata teratas per topik → topik
# dominan tiap dokumen.
#
# **Data demo:** news4 (4 topik). Label HANYA dipakai untuk membandingkan topik (NMI/ARI),
# bukan untuk melatih.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/classification_en"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/news4_train.csv"

TEXT_COL = "text"
LABEL_COL = "label"      # hanya untuk evaluasi, hapus bila soal tidak punya label
ID_COL = "id"
N_TOPICS = 4
N_TOP_WORDS = 10
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
data[TEXT_COL] = data[TEXT_COL].fillna("")
print(data.shape)
print(data[LABEL_COL].value_counts())
data.head()

# %% [markdown]
# ## EDA: panjang dokumen

# %%
word_counts = data[TEXT_COL].str.split().str.len()
word_counts.hist(bins=50)
plt.title("Jumlah kata per dokumen")
plt.show()

# %% [markdown]
# ## Model
# LDA bekerja dengan jumlah kata (`CountVectorizer`); NMF dan LSA lebih cocok dengan TF-IDF.
# Pilih SATU pasangan: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
vectorizer = CountVectorizer(stop_words="english", min_df=3, max_df=0.5)
# vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5)   # untuk NMF / LSA

model = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="batch", max_iter=30,
                                  random_state=SEED)
# model = NMF(n_components=N_TOPICS, init="nndsvda", max_iter=500, random_state=SEED)
# model = NMF(n_components=N_TOPICS, beta_loss="kullback-leibler", solver="mu", max_iter=500,
#             random_state=SEED)
# model = TruncatedSVD(n_components=N_TOPICS, random_state=SEED)   # LSA

X = vectorizer.fit_transform(data[TEXT_COL])
doc_topics = model.fit_transform(X)   # baris = dokumen, kolom = bobot tiap topik
print(doc_topics.shape)

# %% [markdown]
# ## Kata teratas tiap topik (bahan memberi nama topik)

# %%
vocabulary = vectorizer.get_feature_names_out()
for topic_number in range(N_TOPICS):
    word_weight = model.components_[topic_number]
    top_columns = np.argsort(word_weight)[::-1][:N_TOP_WORDS]
    print("Topik", topic_number, ":", " ".join(vocabulary[top_columns]))

# %% [markdown]
# ## Evaluasi
# Topik dominan tiap dokumen dibandingkan dengan label asli (hanya bila label ada).

# %%
topics = doc_topics.argmax(axis=1)
print("NMI:", round(normalized_mutual_info_score(data[LABEL_COL], topics), 4))
print("ARI:", round(adjusted_rand_score(data[LABEL_COL], topics), 4))
print(pd.crosstab(data[LABEL_COL], topics))

# %% [markdown]
# ## Simpan hasil (topik dominan tiap dokumen)

# %%
result = pd.DataFrame()
result["id"] = data[ID_COL]
result["topik"] = topics
result["bobot_topik"] = doc_topics.max(axis=1)
result.to_csv("hasil.csv", index=False)
result.head()
