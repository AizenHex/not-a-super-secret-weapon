# %% [markdown]
# # Teks — Topic Modeling dengan BERTopic (embedding kalimat + clustering)
# **Kapan dipakai:** ingin topik yang lebih "bermakna" daripada LDA/NMF (`text/topik`),
# terutama untuk teks pendek. Jumlah topik tidak wajib ditentukan di awal.
#
# **Cara kerja:**
# 1. Embedding kalimat (MiniLM lokal) → tiap dokumen jadi vektor angka.
# 2. UMAP mengecilkan dimensi, HDBSCAN mengelompokkan.
# 3. Kata khas tiap kelompok dicari dengan c-TF-IDF.
#
# Dokumen yang tidak masuk kelompok mana pun = topik -1.
#
# **Data demo:** news4 (4 topik). Label HANYA dipakai untuk membandingkan (NMI/ARI).

# %%
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/classification_en"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/news4_train.csv"
EMBEDDING_MODEL = "Z:/GEMASTIK-DATABASE/pretrained/minilm_en"
# teks Indonesia: EMBEDDING_MODEL = "Z:/GEMASTIK-DATABASE/pretrained/minilm"

TEXT_COL = "text"
LABEL_COL = "label"       # hanya untuk evaluasi, hapus bila soal tidak punya label
MIN_TOPIC_SIZE = 50       # kelompok lebih kecil dari ini digabung / jadi -1
N_TOPICS = 5              # target topik + 1 (topik -1 ikut dihitung). None = otomatis
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
data[TEXT_COL] = data[TEXT_COL].fillna("")
texts = list(data[TEXT_COL])
print(data.shape)
print(data[LABEL_COL].value_counts())

# %% [markdown]
# ## Embedding kalimat
# Dihitung sekali; saat mencoba pengaturan BERTopic lain, sel ini tidak perlu dijalankan ulang.

# %%
encoder = SentenceTransformer(EMBEDDING_MODEL)
embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False)
print(embeddings.shape)

# %% [markdown]
# ## BERTopic
# `random_state` di UMAP supaya hasil bisa diulang. Stopword dibuang hanya saat memilih kata
# topik.

# %%
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                  random_state=SEED)
hdbscan_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, metric="euclidean",
                        prediction_data=True)
word_counter = CountVectorizer(stop_words="english", min_df=2)
# teks Indonesia: word_counter = CountVectorizer(stop_words=daftar_stopword_id, min_df=2)

model = BERTopic(
    embedding_model=encoder,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=word_counter,
    nr_topics=N_TOPICS,
)
topics, probabilities = model.fit_transform(texts, embeddings)
topic_info = model.get_topic_info()
topic_info.head(10)

# %% [markdown]
# ## Kata kunci tiap topik

# %%
for topic_number in topic_info["Topic"]:
    if topic_number == -1:
        continue
    words = []
    for word, weight in model.get_topic(topic_number)[:10]:
        words.append(word)
    print("topik", topic_number, ":", ", ".join(words))

# %% [markdown]
# ## Evaluasi dengan label (HANYA DEMO; saat lomba biasanya tidak ada label)
# Pembanding: LDA di `text/topik` (NMI 0.41, ARI 0.39).

# %%
data["topik"] = topics
print("dokumen tanpa topik (-1):", round((data["topik"] == -1).mean(), 3))
print("NMI:", round(normalized_mutual_info_score(data[LABEL_COL], data["topik"]), 3))
print("ARI:", round(adjusted_rand_score(data[LABEL_COL], data["topik"]), 3))
print(pd.crosstab(data[LABEL_COL], data["topik"]))

# %% [markdown]
# ## Simpan topik tiap dokumen + info topik

# %%
result = pd.DataFrame()
result["text"] = data[TEXT_COL]
result["topik"] = data["topik"]
result.to_csv("topik_dokumen.csv", index=False)
topic_info.to_csv("info_topik.csv", index=False)
result.head()
