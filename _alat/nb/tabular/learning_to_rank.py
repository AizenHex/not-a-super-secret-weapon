# %% [markdown]
# # Tabular — Learning to Rank (urutkan item per query)
# **Kapan dipakai:** untuk tiap query (kata kunci pencarian, pengguna, pertanyaan) ada banyak
# item, dan tugasnya mengurutkan item paling relevan di atas.
#
# **Metrik:** NDCG@10 = seberapa bagus urutan 10 item teratas (1.0 = sempurna), dirata-rata
# per query. Pembanding: urutan mesin pencari biasa (hanya kolom `skor_teks`).
#
# **Data demo SINTETIS:** pencarian produk e-commerce, query = `query`, item = `produk`,
# relevansi 0–3.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/ranking"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/produk_rank_SINTETIS_train.csv"
TEST_CSV = f"{DATA_FOLDER}/produk_rank_SINTETIS_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/produk_rank_SINTETIS_test_labels.csv"   # hanya di data latihan

QUERY = "query"                  # kolom pengelompok (satu query = satu daftar item)
TARGET = "relevansi"             # bilangan bulat ≥ 0 (rating 1–5 → kurangi 1 dulu)
NOT_FEATURES = ["id", QUERY, "produk"]
BASELINE_COL = "skor_teks"       # skor pembanding
TOP_K = 10
SEED = 42

# %% [markdown]
# ## Baca data
# Data diurutkan per query: LGBMRanker membaca baris query yang sama secara berurutan.

# %%
train = pd.read_csv(TRAIN_CSV)
train = train.sort_values(QUERY, kind="stable")
test = pd.read_csv(TEST_CSV)
test = test.sort_values(QUERY, kind="stable")
FEATURES = test.columns.drop(NOT_FEATURES).tolist()
print("train:", train.shape, "| test:", test.shape)
print("feature:", FEATURES)
train.head()

# %% [markdown]
# ## EDA

# %%
items_per_query = train.groupby(QUERY).size()
print("jumlah query:", train[QUERY].nunique())
print("item per query (median):", items_per_query.median())

relevance_counts = train[TARGET].value_counts()
relevance_counts.sort_index().plot.bar()
plt.title("Sebaran relevansi")
plt.show()

correlation = train[FEATURES + [TARGET]].corr(method="spearman")
print(correlation[TARGET].sort_values().round(3))

# %% [markdown]
# ## Split per query
# Query yang sama tidak boleh ada di train dan validasi sekaligus.
# `group` = jumlah baris tiap query, sesuai urutan baris.

# %%
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_rows, val_rows = next(splitter.split(train, groups=train[QUERY]))
train_part = train.iloc[train_rows]
val_part = train.iloc[val_rows].copy()

train_groups = train_part.groupby(QUERY, sort=False).size()
val_groups = val_part.groupby(QUERY, sort=False).size()
print("query latih:", len(train_groups), "| query validasi:", len(val_groups))

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                   random_state=SEED, verbose=-1)
# model = LGBMRanker(objective="rank_xendcg", n_estimators=300, learning_rate=0.05, verbose=-1)
# model = LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=63,
#                    verbose=-1)

model.fit(train_part[FEATURES], train_part[TARGET], group=train_groups,
          eval_set=[(val_part[FEATURES], val_part[TARGET])], eval_group=[val_groups],
          eval_at=[TOP_K])

# %% [markdown]
# ## Evaluasi NDCG@10 per query
# `mean_ndcg` menghitung NDCG tiap query lalu merata-ratakannya.

# %%
def mean_ndcg(table, score_col):
    values = []
    for query_name, group in table.groupby(QUERY):
        true_relevance = [group[TARGET].to_numpy()]
        scores = [group[score_col].to_numpy()]
        values.append(ndcg_score(true_relevance, scores, k=TOP_K))
    return np.mean(values)


val_part["score"] = model.predict(val_part[FEATURES])
print("NDCG@10 model    :", round(mean_ndcg(val_part, "score"), 4))
print("NDCG@10 pembanding:", round(mean_ndcg(val_part, BASELINE_COL), 4))

importance = pd.Series(model.feature_importances_, index=FEATURES)
importance.sort_values().plot.barh()
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission
# Submission: skor per baris + peringkat di dalam query (1 = paling relevan).

# %%
all_groups = train.groupby(QUERY, sort=False).size()
model.fit(train[FEATURES], train[TARGET], group=all_groups)
test["score"] = model.predict(test[FEATURES])
test["rank"] = test.groupby(QUERY)["score"].rank(ascending=False, method="first")

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["skor"] = test["score"]
submission["peringkat"] = test["rank"].astype(int)
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.
# Kunci ikut diurutkan per query supaya barisnya sejajar dengan `test`.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
test_key = test_key.sort_values(QUERY, kind="stable")
test[TARGET] = test_key[TARGET].to_numpy()
print("NDCG@10 test model     :", round(mean_ndcg(test, "score"), 4))
print("NDCG@10 test pembanding:", round(mean_ndcg(test, BASELINE_COL), 4))
