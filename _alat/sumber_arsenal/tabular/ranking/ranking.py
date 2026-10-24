# %% [markdown]
# # Ranking (learning to rank): urutkan item per query/pengguna → NDCG@k, MAP@k
#
# **Butuh:** tabel satu baris per (query, item) dengan fitur + kolom relevansi (0, 1, 2, ...).
# **Menghasilkan:** skor per baris, urutan item per query, `submission.csv`.
#
# Contoh soal: urutan produk hasil pencarian, rekomendasi item per pengguna, prioritas
# dokumen. Kunci: baris satu query saling dibandingkan → fold per QUERY (`GroupKFold`), dan
# model dilatih dengan tujuan ranking (LambdaMART: Burges 2010; LightGBM `lambdarank`).
#
# | cara | ide | sel |
# |---|---|---|
# | aturan 1 fitur | urutkan dengan fitur terkuat | 2 |
# | regresi/klasifikasi biasa | tebak relevansi tiap baris, lalu urutkan | 3 |
# | `LGBMRanker` lambdarank | langsung mengoptimalkan NDCG per query | 4 |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi, urut per query, fold per query, `mean_ndcg` | **ya** |
# | 2 | baseline aturan | **ya** |
# | 3 | regresi biasa | coba |
# | 4 | `LGBMRanker` (butuh `group` = jumlah baris per query) | **ya** |
# | 5 | fitur relatif dalam query (peringkat, selisih dari rata-rata query) | disarankan |
# | 6 | submission (skor atau urutan item per query) | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Produk hasil pencarian: 20–40 produk per query, relevansi 0 (tidak) – 3 (sangat).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ranking/produk_rank_SINTETIS_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ranking/produk_rank_SINTETIS_test.csv")
FOLDER = f"{DEMO_FOLDER}/tabular_extra/ranking"
test_labels = pd.read_csv(f"{FOLDER}/produk_rank_SINTETIS_test_labels.csv")
print(train.shape, test.shape, "| query:", train["query"].nunique(), test["query"].nunique())
print(train["relevansi"].value_counts().sort_index())

# %% [markdown]
# ## 1. Konfigurasi: urutkan per query, fold per query, `mean_ndcg(table, score_column, k)`
# LightGBM ranker butuh baris satu query BERDEMPETAN → `sort_values(QUERY)` wajib.

# %%
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold

QUERY = "query"           # UBAH
TARGET = "relevansi"      # UBAH
FEATURES = ["skor_teks", "kategori_cocok", "harga_per_anggaran", "rating_produk",
            "jumlah_ulasan", "diskon_persen", "jarak_km"]   # UBAH
TOP_K = 10                # UBAH: NDCG@k soal

train = train.sort_values([QUERY, "id"]).reset_index(drop=True)
test = test.sort_values([QUERY, "id"]).reset_index(drop=True)
folds = np.zeros(len(train), dtype=int)
for fold, (_, index_val) in enumerate(GroupKFold(5).split(train, groups=train[QUERY])):
    folds[index_val] = fold


def mean_ndcg(table, score_column, top_k=TOP_K):
    values = []
    for _, rows in table.groupby(QUERY):
        if len(rows) > 1:
            values.append(ndcg_score([rows[TARGET].to_numpy()],
                                     [rows[score_column].to_numpy()], k=top_k))
    return round(float(np.mean(values)), 4)


train["acak"] = np.random.default_rng(42).random(len(train))
print(f"NDCG@{TOP_K} urutan acak:", mean_ndcg(train, "acak"))

# %% [markdown]
# ## 2. Baseline aturan: urutkan dengan SATU fitur (cek tiap fitur, tanda + / -)

# %%
for column in FEATURES:
    train["aturan"] = train[column]
    up = mean_ndcg(train, "aturan")
    train["aturan"] = -train[column]
    down = mean_ndcg(train, "aturan")
    print(f"{column:20s} naik: {up:.4f} | turun: {down:.4f}")

# %% [markdown]
# ## 3. Regresi biasa: tebak relevansi tiap baris (out-of-fold), lalu urutkan
# Sering sudah kuat; tidak tahu bahwa yang dinilai hanya URUTAN di dalam query.

# %%
import lightgbm as lgb
import numpy as np

train["skor_regresi"] = 0.0
for fold in range(folds.max() + 1):
    is_val = folds == fold
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, verbose=-1)
    model.fit(train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET])
    train.loc[is_val, "skor_regresi"] = model.predict(train.loc[is_val, FEATURES])
print(f"NDCG@{TOP_K} regresi:", mean_ndcg(train, "skor_regresi"))

# %% [markdown]
# ## 4. `LGBMRanker` (lambdarank): `group` = banyak baris tiap query, urut sesuai tabel
# `eval_at` = k yang dinilai. Relevansi harus bilangan bulat ≥ 0.

# %%
import lightgbm as lgb
import numpy as np

RANK_PARAMS = {"objective": "lambdarank", "n_estimators": 2000, "learning_rate": 0.05,
               "num_leaves": 31, "min_child_samples": 20, "verbose": -1}   # UBAH


best_rounds = []


def group_sizes(table):
    sizes = table.groupby(QUERY, sort=False).size()   # urutan = urutan muncul di tabel
    return sizes.to_numpy()


def fit_ranker(table_fit, table_val, features):
    model = lgb.LGBMRanker(**RANK_PARAMS)
    model.fit(table_fit[features], table_fit[TARGET], group=group_sizes(table_fit),
              eval_X=table_val[features], eval_y=table_val[TARGET],
              eval_group=[group_sizes(table_val)], eval_at=[TOP_K],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    best_rounds.append(model.best_iteration_)
    return model


print("contoh group (baris per query):", group_sizes(train)[:5])

# %% [markdown]
# CV per query dengan `fit_ranker`.

# %%
train["skor_ranker"] = 0.0
for fold in range(folds.max() + 1):
    is_val = folds == fold
    model = fit_ranker(train[~is_val], train[is_val], FEATURES)
    train.loc[is_val, "skor_ranker"] = model.predict(train.loc[is_val, FEATURES])
print(f"NDCG@{TOP_K} lambdarank:", mean_ndcg(train, "skor_ranker"),
      "| putaran terbaik per fold:", best_rounds)

# %% [markdown]
# ## 5. Fitur relatif dalam query: "lebih murah/lebih dekat dari produk lain di query ini"
# Nilai mentah sulit dibandingkan antar-query; posisi relatif di dalam query lebih bermakna.

# %%
import lightgbm as lgb

RELATIVE = ["skor_teks", "harga_per_anggaran", "rating_produk", "jarak_km",
            "jumlah_ulasan"]   # UBAH


def add_relative(table):
    by_query = table.groupby(QUERY)
    for column in RELATIVE:
        table[f"{column}_rank_q"] = by_query[column].rank(pct=True)
        table[f"{column}_minus_mean_q"] = table[column] - by_query[column].transform("mean")
    table["query_size"] = by_query[QUERY].transform("size")
    return table


train, test = add_relative(train), add_relative(test)
FEATURES_PLUS = list(FEATURES)
for column in train.columns:
    if column.endswith("_q") or column == "query_size":
        FEATURES_PLUS.append(column)
print(len(FEATURES_PLUS), "fitur:", FEATURES_PLUS[len(FEATURES):][:4], "...")

# %% [markdown]
# CV ulang dengan fitur tambahan.

# %%
best_rounds = []
train["skor_ranker_plus"] = 0.0
for fold in range(folds.max() + 1):
    is_val = folds == fold
    model = fit_ranker(train[~is_val], train[is_val], FEATURES_PLUS)
    train.loc[is_val, "skor_ranker_plus"] = model.predict(train.loc[is_val, FEATURES_PLUS])
print(f"NDCG@{TOP_K} lambdarank + fitur relatif:", mean_ndcg(train, "skor_ranker_plus"))

# %% [markdown]
# ## 6. Latih di semua train → skor test → `submission.csv`
# Format umum: satu baris per (query, item) + skor, ATAU satu baris per query berisi daftar
# item terurut (dipisah spasi). Dua-duanya dibuat; pakai yang diminta soal.

# %%
import lightgbm as lgb
import numpy as np
import pandas as pd

final_params = dict(RANK_PARAMS)
final_params["n_estimators"] = int(np.mean(best_rounds) * 1.1)
final_model = lgb.LGBMRanker(**final_params)
final_model.fit(train[FEATURES_PLUS], train[TARGET], group=group_sizes(train))
test["skor"] = final_model.predict(test[FEATURES_PLUS])
submission = pd.DataFrame()
submission["id"] = test["id"]          # UBAH
submission["skor"] = test["skor"]      # UBAH
submission.to_csv("submission.csv", index=False)
ordered = test.sort_values([QUERY, "skor"], ascending=[True, False])
ranked_lists = ordered.groupby(QUERY)["produk"].apply(" ".join)   # UBAH: kolom item
ranked_lists.reset_index().to_csv("submission_urutan.csv", index=False)
print(ranked_lists.head(3).str[:70])

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
answer = dict(zip(test_labels["id"], test_labels["relevansi"]))
test[TARGET] = test["id"].map(answer)
print(f"NDCG@{TOP_K} test lambdarank + fitur relatif:", mean_ndcg(test, "skor"))
test["aturan"] = test["skor_teks"]
print(f"NDCG@{TOP_K} test aturan skor_teks:", mean_ndcg(test, "aturan"))
