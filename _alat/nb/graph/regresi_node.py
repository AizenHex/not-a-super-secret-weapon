# %% [markdown]
# # Graph — Regresi Nilai Simpul
# **Kapan dipakai:** tiap simpul punya angka (trafik halaman, harga, skor risiko) yang diketahui
# untuk sebagian simpul dan harus ditebak untuk simpul lain.
#
# **Idenya:** struktur graf sendiri sudah informatif (simpul penting cenderung bernilai besar),
# dan tetangga cenderung bernilai mirip. Feature: derajat, clustering, PageRank, derajat
# tetangga, core number, segitiga, dan rata-rata target tetangga yang diketahui.
#
# **Langkah:**
# 1. Split simpul train/uji, hitung feature graf tiap simpul.
# 2. Latih model regresi, bandingkan dengan dan tanpa feature target tetangga.
# 3. Simpan prediksi simpul uji (kembali ke skala asli).
#
# **Data demo:** artikel Wikipedia "chameleon" (2.277 simpul, tautan antar artikel, target =
# rata-rata kunjungan bulanan). Target dilogaritmakan karena sangat miring.
# Prediksi link: lihat notebook `prediksi_link`.

# %%
import networkx as nx
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/graph/wikipedia_chameleon"   # GANTI saat lomba
EDGES_CSV = f"{DATA_FOLDER}/edges.csv"
TARGET_CSV = f"{DATA_FOLDER}/target.csv"

NODE_ID_COL = "id"
SOURCE_COL = "id1"
TARGET_COL = "id2"
VALUE_COL = "target"
TEST_FRACTION = 0.2     # porsi simpul yang disembunyikan targetnya
SEED = 42

# %% [markdown]
# ## Baca data

# %%
edges = pd.read_csv(EDGES_CSV)
values = pd.read_csv(TARGET_CSV)
graph = nx.Graph()
graph.add_nodes_from(values[NODE_ID_COL])
graph.add_edges_from(zip(edges[SOURCE_COL], edges[TARGET_COL]))
graph.remove_edges_from(nx.selfloop_edges(graph))     # buang sisi dari simpul ke dirinya sendiri
values["target_log"] = np.log1p(values[VALUE_COL])    # log(1 + x) meredam nilai yang sangat besar
print(graph.number_of_nodes(), "simpul,", graph.number_of_edges(), "sisi")
print(values[VALUE_COL].describe().round(0))

# %% [markdown]
# ## Langkah 1a — split simpul dan target yang boleh dilihat
# `known_target` hanya berisi simpul train, supaya target simpul uji tidak bocor ke feature.

# %%
train_ids, test_ids = train_test_split(values[NODE_ID_COL], test_size=TEST_FRACTION,
                                       random_state=SEED)
is_train = values[NODE_ID_COL].isin(train_ids)
known_target = {}
for node, value in zip(values[NODE_ID_COL][is_train], values["target_log"][is_train]):
    known_target[node] = value
train_mean = values["target_log"][is_train].mean()
print("simpul train:", len(train_ids), "| simpul uji:", len(test_ids))

# %% [markdown]
# ## Langkah 1b — feature graf tiap simpul
# `neighbor_target` = rata-rata target tetangga yang diketahui; simpul tanpa tetangga yang
# diketahui diberi rata-rata train.

# %%
features = pd.DataFrame()
features["id"] = values[NODE_ID_COL]
features["degree"] = features["id"].map(dict(graph.degree()))
features["clustering"] = features["id"].map(nx.clustering(graph))
features["pagerank"] = features["id"].map(nx.pagerank(graph))
features["neighbor_degree"] = features["id"].map(nx.average_neighbor_degree(graph))
features["triangles"] = features["id"].map(nx.triangles(graph))
features["core"] = features["id"].map(nx.core_number(graph))

neighbor_targets = []
for node in features["id"]:
    known = []
    for neighbor in graph.neighbors(node):
        if neighbor in known_target:
            known.append(known_target[neighbor])
    if len(known) > 0:
        neighbor_targets.append(np.mean(known))
    else:
        neighbor_targets.append(train_mean)
features["neighbor_target"] = neighbor_targets
features.describe().round(3)

# %% [markdown]
# ## Langkah 2 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.
# Pembanding: tebak rata-rata train = R² 0.

# %%
FEATURE_COLS = ["degree", "clustering", "pagerank", "neighbor_degree", "triangles", "core",
                "neighbor_target"]
X_train = features[FEATURE_COLS][is_train]
X_test = features[FEATURE_COLS][~is_train]
y_train = values["target_log"][is_train]
y_test = values["target_log"][~is_train]

model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=4, random_state=SEED)
# model = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, n_jobs=1,
#                       random_state=SEED, verbose=-1)
# model = Ridge(alpha=1.0)                     # baseline linear, sangat cepat

model.fit(X_train, y_train)
pred_test = model.predict(X_test)
print("R2 uji (skala log):", round(r2_score(y_test, pred_test), 4))

# %% [markdown]
# ## Pembanding: tanpa feature target tetangga
# Selisih R² = seberapa besar manfaat "tetangga bernilai mirip".

# %%
structure_cols = FEATURE_COLS[:-1]              # semua feature kecuali neighbor_target
structure_model = clone(model)
structure_model.fit(X_train[structure_cols], y_train)
pred_structure = structure_model.predict(X_test[structure_cols])
print("R2 tanpa neighbor_target:", round(r2_score(y_test, pred_structure), 4))

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importance.sort_values(ascending=False)

# %% [markdown]
# ## Langkah 3 — submission
# Prediksi dikembalikan ke skala asli dengan `expm1` (kebalikan `log1p`).

# %%
predictions = np.expm1(pred_test)

submission = pd.DataFrame()
submission["id"] = values[NODE_ID_COL][~is_train]
submission["target"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
