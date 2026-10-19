# %% [markdown]
# # Graph — Klasifikasi Simpul (tanpa deep learning)
# **Kapan dipakai:** tiap simpul (makalah, akun, pengguna) punya feature sendiri dan terhubung
# dengan simpul lain; tebak label simpul yang belum diketahui.
#
# **Idenya:** feature asli simpul ditambah feature struktur graf (derajat, PageRank, clustering,
# core number, embedding spektral), lalu model tabular biasa.
#
# **Langkah:**
# 1. Bangun graf, hitung feature struktur tiap simpul.
# 2. Split simpul, latih model, evaluasi.
# 3. Simpan prediksi simpul uji.
#
# **Data demo:** Cora (7 topik makalah, feature kata `w0..w1432`). Versi GNN: notebook `gcn`.

# %%
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/graph/cora"   # GANTI saat lomba
NODES_CSV = f"{DATA_FOLDER}/nodes.csv"
EDGES_CSV = f"{DATA_FOLDER}/edges.csv"

NODE_ID_COL = "id"
SOURCE_COL = "source"
TARGET_COL = "target"
LABEL_COL = "label"
DROP_COLS = ["id", "label"]     # kolom yang bukan feature
SPECTRAL_DIM = 16               # jumlah dimensi embedding spektral
SEED = 42

# %% [markdown]
# ## Baca data

# %%
nodes = pd.read_csv(NODES_CSV)
edges = pd.read_csv(EDGES_CSV)
missing = nodes.isna().sum()
print(nodes.shape, edges.shape, "| missing:", missing.sum())
nodes[DROP_COLS].head()

# %% [markdown]
# ## EDA: jumlah simpul per label

# %%
label_counts = nodes[LABEL_COL].value_counts()
print(label_counts)
label_counts.plot.barh(figsize=(6, 3))
plt.show()

# %% [markdown]
# ## Bangun graf
# `adjacency` = matriks simpul x simpul, 1 bila ada sisi (urutan baris = urutan `nodes`).

# %%
graph = nx.Graph()
graph.add_nodes_from(nodes[NODE_ID_COL])
graph.add_edges_from(zip(edges[SOURCE_COL], edges[TARGET_COL]))
graph.remove_edges_from(nx.selfloop_edges(graph))     # buang sisi dari simpul ke dirinya sendiri
node_order = list(nodes[NODE_ID_COL])
adjacency = nx.to_scipy_sparse_array(graph, nodelist=node_order, format="csr")
print(graph.number_of_nodes(), "simpul,", graph.number_of_edges(), "sisi")

# %% [markdown]
# ## Langkah 1 — feature struktur graf
# - **degree** jumlah tetangga, **pagerank** pengaruh, **clustering** tetangga saling kenal,
#   **core** seberapa dalam di inti graf.
# - **Embedding spektral:** posisi simpul di "peta" graf (simpul yang berdekatan = mirip).

# %%
structure = pd.DataFrame()
structure["degree"] = nodes[NODE_ID_COL].map(dict(graph.degree()))
structure["pagerank"] = nodes[NODE_ID_COL].map(nx.pagerank(graph))
structure["clustering"] = nodes[NODE_ID_COL].map(nx.clustering(graph))
structure["core"] = nodes[NODE_ID_COL].map(nx.core_number(graph))

spectral = SpectralEmbedding(n_components=SPECTRAL_DIM, affinity="precomputed",
                             random_state=SEED)
spectral_values = spectral.fit_transform(adjacency.toarray())
for dim in range(SPECTRAL_DIM):
    structure[f"spectral{dim}"] = spectral_values[:, dim]

word_features = nodes.drop(columns=DROP_COLS)
all_features = pd.concat([word_features, structure], axis=1)
print("jumlah feature:", all_features.shape[1])
structure.head()

# %% [markdown]
# ## Split simpul
# Untuk melihat nilai tambah struktur graf, ganti `X = all_features` dengan `X = word_features`.

# %%
X = all_features
# X = word_features                 # hanya feature kata, tanpa struktur graf
y = nodes[LABEL_COL]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y,
                                                  random_state=SEED)

# %% [markdown]
# ## Langkah 2 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
# model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500,
#                                                                random_state=SEED))
# model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
#                        random_state=SEED, verbose=-1)
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = ExtraTreesClassifier(n_estimators=500, random_state=SEED)
# model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,),
#                                                       max_iter=1000, random_state=SEED))

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi

# %%
pred_val = model.predict(X_val)
print("F1 macro:", f1_score(y_val, pred_val, average="macro"))
print("Akurasi:", accuracy_score(y_val, pred_val))
print(classification_report(y_val, pred_val))
ConfusionMatrixDisplay.from_predictions(y_val, pred_val, xticks_rotation=45, normalize="true")
plt.show()

# %% [markdown]
# ## Langkah 3 — simpan prediksi simpul uji
# Cora tidak punya file test, jadi yang disimpan adalah prediksi untuk simpul validasi.
# Saat lomba: simpul tanpa label = test. Latih di semua simpul berlabel, prediksi simpul test.

# %%
predictions = model.predict(X_val)

submission = pd.DataFrame()
submission["id"] = nodes[NODE_ID_COL][X_val.index]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
