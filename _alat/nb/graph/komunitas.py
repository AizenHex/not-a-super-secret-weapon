# %% [markdown]
# # Graph — Deteksi Komunitas
# **Kapan dipakai:** "kelompokkan simpul (orang, akun, makalah) yang saling terhubung rapat",
# tanpa label. Contoh: kelompok pertemanan, cluster sitasi, jaringan penipuan.
#
# **Langkah:**
# 1. Baca sisi (edges) → graf networkx, ambil komponen terbesar.
# 2. Jalankan algoritma komunitas, nilai dengan modularity (makin tinggi = makin rapat di dalam).
# 3. Simpan komunitas tiap simpul. Bila ada label asli, bandingkan dengan NMI.
#
# **Data demo:** Cora (2.708 makalah, kolom `label` = topik). Alternatif kecil: Karate Club.

# %%
import networkx as nx
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/graph/cora"   # GANTI saat lomba
# DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/graph/karate"   # graf kecil, label = "club"
NODES_CSV = f"{DATA_FOLDER}/nodes.csv"
EDGES_CSV = f"{DATA_FOLDER}/edges.csv"

NODE_ID_COL = "id"
SOURCE_COL = "source"
TARGET_COL = "target"
LABEL_COL = "label"         # label asli, hanya untuk cek (Karate: "club")
SEED = 42

# %% [markdown]
# ## Baca data dan ambil komponen terbesar
# Komponen = kumpulan simpul yang saling tersambung. Simpul yang terpisah sendiri tidak punya
# komunitas yang bermakna, jadi dipakai komponen terbesar saja.

# %%
nodes = pd.read_csv(NODES_CSV)
edges = pd.read_csv(EDGES_CSV)
graph = nx.from_pandas_edgelist(edges, SOURCE_COL, TARGET_COL)
graph.remove_edges_from(nx.selfloop_edges(graph))     # buang sisi dari simpul ke dirinya sendiri

largest = set()
for component in nx.connected_components(graph):
    if len(component) > len(largest):
        largest = component
graph = graph.subgraph(largest).copy()
print(graph.number_of_nodes(), "simpul,", graph.number_of_edges(), "sisi")
edges.head()

# %% [markdown]
# ## Deteksi komunitas
# Pilih SATU algoritma: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
communities = nx.community.louvain_communities(graph, seed=SEED)   # cepat, umumnya terbaik
# communities = nx.community.greedy_modularity_communities(graph)  # lebih lambat, hasil mirip
# communities = nx.community.asyn_lpa_communities(graph, seed=SEED)   # label propagation
# communities = nx.community.label_propagation_communities(graph)
# communities = nx.community.kernighan_lin_bisection(graph, seed=SEED)   # tepat 2 kelompok
# communities = nx.community.asyn_fluidc(graph, k=4, seed=SEED)    # tepat k kelompok
communities = list(communities)

print("jumlah komunitas:", len(communities))
print("modularity:", round(nx.community.modularity(graph, communities), 4))

# %% [markdown]
# ## Simpan komunitas tiap simpul

# %%
community_of = {}
for number, members in enumerate(communities):
    for node in members:
        community_of[node] = number

node_ids = list(graph.nodes)
node_communities = []
for node in node_ids:
    node_communities.append(community_of[node])

result = pd.DataFrame()
result["id"] = node_ids
result["community"] = node_communities
result.to_csv("hasil.csv", index=False)
result["community"].value_counts().head(10)

# %% [markdown]
# ## Cek dengan label asli (HAPUS bila soal tidak punya label)
# NMI = kecocokan komunitas dengan label asli (1 = sama persis, 0 = tidak berhubungan).

# %%
label_of = {}
for node, label in zip(nodes[NODE_ID_COL], nodes[LABEL_COL]):
    label_of[node] = str(label)
true_labels = []
for node in node_ids:
    true_labels.append(label_of[node])
print("NMI:", round(normalized_mutual_info_score(true_labels, node_communities), 4))
