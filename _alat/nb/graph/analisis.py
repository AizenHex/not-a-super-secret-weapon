# %% [markdown]
# # Graph — Analisis Dasar (statistik dan sentralitas)
# **Kapan dipakai:** langkah pertama pada data graf: seberapa besar, seberapa rapat, simpul mana
# yang paling penting. Hasilnya juga bisa jadi feature untuk model tabular.
#
# **Langkah:**
# 1. Baca simpul dan sisi → graf networkx.
# 2. Statistik graf (kerapatan, komponen, clustering, diameter) dan sebaran derajat.
# 3. Sentralitas tiap simpul (derajat, PageRank, betweenness, closeness), simpan ke csv.
#
# **Data demo:** Cora (sitasi 2.708 makalah) dan Karate Club (untuk gambar).

# %%
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/graph"   # GANTI saat lomba
NODES_CSV = f"{DATA_FOLDER}/cora/nodes.csv"
EDGES_CSV = f"{DATA_FOLDER}/cora/edges.csv"
KARATE_NODES_CSV = f"{DATA_FOLDER}/karate/nodes.csv"
KARATE_EDGES_CSV = f"{DATA_FOLDER}/karate/edges.csv"

NODE_ID_COL = "id"
SOURCE_COL = "source"
TARGET_COL = "target"
BETWEENNESS_SAMPLES = 200   # betweenness dihitung dari 200 simpul acak (versi penuh lambat)
TOP = 10
SEED = 42

# %% [markdown]
# ## Baca data
# Simpul ditambahkan dulu supaya simpul tanpa sisi tetap ikut dihitung.

# %%
nodes = pd.read_csv(NODES_CSV, usecols=[NODE_ID_COL, "label"])
edges = pd.read_csv(EDGES_CSV)
graph = nx.Graph()
graph.add_nodes_from(nodes[NODE_ID_COL])
graph.add_edges_from(zip(edges[SOURCE_COL], edges[TARGET_COL]))
graph.remove_edges_from(nx.selfloop_edges(graph))     # buang sisi dari simpul ke dirinya sendiri
print(graph.number_of_nodes(), "simpul,", graph.number_of_edges(), "sisi")
edges.head()

# %% [markdown]
# ## Statistik graf
# Jalur terpendek dan diameter hanya bisa dihitung di komponen terbesar (LCC), karena antar
# komponen tidak ada jalur.

# %%
components = list(nx.connected_components(graph))
largest = set()
for component in components:
    if len(component) > len(largest):
        largest = component
lcc = graph.subgraph(largest).copy()

degree_of = dict(graph.degree())        # simpul -> jumlah tetangga
stats = {}
stats["simpul"] = graph.number_of_nodes()
stats["sisi"] = graph.number_of_edges()
stats["kerapatan"] = nx.density(graph)
stats["derajat rata-rata"] = sum(degree_of.values()) / len(degree_of)
stats["jumlah komponen"] = len(components)
stats["komponen terbesar"] = len(lcc)
stats["clustering rata-rata"] = nx.average_clustering(graph)
stats["transitivitas"] = nx.transitivity(graph)
stats["jalur terpendek rata-rata (LCC)"] = nx.average_shortest_path_length(lcc)
stats["diameter (LCC)"] = nx.diameter(lcc)
stats = pd.Series(stats)
stats.to_csv("statistik.csv")
print(stats)

# %% [markdown]
# ## Sebaran derajat
# Sumbu y skala log: banyak simpul berderajat kecil, sedikit hub berderajat besar.

# %%
degrees = pd.Series(degree_of)
degrees.hist(bins=50)
plt.title("Sebaran derajat")
plt.xlabel("derajat")
plt.yscale("log")
plt.show()

# %% [markdown]
# ## Sentralitas tiap simpul
# - **Derajat:** jumlah tetangga.  **PageRank:** pengaruh (ditunjuk oleh simpul penting).
# - **Betweenness:** peran jembatan antar kelompok.  **Closeness:** dekat ke semua simpul lain.

# %%
pagerank = nx.pagerank(graph)
betweenness = nx.betweenness_centrality(graph, k=BETWEENNESS_SAMPLES, seed=SEED)
closeness = nx.closeness_centrality(graph)

centrality = pd.DataFrame()
centrality["id"] = nodes[NODE_ID_COL]
centrality["label"] = nodes["label"]
centrality["degree"] = centrality["id"].map(degree_of)
centrality["pagerank"] = centrality["id"].map(pagerank)
centrality["betweenness"] = centrality["id"].map(betweenness)
centrality["closeness"] = centrality["id"].map(closeness)
centrality.to_csv("sentralitas.csv", index=False)
top_nodes = centrality.sort_values("pagerank", ascending=False)
top_nodes.head(TOP)

# %% [markdown]
# ## Gambar graf kecil: Karate Club (warna = klub)

# %%
karate_nodes = pd.read_csv(KARATE_NODES_CSV)
karate_edges = pd.read_csv(KARATE_EDGES_CSV)
karate = nx.Graph()
karate.add_nodes_from(karate_nodes[NODE_ID_COL])
karate.add_edges_from(zip(karate_edges[SOURCE_COL], karate_edges[TARGET_COL]))
colors = []
for club in karate_nodes["club"]:
    if club == "Mr. Hi":
        colors.append("tab:blue")
    else:
        colors.append("tab:orange")
positions = nx.spring_layout(karate, seed=SEED)
plt.figure(figsize=(6, 5))
nx.draw(karate, positions, node_color=colors, with_labels=True)
plt.savefig("karate.png", dpi=120)
plt.show()
