# %% [markdown]
# # Visualisasi — Graf / Jaringan
# **Kapan dipakai:** "siapa yang paling berpengaruh? apakah ada kelompok (komunitas)? seperti apa
# pola koneksinya?"
#
# **Isi:** layout spring dan kamada-kawai, ukuran simpul = derajat, warna = komunitas, sentralitas
# teratas, matriks adjacency terurut, sebaran derajat, jaringan ego pada graf besar (Cora), graf
# interaktif plotly. Semua gambar disimpan di folder notebook.
#
# **Data demo:** Zachary Karate Club (34 simpul, kecil, mudah dibaca) dan Cora (2.708 makalah,
# sitasi).

# %%
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/graph"   # GANTI saat lomba
KARATE_EDGES_CSV = f"{DATA_FOLDER}/karate/edges.csv"
KARATE_NODES_CSV = f"{DATA_FOLDER}/karate/nodes.csv"
CORA_EDGES_CSV = f"{DATA_FOLDER}/cora/edges.csv"
CORA_NODES_CSV = f"{DATA_FOLDER}/cora/nodes.csv"

SOURCE_COL = "source"     # kolom asal sisi
TARGET_COL = "target"     # kolom tujuan sisi
SEED = 42

# %% [markdown]
# ## Gaya grafik (sama di semua notebook visualisasi)
# Font besar supaya terbaca di slide, palet tetap, garis tepi atas/kanan dimatikan.

# %%
PALETTE = ["#2E6FB7", "#F28E2B", "#3BA272", "#D1495B", "#7B5EA7", "#9AA0A6"]
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = PALETTE
plt.rcParams.update({
    "figure.figsize": (8, 5), "figure.dpi": 100, "savefig.dpi": 200,
    "font.size": 14, "axes.titlesize": 18, "axes.titleweight": "bold", "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})
sns.set_palette(PALETTE)

# %% [markdown]
# ## Baca data (Karate Club) dan cari komunitas
# Komunitas dicari dengan Louvain: simpul dikelompokkan supaya sisi di dalam kelompok banyak dan
# sisi antar kelompok sedikit. Algoritma tidak melihat label klub.

# %%
edges = pd.read_csv(KARATE_EDGES_CSV)
nodes = pd.read_csv(KARATE_NODES_CSV)
karate = nx.from_pandas_edgelist(edges, SOURCE_COL, TARGET_COL)

club_of = {}
for node, club in zip(nodes["id"], nodes["club"]):
    club_of[node] = club

communities = nx.community.louvain_communities(karate, seed=SEED)
community_of = {}
for i, group in enumerate(communities):
    for node in group:
        community_of[node] = i

degree_of = dict(karate.degree())     # simpul -> jumlah tetangga
print(karate.number_of_nodes(), "simpul,", karate.number_of_edges(), "sisi,",
      len(communities), "komunitas Louvain")
edges.head(3)

# %% [markdown]
# ## 1. Layout spring (Fruchterman-Reingold)
# **Kapan dipakai:** gambaran awal graf kecil-menengah; layout paling umum.
# **Cara baca:** simpul yang terhubung ditarik berdekatan, yang tidak terhubung saling menjauh.
# Warna = kelas asli (klub). Posisi bukan koordinat bermakna, hanya kedekatan relatif.

# %%
positions = nx.spring_layout(karate, seed=SEED)
club_colors = []
for node in karate.nodes:
    if club_of[node] == "Mr. Hi":
        club_colors.append(BLUE)
    else:
        club_colors.append(ORANGE)
plt.figure(figsize=(8, 6))
nx.draw_networkx(karate, positions, node_color=club_colors, node_size=350, edge_color=GREY,
                 font_size=9, font_color="white")
plt.title("Karate Club: layout spring")
plt.axis("off")
plt.savefig("graf_spring.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Layout kamada-kawai (dan alternatif lain)
# **Kapan dipakai:** graf kecil dan ingin jarak gambar mendekati jarak jalur terpendek;
# hasilnya deterministik dan rapi. **Cara baca:** sama dengan spring, tapi jarak antar simpul
# lebih mencerminkan jumlah langkah di graf.

# %%
layout = nx.kamada_kawai_layout(karate)
# layout = nx.circular_layout(karate)    # simpul melingkar, cocok untuk graf sangat kecil
# layout = nx.shell_layout(karate)       # lingkaran berlapis
# layout = nx.spectral_layout(karate)    # dari eigenvector, cepat untuk graf besar
plt.figure(figsize=(8, 6))
nx.draw_networkx(karate, layout, node_color=club_colors, node_size=350, edge_color=GREY,
                 font_size=9, font_color="white")
plt.title("Karate Club: layout Kamada-Kawai")
plt.axis("off")
plt.savefig("graf_kamada_kawai.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Ukuran = derajat, warna = komunitas
# **Kapan dipakai:** menunjukkan hub (simpul terhubung banyak) sekaligus kelompok alaminya.
# **Cara baca:** simpul besar = banyak koneksi; warna sama = satu komunitas hasil Louvain.
# Bandingkan dengan grafik 1: apakah komunitas = klub?

# %%
community_colors = []
node_sizes = []
for node in karate.nodes:
    color_number = community_of[node] % len(PALETTE)    # warna diulang bila komunitas > 6
    community_colors.append(PALETTE[color_number])
    node_sizes.append(degree_of[node] * 60)
plt.figure(figsize=(8, 6))
nx.draw_networkx(karate, positions, node_color=community_colors, node_size=node_sizes,
                 edge_color=GREY, font_size=9, font_color="white")
plt.title("Ukuran = derajat, warna = komunitas")
plt.axis("off")
plt.savefig("graf_derajat_komunitas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Sentralitas teratas
# **Kapan dipakai:** menjawab "siapa paling penting?" dengan definisi yang jelas: derajat (paling
# banyak teman), betweenness (jembatan antar kelompok), PageRank (dikenal oleh yang terkenal).
# **Cara baca:** tiga panel, 5 simpul teratas tiap ukuran; simpul yang muncul di ketiganya = kunci.

# %%
centrality = {}
centrality["derajat"] = nx.degree_centrality(karate)
centrality["betweenness"] = nx.betweenness_centrality(karate)
centrality["PageRank"] = nx.pagerank(karate)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, name in zip(axes, centrality):
    values = pd.Series(centrality[name])
    top = values.nlargest(5).sort_values()
    ax.barh(top.index.astype(str), top.values, color=BLUE)
    ax.set_title(name)
    ax.set_xlabel("nilai")
axes[0].set_ylabel("simpul")
plt.savefig("graf_sentralitas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Matriks adjacency terurut per komunitas
# **Kapan dipakai:** graf padat/besar yang gambar simpul-garisnya menjadi gumpalan; menguji
# apakah komunitas benar-benar padat di dalam. **Cara baca:** titik biru = ada sisi.
# Blok padat di diagonal = komunitas yang rapat; titik di luar blok = hubungan antar komunitas.

# %%
ordered_nodes = []
for group in communities:            # simpul komunitas 0 dulu, lalu komunitas 1, dst.
    for node in sorted(group):
        ordered_nodes.append(node)
adjacency = nx.to_numpy_array(karate, nodelist=ordered_nodes)   # 1 = ada sisi
plt.figure(figsize=(6, 6))
sns.heatmap(adjacency, cmap=["white", BLUE], cbar=False, square=True,
            xticklabels=ordered_nodes, yticklabels=ordered_nodes)
plt.title("Adjacency terurut per komunitas")
plt.grid(False)
plt.savefig("graf_adjacency.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Graf besar: Cora (sitasi makalah)

# %%
cora_edges = pd.read_csv(CORA_EDGES_CSV)
cora_nodes = pd.read_csv(CORA_NODES_CSV, usecols=["id", "label"])
cora = nx.from_pandas_edgelist(cora_edges, SOURCE_COL, TARGET_COL)
topic_of = {}
for node, topic in zip(cora_nodes["id"], cora_nodes["label"]):
    topic_of[node] = topic
print(cora.number_of_nodes(), "simpul,", cora.number_of_edges(), "sisi,",
      nx.number_connected_components(cora), "komponen terhubung")

# %% [markdown]
# ## 6. Sebaran derajat (skala log-log)
# **Kapan dipakai:** memeriksa apakah ada hub ekstrem (jaringan sosial/sitasi biasanya begitu).
# **Cara baca:** sumbu x = derajat, y = jumlah simpul. Garis hampir lurus menurun = sedikit hub
# raksasa dan banyak simpul biasa (power law).

# %%
cora_degree = pd.Series(dict(cora.degree()))       # simpul -> derajat
degree_counts = cora_degree.value_counts().sort_index()
plt.scatter(degree_counts.index, degree_counts.values, color=BLUE, s=40)
plt.xscale("log")
plt.yscale("log")
plt.title("Sebaran derajat Cora")
plt.xlabel("derajat (jumlah sitasi/koneksi)")
plt.ylabel("jumlah makalah")
plt.savefig("graf_distribusi_derajat.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Jaringan ego (lingkungan satu simpul pada graf besar)
# **Kapan dipakai:** graf terlalu besar untuk digambar utuh; fokus pada satu simpul penting.
# **Cara baca:** simpul besar = hub dengan derajat tertinggi, titik lain = tetangga langsungnya.
# Warna = topik makalah. Tetangga sewarna = hub dikelilingi topik sejenis (homofili); garis
# antar tetangga = saling mengutip.

# %%
hub = cora_degree.idxmax()                  # simpul dengan derajat tertinggi
ego = nx.ego_graph(cora, hub, radius=1)     # radius=2 untuk tetangga dari tetangga
topics = sorted(cora_nodes["label"].unique())    # 7 topik, warna diulang bila > 6
topic_colors = []
ego_sizes = []
for node in ego.nodes:
    topic_number = topics.index(topic_of[node])
    topic_colors.append(PALETTE[topic_number % len(PALETTE)])
    if node == hub:
        ego_sizes.append(800)
    else:
        ego_sizes.append(40)
ego_positions = nx.spring_layout(ego, seed=SEED, k=0.3)
plt.figure(figsize=(8, 8))
nx.draw_networkx(ego, ego_positions, node_color=topic_colors, node_size=ego_sizes,
                 with_labels=False, edge_color=GREY, width=0.5)
plt.title(f"Ego-network hub (derajat {cora_degree[hub]}), {ego.number_of_nodes()} simpul")
plt.axis("off")
plt.savefig("graf_ego_cora.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Graf interaktif (plotly)
# **Kapan dipakai:** lampiran interaktif; menelusuri siapa terhubung ke siapa lewat hover.
# **Cara baca:** arahkan kursor ke simpul untuk melihat ID, klub, derajat, dan komunitasnya;
# zoom untuk detail. Ukuran = derajat, warna = klub.

# %%
line_x = []
line_y = []
for start, end in karate.edges:       # None = putus garis sebelum sisi berikutnya
    line_x += [positions[start][0], positions[end][0], None]
    line_y += [positions[start][1], positions[end][1], None]
fig = go.Figure(go.Scatter(x=line_x, y=line_y, mode="lines", line={"color": GREY, "width": 1},
                           hoverinfo="none"))
node_x = []
node_y = []
plotly_sizes = []
hover_texts = []
for node in karate.nodes:
    node_x.append(positions[node][0])
    node_y.append(positions[node][1])
    plotly_sizes.append(degree_of[node] * 3 + 8)
    hover_texts.append(f"simpul {node}, {club_of[node]}, derajat {degree_of[node]}, "
                       f"komunitas {community_of[node]}")
fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", text=hover_texts,
                         marker={"size": plotly_sizes, "color": club_colors}, hoverinfo="text"))
fig.update_layout(title="Karate Club (interaktif)", showlegend=False, plot_bgcolor="white",
                  xaxis={"visible": False}, yaxis={"visible": False})
fig.write_html("graf_interaktif.html", include_plotlyjs=True)
fig.show()
