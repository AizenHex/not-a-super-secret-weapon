# %% [markdown]
# # Visualisasi — Embedding dan Klaster (melihat data berdimensi tinggi dalam 2D)
# **Kapan dipakai:** "apakah data punya kelompok alami? berapa jumlah klaster yang masuk akal?
# apakah embedding (fitur gambar/teks) memisahkan kelas?"
#
# **Isi:** PCA, t-SNE, UMAP 2D diwarnai label, scree plot, elbow, silhouette plot, dendrogram,
# UMAP interaktif. Semua gambar disimpan di folder notebook.
#
# **Data demo:** Digits (tulisan tangan 0-9, 64 piksel). Untuk embedding gambar atau teks: ganti
# `X` dengan matriks embedding (baris x dimensi) dan `labels` dengan kelasnya.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/digits.csv"

LABEL_COL = "digit"    # kelas asli (hanya untuk mewarnai)
N_CLUSTERS = 10        # k untuk KMeans (lihat elbow dan silhouette)
DENDROGRAM_ROWS = 80   # dendrogram hanya untuk data kecil
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
# ## Baca data (fitur dibakukan: rata-rata 0, simpangan 1)

# %%
table = pd.read_csv(DATA_CSV)
labels = table[LABEL_COL]
X = StandardScaler().fit_transform(table.drop(columns=[LABEL_COL]))
print(X.shape, labels.nunique(), "kelas")
class_colors = sns.color_palette("tab10", labels.nunique())

# %% [markdown]
# ## 1. PCA 2D (cepat, linear)
# **Kapan dipakai:** langkah pertama melihat struktur data, cepat dan hasilnya selalu sama.
# **Cara baca:** tiap titik = satu data, warna = kelas asli. Kelas yang terpisah rapi mudah
# dibedakan model sederhana, yang menumpuk butuh model non-linear.

# %%
pca = PCA(n_components=2, random_state=SEED)
pca_points = pca.fit_transform(X)
sns.scatterplot(x=pca_points[:, 0], y=pca_points[:, 1], hue=labels, palette=class_colors, s=18,
                legend="full")
plt.title("PCA 2D data Digits")
plt.xlabel(f"komponen 1 ({pca.explained_variance_ratio_[0]:.0%} variansi)")
plt.ylabel(f"komponen 2 ({pca.explained_variance_ratio_[1]:.0%} variansi)")
plt.legend(title="digit", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.savefig("embed_pca.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. t-SNE 2D
# **Kapan dipakai:** melihat kelompok lokal dengan jelas (embedding gambar/teks).
# **Cara baca:** titik berdekatan = mirip. Jarak ANTAR gerombol dan ukuran gerombol tidak bermakna,
# hanya siapa bertetangga dengan siapa. Hasil berubah tiap seed, jangan menarik kesimpulan kuat.

# %%
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
tsne_points = tsne.fit_transform(X)
sns.scatterplot(x=tsne_points[:, 0], y=tsne_points[:, 1], hue=labels, palette=class_colors, s=18,
                legend=False)
plt.title("t-SNE 2D data Digits")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("embed_tsne.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. UMAP 2D
# **Kapan dipakai:** seperti t-SNE tapi lebih cepat dan lebih menjaga struktur global.
# **Cara baca:** sama dengan t-SNE. Atur `n_neighbors` (kecil = detail lokal, besar = gambaran
# global) dan `min_dist` (kecil = gerombol rapat).

# %%
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
umap_points = reducer.fit_transform(X)
sns.scatterplot(x=umap_points[:, 0], y=umap_points[:, 1], hue=labels, palette=class_colors, s=18,
                legend=False)
plt.title("UMAP 2D data Digits")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
# Alternatif tanpa label (data tanpa kelas): sns.scatterplot(x=..., y=..., color=BLUE, s=18)
plt.savefig("embed_umap.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Scree plot (berapa dimensi PCA yang cukup?)
# **Kapan dipakai:** memilih jumlah komponen PCA untuk fitur atau kompresi.
# **Cara baca:** garis naik menunjukkan variansi kumulatif. Ambil jumlah komponen saat mencapai
# 90-95% (garis putus-putus).

# %%
full_pca = PCA(random_state=SEED)
full_pca.fit(X)
cumulative = np.cumsum(full_pca.explained_variance_ratio_) * 100
enough = int(np.argmax(cumulative >= 90)) + 1     # komponen pertama yang mencapai 90%
plt.plot(range(1, len(cumulative) + 1), cumulative, color=BLUE, linewidth=3)
plt.axhline(90, color=ORANGE, linestyle="--")
plt.axvline(enough, color=ORANGE, linestyle="--")
plt.title(f"{enough} komponen menjelaskan 90% variansi")
plt.xlabel("jumlah komponen PCA")
plt.ylabel("variansi kumulatif (%)")
plt.savefig("embed_scree.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Elbow (inertia vs jumlah klaster)
# **Kapan dipakai:** memilih jumlah klaster KMeans bila tidak ada petunjuk.
# **Cara baca:** inertia selalu turun. Pilih k di "siku" tempat penurunan mulai melandai.
# Kalau tidak ada siku jelas, lihat silhouette.

# %%
cluster_counts = range(2, 15)
inertia = []
for n_clusters in cluster_counts:
    trial = KMeans(n_clusters, n_init=5, random_state=SEED)
    trial.fit(X)
    inertia.append(trial.inertia_)
plt.plot(cluster_counts, inertia, marker="o", color=BLUE, linewidth=3)
plt.title("Elbow: inertia vs jumlah klaster")
plt.xlabel("jumlah klaster k")
plt.ylabel("inertia")
plt.savefig("embed_elbow.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Silhouette plot
# **Kapan dipakai:** menilai kualitas klaster per klaster, bukan hanya satu angka.
# **Cara baca:** lebar bar = ukuran klaster, panjang = seberapa pas tiap titik di klasternya
# (1 = pas, 0 = di perbatasan, negatif = salah klaster). Bar di bawah garis merah (rata-rata)
# adalah klaster lemah.

# %%
kmeans = KMeans(N_CLUSTERS, n_init=5, random_state=SEED)
clusters = kmeans.fit_predict(X)
silhouette_values = silhouette_samples(X, clusters)
bottom = 0
for cluster in range(N_CLUSTERS):
    cluster_values = np.sort(silhouette_values[clusters == cluster])
    plt.fill_betweenx(np.arange(bottom, bottom + len(cluster_values)), 0, cluster_values,
                      color=class_colors[cluster])
    plt.text(-0.08, bottom + len(cluster_values) / 2, str(cluster))
    bottom = bottom + len(cluster_values) + 10     # jarak 10 antar klaster
plt.axvline(silhouette_score(X, clusters), color=RED, linestyle="--", label="rata-rata")
plt.title("Silhouette per klaster")
plt.xlabel("nilai silhouette")
plt.ylabel("klaster")
plt.yticks([])
plt.legend()
plt.savefig("embed_silhouette.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Klaster vs kelas asli di peta UMAP
# **Kapan dipakai:** mengecek apakah klaster hasil model sesuai dengan kelas asli (atau menamai
# klaster). **Cara baca:** kiri = klaster KMeans, kanan = kelas asli. Warna kiri mirip kanan =
# klaster cocok dengan kelas.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(umap_points[:, 0], umap_points[:, 1], c=clusters, cmap="tab10", s=12)
axes[0].set_title("Klaster KMeans")
axes[1].scatter(umap_points[:, 0], umap_points[:, 1], c=labels, cmap="tab10", s=12)
axes[1].set_title("Kelas asli")
plt.savefig("embed_klaster_vs_kelas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Dendrogram (klaster hierarki)
# **Kapan dipakai:** data kecil (< 500 baris), ingin tahu struktur bersarang dan jumlah klaster
# lewat tinggi pemotongan. **Cara baca:** gabungan di bawah = sangat mirip. Potong garis
# horizontal di tinggi tertentu, jumlah garis vertikal yang terpotong = jumlah klaster.

# %%
links = linkage(X[:DENDROGRAM_ROWS], method="ward")
cut_height = 0.7 * links[-1, 2]     # warna berbeda di bawah 70% tinggi gabungan terakhir
plt.figure(figsize=(12, 4.5))
dendrogram(links, labels=labels.to_numpy()[:DENDROGRAM_ROWS], leaf_font_size=9,
           color_threshold=cut_height)
plt.title(f"Dendrogram {DENDROGRAM_ROWS} data pertama (Ward)")
plt.xlabel("label asli")
plt.ylabel("jarak antar klaster")
# Alternatif ambil 10 klaster: scipy.cluster.hierarchy.fcluster(links, 10, "maxclust")
plt.savefig("embed_dendrogram.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. UMAP interaktif (plotly)
# **Kapan dipakai:** menelusuri titik aneh atau salah letak (hover label/ID, klik legenda).
# **Cara baca:** arahkan kursor ke titik untuk melihat label. Klik nama kelas di legenda untuk
# menyembunyikan/menampilkan kelas itu.

# %%
plot_table = pd.DataFrame()
plot_table["umap1"] = umap_points[:, 0]
plot_table["umap2"] = umap_points[:, 1]
plot_table["digit"] = labels.astype(str)
plot_table["klaster"] = clusters
fig = px.scatter(plot_table, x="umap1", y="umap2", color="digit", hover_data=["klaster"],
                 color_discrete_sequence=px.colors.qualitative.T10,
                 title="UMAP Digits (interaktif)")
fig.write_html("embed_umap_interaktif.html", include_plotlyjs=True)
fig.show()
