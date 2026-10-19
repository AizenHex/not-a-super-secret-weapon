# %% [markdown]
# # Visualisasi — Kategori (membandingkan kelompok)
# **Kapan dipakai:** "kategori mana yang terbanyak? berapa porsinya? kategori mana yang paling
# berisiko?"
#
# **Isi:** bar terurut, top-N + "lainnya", proporsi 100% bertumpuk, target rate per kategori
# (dengan jumlah data dan selang kepercayaan), heatmap target rate dua kategori, treemap interaktif.
# Semua gambar disimpan di folder notebook.
#
# **Data demo:** Online Retail (banyak kategori) dan Titanic (target selamat).

# %%
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan"   # GANTI saat lomba
RETAIL_CSV = f"{DATA_FOLDER}/unsupervised/online_retail.csv"
TRAIN_CSV = f"{DATA_FOLDER}/tabular/classification/titanic_train.csv"

MANY_COL = "Description"      # kategori dengan banyak nilai (untuk top-N)
COUNTRY_COL = "Country"
TARGET = "survived"           # target 0/1
CATEGORY_COL = "embark_town"
TOP_N = 10

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
# ## Baca data

# %%
retail = pd.read_csv(RETAIL_CSV, usecols=[MANY_COL, COUNTRY_COL])
train = pd.read_csv(TRAIN_CSV)
train["jenis_kelamin"] = train["sex"].map({"male": "laki-laki", "female": "perempuan"})
print(retail.shape, train.shape)
print("negara:", retail[COUNTRY_COL].nunique(), "| produk:", retail[MANY_COL].nunique())

# %% [markdown]
# ## 1. Bar terurut (horizontal)
# **Kapan dipakai:** membandingkan jumlah antar kategori, paling mudah dibaca manusia.
# **Cara baca:** urutkan dari terbesar, bar horizontal supaya nama panjang terbaca.
# Mulai sumbu dari 0 supaya panjang bar jujur.

# %%
counts = retail[COUNTRY_COL].value_counts().head(8)
counts = counts.sort_values()   # terkecil di bawah supaya terbesar di atas pada bar horizontal
counts.plot.barh(color=BLUE)
for i, value in enumerate(counts):
    plt.text(value, i, f" {value:,}", va="center")
plt.title("Transaksi per negara (8 teratas)")
plt.xlabel("jumlah baris transaksi")
plt.xlim(0, counts.max() * 1.15)
# Alternatif seaborn: sns.countplot(data=retail, y="Country", order=counts.index[::-1])
# Alternatif lollipop: plt.hlines(counts.index, 0, counts.values)
plt.savefig("kategori_bar_terurut.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Top-N + "lainnya"
# **Kapan dipakai:** kategori terlalu banyak (produk, kota, merek) sehingga grafik penuh.
# **Cara baca:** N kategori terbesar ditampilkan, sisanya digabung jadi satu bar abu-abu.
# Bar "lainnya" yang besar berarti datanya tersebar luas.

# %%
all_counts = retail[MANY_COL].value_counts()
summary = all_counts.head(TOP_N).copy()
summary["(lainnya)"] = all_counts.iloc[TOP_N:].sum()
colors = [BLUE] * TOP_N + [GREY]
summary = summary.iloc[::-1]   # dibalik: terbesar di atas
colors = colors[::-1]
summary.plot.barh(color=colors, figsize=(9, 6))
plt.title(f"{TOP_N} produk terlaris vs sisanya ({all_counts.size:,} produk)")
plt.xlabel("jumlah baris transaksi")
plt.savefig("kategori_top_n_lainnya.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Proporsi 100% bertumpuk
# **Kapan dipakai:** membandingkan komposisi antar kelompok yang jumlahnya berbeda-beda.
# **Cara baca:** tiap bar panjangnya 100%, lihat seberapa besar potongan tiap warna.
# Garis putus-putus 50% membantu membandingkan.

# %%
shares = pd.crosstab(train["class"], train["jenis_kelamin"], normalize="index") * 100
shares.plot.barh(stacked=True, color=[ORANGE, BLUE])
plt.axvline(50, color="white", linestyle="--")
plt.title("Komposisi jenis kelamin per kelas tiket")
plt.xlabel("persen penumpang")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
# Alternatif jumlah mutlak (bukan persen): pd.crosstab(...).plot.barh(stacked=True)
plt.savefig("kategori_proporsi_100.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Target rate per kategori
# **Kapan dipakai:** mengecek apakah sebuah kategori berpengaruh ke target (fitur layak dipakai?).
# **Cara baca:** tinggi bar = persen target=1, garis hitam = selang kepercayaan 95%,
# angka n = jumlah data. Selang lebar / n kecil = jangan terlalu percaya.

# %%
overall_rate = train[TARGET].mean() * 100
train["persen"] = train[TARGET] * 100
sns.barplot(data=train, x=CATEGORY_COL, y="persen", errorbar=("ci", 95), color=BLUE)
category_counts = train[CATEGORY_COL].value_counts()
for i, label in enumerate(plt.gca().get_xticklabels()):
    plt.text(i, 3, f"n={category_counts[label.get_text()]}", ha="center", color="white")
plt.axhline(overall_rate, color=RED, linestyle="--", label=f"rata-rata {overall_rate:.0f}%")
plt.title("Persen selamat per kota naik")
plt.xlabel("kota naik")
plt.ylabel("persen selamat")
plt.legend()
plt.savefig("kategori_target_rate.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Heatmap target rate dua kategori
# **Kapan dipakai:** melihat interaksi dua kategori terhadap target.
# **Cara baca:** warna gelap = persen target tinggi, angka di dalam sel = persen (selamat).

# %%
rate_table = train.pivot_table(index="class", columns="jenis_kelamin", values=TARGET,
                               aggfunc="mean") * 100
sns.heatmap(rate_table, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "persen selamat"})
plt.title("Persen selamat: kelas tiket x jenis kelamin")
plt.xlabel("")
plt.ylabel("")
# Alternatif dengan jumlah data per sel: train.pivot_table(..., aggfunc="count")
plt.savefig("kategori_heatmap_target.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Treemap interaktif (plotly)
# **Kapan dipakai:** kategori bertingkat atau banyak, ingin lihat porsi dan menelusuri.
# **Cara baca:** luas kotak = jumlah data, klik kotak untuk masuk ke dalamnya.

# %%
country_counts = retail[COUNTRY_COL].value_counts()
countries = pd.DataFrame()
countries["negara"] = country_counts.index
countries["jumlah"] = country_counts.to_numpy()
fig = px.treemap(countries, path=["negara"], values="jumlah", title="Transaksi per negara",
                 color_discrete_sequence=PALETTE)
fig.write_html("kategori_treemap.html", include_plotlyjs=True)
fig.show()
