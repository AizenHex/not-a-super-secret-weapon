# %% [markdown]
# # Visualisasi — Distribusi (sebaran satu angka)
# **Kapan dipakai:** "seperti apa sebaran kolom ini? miring? ada outlier? beda antar kelompok?"
#
# **Isi:** histogram+KDE, box, violin, ECDF, sebaran per kelas target, skew sebelum/sesudah log,
# versi interaktif plotly. Semua gambar disimpan di folder notebook (PNG 200 dpi).
#
# **Data demo:** Titanic (umur, tarif) dan Diamonds (harga, miring ke kanan).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/classification/titanic_train.csv"
DIAMONDS_CSV = f"{DATA_FOLDER}/regression/diamonds_train.csv"

MAIN_COL = "age"        # kolom angka yang disorot
GROUP_COL = "pclass"    # kolom kategori untuk membandingkan sebaran
TARGET = "survived"     # kelas target (0/1)
SKEWED_COL = "price"    # kolom miring ke kanan (pakai data diamonds)

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
train = pd.read_csv(TRAIN_CSV)
train["nasib"] = train[TARGET].map({0: "meninggal", 1: "selamat"})
diamonds = pd.read_csv(DIAMONDS_CSV)
print(train.shape, diamonds.shape)
train[[MAIN_COL, "fare"]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1)

# %% [markdown]
# ## 1. Histogram + KDE
# **Kapan dipakai:** pertama kali melihat sebaran satu kolom angka.
# **Cara baca:** batang tinggi = banyak data di rentang itu, garis halus (KDE) = bentuk sebarannya.
# Satu puncak = wajar, dua puncak = ada dua kelompok campur, ekor panjang = miring.

# %%
sns.histplot(train[MAIN_COL], bins=30, kde=True, color=BLUE)
plt.axvline(train[MAIN_COL].median(), color=ORANGE, linestyle="--", label="median")
plt.title("Sebaran umur penumpang")
plt.xlabel("umur (tahun)")
plt.ylabel("jumlah penumpang")
plt.legend()
# Alternatif Matplotlib murni: plt.hist(train[MAIN_COL].dropna(), bins=30)
# Alternatif banyak kolom sekaligus: train[["age", "fare"]].hist(bins=30, figsize=(10, 4))
plt.savefig("distribusi_hist_kde.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Box plot
# **Kapan dipakai:** membandingkan median dan outlier beberapa kelompok sekaligus.
# **Cara baca:** kotak = 50% data tengah (Q1–Q3), garis dalam kotak = median, titik di luar kumis =
# kandidat outlier (di luar 1.5 x IQR).

# %%
sns.boxplot(data=train, x=GROUP_COL, y="fare", color=BLUE)
plt.yscale("log")
plt.title("Tarif per kelas tiket (skala log)")
plt.xlabel("kelas tiket")
plt.ylabel("tarif (skala log)")
# Alternatif untuk data besar (lebih banyak kuantil): sns.boxenplot(data=train, x=..., y=...)
plt.savefig("distribusi_box.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Violin plot
# **Kapan dipakai:** ingin tahu bentuk sebaran (bukan hanya median) tiap kelompok.
# **Cara baca:** lebar "biola" = banyaknya data di nilai itu, kotak kecil di dalam = box plot.

# %%
sns.violinplot(data=train, x=GROUP_COL, y=MAIN_COL, inner="quartile", color=BLUE)
plt.title("Sebaran umur per kelas tiket")
plt.xlabel("kelas tiket")
plt.ylabel("umur (tahun)")
# Alternatif titik asli (data kecil): sns.stripplot(data=train, x=..., y=..., alpha=0.4)
# Alternatif dua sisi per kelas target: sns.violinplot(..., hue="nasib", split=True)
plt.savefig("distribusi_violin.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. ECDF (distribusi kumulatif)
# **Kapan dipakai:** membandingkan dua kelompok tanpa memilih jumlah bin, menjawab "berapa persen
# data di bawah nilai X?". **Cara baca:** kurva di sebelah kiri = nilainya cenderung lebih kecil.
# Tinggi kurva di nilai X = proporsi data yang <= X.

# %%
sns.ecdfplot(data=train, x="fare", hue=GROUP_COL, palette=[BLUE, ORANGE, GREEN])
plt.xscale("log")
plt.title("Berapa persen penumpang membayar <= tarif tertentu")
plt.xlabel("tarif (skala log)")
plt.ylabel("proporsi kumulatif")
plt.savefig("distribusi_ecdf.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Sebaran per kelas target
# **Kapan dipakai:** menilai apakah sebuah fitur memisahkan kelas (jadi fitur berguna).
# **Cara baca:** makin berbeda bentuk dua kurva, makin informatif fitur itu. Kalau menumpuk, lemah.

# %%
sns.histplot(data=train, x=MAIN_COL, hue="nasib", bins=25, stat="density", common_norm=False,
             element="step", palette=[RED, BLUE])
plt.title("Sebaran umur: selamat vs meninggal")
plt.xlabel("umur (tahun)")
plt.ylabel("kepadatan")
# Alternatif garis halus: sns.kdeplot(data=train, x=..., hue="nasib", common_norm=False, fill=True)
plt.savefig("distribusi_per_kelas.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Skew sebelum dan sesudah transformasi log
# **Kapan dipakai:** target/fitur miring ke kanan (harga, pendapatan, jumlah klik). Model linear
# dan jaringan saraf lebih stabil setelah log. **Cara baca:** skewness mendekati 0 = simetris.

# %%
before = diamonds[SKEWED_COL]
after = np.log1p(diamonds[SKEWED_COL])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(before, bins=40, color=ORANGE, ax=axes[0])
axes[0].set_title(f"Asli (skew = {stats.skew(before):.2f})")
axes[0].set_xlabel("harga")
sns.histplot(after, bins=40, color=BLUE, ax=axes[1])
axes[1].set_title(f"log1p (skew = {stats.skew(after):.2f})")
axes[1].set_xlabel("log(1 + harga)")
# Alternatif untuk nilai negatif: from sklearn.preprocessing import PowerTransformer (Yeo-Johnson)
plt.savefig("distribusi_skew_log.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Versi interaktif (plotly)
# **Kapan dipakai:** menjelajah sendiri (hover angka, zoom) atau lampiran HTML untuk juri.
# **Cara baca:** arahkan kursor ke batang untuk melihat jumlah, panel atas = box plot.
# File HTML berdiri sendiri (jalan offline).

# %%
fig = px.histogram(train, x=MAIN_COL, color="nasib", nbins=30, marginal="box", opacity=0.7,
                   barmode="overlay", color_discrete_sequence=[RED, BLUE],
                   title="Sebaran umur per nasib (interaktif)")
fig.write_html("distribusi_interaktif.html", include_plotlyjs=True)
fig.show()
