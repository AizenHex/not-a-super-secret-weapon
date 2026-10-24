# %% [markdown]
# # Segmentasi tanpa label: RFM pelanggan → KMeans → profil & nama segmen
#
# **Butuh:** tabel transaksi (id pelanggan, tanggal, nilai) ATAU tabel fitur per entitas.
# **Menghasilkan:** `customers["segment"]`, tabel profil per segmen, plot 2D,
# `outputs/segmen.csv`.
#
# RFM (Hughes 1994) = cara standar segmentasi pelanggan: **R**ecency (berapa hari sejak
# belanja terakhir), **F**requency (berapa kali belanja), **M**onetary (total belanja).
#
# | langkah | kenapa |
# |---|---|
# | log + standarisasi | KMeans memakai jarak; nilai miring/berskala besar mendominasi |
# | pilih k (inertia + silhouette) | tidak ada jawaban pasti; pilih yang bisa dijelaskan |
# | profil median per segmen | bahan presentasi: "segmen 2 = pelanggan setia bernilai tinggi" |
# | stabilitas (ARI antar seed) | segmen yang berubah-ubah tiap seed = tidak meyakinkan |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | bersihkan transaksi (retur, harga 0) | **ya** |
# | 2 | RFM per pelanggan (+ fitur tambahan) | **ya** |
# | 3 | log + standarisasi | **ya** |
# | 4 | pilih k: inertia (siku) + silhouette | **ya** |
# | 5 | KMeans final + stabilitas antar seed | **ya** |
# | 6 | profil + nama segmen | **ya** |
# | 7 | plot PCA 2D | disarankan |
# | 8 | alternatif: GaussianMixture / skor RFM kuantil (aturan) | opsional |
# | 9 | simpan | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Transaksi toko online Inggris 2010–2011 (1500 pelanggan).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

sales = pd.read_csv(f"{DEMO_FOLDER}/unsupervised/online_retail.csv",
                    parse_dates=["InvoiceDate"], dtype={"InvoiceNo": str})
print(sales.shape)
print(sales.head(3))

# %% [markdown]
# ## 1. Bersihkan: buang retur (jumlah negatif / nomor diawali "C") dan harga ≤ 0

# %%
CUSTOMER = "CustomerID"   # UBAH
DATE = "InvoiceDate"      # UBAH
INVOICE = "InvoiceNo"     # UBAH

is_return = (sales["Quantity"] <= 0) | sales[INVOICE].str.startswith("C")
clean = sales[~is_return & (sales["UnitPrice"] > 0)].copy()
clean["value"] = clean["Quantity"] * clean["UnitPrice"]       # UBAH: kolom nilai belanja
print("baris:", len(sales), "→", len(clean), "| retur dibuang:", int(is_return.sum()))

# %% [markdown]
# ## 2. RFM per pelanggan (+ rata-rata nilai per transaksi, lama jadi pelanggan, porsi retur)
# Tanggal acuan = sehari setelah transaksi terakhir di data.

# %%
import pandas as pd

reference = clean[DATE].max() + pd.Timedelta(days=1)
by_customer = clean.groupby(CUSTOMER)
customers = by_customer.agg(last_date=(DATE, "max"), first_date=(DATE, "min"),
                            frequency=(INVOICE, "nunique"), monetary=("value", "sum"),
                            products=("StockCode", "nunique"))
customers["recency"] = (reference - customers["last_date"]).dt.days
customers["tenure"] = (reference - customers["first_date"]).dt.days
customers["avg_basket"] = customers["monetary"] / customers["frequency"]
return_share = is_return.groupby(sales[CUSTOMER]).mean()
customers["return_share"] = return_share.reindex(customers.index).fillna(0)
FEATURES = ["recency", "frequency", "monetary", "avg_basket", "tenure"]   # UBAH
print(customers[FEATURES].describe().round(1))

# %% [markdown]
# ## 3. log + standarisasi (rata-rata 0, simpangan 1)
# Frequency/monetary sangat miring (sedikit pelanggan raksasa) → log1p dulu.

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler

SKEWED = ["frequency", "monetary", "avg_basket"]   # UBAH
matrix = customers[FEATURES].copy()
for column in SKEWED:
    matrix[column] = np.log1p(matrix[column].clip(lower=0))
scaled = StandardScaler().fit_transform(matrix)
print(scaled.shape, "| rata-rata:", scaled.mean(axis=0).round(2))

# %% [markdown]
# ## 4. Pilih k: inertia (cari "siku") + silhouette (−1..1, makin tinggi makin terpisah)
# Lomba/bisnis: k 3–6 biasanya paling mudah dijelaskan. Silhouette tertinggi tidak wajib.

# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

K_RANGE = range(2, 9)   # UBAH
inertia, silhouette = [], []
for k_value in K_RANGE:
    model = KMeans(n_clusters=k_value, n_init=10, random_state=42).fit(scaled)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(scaled, model.labels_))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(list(K_RANGE), inertia, marker="o")
axes[0].set_title("inertia (cari siku)")
axes[1].plot(list(K_RANGE), silhouette, marker="o")
axes[1].set_title("silhouette")
plt.show()
for k_value, value in zip(K_RANGE, silhouette):
    print(f"k={k_value}: silhouette {value:.3f}")

# %% [markdown]
# ## 5. KMeans final + stabilitas: jalankan ulang dengan seed lain, bandingkan (ARI)
# ARI ≈ 1 = segmen sama persis walau seed beda → segmen stabil.

# %%
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

N_SEGMENTS = 4   # UBAH: hasil sel 4
final = KMeans(n_clusters=N_SEGMENTS, n_init=10, random_state=42).fit(scaled)
customers["segment"] = final.labels_
agreement = []
for seed in [1, 2, 3]:
    other = KMeans(n_clusters=N_SEGMENTS, n_init=10, random_state=seed).fit(scaled)
    agreement.append(adjusted_rand_score(final.labels_, other.labels_))
print("ukuran segmen:", np.bincount(final.labels_).tolist())
print("ARI antar seed:", np.round(agreement, 3))

# %% [markdown]
# ## 6. Profil per segmen (median + ukuran) → beri nama
# Bandingkan tiap segmen dengan median semua pelanggan: lebih baru? lebih sering? lebih besar?

# %%
import pandas as pd

profile = customers.groupby("segment")[FEATURES].median()
profile["size"] = customers["segment"].value_counts()
profile["share_of_revenue"] = customers.groupby("segment")["monetary"].sum() / \
    customers["monetary"].sum()
profile = profile.sort_values("monetary", ascending=False)
print(profile.round(2).to_string())
# contoh demo: 2 = baru belanja + sering + besar ("juara"), 3 = lama tidak belanja ("hilang")
SEGMENT_NAMES = {}   # UBAH setelah membaca profil, mis. {2: "juara", 3: "hilang", ...}
customers["segment_name"] = customers["segment"].map(SEGMENT_NAMES)
customers["segment_name"] = customers["segment_name"].fillna("segmen_" +
                                                             customers["segment"].astype(str))

# %% [markdown]
# ## 7. Plot 2D (PCA): apakah segmen terlihat terpisah?

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
points = pca.fit_transform(scaled)
fig, ax = plt.subplots(figsize=(6, 5))
for segment in sorted(customers["segment"].unique()):
    chosen = (customers["segment"] == segment).to_numpy()
    ax.scatter(points[chosen, 0], points[chosen, 1], s=6, label=f"segmen {segment}")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.legend()
plt.show()

# %% [markdown]
# ## 8. Alternatif
# - **GaussianMixture**: kelompok boleh lonjong & tumpang tindih; ada proba keanggotaan.
# - **Skor RFM kuantil (aturan)**: R, F, M masing-masing 1–5 menurut kuantil → mudah
#   dijelaskan ke juri ("555 = juara", "1xx = hilang"). Tidak butuh model sama sekali.

# %%
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

mixture = GaussianMixture(n_components=N_SEGMENTS, random_state=42).fit(scaled)
customers["segment_gmm"] = mixture.predict(scaled)
print("ARI KMeans vs GMM:", round(adjusted_rand_score(customers["segment"],
                                                      customers["segment_gmm"]), 3))
customers["r_score"] = pd.qcut(customers["recency"].rank(method="first"), 5,
                               labels=[5, 4, 3, 2, 1]).astype(int)   # baru = 5
customers["f_score"] = pd.qcut(customers["frequency"].rank(method="first"), 5,
                               labels=[1, 2, 3, 4, 5]).astype(int)
customers["m_score"] = pd.qcut(customers["monetary"].rank(method="first"), 5,
                               labels=[1, 2, 3, 4, 5]).astype(int)
customers["rfm_code"] = customers["r_score"].astype(str) + customers["f_score"].astype(str) + \
    customers["m_score"].astype(str)
print(customers["rfm_code"].value_counts().head(5))

# %% [markdown]
# ## 9. Simpan `outputs/segmen.csv` (satu baris per pelanggan + segmen + fitur)

# %%
import os

os.makedirs("outputs", exist_ok=True)
columns = FEATURES + ["segment", "segment_name", "segment_gmm", "rfm_code"]
customers[columns].to_csv("outputs/segmen.csv")
profile.to_csv("outputs/segmen_profil.csv")
print(customers[columns].head())
