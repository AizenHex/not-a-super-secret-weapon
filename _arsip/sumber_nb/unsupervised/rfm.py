# %% [markdown]
# # Unsupervised — Segmentasi Pelanggan RFM (Recency, Frequency, Monetary)
# **Kapan dipakai:** ada data transaksi (pelanggan, tanggal, nilai belanja) dan diminta
# mengelompokkan pelanggan untuk strategi pemasaran.
#
# **Idenya:** tiap pelanggan diringkas jadi 3 angka:
# - **recency** = berapa hari sejak belanja terakhir (kecil = baru saja belanja).
# - **frequency** = berapa kali belanja (jumlah invoice).
# - **monetary** = total uang yang dibelanjakan.
#
# Ketiganya di-log (sangat miring), disamakan skalanya, lalu di-cluster. Tiap segmen diberi
# nama dari profil rata-ratanya.
#
# **Langkah:**
# 1. Bersihkan transaksi (buang retur dan harga 0).
# 2. Hitung RFM per pelanggan.
# 3. Cluster, lihat profil segmen, simpan segmen tiap pelanggan.
#
# **Data demo:** Online Retail UCI (toko online Inggris 2010–2011).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/unsupervised"   # GANTI saat lomba
TRANSACTIONS_CSV = f"{DATA_FOLDER}/online_retail.csv"

CUSTOMER_COL = "CustomerID"
INVOICE_COL = "InvoiceNo"
DATE_COL = "InvoiceDate"
QUANTITY_COL = "Quantity"
PRICE_COL = "UnitPrice"
CANCEL_PREFIX = "C"      # invoice berawalan C = pembatalan
N_CLUSTERS = 4
SEED = 42

# %% [markdown]
# ## Baca data

# %%
transactions = pd.read_csv(TRANSACTIONS_CSV, parse_dates=[DATE_COL])
print(transactions.shape)
print("pelanggan:", transactions[CUSTOMER_COL].nunique())
print("invoice:", transactions[INVOICE_COL].nunique())
transactions.head()

# %% [markdown]
# ## Langkah 1 — bersihkan transaksi

# %%
is_valid = (transactions[QUANTITY_COL] > 0) & (transactions[PRICE_COL] > 0)
invoice_text = transactions[INVOICE_COL].astype(str)
is_cancel = invoice_text.str.startswith(CANCEL_PREFIX)
transactions = transactions[is_valid & ~is_cancel].copy()
transactions["total"] = transactions[QUANTITY_COL] * transactions[PRICE_COL]
print("setelah bersih:", transactions.shape)

month = transactions[DATE_COL].dt.to_period("M")
monthly_sales = transactions.groupby(month)["total"].sum()
monthly_sales.plot.bar(figsize=(10, 3))
plt.title("Omzet per bulan")
plt.show()

# %% [markdown]
# ## Langkah 2 — RFM per pelanggan
# Tanggal acuan = sehari setelah transaksi terakhir di data.

# %%
reference_date = transactions[DATE_COL].max() + pd.DateOffset(days=1)
per_customer = transactions.groupby(CUSTOMER_COL)
last_purchase = per_customer[DATE_COL].max()

rfm = pd.DataFrame()
rfm["recency"] = (reference_date - last_purchase).dt.days
rfm["frequency"] = per_customer[INVOICE_COL].nunique()
rfm["monetary"] = per_customer["total"].sum()
print(rfm.describe().round(1))

scaler = StandardScaler()
X = scaler.fit_transform(np.log1p(rfm))      # log(1 + x) meredam nilai yang sangat besar

# %% [markdown]
# ## Pilih jumlah segmen
# Silhouette tinggi = segmen rapat dan saling terpisah. Tulis pilihan di `N_CLUSTERS`.
# Jumlah 3–5 biasanya paling mudah dijelaskan saat presentasi.

# %%
for count in range(2, 9):
    kmeans = KMeans(n_clusters=count, n_init=10, random_state=SEED)
    trial_labels = kmeans.fit_predict(X)
    print(count, "segmen | silhouette:", round(silhouette_score(X, trial_labels), 3))

# %% [markdown]
# ## Langkah 3 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
# model = GaussianMixture(n_components=N_CLUSTERS, random_state=SEED)
# model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")

labels = model.fit_predict(X)
rfm["segment"] = labels
print("silhouette:", round(silhouette_score(X, labels), 3))

# %% [markdown]
# ## Profil segmen (bahan presentasi)
# Contoh penamaan: recency kecil + frequency/monetary besar = "Pelanggan setia",
# recency besar + frequency kecil = "Hampir hilang".

# %%
profile = rfm.groupby("segment").median()
profile["n_customers"] = rfm.groupby("segment").size()
segment_money = rfm.groupby("segment")["monetary"].sum()
profile["sales_share_%"] = segment_money / rfm["monetary"].sum() * 100
print(profile.round(1))

plt.scatter(np.log1p(rfm["recency"]), np.log1p(rfm["monetary"]), c=labels, cmap="tab10", s=6)
plt.xlabel("log recency")
plt.ylabel("log monetary")
plt.show()

# %% [markdown]
# ## Simpan segmen tiap pelanggan

# %%
submission = pd.DataFrame()
submission["CustomerID"] = rfm.index
submission["recency"] = rfm["recency"].to_numpy()
submission["frequency"] = rfm["frequency"].to_numpy()
submission["monetary"] = rfm["monetary"].to_numpy()
submission["segment"] = labels
submission.to_csv("submission.csv", index=False)
submission.head()
