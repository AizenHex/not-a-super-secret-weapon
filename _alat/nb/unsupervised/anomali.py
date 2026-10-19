# %% [markdown]
# # Unsupervised — Deteksi Anomali (data tabel)
# **Kapan dipakai:** diminta mencari baris yang "aneh" (transaksi curang, mesin rusak, pasien
# tidak biasa) dan label anomali tidak ada atau sangat sedikit.
#
# **Idenya:** model anomali mempelajari pola mayoritas data, lalu memberi skor tiap baris:
# makin besar skor = makin menyimpang. Baris dengan skor tertinggi ditandai anomali.
#
# **Langkah:**
# 1. Baca data, samakan skala feature.
# 2. Latih model anomali, hitung skor dan tanda anomali tiap baris.
# 3. Simpan skor. Cek dengan kunci jawaban di sel terakhir (hanya data demo).
#
# **Data demo:** Breast cancer versi anomali (kolom `is_anomaly`: 1 = anomali, hanya untuk cek).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/breast_cancer_anomaly.csv"

LABEL_COL = "is_anomaly"     # hanya untuk cek, saat lomba biasanya tidak ada
ANOMALY_FRACTION = 0.06      # perkiraan porsi anomali
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
print(data.shape)
missing_per_column = data.isna().sum()
print("missing total:", missing_per_column.sum())
data.head()

# %% [markdown]
# ## Langkah 1 — samakan skala feature
# Gambar PCA 2D = semua kolom diringkas jadi 2 sumbu, supaya data bisa dilihat.

# %%
X = data.drop(columns=[LABEL_COL])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=12)
plt.title("Data dalam PCA 2D")
plt.show()

# %% [markdown]
# ## Langkah 2 — model anomali
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.
# Semua model `pyod` dipakai dengan cara yang sama: `model.fit(X)`, lalu
# `model.decision_function(X)` = skor dan `model.labels_` = tanda (1 = anomali).

# %%
model = IForest(n_estimators=300, contamination=ANOMALY_FRACTION, random_state=SEED)
# model = LOF(n_neighbors=20, contamination=ANOMALY_FRACTION)    # kepadatan tetangga
# model = KNN(n_neighbors=10, contamination=ANOMALY_FRACTION)    # jarak ke tetangga terdekat
# model = OCSVM(contamination=ANOMALY_FRACTION)                  # batas di sekitar data normal
# model = ECOD(contamination=ANOMALY_FRACTION)                   # tanpa parameter, sangat cepat
# model = COPOD(contamination=ANOMALY_FRACTION)                  # tanpa parameter, sangat cepat
# model = HBOS(contamination=ANOMALY_FRACTION)                   # histogram per feature

model.fit(X_scaled)
scores = model.decision_function(X_scaled)
flags = model.labels_
print("ditandai anomali:", flags.sum(), "dari", len(flags))

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=scores, cmap="viridis", s=12)
plt.title("Skor anomali (makin terang = makin aneh)")
plt.colorbar()
plt.show()

# %% [markdown]
# ## Langkah 3 — simpan skor

# %%
submission = pd.DataFrame()
submission["anomaly_score"] = scores
submission["is_anomaly"] = flags
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# AUC = seberapa sering anomali diberi skor lebih tinggi dari baris normal (1 = sempurna).
# AP = presisi rata-rata (cocok bila anomali sedikit).

# %%
truth = data[LABEL_COL]
print("AUC:", round(roc_auc_score(truth, scores), 3))
print("AP :", round(average_precision_score(truth, scores), 3))
print("F1 anomali:", round(f1_score(truth, flags), 3))
