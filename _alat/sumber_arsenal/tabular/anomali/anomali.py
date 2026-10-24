# %% [markdown]
# # Deteksi anomali tabel tanpa label: skor keanehan per baris → daftar baris paling aneh
#
# **Butuh:** tabel fitur angka (tanpa label, atau label hanya sedikit untuk menilai).
# **Menghasilkan:** kolom `anomaly_score` (makin besar makin aneh), `is_anomaly` (k teratas
# atau ambang), tabel perbandingan metode, `outputs/anomali.csv`.
#
# | metode | ide | cocok untuk |
# |---|---|---|
# | robust z-score (median/MAD) | jauh dari median per kolom | anomali 1 kolom, mudah dijelaskan |
# | Mahalanobis | jauh dari pusat, memperhitungkan korelasi | data kira-kira normal |
# | IsolationForest (Liu dkk. 2008) | titik aneh cepat terisolasi pohon acak | umum, data besar |
# | LOF (Breunig dkk. 2000) | kepadatan lebih rendah dari tetangganya | kelompok berbeda kepadatan |
# | gabungan peringkat | rata-rata peringkat beberapa metode | paling stabil |
#
# Studi pembanding besar ADBench (Han dkk. NeurIPS 2022, 57 dataset): tidak ada metode
# tanpa label yang selalu menang; IsolationForest termasuk yang paling konsisten → baseline.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + standarisasi robust | **ya** |
# | 2 | robust z-score | **ya** |
# | 3 | Mahalanobis (kovarians robust) | coba |
# | 4 | IsolationForest | **ya** |
# | 5 | LOF | coba |
# | 6 | gabungan peringkat | disarankan |
# | 7 | nilai dengan label (kalau ada sedikit) | kalau ada |
# | 8 | jelaskan baris teraneh: kolom mana yang menyimpang | untuk laporan |
# | 9 | pilih ambang/k + simpan | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Sel kanker payudara: 357 normal (jinak) + 22 "anomali" (ganas). Label hanya untuk menilai.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

table = pd.read_csv(f"{DEMO_FOLDER}/unsupervised/breast_cancer_anomaly.csv")
print(table.shape, "| porsi anomali:", round(table["is_anomaly"].mean(), 3))

# %% [markdown]
# ## 1. Konfigurasi + standarisasi robust (median & IQR, tahan pencilan)

# %%
from sklearn.preprocessing import RobustScaler

LABEL = "is_anomaly"   # UBAH: None kalau tidak ada label sama sekali
FEATURES = []
for column in table.columns:
    if column != LABEL:
        FEATURES.append(column)
CONTAMINATION = 0.06   # UBAH: perkiraan porsi anomali (tebakan kasar boleh)
scaled = RobustScaler().fit_transform(table[FEATURES])
print(len(FEATURES), "fitur |", scaled.shape)
scores = {}

# %% [markdown]
# ## 2. Robust z-score: |nilai − median| / (1,4826 × MAD), ambil yang terbesar per baris
# Aturan umum: z > 3,5 = pencilan (Iglewicz & Hoaglin 1993).

# %%
import numpy as np

values = table[FEATURES].to_numpy(dtype=float)
median = np.median(values, axis=0)
mad = np.median(np.abs(values - median), axis=0) * 1.4826
robust_z = np.abs(values - median) / np.where(mad == 0, 1, mad)
scores["robust_z_max"] = robust_z.max(axis=1)
scores["robust_z_mean"] = robust_z.mean(axis=1)
print("baris dengan z > 3,5 di minimal satu kolom:", int((robust_z.max(axis=1) > 3.5).sum()))

# %% [markdown]
# ## 3. Mahalanobis dengan kovarians robust (MinCovDet): jarak yang memperhitungkan korelasi

# %%
import numpy as np
from sklearn.covariance import MinCovDet

covariance = MinCovDet(random_state=42, support_fraction=0.9).fit(scaled)
scores["mahalanobis"] = np.sqrt(covariance.mahalanobis(scaled))
print("contoh skor:", scores["mahalanobis"][:5].round(2))

# %% [markdown]
# ## 4. IsolationForest: `score_samples` dibalik (minus) supaya besar = aneh

# %%
from sklearn.ensemble import IsolationForest

forest = IsolationForest(n_estimators=500, contamination=CONTAMINATION, random_state=42)
forest.fit(scaled)
scores["isolation_forest"] = -forest.score_samples(scaled)
print("ditandai anomali oleh IsolationForest:", int((forest.predict(scaled) == -1).sum()))

# %% [markdown]
# ## 5. LOF (Local Outlier Factor): kepadatan dibanding tetangga
# `n_neighbors` 20–50 umum dipakai; makin besar makin "global".

# %%
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=30)
lof.fit(scaled)
scores["lof"] = -lof.negative_outlier_factor_
print("contoh skor LOF:", scores["lof"][:5].round(2))

# %% [markdown]
# ## 6. Gabungan peringkat: rata-rata peringkat persentil semua metode
# Skala skor tiap metode berbeda → yang digabung PERINGKAT-nya, bukan nilainya.

# %%
import numpy as np
import pandas as pd

COMBINE = ["robust_z_mean", "mahalanobis", "isolation_forest", "lof"]   # UBAH
ranks = pd.DataFrame(scores).rank(pct=True)
scores["gabungan"] = ranks[COMBINE].mean(axis=1).to_numpy()
table["anomaly_score"] = scores["gabungan"]
agreement = pd.DataFrame(scores).corr(method="spearman")
print(agreement.round(2).to_string())   # korelasi rendah = metode saling melengkapi

# %% [markdown]
# ## 7. Nilai dengan label (kalau ada sebagian): ROC-AUC, PR-AUC, presisi di k teratas
# Presisi@k = dari k baris teraneh, berapa yang benar anomali (berguna kalau hanya sanggup
# memeriksa k baris).

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

truth = table[LABEL].to_numpy()
top_k = int(truth.sum())   # UBAH: jumlah baris yang sanggup diperiksa
rows = []
for name, score in scores.items():
    top = np.argsort(-score)[:top_k]
    rows.append({"metode": name, "roc_auc": round(roc_auc_score(truth, score), 4),
                 "pr_auc": round(average_precision_score(truth, score), 4),
                 f"presisi@{top_k}": round(truth[top].mean(), 4)})
board = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
print(board.to_string(index=False))

# %% [markdown]
# ## 8. Jelaskan baris teraneh: kolom dengan robust z terbesar
# Bahan laporan: "baris 123 aneh karena `worst area` 6× lebih jauh dari biasanya".

# %%
import numpy as np

SHOW = 5   # UBAH
for row in np.argsort(-table["anomaly_score"].to_numpy())[:SHOW]:
    largest = np.argsort(-robust_z[row])[:3]
    reasons = []
    for column_index in largest:
        reasons.append(f"{FEATURES[column_index]} z={robust_z[row, column_index]:.1f}")
    print(f"baris {row} skor {table['anomaly_score'].iloc[row]:.3f} | " + ", ".join(reasons))

# %% [markdown]
# ## 9. Pilih batas: k teratas ATAU porsi `CONTAMINATION` → `is_anomaly_pred` + simpan

# %%
import os

import numpy as np

cutoff = np.quantile(table["anomaly_score"], 1 - CONTAMINATION)
table["is_anomaly_pred"] = (table["anomaly_score"] >= cutoff).astype(int)
os.makedirs("outputs", exist_ok=True)
table.to_csv("outputs/anomali.csv", index=False)
print("ditandai:", int(table["is_anomaly_pred"].sum()), "baris | batas skor:", round(cutoff, 3))
print(pd.crosstab(table[LABEL], table["is_anomaly_pred"]))   # hanya demo
