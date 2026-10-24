# %% [markdown]
# # Timeseries — Deteksi Titik Perubahan (change point)
# **Kapan dipakai:** "kapan pola data berubah?" (kebijakan baru, alat diganti, krisis). Beda dengan
# anomali: anomali = satu titik aneh, titik perubahan = rata-rata/ragam **bergeser permanen**.
#
# **Langkah:** CUSUM manual (melihat kasar) → `ruptures` mencari titik perubahan → rata-rata tiap
# segmen + uji t (apakah pergeseran nyata?).
#
# **Data demo:** debit Sungai Nil 1871–1970, turun permanen sekitar 1898 (bendungan Aswan).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/timeseries/changepoint"   # GANTI saat lomba
SERIES_CSV = f"{DATA_FOLDER}/nile.csv"

TIME_COL = "year"
VALUE_COL = "volume"
KNOWN_YEAR = 1898     # perubahan yang diketahui (hanya untuk cek data demo)
MIN_SIZE = 5          # panjang segmen minimum

# %% [markdown]
# ## Baca data

# %%
table = pd.read_csv(SERIES_CSV)
signal = table[VALUE_COL].to_numpy(dtype=float)
print(table.shape, "| missing:", table[VALUE_COL].isna().sum())
table.head()

# %% [markdown]
# ## EDA: plot + rata-rata bergulir

# %%
plt.figure(figsize=(10, 3))
plt.plot(table[TIME_COL], signal, label=VALUE_COL)
plt.plot(table[TIME_COL], table[VALUE_COL].rolling(10, center=True).mean(), label="rata 10")
plt.legend()
plt.show()

# %% [markdown]
# ## CUSUM manual: jumlah kumulatif selisih dari rata-rata
# Titik puncak/lembah kurva = kandidat titik perubahan rata-rata.

# %%
cusum = np.cumsum(signal - signal.mean())
print("CUSUM ekstrem di:", table[TIME_COL][np.argmax(np.abs(cusum))])
plt.figure(figsize=(10, 3))
plt.plot(table[TIME_COL], cusum)
plt.title("CUSUM")
plt.show()

# %% [markdown]
# ## Model
# Pilih SATU detektor: hapus `#` di baris yang ingin dipakai, beri `#` pada baris Pelt.
# `pen` (penalti) besar → sedikit titik. `model="l2"` = mencari pergeseran rata-rata, `"rbf"` =
# perubahan sebaran apa pun. Dynp tidak menerima `pen`: ganti baris `predict` dengan
# `detector.predict(n_bkps=1)` (jumlah titik ditulis langsung).

# %%
detector = rpt.Pelt(model="l2", min_size=MIN_SIZE, jump=1)
# detector = rpt.Binseg(model="l2", min_size=MIN_SIZE, jump=1)
# detector = rpt.BottomUp(model="l2", min_size=MIN_SIZE, jump=1)
# detector = rpt.Window(width=20, model="l2", jump=1)
# detector = rpt.Pelt(model="rbf", min_size=MIN_SIZE, jump=1)     # pakai penalty = 3
# detector = rpt.Dynp(model="l2", min_size=MIN_SIZE, jump=1)
detector.fit(signal)
penalty = np.log(len(signal)) * signal.var()
breakpoints = detector.predict(pen=penalty)

change_points = breakpoints[:-1]   # elemen terakhir selalu = panjang data, bukan titik perubahan
change_times = list(table[TIME_COL][change_points])
print("titik perubahan (nomor baris):", change_points, "| waktu:", change_times)

# %% [markdown]
# ## Evaluasi: rata-rata tiap segmen + uji t (apakah pergeseran nyata?)

# %%
segments = np.zeros(len(signal), dtype=int)
for point in change_points:
    segments[point:] = segments[point:] + 1     # setelah tiap titik, nomor segmen naik 1
table["segment"] = segments
segment_summary = table.groupby("segment")[VALUE_COL].agg(["count", "mean", "std"])
print(segment_summary.round(1))

before = signal[:change_points[0]]
after = signal[change_points[0]:]
print("uji t Welch p =", stats.ttest_ind(before, after, equal_var=False).pvalue)
for change_time in change_times:
    print("selisih dengan tahun acuan:", change_time - KNOWN_YEAR)

rpt.display(signal, breakpoints, figsize=(10, 3))
plt.title("Segmen hasil deteksi")
plt.show()

# %% [markdown]
# ## Simpan hasil (segmen per baris)

# %%
result = pd.DataFrame()
result["year"] = table[TIME_COL]
result["volume"] = table[VALUE_COL]
result["segmen"] = table["segment"]
result["rata_segmen"] = table.groupby("segment")[VALUE_COL].transform("mean")
result.to_csv("hasil.csv", index=False)
result.iloc[change_points[0] - 3:change_points[0] + 3]
