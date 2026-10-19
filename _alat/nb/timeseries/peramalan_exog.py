# %% [markdown]
# # Timeseries — Peramalan dengan Variabel Eksogen (SARIMAX)
# **Kapan dipakai:** target dipengaruhi deret lain yang nilainya **diketahui untuk periode
# ramalan** (harga BBM, jadwal promo, libur, suhu ramalan BMKG).
#
# **Langkah:** split waktu → SARIMAX dengan `exog=` → bandingkan dengan naive (nilai terakhir).
# Catatan: uji memakai nilai eksogen aktual periode uji. Di lomba, pastikan nilai eksogen masa
# depan tersedia; kalau tidak, ramal dulu eksogennya atau pakai `peramalan_statistik`.
#
# **Data demo:** PDB riil AS kuartalan (`realgdp`) dengan eksogen `unemp`, `tbilrate`, `realinv`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/timeseries/forecasting"   # GANTI saat lomba
SERIES_CSV = f"{DATA_FOLDER}/macro_quarterly.csv"

DATE_COL = "date"
TARGET = "realgdp"
EXOG_COLS = ["unemp", "tbilrate", "realinv"]
FREQUENCY = "QS"      # QS = kuartalan, MS = bulanan, D = harian
SEASON_LENGTH = 4     # musim kuartalan
HORIZON = 8           # jumlah langkah ramalan (8 kuartal = 2 tahun)

# %% [markdown]
# ## Baca data dan EDA

# %%
table = pd.read_csv(SERIES_CSV, parse_dates=[DATE_COL], index_col=DATE_COL)
table = table.asfreq(FREQUENCY)
missing = table.isna().sum()
print(table.shape, "| missing:", missing.sum())
table.head()

# %%
table[TARGET].plot(figsize=(10, 3), title=TARGET)
plt.show()
table[EXOG_COLS].plot(subplots=True, figsize=(10, 6))
plt.show()

# %% [markdown]
# ## Split waktu (bukan acak)

# %%
y = table[TARGET]
X = table[EXOG_COLS]
y_train = y.iloc[:-HORIZON]
y_test = y.iloc[-HORIZON:]
X_train = X.iloc[:-HORIZON]
X_test = X.iloc[-HORIZON:]
print("latih:", len(y_train), "| uji:", len(y_test))

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris aktif.
# `order=(p, d, q)`: p = lag AR, d = berapa kali di-differencing, q = lag MA.

# %%
def forecast(y_part, X_part, X_future):
    model = SARIMAX(y_part, exog=X_part, order=(1, 1, 1))
    # model = SARIMAX(y_part, exog=X_part, order=(1, 0, 1),
    #                 seasonal_order=(0, 1, 1, SEASON_LENGTH))
    # model = SARIMAX(y_part, exog=X_part, order=(2, 1, 0))
    # model = SARIMAX(y_part, exog=X_part, order=(1, 1, 1), trend="t")
    # model = SARIMAX(y_part, exog=X_part, order=(0, 1, 1))
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(len(X_future), exog=X_future))


print("uji coba 2 langkah:", forecast(y_train, X_train, X_test.iloc[:2]).round(1))

# %% [markdown]
# ## Evaluasi di data uji (MAE, RMSE, MAPE) dibanding naive

# %%
predictions = forecast(y_train, X_train, X_test)
naive = np.repeat(y_train.iloc[-1], HORIZON)       # nilai terakhir diulang
print("Model SARIMAX MAE:", round(mean_absolute_error(y_test, predictions), 2),
      "| RMSE:", round(root_mean_squared_error(y_test, predictions), 2),
      "| MAPE(%):", round(100 * mean_absolute_percentage_error(y_test, predictions), 2))
print("Naive         MAE:", round(mean_absolute_error(y_test, naive), 2),
      "| RMSE:", round(root_mean_squared_error(y_test, naive), 2),
      "| MAPE(%):", round(100 * mean_absolute_percentage_error(y_test, naive), 2))

ax = y_train.iloc[-5 * SEASON_LENGTH:].plot(label="latih")
y_test.plot(ax=ax, label="aktual")
ax.plot(y_test.index, predictions, label="ramalan model")
ax.legend()
plt.show()

# %% [markdown]
# ## Simpan hasil
# Tidak dilatih ulang di semua data: ramalan ke depan butuh nilai eksogen masa depan yang tidak
# ada di data demo. Saat lomba: `forecast(y, X, X_future)` dengan `X_future` = tabel eksogen
# periode ramalan.

# %%
result = pd.DataFrame()
result["tanggal"] = y_test.index
result["aktual"] = y_test.to_numpy()
result["ramalan"] = predictions
result.to_csv("hasil.csv", index=False)
result.head()
