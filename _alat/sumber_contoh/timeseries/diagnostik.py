# %% [markdown]
# # Timeseries — Diagnostik (tren, musim, stasioner)
# **Kapan dipakai:** langkah pertama sebelum meramal. Hasilnya menentukan pilihan model:
# ada musim → model musiman, tidak stasioner → differencing (ARIMA `d`).
#
# **Langkah:** plot → dekomposisi (tren + musiman + residu) → ACF/PACF → uji ADF.
# Tidak ada model dan tidak ada split: semua data dipakai.
# ADF: p-value < 0.05 berarti deret stasioner.
#
# **Data demo:** CO2 bulanan (`value`).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/timeseries/forecasting"   # GANTI saat lomba
SERIES_CSV = f"{DATA_FOLDER}/co2_monthly.csv"

DATE_COL = "date"
VALUE_COL = "value"
FREQUENCY = "MS"    # MS = bulanan, QS = kuartalan, D = harian
PERIOD = 12         # panjang musim
MAX_LAG = 40        # jumlah lag untuk ACF/PACF

# %% [markdown]
# ## Baca data dan plot

# %%
table = pd.read_csv(SERIES_CSV, parse_dates=[DATE_COL], index_col=DATE_COL)
series = table[VALUE_COL].asfreq(FREQUENCY)
print(series.shape, "| missing:", series.isna().sum())
series.plot(figsize=(10, 3), title="Data mentah")
plt.show()

# %% [markdown]
# ## Dekomposisi (tren, musiman, residu)

# %%
decomposition = seasonal_decompose(series, model="additive", period=PERIOD)
decomposition.plot()
plt.tight_layout()
plt.show()

result = pd.DataFrame()
result["tanggal"] = series.index
result["tren"] = decomposition.trend.to_numpy()
result["musiman"] = decomposition.seasonal.to_numpy()
result["residu"] = decomposition.resid.to_numpy()
result.to_csv("hasil.csv", index=False)
result.head()

# %% [markdown]
# ## ACF dan PACF
# ACF turun pelan = ada tren. Puncak di lag 12, 24, ... = musim tahunan.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
plot_acf(series, lags=MAX_LAG, ax=axes[0])
plot_pacf(series, lags=MAX_LAG, ax=axes[1], method="ywm")
plt.show()

# %% [markdown]
# ## Uji ADF (indeks [1] adalah p-value)

# %%
first_difference = series.diff().dropna()
seasonal_difference = series.diff(PERIOD).dropna()
print("ADF data asli       p-value:", round(adfuller(series)[1], 4))
print("ADF selisih pertama p-value:", round(adfuller(first_difference)[1], 4))
print("ADF selisih musiman p-value:", round(adfuller(seasonal_difference)[1], 4))
