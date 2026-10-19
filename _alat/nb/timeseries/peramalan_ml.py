# %% [markdown]
# # Timeseries — Peramalan ML (fitur lag + model tabular)
# **Kapan dipakai:** satu deret waktu dan model statistik kurang pas (pola tidak linear, ada
# hari libur/fitur kalender). Model tabular biasa dilatih dari fitur lag.
#
# **Langkah:** fitur = nilai lag, rata-rata bergulir, bulan → model regresi → ramalan ke depan
# **rekursif**: tiap langkah memakai ramalan langkah sebelumnya sebagai lag.
#
# **Data demo:** CO2 bulanan Mauna Loa.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/timeseries/forecasting"   # GANTI saat lomba
SERIES_CSV = f"{DATA_FOLDER}/co2_monthly.csv"

DATE_COL = "date"
VALUE_COL = "value"
FREQUENCY = "MS"            # MS = awal bulan, D = harian, h = per jam
LAGS = [1, 2, 3, 6, 12]     # nilai k langkah sebelumnya dipakai sebagai fitur
HORIZON = 36                # jumlah langkah ramalan, data uji = HORIZON titik terakhir
SEED = 42

# %% [markdown]
# ## Baca data dan split waktu

# %%
table = pd.read_csv(SERIES_CSV, parse_dates=[DATE_COL], index_col=DATE_COL)
y = table[VALUE_COL].asfreq(FREQUENCY)
train = y.iloc[:-HORIZON]
test = y.iloc[-HORIZON:]
print(y.shape, "| missing:", y.isna().sum(), "| latih:", len(train), "| uji:", len(test))

# %% [markdown]
# ## Fitur lag, rata-rata bergulir, dan bulan
# `shift(1)` dulu sebelum rolling supaya nilai hari itu sendiri tidak ikut (bocor).

# %%
def make_features(series):
    features = pd.DataFrame(index=series.index)
    for lag in LAGS:
        features[f"lag_{lag}"] = series.shift(lag)
    previous = series.shift(1)
    features["mean_3"] = previous.rolling(3).mean()
    features["mean_12"] = previous.rolling(12).mean()
    features["month"] = series.index.month
    return features


X_train = make_features(train).dropna()     # baris awal tanpa lag lengkap dibuang
y_train = train.loc[X_train.index]
print(X_train.shape)
X_train.tail()

# %% [markdown]
# ## Ramalan rekursif (tiap langkah memakai ramalan sebelumnya)
# Tanggal baru ditambah satu per satu, nilainya NaN, fitur baris terakhir dihitung, ditebak,
# lalu tebakan dimasukkan sebagai nilai tanggal itu.

# %%
STEP = pd.tseries.frequencies.to_offset(FREQUENCY)     # jarak antar tanggal


def recursive_forecast(model, history, steps):
    values = list(history.to_numpy())
    dates = list(history.index)
    for _ in range(steps):
        dates.append(dates[-1] + STEP)
        series = pd.Series(values + [np.nan], index=dates)
        last_row = make_features(series).iloc[[-1]]
        values.append(float(model.predict(last_row)[0]))
    return np.array(values[-steps:])

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris RandomForest.
# Catatan: model pohon tidak bisa meramal di atas nilai tertinggi data latih (tren naik
# terpotong). Untuk tren kuat, coba Ridge.

# %%
model = RandomForestRegressor(n_estimators=300, random_state=SEED)
# model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
# model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = ExtraTreesRegressor(n_estimators=300, random_state=SEED)
# model = HistGradientBoostingRegressor(random_state=SEED)
# model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED)
# model = CatBoostRegressor(iterations=300, verbose=0, allow_writing_files=False,
#                           random_seed=SEED)
# model = KNeighborsRegressor(n_neighbors=5)
# model = make_pipeline(StandardScaler(), SVR(C=10))

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di data uji (MAE, RMSE, MAPE)

# %%
predictions = recursive_forecast(model, train, HORIZON)
print("MAE :", round(mean_absolute_error(test, predictions), 2))
print("RMSE:", round(root_mean_squared_error(test, predictions), 2))
print("MAPE (%):", round(100 * mean_absolute_percentage_error(test, predictions), 2))

ax = train.iloc[-60:].plot(label="latih")
test.plot(ax=ax, label="aktual")
ax.plot(test.index, predictions, label="ramalan model")
ax.legend()
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + hasil ramalan ke depan

# %%
X_all = make_features(y).dropna()
model.fit(X_all, y.loc[X_all.index])
future_values = recursive_forecast(model, y, HORIZON)
future_dates = pd.date_range(start=y.index[-1], periods=HORIZON + 1, freq=FREQUENCY)[1:]

result = pd.DataFrame()
result["tanggal"] = future_dates
result["ramalan"] = future_values
result.to_csv("hasil.csv", index=False)
result.head()
