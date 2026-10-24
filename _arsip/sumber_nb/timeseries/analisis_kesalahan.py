# %% [markdown]
# # Timeseries — Analisis Kesalahan Ramalan (kapan dan kenapa ramalan meleset?)
# **Kapan dipakai:** **setelah** model ramalan jadi, untuk mencari perbaikan yang masuk akal.
#
# **Langkah:** model ramalan → residual (aktual − ramalan) → dibedah:
# - kapan meleset: per jam, per hari dalam seminggu, hari-hari terburuk (libur? badai?)
# - bias: apakah ramalan selalu terlalu tinggi/rendah
# - residual masih berpola? (autokorelasi + uji Ljung-Box). Kalau ya, masih ada fitur yang hilang.
#
# **Data demo:** penumpang taksi NYC per jam. Model: LightGBM dengan lag ≥ 24 jam (ramal H+1 hari).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/timeseries/anomaly"   # GANTI saat lomba
SERIES_CSV = f"{DATA_FOLDER}/nyc_taxi.csv"

DATE_COL = "timestamp"
VALUE_COL = "value"
FREQUENCY = "h"            # data 30 menit dijumlahkan jadi per jam
LAGS = [24, 48, 168]       # semua lag >= 24: model bisa meramal 1 hari ke depan
TEST_WEEKS = 8
SEED = 42

# %% [markdown]
# ## Baca data

# %%
table = pd.read_csv(SERIES_CSV, parse_dates=[DATE_COL], index_col=DATE_COL)
series = table[VALUE_COL].resample(FREQUENCY).sum()
print(series.shape, series.index.min(), series.index.max())

# %% [markdown]
# ## Fitur dan split waktu (`TEST_WEEKS` minggu terakhir = uji)

# %%
features = pd.DataFrame(index=series.index)
for lag in LAGS:
    features[f"lag_{lag}"] = series.shift(lag)
day_before = series.shift(24)
features["mean_24_before"] = day_before.rolling(24).mean()
features["hour"] = series.index.hour
features["weekday"] = series.index.dayofweek
features = features.dropna()
target = series.loc[features.index]

split_time = features.index.max() - pd.DateOffset(weeks=TEST_WEEKS)
X_train = features[features.index <= split_time]
X_test = features[features.index > split_time]
y_train = target[target.index <= split_time]
y_test = target[target.index > split_time]
print("latih:", X_train.shape, "| uji:", X_test.shape)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris LightGBM.

# %%
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = HistGradientBoostingRegressor(random_state=SEED)
# model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED)
# model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("MAE uji:", round(mean_absolute_error(y_test, predictions), 1))

# %% [markdown]
# ## Tabel residual

# %%
result = pd.DataFrame(index=y_test.index)
result["actual"] = y_test
result["forecast"] = predictions
result["residual"] = result["actual"] - result["forecast"]
result["abs_error"] = result["residual"].abs()
result["hour"] = result.index.hour
result["day"] = result.index.day_name()
result["date"] = result.index.date
print("bias (rata-rata residual):", round(result["residual"].mean(), 1),
      "| positif = ramalan terlalu rendah")

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(result.index, result["actual"], label="aktual", linewidth=0.8)
axes[0].plot(result.index, result["forecast"], label="ramalan", linewidth=0.8)
axes[0].legend()
axes[1].plot(result.index, result["residual"], color="tab:red", linewidth=0.8)
axes[1].axhline(0, color="black")
axes[1].set_ylabel("residual")
plt.savefig("residual_waktu.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Kapan paling meleset? (jam, hari, tanggal terburuk)
# Di demo, tanggal terburuk = 26–27 Jan 2015 (badai salju NYC), 25 Des, 1 Jan. Perbaikan yang
# masuk akal: fitur `libur` / `cuaca`, bukan model yang lebih rumit.

# %%
mae_per_hour = result.groupby("hour")["abs_error"].mean()
mae_per_day = result.groupby("day")["abs_error"].mean()
mae_per_day = mae_per_day.sort_values()
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
mae_per_hour.plot.bar(ax=axes[0], title="MAE per jam")
mae_per_day.plot.barh(ax=axes[1], title="MAE per hari")
plt.tight_layout()
plt.savefig("mae_jam_hari.png", dpi=150)
plt.show()

mae_per_date = result.groupby("date")["abs_error"].mean()
mae_per_date = mae_per_date.sort_values(ascending=False)
print("5 tanggal terburuk (cek: libur nasional? cuaca ekstrem? event?):")
print(mae_per_date.head(5).round(1))

# %% [markdown]
# ## Residual masih berpola?
# Batang ACF yang keluar dari area biru = residual masih berkorelasi → ada pola yang belum
# ditangkap (mis. lag 1–23 jam tidak dipakai karena ramalan H+1 hari).
# Ljung-Box p-value < 0.05 = residual **bukan** noise acak.

# %%
plot_acf(result["residual"], lags=48)
plt.savefig("acf_residual.png", dpi=150, bbox_inches="tight")
plt.show()
print(acorr_ljungbox(result["residual"], lags=[24, 168]))

# %% [markdown]
# ## Galat vs besar nilai (meleset lebih besar saat ramai?) + simpan

# %%
result["actual_group"] = pd.qcut(result["actual"], 5)
mae_per_group = result.groupby("actual_group", observed=True)["abs_error"].mean()
print(mae_per_group.round(1))
result.to_csv("residual.csv")
