# %% [markdown]
# # Timeseries — Peramalan Global (ratusan deret sekaligus, satu model)
# **Kapan dipakai:** banyak deret sejenis (harga komoditas per provinsi, penjualan per toko),
# diminta ramalan periode berikutnya. Pola Datavidia 2024: 13 komoditas x 34 provinsi = 442 deret,
# ramal 92 hari ke depan.
#
# **Langkah:** **satu** LightGBM untuk semua deret. Fitur = harga 92+ hari lalu (lag), rata-rata
# bergulir, kalender, dan kode provinsi/komoditas. Semua lag >= `HORIZON`, jadi fitur test sudah
# tersedia tanpa ramalan rekursif. Target = rasio harga terhadap lag terpendek (skala antar
# komoditas jadi sama). Metrik: MAPE.
#
# **Data demo:** harga pangan SINTETIS (format sama dengan soal, ada harga kosong).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/timeseries/global"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/harga_pangan_SINTETIS_train.csv"
TEST_CSV = f"{DATA_FOLDER}/harga_pangan_SINTETIS_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/harga_pangan_SINTETIS_test_labels.csv"   # hanya di data latihan

DATE_COL = "date"
TARGET = "harga"
PROVINCE_COL = "provinsi"
COMMODITY_COL = "komoditas"
SERIES_COLS = [PROVINCE_COL, COMMODITY_COL]   # kolom yang menandai satu deret
HORIZON = 92                                  # panjang periode test (hari)
LAGS = [92, 99, 106, 122, 182, 365]           # semua >= HORIZON, yang pertama = HORIZON
EXAMPLE_PROVINCE = "Aceh"                     # untuk grafik contoh
EXAMPLE_COMMODITY = "cabai_merah"
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL])
test = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL])
print("train:", train.shape, "| test:", test.shape)
print("jumlah deret:", train.groupby(SERIES_COLS).ngroups)
print("train:", train[DATE_COL].min().date(), "s.d.", train[DATE_COL].max().date())
print("test :", test[DATE_COL].min().date(), "s.d.", test[DATE_COL].max().date())
print("harga kosong di train:", round(train[TARGET].isna().mean() * 100, 1), "%")
train.head()

# %% [markdown]
# ## EDA

# %%
example = train[train[PROVINCE_COL] == EXAMPLE_PROVINCE]
example = example.pivot_table(index=DATE_COL, columns=COMMODITY_COL, values=TARGET)
(example / example.mean()).plot(figsize=(11, 4), legend=False,
                                title=f"{EXAMPLE_PROVINCE}, harga / rata-rata")
plt.show()

spread = train.groupby(COMMODITY_COL)[TARGET].agg(["median", "std"])
print(spread.round(0))
is_missing = train[TARGET].isna()
missing_share = is_missing.groupby(train[COMMODITY_COL]).mean()
print(missing_share.round(3))

# %% [markdown]
# ## Fitur
# Train dan test digabung agar lag test bisa mengambil harga dari train. Harga kosong diisi nilai
# terakhir (ffill) per deret. Urutan baris tetap: train dulu, lalu test apa adanya.
# `groupby(...).rolling(...)` menghasilkan index bertingkat; `reset_index(level=SERIES_COLS)`
# mengembalikannya ke nomor baris asli supaya bisa ditempel.

# %%
BASE_LAG = f"lag_{HORIZON}"      # lag terpendek
all_rows = pd.concat([train, test.drop(columns="id")], ignore_index=True)
groups = all_rows.groupby(SERIES_COLS)
all_rows["filled"] = groups[TARGET].ffill()
for lag in LAGS:
    all_rows[f"lag_{lag}"] = groups["filled"].shift(lag)

groups = all_rows.groupby(SERIES_COLS)
mean_7 = groups[BASE_LAG].rolling(7).mean()
all_rows["mean_7"] = mean_7.reset_index(level=SERIES_COLS, drop=True)
mean_30 = groups[BASE_LAG].rolling(30).mean()
all_rows["mean_30"] = mean_30.reset_index(level=SERIES_COLS, drop=True)
all_rows["trend_30"] = all_rows[BASE_LAG] / groups[BASE_LAG].shift(30)
all_rows["month"] = all_rows[DATE_COL].dt.month
all_rows["weekday"] = all_rows[DATE_COL].dt.dayofweek
all_rows["day_of_year"] = all_rows[DATE_COL].dt.dayofyear
all_rows[PROVINCE_COL] = all_rows[PROVINCE_COL].astype("category")
all_rows[COMMODITY_COL] = all_rows[COMMODITY_COL].astype("category")
all_rows["ratio"] = all_rows[TARGET] / all_rows[BASE_LAG]     # target model

FEATURES = SERIES_COLS + ["mean_7", "mean_30", "trend_30", "month", "weekday", "day_of_year"]
for lag in LAGS:
    FEATURES.append(f"lag_{lag}")
all_rows[FEATURES].tail()

# %% [markdown]
# ## Split waktu: 92 hari terakhir train = validasi (meniru jarak train → test)
# `naive` = harga terakhir sebelum validasi, dipakai sebagai pembanding.

# %%
VAL_START = train[DATE_COL].max() - pd.DateOffset(days=HORIZON - 1)
TEST_START = test[DATE_COL].min()
before_val = all_rows["filled"].where(all_rows[DATE_COL] < VAL_START)
all_rows["naive"] = before_val.groupby([all_rows[PROVINCE_COL], all_rows[COMMODITY_COL]],
                                       observed=True).ffill()

labeled = all_rows[all_rows[TARGET].notna() & all_rows[BASE_LAG].notna()]
fit_rows = labeled[labeled[DATE_COL] < VAL_START]
val_rows = labeled[(labeled[DATE_COL] >= VAL_START) & (labeled[DATE_COL] < TEST_START)]
print("latih:", fit_rows.shape, "| validasi:", val_rows.shape,
      "| validasi mulai:", VAL_START.date())

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris aktif.
# Kolom provinsi/komoditas bertipe category, dipakai langsung oleh semua model di bawah.
# CatBoost paling lambat (±6 menit di laptop), yang lain < 1 menit.

# %%
model = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, objective="l1",
                      random_state=SEED, verbose=-1)
# model = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, random_state=SEED,
#                       verbose=-1)
# model = HistGradientBoostingRegressor(loss="absolute_error", categorical_features="from_dtype",
#                                       max_iter=400, random_state=SEED)
# model = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8,
#                      enable_categorical=True, tree_method="hist", random_state=SEED)
# model = CatBoostRegressor(iterations=600, learning_rate=0.08, loss_function="MAE",
#                           cat_features=SERIES_COLS, verbose=0, allow_writing_files=False,
#                           random_seed=SEED)

# %% [markdown]
# ## Evaluasi: MAPE model vs naive (harga terakhir sebelum validasi)

# %%
model.fit(fit_rows[FEATURES], fit_rows["ratio"])
val_predictions = model.predict(val_rows[FEATURES]) * val_rows[BASE_LAG]
mape_model = mean_absolute_percentage_error(val_rows[TARGET], val_predictions) * 100
mape_naive = mean_absolute_percentage_error(val_rows[TARGET], val_rows["naive"]) * 100
print("MAPE model:", round(mape_model, 3), "% | naive:", round(mape_naive, 3), "%")

errors = abs(val_rows[TARGET] - val_predictions) / val_rows[TARGET]
per_commodity = errors.groupby(val_rows[COMMODITY_COL], observed=True).mean() * 100
print(per_commodity.round(2).sort_values())

one_series = val_rows[PROVINCE_COL] == EXAMPLE_PROVINCE
one_series = one_series & (val_rows[COMMODITY_COL] == EXAMPLE_COMMODITY)
plt.plot(val_rows[DATE_COL][one_series], val_rows[TARGET][one_series], label="aktual")
plt.plot(val_rows[DATE_COL][one_series], val_predictions[one_series], label="ramalan")
plt.title(f"{EXAMPLE_PROVINCE}, {EXAMPLE_COMMODITY} (validasi)")
plt.legend()
plt.show()

# %% [markdown]
# ## Latih ulang di semua data train + submission

# %%
all_train_rows = labeled[labeled[DATE_COL] < TEST_START]
model.fit(all_train_rows[FEATURES], all_train_rows["ratio"])
test_rows = all_rows[all_rows[DATE_COL] >= TEST_START]
predictions = model.predict(test_rows[FEATURES]) * test_rows[BASE_LAG].to_numpy()

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["harga"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
has_price = test_key.notna()
test_mape = mean_absolute_percentage_error(test_key[has_price], predictions[has_price]) * 100
print("MAPE test:", round(test_mape, 3), "%")
