# %% [markdown]
# # Timeseries — Sisa Umur Pakai (RUL, predictive maintenance)
# **Kapan dipakai:** data sensor mesin per siklus, tebak berapa siklus lagi mesin rusak.
#
# **Langkah:** label RUL = siklus maksimum mesin − siklus sekarang, dipotong di `RUL_MAX` (awal
# umur mesin dianggap sehat) → fitur bergulir per mesin → split **per mesin** (bukan per baris)
# → model → latih ulang → tebak RUL di siklus terakhir tiap mesin test.
#
# **Data demo:** NASA C-MAPSS FD001. Train = 100 mesin dijalankan sampai rusak. Test = 100 mesin
# dipotong di tengah jalan.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/timeseries/rul"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/cmapss_train.csv"
TEST_CSV = f"{DATA_FOLDER}/cmapss_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/cmapss_test_labels.csv"   # hanya ada di data latihan

UNIT_COL = "unit"
CYCLE_COL = "siklus"
# sensor yang berubah selama umur mesin (dipilih dari EDA std di bawah)
SENSOR_COLS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17",
               "s20", "s21"]
WINDOW = 10        # panjang rata-rata bergulir per mesin
RUL_MAX = 125
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print(train.shape, test.shape)
print("mesin train:", train[UNIT_COL].nunique(), "| mesin test:", test[UNIT_COL].nunique())
train.head()

# %% [markdown]
# ## EDA: sensor yang konstan tidak berguna, umur mesin, contoh tren sensor

# %%
sensor_std = train.drop(columns=[UNIT_COL, CYCLE_COL]).std()
print(sensor_std.sort_values().round(4))
lifetimes = train.groupby(UNIT_COL)[CYCLE_COL].max()
print(lifetimes.describe().round(1))

for unit in [1, 2, 3]:
    machine = train[train[UNIT_COL] == unit]
    plt.plot(machine[CYCLE_COL], machine["s11"], label=f"mesin {unit}")
plt.title("s11 naik menjelang rusak")
plt.legend()
plt.show()

# %% [markdown]
# ## Label RUL + fitur bergulir per mesin
# Per sensor: rata-rata dan simpangan `WINDOW` siklus terakhir, serta selisih dari siklus pertama.

# %%
lifetime = train.groupby(UNIT_COL)[CYCLE_COL].transform("max")
train["RUL"] = lifetime - train[CYCLE_COL]
train["RUL"] = train["RUL"].clip(upper=RUL_MAX)


def add_features(table):
    table = table.copy()
    groups = table.groupby(UNIT_COL)[SENSOR_COLS]
    rolling_mean = groups.rolling(WINDOW, min_periods=1).mean()
    rolling_std = groups.rolling(WINDOW, min_periods=2).std()
    first_values = groups.transform("first")
    for column in SENSOR_COLS:
        table[f"{column}_mean"] = rolling_mean[column].droplevel(0)   # buang level unit
        table[f"{column}_std"] = rolling_std[column].droplevel(0)
        table[f"{column}_diff"] = table[column] - first_values[column]
    return table


train = add_features(train)
test = add_features(test)
FEATURES = [CYCLE_COL] + SENSOR_COLS
for column in SENSOR_COLS:
    FEATURES = FEATURES + [f"{column}_mean", f"{column}_std", f"{column}_diff"]
print("jumlah fitur:", len(FEATURES))

# %% [markdown]
# ## Split per mesin (25% mesin jadi validasi)
# Simpangan di siklus pertama kosong (NaN) → diisi 0.

# %%
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
train_index, val_index = next(splitter.split(train, groups=train[UNIT_COL]))
train_part = train.iloc[train_index]
val_part = train.iloc[val_index]
X_train = train_part[FEATURES].fillna(0)
y_train = train_part["RUL"]
X_val = val_part[FEATURES].fillna(0)
y_val = val_part["RUL"]
print("mesin latih:", train_part[UNIT_COL].nunique(), "| validasi:", val_part[UNIT_COL].nunique())

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris LightGBM.

# %%
model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, random_state=SEED,
                      verbose=-1)
# model = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED)
# model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED)
# model = HistGradientBoostingRegressor(random_state=SEED)
# model = CatBoostRegressor(iterations=500, random_seed=SEED, verbose=0)
# model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi: RMSE semua baris validasi + RMSE bila mesin dipotong acak (meniru test)

# %%
val_predictions = model.predict(X_val)
val_predictions = np.clip(val_predictions, 0, RUL_MAX)
print("RMSE semua baris validasi:", round(root_mean_squared_error(y_val, val_predictions), 2))

rng = np.random.default_rng(SEED)
cut_rows = []
for unit, machine in val_part.groupby(UNIT_COL):
    cut_cycle = rng.integers(31, machine[CYCLE_COL].max())   # mesin dipotong di siklus acak
    cut_rows.append(machine[machine[CYCLE_COL] == cut_cycle])
cut_val = pd.concat(cut_rows)
cut_predictions = model.predict(cut_val[FEATURES].fillna(0))
cut_predictions = np.clip(cut_predictions, 0, RUL_MAX)
cut_rmse = root_mean_squared_error(cut_val["RUL"], cut_predictions)
print("RMSE siklus terakhir (dipotong acak):", round(cut_rmse, 2))

first_unit = val_part[UNIT_COL].iloc[0]
is_first = (val_part[UNIT_COL] == first_unit).to_numpy()
plt.plot(val_part[CYCLE_COL][is_first], y_val[is_first], label="RUL asli (dipotong)")
plt.plot(val_part[CYCLE_COL][is_first], val_predictions[is_first], label="prediksi")
plt.legend()
plt.show()

# %% [markdown]
# ## Latih ulang di semua mesin train + submission (siklus terakhir tiap mesin test)

# %%
final_model = clone(model)
final_model.fit(train[FEATURES].fillna(0), train["RUL"])
last_cycles = test.groupby(UNIT_COL).tail(1)
prediksi = final_model.predict(last_cycles[FEATURES].fillna(0))
prediksi = np.clip(prediksi, 0, RUL_MAX)

submission = pd.DataFrame()
submission["unit"] = last_cycles[UNIT_COL].to_numpy()
submission["RUL"] = prediksi
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["RUL"]
print("RMSE test:", round(root_mean_squared_error(test_key, prediksi), 2))
print("RMSE test, kunci dipotong 125:",
      round(root_mean_squared_error(test_key.clip(upper=RUL_MAX), prediksi), 2))
