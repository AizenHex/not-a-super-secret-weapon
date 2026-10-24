# %% [markdown]
# # Timeseries — VAR (beberapa deret saling mempengaruhi, diramal bersama)
# **Kapan dipakai:** beberapa deret yang saling terkait (inflasi, kurs, suku bunga) dan semuanya
# harus diramal, tanpa nilai masa depan dari luar.
#
# **Langkah:** uji stasioner (ADF) → log + differencing → pilih jumlah lag (AIC) → ramal →
# kembalikan ke skala asli → MAE per variabel vs naive. Bonus: uji Granger (apakah X membantu
# meramal Y?).
#
# **Data demo:** makro ekonomi AS kuartalan (`realgdp`, `realcons`, `realinv`, `unemp`, `infl`).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/timeseries/forecasting"   # GANTI saat lomba
SERIES_CSV = f"{DATA_FOLDER}/macro_quarterly.csv"

DATE_COL = "date"
LOG_COLS = ["realgdp", "realcons", "realinv"]   # deret level positif → log lalu diff
PLAIN_COLS = ["unemp", "infl"]                  # deret persen → diff saja
GRANGER_CAUSE = "realinv"     # apakah kolom ini membantu meramal ...
GRANGER_TARGET = "realgdp"    # ... kolom ini?
FREQUENCY = "QS"
HORIZON = 8          # jumlah kuartal ramalan, data uji = HORIZON titik terakhir
MAX_LAG = 8

# %% [markdown]
# ## Baca data

# %%
table = pd.read_csv(SERIES_CSV, parse_dates=[DATE_COL], index_col=DATE_COL)
table = table[LOG_COLS + PLAIN_COLS].asfreq(FREQUENCY)
missing = table.isna().sum()
print(table.shape, "| missing:", missing.sum())
(table / table.iloc[0].abs().clip(lower=1)).plot(subplots=True, figsize=(10, 7))
plt.show()

# %% [markdown]
# ## Uji stasioner ADF: p < 0.05 → stasioner (boleh masuk VAR)

# %%
transformed = pd.concat([np.log(table[LOG_COLS]), table[PLAIN_COLS]], axis=1)
differenced = transformed.diff().dropna()
for column in table.columns:
    p_original = adfuller(table[column])[1]
    p_differenced = adfuller(differenced[column])[1]
    print(column, "| p asli", round(p_original, 3), "| p setelah diff", round(p_differenced, 3))

# %% [markdown]
# ## Split waktu

# %%
train = differenced.iloc[:-HORIZON]
test = table.iloc[-HORIZON:]
last_train_values = transformed.iloc[-HORIZON - 1]     # nilai (log) terakhir sebelum uji
print("latih:", train.shape, "| uji:", test.shape)

# %% [markdown]
# ## Model
# Pilih SATU cara memilih lag: hapus `#` di baris yang ingin dipakai, beri `#` pada baris AIC.

# %%
var = VAR(train)
print(var.select_order(MAX_LAG).summary())
model = var.fit(maxlags=MAX_LAG, ic="aic")
# model = var.fit(maxlags=MAX_LAG, ic="bic")
# model = var.fit(maxlags=MAX_LAG, ic="hqic")
# model = var.fit(2)                          # jumlah lag ditulis langsung
print("lag terpilih:", model.k_ar)

# %% [markdown]
# ## Ramal lalu kembalikan ke skala asli
# Diff dijumlah kumulatif + nilai terakhir, lalu `exp` untuk kolom log.

# %%
def undo_transform(differenced_forecast, last_values):
    level = differenced_forecast.cumsum() + last_values
    level[LOG_COLS] = np.exp(level[LOG_COLS])
    return level


recent = train.to_numpy()[-model.k_ar:]     # k_ar baris terakhir = input ramalan
forecast = model.forecast(recent, steps=HORIZON)
forecast = pd.DataFrame(forecast, index=test.index, columns=differenced.columns)
predictions = undo_transform(forecast, last_train_values)
predictions.round(2)

# %% [markdown]
# ## Evaluasi: MAE per variabel, VAR vs naive (nilai terakhir)
# Data demo: periode uji = krisis 2008, VAR (yang meneruskan tren) kalah dari naive untuk GDP.
# Itu wajar, guncangan besar tidak bisa diramal. Selalu laporkan pembanding naive.

# %%
naive = table.iloc[-HORIZON - 1]
mae_var = []
mae_naive = []
mape_var = []
for column in table.columns:
    mae_var.append(mean_absolute_error(test[column], predictions[column]))
    mae_naive.append(mean_absolute_error(test[column], np.repeat(naive[column], HORIZON)))
    relative_error = abs(test[column] - predictions[column]) / abs(test[column])
    mape_var.append(100 * relative_error.mean())
scores = pd.DataFrame(index=table.columns)
scores["MAE_VAR"] = mae_var
scores["MAE_naive"] = mae_naive
scores["MAPE_VAR_%"] = mape_var
print(scores.round(3))

fig, axes = plt.subplots(len(table.columns), 1, figsize=(10, 9), sharex=True)
for ax, column in zip(axes, table.columns):
    ax.plot(table[column].iloc[-40:], label="aktual")
    ax.plot(predictions[column], label="VAR")
    ax.set_title(column)
axes[0].legend()
plt.show()

# %% [markdown]
# ## Uji Granger: apakah `GRANGER_CAUSE` membantu meramal `GRANGER_TARGET`? (p < 0.05 → ya)

# %%
granger = grangercausalitytests(differenced[[GRANGER_TARGET, GRANGER_CAUSE]], maxlag=4)
for lag in granger:
    p_value = granger[lag][0]["ssr_ftest"][1]
    print("lag", lag, ": p =", round(p_value, 4))

# %% [markdown]
# ## Latih ulang di semua data + hasil ramalan ke depan

# %%
model = VAR(differenced).fit(model.k_ar)
future_dates = pd.date_range(start=table.index[-1], periods=HORIZON + 1, freq=FREQUENCY)[1:]
forecast = model.forecast(differenced.to_numpy()[-model.k_ar:], steps=HORIZON)
forecast = pd.DataFrame(forecast, index=future_dates, columns=differenced.columns)
future_values = undo_transform(forecast, transformed.iloc[-1])

result = pd.DataFrame()
result["tanggal"] = future_dates
for column in table.columns:
    result[column] = future_values[column].to_numpy()
result.to_csv("hasil.csv", index=False)
result.round(2)
