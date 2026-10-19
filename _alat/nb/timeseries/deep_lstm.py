# %% [markdown]
# # Timeseries — Peramalan Deep Learning (LSTM / GRU / 1D-CNN, torch)
# **Kapan dipakai:** deret panjang (ribuan titik) dengan pola rumit. Data kecil (ratusan titik)?
# Model statistik/ML sering menang, jadi selalu bandingkan dengan naive.
#
# **Langkah:** jendela geser: `WINDOW` nilai terakhir → ramal `HORIZON` langkah sekaligus (direct
# multi-output, tanpa rekursi). Tiap jendela dikurangi nilai terakhirnya → model belajar
# **perubahan**, jadi tren naik tetap bisa diikuti. Pembanding: naive musiman (nilai tahun lalu).
#
# **Data demo:** CO2 bulanan Mauna Loa.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from torch import nn
from torch.nn import functional

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/timeseries/forecasting"   # GANTI saat lomba
SERIES_CSV = f"{DATA_FOLDER}/co2_monthly.csv"

DATE_COL = "date"
VALUE_COL = "value"
FREQUENCY = "MS"
SEASON_LENGTH = 12    # panjang musim
WINDOW = 48           # panjang input (jumlah nilai terakhir yang dilihat model)
HORIZON = 36          # jumlah langkah ramalan, data uji = HORIZON titik terakhir
HIDDEN_SIZE = 64
EPOCHS = 400
SEED = 42

# %% [markdown]
# ## Baca data dan split waktu

# %%
table = pd.read_csv(SERIES_CSV, parse_dates=[DATE_COL], index_col=DATE_COL)
y = table[VALUE_COL].asfreq(FREQUENCY)
y = y.interpolate()       # tanggal kosong diisi garis lurus antar tetangga
train = y.iloc[:-HORIZON]
test = y.iloc[-HORIZON:]
print(y.shape, "| latih:", len(train), "| uji:", len(test))
y.plot(figsize=(10, 3), title=VALUE_COL)
plt.show()

# %% [markdown]
# ## Jendela geser
# Input = `WINDOW` nilai, target = `HORIZON` nilai berikutnya, keduanya dikurangi nilai terakhir
# jendela. `SCALE` = simpangan baku perubahan, agar angka input sekitar -1..1.

# %%
def make_windows(series):
    values = series.to_numpy(dtype=np.float32)
    inputs = []
    targets = []
    for start in range(len(values) - WINDOW - HORIZON + 1):
        last = values[start + WINDOW - 1]
        inputs.append(values[start:start + WINDOW] - last)
        targets.append(values[start + WINDOW:start + WINDOW + HORIZON] - last)
    return np.array(inputs), np.array(targets)


train_inputs, train_targets = make_windows(train)
SCALE = float(train_targets.std())
X_train = torch.tensor(train_inputs / SCALE).unsqueeze(-1)     # bentuk (jendela, WINDOW, 1)
y_train = torch.tensor(train_targets / SCALE)
print("jumlah jendela:", X_train.shape, y_train.shape)

# %% [markdown]
# ## Model
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok LSTM.

# %%
torch.manual_seed(SEED)
rnn = nn.LSTM(1, HIDDEN_SIZE, num_layers=2, batch_first=True)
# rnn = nn.GRU(1, HIDDEN_SIZE, num_layers=2, batch_first=True)
head = nn.Linear(HIDDEN_SIZE, HORIZON)
network = nn.ModuleList([rnn, head])     # semua lapisan yang dilatih


def predict_change(X):
    output, _ = rnn(X)
    return head(output[:, -1])          # keadaan di langkah terakhir → HORIZON angka


# Alternatif 1D-CNN:
# cnn = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
#                     nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
#                     nn.Flatten(), nn.Linear(32 * WINDOW, HORIZON))
# network = nn.ModuleList([cnn])
# def predict_change(X):
#     return cnn(X.transpose(1, 2))     # Conv1d butuh bentuk (jendela, saluran, WINDOW)

print(predict_change(X_train[:2]).shape)

# %% [markdown]
# ## Latih (full batch, Adam, loss L1)

# %%
def train_network(X, y_target):
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-3)
    network.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = functional.l1_loss(predict_change(X), y_target)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print("epoch", epoch + 1, "loss", round(loss.item(), 4))


def forecast(series):
    last = float(series.iloc[-1])
    window = series.to_numpy(dtype=np.float32)[-WINDOW:]
    X = torch.tensor((window - last) / SCALE).reshape(1, WINDOW, 1)
    network.eval()
    with torch.no_grad():
        change = predict_change(X)
    return change.numpy().ravel() * SCALE + last


train_network(X_train, y_train)

# %% [markdown]
# ## Evaluasi di data uji: model vs naive musiman

# %%
predictions = forecast(train)
naive = np.resize(train.iloc[-SEASON_LENGTH:].to_numpy(), HORIZON)   # ulang musim terakhir
print("MAE model        :", round(mean_absolute_error(test, predictions), 3))
print("MAE naive musiman:", round(mean_absolute_error(test, naive), 3))
print("MAPE model (%):", round(100 * mean_absolute_percentage_error(test, predictions), 3))

ax = train.iloc[-60:].plot(label="latih")
test.plot(ax=ax, label="aktual")
ax.plot(test.index, predictions, label="ramalan model")
ax.legend()
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + hasil ramalan ke depan
# Bobot dilanjutkan dari model di atas (tambahan `EPOCHS` langkah dengan data lengkap).

# %%
all_inputs, all_targets = make_windows(y)
train_network(torch.tensor(all_inputs / SCALE).unsqueeze(-1), torch.tensor(all_targets / SCALE))
future_values = forecast(y)
future_dates = pd.date_range(start=y.index[-1], periods=HORIZON + 1, freq=FREQUENCY)[1:]

result = pd.DataFrame()
result["tanggal"] = future_dates
result["ramalan"] = future_values
result.to_csv("hasil.csv", index=False)
result.head()
