# %% [markdown]
# # Simpan dan muat: tabel, array, model, dict, file jawaban
#
# **Butuh:** variabel yang mau disimpan. **Menghasilkan:** file di `outputs/` + cara memuatnya
# kembali setelah kernel restart (tanpa mengulang tahap yang lama).
#
# ## Cara menempel ke notebook-mu
# Tiap bagian punya **dua sel**: sel *simpan* ditempel tepat setelah variabel selesai dihitung,
# sel *muat* ditempel di notebook/tahap berikutnya. Setelah restart: jalankan sel import + sel
# *muat*, lewati sel yang menghitung ulang.
#
# | yang disimpan | format | bagian |
# |---|---|---|
# | tabel pandas | `.csv` (dibaca manusia) / `.pkl` (tipe kolom tetap, mis. tanggal) | 1 |
# | array numpy (fitur, embedding, gambar) | `.npy` / `.npz` | 2 |
# | model sklearn (termasuk `Pipeline`) | `.joblib` | 3 |
# | model PyTorch (+ lanjut training) | `.pt` | 4 |
# | dict / list (konfigurasi, peta label) | `.json` | 5 |
# | objek Python lain apa saja | `.pkl` | 6 |
# | **file jawaban** (`submission.csv`) + cek dengan `sample_submission.csv` | `.csv` | 7 |
# | cadangkan seluruh `outputs/` jadi satu zip | `.zip` | 8 |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(0)
table = pd.DataFrame({"id": range(1, 7), "income": rng.normal(5, 1, 6),
                      "date": pd.date_range("2026-10-01", periods=6)})
features = rng.normal(0, 1, (100, 8)).astype(np.float32)
labels = rng.integers(0, 2, 100)
model = LogisticRegression().fit(features, labels)
network = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
label_names = {0: "tidak", 1: "ya"}
test = pd.DataFrame({"Id": [101, 102, 103, 104]})
X_test = rng.normal(0, 1, (4, 8)).astype(np.float32)
sample = pd.DataFrame({"Id": [101, 102, 103, 104], "label": ["ya", "ya", "ya", "ya"]})
sample.to_csv("sample_submission.csv", index=False)
print(table)

# %% [markdown]
# ## 1. Tabel pandas
# `index=False`: nomor baris tidak ikut jadi kolom. CSV mengubah tanggal jadi teks → saat muat
# tulis `parse_dates=[...]`. Tidak mau repot tipe kolom? Pakai `.pkl` (hanya dibaca pandas).

# %%
table.to_csv("outputs/table.csv", index=False)
table.to_pickle("outputs/table.pkl")

# %%
import pandas as pd

table = pd.read_csv("outputs/table.csv", parse_dates=["date"])   # UBAH: kolom tanggal
table = pd.read_pickle("outputs/table.pkl")                       # atau: tipe kolom persis sama
print(table.dtypes)

# %% [markdown]
# ## 2. Array numpy
# Satu array → `.npy`. Banyak array sekaligus → `.npz` (seperti dict: nama → array).
# Urutan baris array harus sama dengan tabelnya: simpan juga tabelnya (bagian 1).

# %%
import numpy as np

np.save("outputs/features.npy", features)
np.savez("outputs/train_arrays.npz", features=features, labels=labels)

# %%
import numpy as np

features = np.load("outputs/features.npy")
arrays = np.load("outputs/train_arrays.npz")
features = arrays["features"]
labels = arrays["labels"]
print(features.shape, labels.shape)

# %% [markdown]
# ## 3. Model sklearn
# `joblib` ikut terpasang bersama sklearn. Versi sklearn saat muat harus sama dengan saat simpan.

# %%
import joblib

joblib.dump(model, "outputs/model.joblib")

# %%
import joblib

model = joblib.load("outputs/model.joblib")
print(model.predict(features[:5]))

# %% [markdown]
# ## 4. Model PyTorch
# (a) Hanya bobot (`state_dict`): untuk prediksi. Saat muat, **buat dulu model dengan kode yang
# sama persis** (kelas/`nn.Sequential` yang sama), baru isi bobotnya.
# (b) *Checkpoint*: bobot + optimizer + epoch → untuk melanjutkan training yang terputus.

# %%
import torch

torch.save(network.state_dict(), "outputs/network.pt")                       # (a)
epoch = 5                                                                    # UBAH: dari loop
checkpoint = {"model": network.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
torch.save(checkpoint, "outputs/checkpoint.pt")                              # (b)

# %%
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
network = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))   # UBAH: model yang sama
network.load_state_dict(torch.load("outputs/network.pt", map_location=DEVICE))
network = network.to(DEVICE)
network.eval()   # mode prediksi (dropout/batchnorm berhenti berubah)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)   # (b) lanjut training
checkpoint = torch.load("outputs/checkpoint.pt", map_location=DEVICE)
network.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1   # loop training: for epoch in range(start_epoch, EPOCHS)
print("lanjut dari epoch", start_epoch)

# %% [markdown]
# ## 5. Dict / list → JSON
# Kunci JSON selalu teks: `{0: "tidak"}` dimuat kembali sebagai `{"0": "tidak"}` → ubah ke int.
# `ensure_ascii=False`: huruf non-Latin (aksara Jawa, dll.) tetap terbaca di file.

# %%
import json

with open("outputs/label_names.json", "w", encoding="utf-8") as file:
    json.dump(label_names, file, ensure_ascii=False, indent=2)

# %%
import json

with open("outputs/label_names.json", encoding="utf-8") as file:
    loaded = json.load(file)
label_names = {}
for key, value in loaded.items():
    label_names[int(key)] = value   # UBAH: hapus int() kalau kuncinya memang teks
print(label_names)

# %% [markdown]
# ## 6. Objek Python lain → pickle
# Apa saja bisa (tokenizer, encoder, tuple berisi beberapa objek). Hanya buka pickle buatan
# sendiri: pickle dari orang lain bisa menjalankan kode.

# %%
import pickle

bundle = {"label_names": label_names, "feature_count": features.shape[1]}   # UBAH
with open("outputs/bundle.pkl", "wb") as file:
    pickle.dump(bundle, file)

# %%
import pickle

with open("outputs/bundle.pkl", "rb") as file:
    bundle = pickle.load(file)
print(bundle)

# %% [markdown]
# ## 7. File jawaban (pola tetap, ATURAN_KODE §1)
# Tabel baru → kolom ditambah satu per satu → simpan. Urutan baris = urutan `test`.
# Nama kolom disalin persis dari `sample_submission.csv`. Model menebak angka tapi jawaban
# diminta teks? Ubah lewat `map` dulu.

# %%
import pandas as pd

predictions = model.predict(X_test)
predicted_names = pd.Series(predictions).map(label_names)   # UBAH: hapus kalau jawaban angka

submission = pd.DataFrame()
submission["Id"] = test["Id"]                  # UBAH: nama kolom dari sample_submission
submission["label"] = predicted_names.values   # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head())

# %% [markdown]
# Cek file jawaban dengan contoh dari panitia (sel terpisah, hanya mencetak).
# Ketiga baris harus `True`.

# %%
import pandas as pd

sample = pd.read_csv("sample_submission.csv")
written = pd.read_csv("submission.csv")
print("kolom sama   :", written.columns.tolist() == sample.columns.tolist())
print("jumlah sama  :", len(written) == len(sample))
missing_per_column = written.isna().sum()
print("tanpa kosong :", missing_per_column.sum() == 0)
print(written["label"].value_counts())   # UBAH: semua satu kelas → curigai model

# %% [markdown]
# ## 8. Cadangkan seluruh `outputs/` jadi satu zip
# Untuk dipindah ke flashdisk / laptop lain. Membuka: klik kanan → Extract, atau
# `shutil.unpack_archive("cadangan_outputs.zip", "outputs")`.

# %%
import shutil

shutil.make_archive("cadangan_outputs", "zip", "outputs")   # → cadangan_outputs.zip
print("selesai: cadangan_outputs.zip")
