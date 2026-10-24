# %% [markdown]
# # Preprocessing — Angka, uang, satuan, tanggal, lama waktu dari teks
# **Kapan dipakai:** angka ditulis manusia, misalnya `Rp1.500.000,-`, `1,5 jt`, `3.700rb`,
# `2-3 juta`, `0,7 kwintal`, `12 Agu 2024`, `45516` (tanggal Excel), `2 thn 3 bln`.
#
# **Langkah:** angka + satuan → tanggal → lama waktu → simpan. Aturan angka gaya Indonesia
# yang dipakai `to_number`:
# - titik diikuti **tepat 3 digit** = pemisah ribuan (`1.500.000`, `70.500`)
# - koma = desimal (`1,5`)
# - titik lain = desimal gaya Inggris (`2.1 juta`)
#
# Satuan dibaca dari tabel `*_SCALE` di Konfigurasi. Satuan yang tidak ada di tabel → NaN, supaya
# terlihat di laporan "gagal dibaca", lalu tambahkan satuannya ke tabel.
#
# **Data demo:** `kotor_SINTETIS` (dibuat oleh `_alat/buat_data_kotor.py`).

# %%
import re

import numpy as np
import pandas as pd

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/preprocessing"   # GANTI saat lomba
DIRTY_CSV = f"{DATA_FOLDER}/kotor_SINTETIS.csv"
KEY_CSV = f"{DATA_FOLDER}/kotor_SINTETIS_kunci.csv"   # hanya ada di data latihan

MONEY_SCALE = {"": 1, "rb": 1e3, "ribu": 1e3, "k": 1e3, "jt": 1e6, "juta": 1e6, "m": 1e9,
               "miliar": 1e9, "milyar": 1e9}
WEIGHT_SCALE = {"": 1, "kg": 1, "g": 1e-3, "gr": 1e-3, "gram": 1e-3, "ons": 0.1,
                "kwintal": 100, "ton": 1000}     # tanpa satuan dianggap kg
MONTHS = {"januari": "01", "februari": "02", "maret": "03", "april": "04", "mei": "05",
          "juni": "06", "juli": "07", "agustus": "08", "september": "09", "oktober": "10",
          "november": "11", "desember": "12", "jan": "01", "feb": "02", "mar": "03", "apr": "04",
          "jun": "06", "jul": "07", "agu": "08", "agt": "08", "sep": "09", "okt": "10",
          "nov": "11", "des": "12"}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%m-%y", "%d %m %Y"]
LESS_THAN_YEAR_MONTHS = 6       # "kurang dari 1 tahun" dianggap 6 bulan
# pola: angka [- angka2 kalau rentang] satuan
VALUE_PATTERN = r"(?P<number>\d[\d.,]*)(?:\s*-\s*(?P<number2>\d[\d.,]*))?\s*(?P<unit>[a-z]*)"

# %% [markdown]
# ## Baca data

# %%
table = pd.read_csv(DIRTY_CSV, dtype=str)
print(table.shape)
table[["harga", "berat", "tanggal_daftar", "lama_kerja"]].head(10)

# %% [markdown]
# ## Angka + satuan (uang dan berat)
# Rentang `2-3 juta` diambil nilai tengahnya.

# %%
def to_number(text):
    # '1.500.000' -> 1500000, '70.500' -> 70500, '1,5' -> 1.5, '2.1' -> 2.1
    text = text.strip(".,")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", text):
        text = text.replace(".", "")
    return float(text.replace(",", "."))


def read_value(text, scale):
    found = re.search(VALUE_PATTERN, str(text).lower())
    if found is None:
        return np.nan
    value = to_number(found["number"])
    if found["number2"]:
        value = (value + to_number(found["number2"])) / 2
    return value * scale.get(found["unit"], np.nan)


for text in ["Rp1.500.000,-", "± 4,3 jt", "3.700rb", "4-6 juta"]:
    print(f"{text:15s} -> {read_value(text, MONEY_SCALE):,.1f}")
for text in ["51.600 gr", "0,49 kwintal"]:
    print(f"{text:15s} -> {read_value(text, WEIGHT_SCALE):,.3f}")

# %%
prices = []
for text in table["harga"]:
    prices.append(read_value(text, MONEY_SCALE))
price_rp = pd.Series(prices, index=table.index)

weights = []
for text in table["berat"]:
    weights.append(read_value(text, WEIGHT_SCALE))
weight_kg = pd.Series(weights, index=table.index)

print("harga gagal dibaca:", price_rp.isna().sum())
print(table.loc[price_rp.isna(), "harga"].unique()[:5])
print("berat gagal dibaca:", weight_kg.isna().sum())
print(table.loc[weight_kg.isna(), "berat"].unique()[:5])
pd.DataFrame({"harga": table["harga"], "harga_rp": price_rp, "berat": table["berat"],
              "berat_kg": weight_kg}).head(10)

# %% [markdown]
# ## Tanggal
# 1. Buang nama hari (`Rabu, `).
# 2. Ganti nama bulan Indonesia dengan angka (`Agu` → `08`).
# 3. Coba tiap format di `DATE_FORMATS` **berurutan**: baris yang gagal di format pertama dicoba
#    dengan format berikutnya.
# 4. Angka 5 digit dibaca sebagai tanggal Excel (hari sejak 1899-12-30).
#
# Hati-hati `05/04/2021`: format Indonesia adalah hari/bulan (5 April), bukan bulan/hari.

# %%
date_text = table["tanggal_daftar"].str.lower()
date_text = date_text.str.replace(r"^[a-z]+,\s*", "", regex=True)
for name, number in MONTHS.items():
    date_text = date_text.str.replace(rf"\b{name}\b", number, regex=True)

dates = pd.Series(pd.NaT, index=table.index)
for date_format in DATE_FORMATS:
    parsed = pd.to_datetime(date_text, format=date_format, errors="coerce")
    dates = dates.fillna(parsed)

is_serial = date_text.str.fullmatch(r"\d{5}")
serial = pd.to_numeric(date_text.where(is_serial))
dates = dates.fillna(pd.to_datetime(serial, unit="D", origin="1899-12-30"))

print("tanggal gagal dibaca:", dates.isna().sum())
print(table.loc[dates.isna(), "tanggal_daftar"].head())
pd.DataFrame({"mentah": table["tanggal_daftar"], "tanggal": dates}).head(10)

# %% [markdown]
# Fitur turunan dari tanggal yang biasa dipakai model.

# %%
date_features = pd.DataFrame()
date_features["tahun"] = dates.dt.year
date_features["bulan"] = dates.dt.month
date_features["hari_ke"] = dates.dt.dayofweek          # 0 = Senin
date_features["akhir_pekan"] = dates.dt.dayofweek >= 5
date_features["umur_hari"] = (pd.Timestamp("2025-01-01") - dates).dt.days
date_features.head()

# %% [markdown]
# ## Lama waktu → bulan
# Ambil bagian tahun dan bagian bulan terpisah. Bentuk rapat seperti `1th2bl` juga tertangkap
# karena pola tidak memakai `\b`.

# %%
duration = table["lama_kerja"].str.lower()
years = duration.str.extract(r"(\d+(?:,\d+)?)\s*(?:tahun|thn|th)")[0]
years = pd.to_numeric(years.str.replace(",", "."))
months = pd.to_numeric(duration.str.extract(r"(\d+)\s*(?:bulan|bln|bl)")[0])

duration_months = years.fillna(0) * 12 + months.fillna(0)
duration_months = duration_months.where(years.notna() | months.notna())   # dua-duanya kosong → NaN
is_less_than_year = duration.str.contains(r"kurang|<|blm|belum")
duration_months = duration_months.mask(is_less_than_year, LESS_THAN_YEAR_MONTHS)

print("lama gagal dibaca:", duration_months.isna().sum())
pd.DataFrame({"mentah": table["lama_kerja"], "bulan": duration_months}).head(10)

# %% [markdown]
# ## Simpan hasil

# %%
result = pd.DataFrame()
result["id"] = table["id"]
result["harga_rp"] = price_rp
result["berat_kg"] = weight_kg
result["tanggal_daftar"] = dates.dt.strftime("%Y-%m-%d")
result["lama_bulan"] = duration_months
result = pd.concat([result, date_features], axis=1)
result.to_csv("hasil_angka.csv", index=False)
result.describe()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
key = pd.read_csv(KEY_CSV)
for column in ["harga_rp", "berat_kg", "lama_bulan"]:
    both_empty = result[column].isna() & key[column].isna()
    same = np.isclose(result[column], key[column]) | both_empty
    print(f"{column:15s} benar {same.mean():.3f}")
same = result["tanggal_daftar"] == key["tanggal_daftar"]
print(f"{'tanggal_daftar':15s} benar {same.mean():.3f}")
