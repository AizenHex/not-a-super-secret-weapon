# %% [markdown]
# # Membersihkan tabel kotor → tabel siap olah
#
# **Butuh:** satu tabel pandas bernama `table` (dari `pd.read_csv` atau `baca_folder.ipynb`).
# **Menghasilkan:** `table` yang sudah bersih + `outputs/bersih.csv`.
#
# ## Cara menempel ke notebook-mu
# Jalankan bagian 1 (lihat cepat) dulu, lalu pilih bagian sesuai masalah yang terlihat. Urutan yang
# dianjurkan kalau masalahnya banyak: **2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11**.
# Tiap bagian hanya memakai `table` dan daftar kolom di baris `# UBAH`.
#
# | masalah yang terlihat | bagian |
# |---|---|
# | nama kolom ada spasi / huruf besar / aneh | 2 |
# | kosong ditulis `-`, `?`, `NA`, `null`, `""` | 3 |
# | angka terbaca teks: `Rp 1.250.000`, `12,5%`, `1,234.5` | 4 |
# | tanggal teks, format campur | 5 |
# | kategori tidak konsisten: `Jakarta`, ` jakarta`, `JKT` | 6 |
# | `ya/tidak`, `Y/N`, `true/false` → 0/1 | 7 |
# | baris dobel | 8 |
# | nilai kosong (NaN) | 9 |
# | nilai ekstrem (outlier) | 10 |
# | cek akhir + simpan | 11 |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba, ganti dengan `table = pd.read_csv(...)`)

# %%
import pandas as pd

table = pd.DataFrame({
    " ID Pelanggan": [1, 2, 3, 3, 4, 5, 6],
    "Kota ": ["Jakarta", " jakarta", "JKT", "JKT", "Bandung", "bandung ", "-"],
    "Harga (Rp)": ["Rp 1.250.000", "Rp 980.000", "1.100.000", "1.100.000", "?", "Rp 2.000.000",
                   "Rp 15.000.000"],
    "Diskon": ["10%", "12,5%", "0%", "0%", "5%", "NA", "7,5%"],
    "Tgl Daftar": ["01/10/2026", "2026-10-02", "03-10-2026", "03-10-2026", "", "05/10/2026",
                   "2026/10/06"],
    "Member": ["ya", "Tidak", "YA", "YA", "tidak", "y", "n"],
})
print(table)

# %% [markdown]
# ## 1. Lihat cepat: ukuran, tipe, kosong, jumlah nilai unik, contoh isi
# Kolom angka tapi tipenya `object` = ada teks di dalamnya (lihat bagian 3–4).

# %%
import pandas as pd

print("ukuran:", table.shape)
report = pd.DataFrame({
    "dtype": table.dtypes.astype(str),
    "kosong": table.isna().sum(),
    "unik": table.nunique(),
    "contoh": table.iloc[0],
})
print(report.to_string())
print(table.head())

# %% [markdown]
# ## 2. Rapikan nama kolom → huruf kecil, tanpa spasi/simbol
# `" ID Pelanggan"` → `"id_pelanggan"`, `"Harga (Rp)"` → `"harga_rp"`. Setelah ini pakai nama baru.

# %%
import re

new_names = []
for name in table.columns:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)   # selain huruf/angka → _
    new_names.append(name.strip("_"))
table.columns = new_names
print(table.columns.tolist())

# %% [markdown]
# ## 3. Penanda kosong → NaN
# Sudah tahu dari awal? Lebih mudah saat membaca: `pd.read_csv(path, na_values=MISSING_MARKS)`.

# %%
import numpy as np

MISSING_MARKS = ["", "-", "?", "NA", "N/A", "null", "None", "nan"]   # UBAH

for column in table.columns:
    if table[column].dtype == object:
        table[column] = table[column].str.strip()
        table[column] = table[column].replace(MISSING_MARKS, np.nan)
print(table.isna().sum())

# %% [markdown]
# ## 4. Angka berbentuk teks → float
# Format Indonesia: titik = ribuan, koma = desimal (`1.250.000,50`). Format Inggris: koma = ribuan
# (`1,250,000.50`) → tukar dua baris `replace` yang ditandai. `errors="coerce"`: yang tetap gagal
# jadi NaN (dicetak supaya kelihatan).

# %%
import pandas as pd

NUMBER_COLUMNS = ["harga_rp", "diskon"]   # UBAH


def to_number(text):
    text = str(text)
    for mark in ["Rp", "%", " "]:   # UBAH: tanda yang dibuang
        text = text.replace(mark, "")
    text = text.replace(".", "")    # UBAH: titik = ribuan (format Indonesia)
    text = text.replace(",", ".")   # UBAH: koma = desimal
    return text


for column in NUMBER_COLUMNS:
    before = table[column].notna()
    table[column] = pd.to_numeric(table[column].apply(to_number), errors="coerce")
    failed = before & table[column].isna()
    print(column, "| gagal diubah:", failed.sum())
table["diskon"] = table["diskon"] / 100   # UBAH: persen → pecahan (opsional)
print(table[NUMBER_COLUMNS])

# %% [markdown]
# ## 5. Tanggal teks → datetime
# Format campur: coba beberapa format satu per satu, isi yang masih kosong. `%d` hari, `%m` bulan,
# `%Y` tahun 4 digit, `%y` 2 digit, `%H:%M` jam:menit. Urutan format penting (hari dulu atau bulan).

# %%
import pandas as pd

DATE_COLUMN = "tgl_daftar"                                        # UBAH
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]   # UBAH

parsed = pd.Series(pd.NaT, index=table.index)
for date_format in DATE_FORMATS:
    attempt = pd.to_datetime(table[DATE_COLUMN], format=date_format, errors="coerce")
    parsed = parsed.fillna(attempt)
still_text = table[DATE_COLUMN].notna() & parsed.isna()
print("tidak terbaca:", table.loc[still_text, DATE_COLUMN].tolist())
table[DATE_COLUMN] = parsed
print(table[[DATE_COLUMN]])

# %% [markdown]
# ## 6. Kategori tidak konsisten
# Langkah: huruf kecil + buang spasi → lihat daftar nilai → tulis kamus penyeragaman `ALIASES`.

# %%
CATEGORY_COLUMNS = ["kota"]               # UBAH
ALIASES = {"jkt": "jakarta"}              # UBAH: {"nilai_lama": "nilai_baru"}

for column in CATEGORY_COLUMNS:
    table[column] = table[column].str.lower()
    table[column] = table[column].str.strip()
    table[column] = table[column].replace(ALIASES)
    print(table[column].value_counts(dropna=False))

# %% [markdown]
# ## 7. Ya/tidak → 1/0
# Nilai yang tidak ada di `YES_NO` menjadi NaN dan dicetak.

# %%
YES_NO_COLUMNS = ["member"]                                                  # UBAH
YES_NO = {"ya": 1, "y": 1, "yes": 1, "true": 1, "1": 1,
          "tidak": 0, "t": 0, "n": 0, "no": 0, "false": 0, "0": 0}           # UBAH

for column in YES_NO_COLUMNS:
    lowered = table[column].astype(str).str.lower()
    table[column] = lowered.map(YES_NO)
    print(column, "| tidak dikenal:", lowered[table[column].isna()].unique())

# %% [markdown]
# ## 8. Baris dobel
# (a) dobel persis di semua kolom. (b) dobel per kunci (mis. id sama): simpan satu, yang mana
# ditentukan urutan sort (di sini tanggal terbaru; tanggal kosong dianggap paling baru).


# %%
KEY_COLUMNS = ["id_pelanggan"]   # UBAH
SORT_COLUMN = "tgl_daftar"       # UBAH: yang terbesar disimpan

print("dobel persis:", table.duplicated().sum())
table = table.drop_duplicates()
print("dobel per kunci:", table.duplicated(KEY_COLUMNS).sum())
table = table.sort_values(SORT_COLUMN)
table = table.drop_duplicates(KEY_COLUMNS, keep="last")
table = table.sort_index()
table = table.reset_index(drop=True)
print("baris sekarang:", len(table))

# %% [markdown]
# ## 9. Nilai kosong (NaN)
# Pilih per kolom: buang baris, isi median (angka), isi modus (kategori), isi median per kelompok.
# Kolom penanda `<kolom>_kosong` menyimpan informasi "tadinya kosong" (sering membantu model).
# Untuk model: hitung median dari **train saja**, lalu pakai angka yang sama di test.

# %%
FILL_MEDIAN = ["harga_rp", "diskon"]   # UBAH: angka → median
FILL_MODE = ["kota"]                   # UBAH: kategori → nilai terbanyak
GROUP_COLUMN = "kota"                  # UBAH: median per kelompok untuk FILL_GROUP_MEDIAN
FILL_GROUP_MEDIAN = []                 # UBAH
DROP_IF_EMPTY = []                     # UBAH: baris dibuang kalau kolom ini kosong

table = table.dropna(subset=DROP_IF_EMPTY)
for column in FILL_MEDIAN + FILL_MODE + FILL_GROUP_MEDIAN:
    table[f"{column}_kosong"] = table[column].isna().astype(int)
for column in FILL_GROUP_MEDIAN:
    group_median = table.groupby(GROUP_COLUMN)[column].transform("median")
    table[column] = table[column].fillna(group_median)
for column in FILL_MEDIAN:
    table[column] = table[column].fillna(table[column].median())
for column in FILL_MODE:
    table[column] = table[column].fillna(table[column].mode()[0])
print(table.isna().sum())

# %% [markdown]
# ## 10. Nilai ekstrem (outlier) dengan aturan IQR
# Batas = Q1 − 1,5·IQR dan Q3 + 1,5·IQR. Pilih: lihat saja, potong ke batas (`clip`), atau buang.
# Jangan buang baris test: jumlah baris jawaban harus sama dengan jumlah baris test.

# %%
OUTLIER_COLUMNS = ["harga_rp"]   # UBAH

for column in OUTLIER_COLUMNS:
    q1 = table[column].quantile(0.25)
    q3 = table[column].quantile(0.75)
    low = q1 - 1.5 * (q3 - q1)
    high = q3 + 1.5 * (q3 - q1)
    outside = (table[column] < low) | (table[column] > high)
    print(column, "| batas:", low, high, "| di luar:", table.loc[outside, column].tolist())
    table[column] = table[column].clip(low, high)   # UBAH: hapus baris ini kalau hanya melihat

# %% [markdown]
# ## 11. Cek akhir + simpan
# Ulangi laporan bagian 1. Simpan supaya langkah berikutnya bisa mulai dari file ini.

# %%
import os

os.makedirs("outputs", exist_ok=True)
print(table.dtypes)
missing_per_column = table.isna().sum()
print("kosong tersisa:", missing_per_column.sum())
table.to_csv("outputs/bersih.csv", index=False)
print(table)
