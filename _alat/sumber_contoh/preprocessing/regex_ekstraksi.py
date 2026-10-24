# %% [markdown]
# # Preprocessing — Regex: ambil bagian penting dari teks
# **Kapan dipakai:** kolom kode/nomor/email/alamat dengan format campur aduk, dan hanya sebagian
# yang penting (tahun dari kode peserta, operator dari nomor HP, domain dari email, kode pos dari
# alamat).
#
# **Langkah** kerja regex:
# 1. Uji pola di beberapa **contoh** dulu.
# 2. Tulis pola dengan **grup bernama** `(?P<nama>...)`. `str.extract` langsung mengubah tiap grup
#    menjadi kolom.
# 3. Lihat baris yang **tidak cocok**, perbaiki pola, ulangi sampai sisanya kosong atau wajar.
#
# | Potongan | Arti |
# |---|---|
# | `(?P<tahun>\d{4})` | grup bernama: bagian penting, jadi kolom `tahun` |
# | `(?:jl\|jln\|jalan)` | grup tanpa nama: harus cocok, tapi tidak diambil |
# | `[-/. ]?` | satu karakter pemisah, boleh tidak ada |
# | `\d{2,5}` / `\d+` | 2 sampai 5 digit / 1 digit atau lebih |
# | `(?i)` di awal pola | abaikan huruf besar/kecil |
# | `(?x)` di awal pola | boleh spasi dan komentar `#` di dalam pola (pola panjang jadi terbaca) |
# | `^` `$` `\b` | awal teks, akhir teks, batas kata |
# | `.*?` | apa saja, sesedikit mungkin |
# | `\d+(?=\s*kg)` | lookahead: angka yang **diikuti** "kg", tanpa ikut mengambil "kg" |
#
# Pembersihan baris per baris seperti ini tidak memakai target, jadi aman dilakukan sebelum split.
#
# **Data demo:** `kotor_SINTETIS` (dibuat oleh `_alat/buat_data_kotor.py`).

# %%
import pandas as pd

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/preprocessing"   # GANTI saat lomba
DIRTY_CSV = f"{DATA_FOLDER}/kotor_SINTETIS.csv"
KEY_CSV = f"{DATA_FOLDER}/kotor_SINTETIS_kunci.csv"   # hanya ada di data latihan

# kode peserta, dua urutan: tahun di depan (2021-TI-00451) dan prodi di depan (TI-2021-451)
CODE_PATTERN_A = r"(?P<tahun>\d{4}|\d{2})[-/.]?(?P<prodi>[A-Za-z]{2})[-/.]?(?P<nomor>\d{1,5})"
CODE_PATTERN_B = r"(?P<prodi>[A-Za-z]{2})-(?P<tahun>\d{4})-(?P<nomor>\d{1,5})"
EMAIL_PATTERN = r"^(?P<user>[\w.]+)@(?P<domain>[\w.]+\.[a-z]+)$"
ADDRESS_PATTERN = r"""(?ix)
    \brt\.?\s*(?P<rt>\d{1,3})        # RT 03 / rt.3 / RT003
    \W*rw\.?\s*(?P<rw>\d{1,3})       # RW 05 / rw.5 / RW005, dipisah / , atau spasi
    .*?(?P<kota>[a-z]+)              # kata terakhir sebelum kode pos = kota
    [\s,-]*(?P<kodepos>\d{5})?       # kode pos 5 digit, boleh tidak ada
    \s*$                             # sampai akhir teks
"""

# %% [markdown]
# ## Baca data

# %%
# dtype=str: semua kolom dibaca sebagai teks, supaya "0812..." tidak berubah jadi 812...
table = pd.read_csv(DIRTY_CSV, dtype=str)
print(table.shape)
table[["kode_peserta", "no_hp", "email", "alamat"]].head(10)

# %% [markdown]
# ## Kode peserta → tahun, prodi, nomor
# Ada dua urutan: tahun di depan (`2021-TI-00451`, `2021TI00451`, `21/TI/451`, `2021.ti.00451`)
# dan prodi di depan (`TI-2021-451`). Jadi dibuat dua pola (`CODE_PATTERN_A`, `CODE_PATTERN_B`).
# Pola B mengisi baris yang gagal di pola A.

# %%
examples = pd.Series(["2021-TI-00451", "21/TI/451", "No. 2021-TI-00451", "TI-2021-451"])
example_a = examples.str.extract(CODE_PATTERN_A)
example_b = examples.str.extract(CODE_PATTERN_B)
print(example_a.fillna(example_b))

# %%
code_a = table["kode_peserta"].str.extract(CODE_PATTERN_A)
code_b = table["kode_peserta"].str.extract(CODE_PATTERN_B)
code = code_a.fillna(code_b)
year = pd.to_numeric(code["tahun"])
code["tahun"] = year.where(year > 100, year + 2000)      # 21 -> 2021
code["prodi"] = code["prodi"].str.upper()
code["nomor"] = pd.to_numeric(code["nomor"])

not_matched = code.isna().any(axis=1)
print("tidak cocok:", not_matched.sum())
print(table.loc[not_matched, "kode_peserta"].head())
code.head()

# %% [markdown]
# ## Nomor HP → format seragam `08...` + operator
# Lebih mudah **membuang** yang tidak perlu daripada mencocokkan semua bentuk:
# buang semua selain digit, ganti `62` di depan dengan `0`, lalu cek bentuk akhirnya.

# %%
phone = table["no_hp"].str.replace(r"\D", "", regex=True)    # buang semua selain digit
phone = phone.str.replace(r"^62", "0", regex=True)             # 62812... -> 0812...
phone = phone.where(phone.str.fullmatch(r"08\d{8,11}"))        # bukan nomor HP -> NaN
phone_prefix = phone.str[:4]                                   # 0812, 0857, ... (operator)

print("kosong / bukan HP:", phone.isna().sum())
pd.DataFrame({"mentah": table["no_hp"], "bersih": phone, "prefix": phone_prefix}).head(8)

# %% [markdown]
# ## Email → user, domain, email kampus
# Perbaiki dulu bentuk yang ditulis manusia (`budi at gmail dot com`, spasi, titik di akhir),
# baru ambil bagiannya.

# %%
email = table["email"].str.strip()
email = email.str.lower()
email = email.str.replace(r"\s+at\s+", "@", regex=True)
email = email.str.replace(r"\s+dot\s+", ".", regex=True)
email = email.str.rstrip(".")
email_parts = email.str.extract(EMAIL_PATTERN)
email_parts["kampus"] = email_parts["domain"].str.endswith(".ac.id")

print("tidak cocok:", email_parts["domain"].isna().sum())
email_parts.head()

# %% [markdown]
# ## Alamat → RT, RW, kota, kode pos
# Pola panjang (`ADDRESS_PATTERN` di Konfigurasi) ditulis dengan `(?x)`, satu bagian per baris,
# masing-masing dengan komentar.

# %%
address = table["alamat"].str.extract(ADDRESS_PATTERN)
address["rt"] = pd.to_numeric(address["rt"])
address["rw"] = pd.to_numeric(address["rw"])
address["kota"] = address["kota"].str.title()

print("tidak cocok:", address["kota"].isna().sum())
print("tanpa kode pos:", address["kodepos"].isna().sum())
pd.concat([table["alamat"], address], axis=1).head(8)

# %% [markdown]
# ## Bonus: semua kemunculan dan lookahead
# `str.findall` mengambil **semua** kecocokan (hasilnya list). Lookahead `(?=...)` mensyaratkan
# teks sesudahnya tanpa ikut mengambilnya.

# %%
first_addresses = table["alamat"].head(3)
print(list(first_addresses.str.findall(r"\d+")))                    # semua angka di alamat
print(first_addresses.str.extract(r"(?i)no\.?\s*(\d+)(?=\D)"))      # nomor rumah sesudah "No"

# %% [markdown]
# ## Simpan hasil

# %%
result = pd.DataFrame()
result["id"] = table["id"]
result["tahun_masuk"] = code["tahun"]
result["kode_prodi"] = code["prodi"]
result["nomor_urut"] = code["nomor"]
result["no_hp"] = phone
result["prefix_hp"] = phone_prefix
result["email"] = email
result["domain"] = email_parts["domain"]
result["email_kampus"] = email_parts["kampus"]
result["rt"] = address["rt"]
result["rw"] = address["rw"]
result["kota"] = address["kota"]
result["kodepos"] = address["kodepos"]
result.to_csv("hasil_regex.csv", index=False)
result.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# Angka dibandingkan sebagai teks: "2021.0" dan "2021" dianggap sama.

# %%
key = pd.read_csv(KEY_CSV, dtype=str)
for column in ["tahun_masuk", "kode_prodi", "nomor_urut", "no_hp", "email", "rt", "rw", "kota",
               "kodepos"]:
    guess = result[column].astype(str).str.removesuffix(".0")
    answer = key[column].astype(str).str.removesuffix(".0")
    correct = guess == answer
    print(f"{column:12s} benar {correct.mean():.3f}")
