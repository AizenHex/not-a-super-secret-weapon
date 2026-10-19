# %% [markdown]
# # Preprocessing — Kategori kotor → kategori baku (fuzzy matching + tabel peta)
# **Kapan dipakai:** satu kategori ditulis banyak cara, misalnya `Teknik Informatika`,
# `T. Informatika`, `Tehnik Informatika`, `TI`, `informatika`.
#
# **Langkah:**
# 1. Ambil **nilai unik**. Yang dibersihkan cukup puluhan nilai unik, bukan ribuan baris.
# 2. Normalisasi: huruf kecil, tanda baca jadi spasi, singkatan diperluas (`t` → `teknik`,
#    `ti` → `teknik informatika`).
# 3. Cari kategori baku termirip dengan `difflib`, beserta skornya (0–1).
# 4. Simpan **draf peta** ke CSV, urut dari skor terendah, periksa yang skornya di bawah
#    `THRESHOLD`, tulis koreksi di `CORRECTIONS`.
# 5. Terapkan peta dan cek tidak ada nilai yang tertinggal.
#
# Nilai unik boleh diambil dari train + test sekaligus, karena target tidak dipakai.
#
# **Data demo:** kolom `jurusan` di `kotor_SINTETIS` (dibuat oleh `_alat/buat_data_kotor.py`).

# %%
import re
from difflib import SequenceMatcher

import pandas as pd

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/preprocessing"   # GANTI saat lomba
DIRTY_CSV = f"{DATA_FOLDER}/kotor_SINTETIS.csv"
KEY_CSV = f"{DATA_FOLDER}/kotor_SINTETIS_kunci.csv"   # hanya ada di data latihan

COLUMN = "jurusan"
STANDARD = ["Teknik Informatika", "Sistem Informasi", "Manajemen", "Akuntansi", "Statistika",
            "Ilmu Hukum"]
WORD_ABBREVIATIONS = {"t": "teknik", "tek": "teknik", "sist": "sistem", "s": "sistem",
                      "i": "ilmu", "fh": ""}
FULL_ABBREVIATIONS = {"ti": "teknik informatika", "si": "sistem informasi", "stat": "statistika"}
THRESHOLD = 0.85                   # skor di bawah ini diperiksa manual
# Diisi SETELAH melihat draf peta (nilai mentah -> kategori baku)
CORRECTIONS = {"FH - Hukum": "Ilmu Hukum", "Hukum": "Ilmu Hukum", "Akunting": "Akuntansi",
               "Informatika": "Teknik Informatika", "Menejemen": "Manajemen"}

# %% [markdown]
# ## Baca data

# %%
table = pd.read_csv(DIRTY_CSV, dtype=str)
print(table[COLUMN].nunique(), "nilai unik dari", len(table), "baris")
table[COLUMN].value_counts()

# %% [markdown]
# ## Normalisasi + cari kategori baku termirip

# %%
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)     # selain huruf dan angka jadi spasi
    words = []
    for word in text.split():
        words.append(WORD_ABBREVIATIONS.get(word, word))
    text = " ".join(words)
    text = " ".join(text.split())             # spasi ganda dirapikan
    return FULL_ABBREVIATIONS.get(text, text)


def best_match(text):
    best_name = ""
    best_score = 0
    for name in STANDARD:
        score = SequenceMatcher(None, text, name.lower()).ratio()
        if score > best_score:
            best_name = name
            best_score = score
    return best_name, round(best_score, 2)


print(normalize("T. Informatika"), "|", normalize(" TI "), "|", normalize("FH - Hukum"))
print(best_match("tehnik informatika"))

# %% [markdown]
# ## Draf peta (satu baris per nilai unik, skor terendah di atas)

# %%
counts = table[COLUMN].value_counts()
normalized = []
suggestions = []
scores = []
for raw in counts.index:
    clean = normalize(raw)
    name, score = best_match(clean)
    normalized.append(clean)
    suggestions.append(name)
    scores.append(score)

draft = pd.DataFrame()
draft["raw"] = counts.index
draft["count"] = counts.to_numpy()
draft["normalized"] = normalized
draft["suggestion"] = suggestions
draft["score"] = scores
draft = draft.sort_values("score")
draft.to_csv(f"peta_{COLUMN}_DRAF.csv", index=False)
print("perlu dicek manual:", (draft["score"] < THRESHOLD).sum(), "nilai")
draft.head(15)

# %% [markdown]
# Baca baris dengan skor di bawah `THRESHOLD`. Kalau usulannya benar, biarkan. Kalau salah, tulis
# di `CORRECTIONS` (sel Konfigurasi), lalu jalankan ulang. Usulan skor rendah yang ternyata benar
# juga bisa ditulis di `CORRECTIONS` supaya jelas sudah diperiksa.
# Untuk peta yang panjang: buka CSV draf di Excel, perbaiki kolom `suggestion`, lalu baca kembali
# dengan `pd.read_csv`.

# %%
mapping = {}
for raw, suggestion, score in zip(draft["raw"], draft["suggestion"], draft["score"]):
    if score >= THRESHOLD:
        mapping[raw] = suggestion
mapping.update(CORRECTIONS)
table[COLUMN + "_baku"] = table[COLUMN].map(mapping)

not_mapped = table[COLUMN + "_baku"].isna()
print("masih kosong:", not_mapped.sum())
print(table.loc[not_mapped, COLUMN].value_counts())
table[COLUMN + "_baku"].value_counts()

# %% [markdown]
# ## Simpan hasil + peta
# Untuk test di soal: pakai `mapping` yang sama, misalnya `test[COLUMN].map(mapping)`.

# %%
result = pd.DataFrame()
result["id"] = table["id"]
result[COLUMN] = table[COLUMN + "_baku"]
result.to_csv(f"hasil_{COLUMN}.csv", index=False)

mapping_table = pd.DataFrame()
mapping_table["mentah"] = list(mapping.keys())
mapping_table["baku"] = list(mapping.values())
mapping_table.to_csv(f"peta_{COLUMN}.csv", index=False)
result.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
key = pd.read_csv(KEY_CSV, dtype=str)
correct = result[COLUMN] == key[COLUMN]
print("benar:", round(correct.mean(), 3))

check = pd.DataFrame()
check["mentah"] = table[COLUMN]
check["hasil"] = result[COLUMN]
check["kunci"] = key[COLUMN]
check[~correct].head()
