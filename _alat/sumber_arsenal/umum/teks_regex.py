# %% [markdown]
# # Olah teks tanpa model: regex, bersihkan, ambil field, pecah, cocokkan nama mirip
#
# **Butuh:** tabel `table` dengan kolom teks. **Menghasilkan:** kolom baru per bagian.
# Untuk teks sebagai input model (klasifikasi, embedding) lihat folder `teks/`.
#
# ## Cara menempel ke notebook-mu
# Tiap bagian berdiri sendiri; ganti `TEXT_COLUMN` dan pola di baris `# UBAH`.
#
# | ingin | bagian |
# |---|---|
# | contekan pola regex | 1 |
# | bersihkan teks (huruf kecil, buang URL/mention/simbol, kata gaul) | 2 |
# | ambil field dari teks (LOT, tanggal, angka, email, nomor HP) | 3 |
# | pecah satu kolom jadi beberapa (`"Jaksel, DKI"` → kota, provinsi) | 4 |
# | hitung kata, kata terbanyak, buang stopword | 5 |
# | angka/tanggal gaya Indonesia di teks (`1,5 juta`, `3 Okt 2026`) | 6 |
# | cocokkan nama yang ditulis beda ke daftar baku (fuzzy) | 7 |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import pandas as pd

table = pd.DataFrame({"text": [
    "Promo!!! Cek https://toko.id/x @admin harga cuma 1,5 juta, hub 0812-3456-7890",
    "LOT: AB123  EXP: 03/2027  fab:01/2025",
    "gak bgt deh, pengiriman lama bgt 😡 email: budi.s@mail.com",
    "Rapat tgl 3 Okt 2026 jam 14.30 di Jakarta Selatan, DKI Jakarta",
]})
table["address"] = ["Bandung, Jawa Barat", "Jakarta Selatan, DKI Jakarta", "-", "Depok, Jabar"]
print(table)

# %% [markdown]
# ## 1. Contekan regex
#
# | pola | arti | contoh cocok |
# |---|---|---|
# | `\d` / `\D` | angka / bukan angka | `7` |
# | `\w` | huruf, angka, `_` | `a`, `Z`, `5` |
# | `\s` | spasi, tab, enter | |
# | `.` | karakter apa saja (kecuali enter) | |
# | `[A-Z0-9]` | salah satu dari himpunan | `Q`, `4` |
# | `[^0-9]` | **bukan** salah satu | |
# | `+` / `*` / `?` | 1 atau lebih / 0 atau lebih / 0 atau 1 | `\d+` → `2026` |
# | `{2}` / `{2,4}` | tepat 2 / 2 sampai 4 kali | `\d{4}` → `2027` |
# | `^` / `$` | awal / akhir teks | |
# | `( )` | kelompok yang diambil | `LOT:\s*(\w+)` → `AB123` |
# | `(?P<nama> )` | kelompok bernama → nama kolom di `str.extract` | |
# | `a\|b` | a atau b | `EXP\|PER` |
# | `\.` `\(` `\$` | simbol asli (bukan arti khusus) | |
#
# Selalu tulis pola sebagai `r"..."` (raw string) supaya `\` tidak dimakan Python.
# `re.IGNORECASE` / `flags=re.I`: huruf besar-kecil dianggap sama.

# %%
import re

text = "LOT: AB123  EXP: 03/2027  fab:01/2025"
match = re.search(r"LOT:\s*(\w+)", text)       # yang pertama cocok
print(match.group(1))                           # AB123
print(re.findall(r"\d{2}/\d{4}", text))         # semua yang cocok: ['03/2027', '01/2025']
print(re.findall(r"(EXP|FAB)\s*:\s*(\d{2}/\d{4})", text, flags=re.I))   # kelompok → tuple
print(re.sub(r"\s+", " ", text))                # ganti: spasi berlebih → satu spasi
print(re.split(r"\s{2,}", text))                # pecah di 2+ spasi

# %% [markdown]
# ## 2. Bersihkan teks → kolom `clean`
# Urutan: huruf kecil → buang URL/mention/hashtag → buang simbol → ganti kata gaul → rapikan spasi.
# Mau mempertahankan angka? Biarkan `0-9` di pola simbol.

# %%
import re

TEXT_COLUMN = "text"                                                       # UBAH
SLANG = {"gak": "tidak", "bgt": "banget", "hub": "hubungi", "tgl": "tanggal"}   # UBAH


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"[@#]\w+", " ", text)                  # @mention, #hashtag
    text = re.sub(r"[^a-z0-9\s]", " ", text)              # selain huruf, angka, spasi
    words = []
    for word in text.split():                             # split() juga merapikan spasi
        words.append(SLANG.get(word, word))
    return " ".join(words)


print(clean_text("Gak bgt!!! cek https://x.id @admin 😡"))   # tidak banget cek
table["clean"] = table[TEXT_COLUMN].apply(clean_text)
print(table["clean"].tolist())

# %% [markdown]
# ## 3. Ambil field dari teks → kolom baru
# `str.extract`: satu kolom per `(?P<nama>...)`, tidak cocok → NaN. Semua kemunculan:
# `str.findall` (hasil list per baris).

# %%
import re

TEXT_COLUMN = "text"   # UBAH
PATTERNS = {           # UBAH: nama_kolom → pola dengan satu kelompok ( )
    "lot": r"LOT\s*[:.]?\s*([A-Z0-9]+)",
    "exp": r"EXP\s*[:.]?\s*(\d{2}/\d{4})",
    "email": r"([\w.+-]+@[\w-]+\.[\w.]+)",
    "phone": r"((?:\+62|0)8[\d\- ]{8,14}\d)",
}

for name, pattern in PATTERNS.items():
    table[name] = table[TEXT_COLUMN].str.extract(pattern, flags=re.I, expand=False)
table["all_numbers"] = table[TEXT_COLUMN].str.findall(r"\d+")
print(table[["lot", "exp", "email", "phone"]])
digits_only = table["phone"].str.replace(r"\D", "", regex=True)   # buang selain angka
print(digits_only.tolist())

# %% [markdown]
# ## 4. Pecah satu kolom jadi beberapa
# `str.split(SEPARATOR, n=1, expand=True)`: maksimal 2 bagian. Hasil kosong → NaN.

# %%
SPLIT_COLUMN = "address"   # UBAH
SEPARATOR = ","            # UBAH

parts = table[SPLIT_COLUMN].str.split(SEPARATOR, n=1, expand=True)
table["city"] = parts[0].str.strip()
table["province"] = parts[1].str.strip()
print(table[["address", "city", "province"]])

# %% [markdown]
# ## 5. Hitung kata, kata terbanyak, buang stopword
# Stopword = kata yang terlalu umum (`yang`, `di`, `dan`). Daftar ditulis sendiri supaya jelas.

# %%
from collections import Counter

STOPWORDS = {"yang", "di", "dan", "ke", "dari", "ini", "itu", "untuk", "dengan",
             "jam", "cek"}   # UBAH


table["n_words"] = table["clean"].str.split().str.len()
counter = Counter()
for text in table["clean"]:
    for word in text.split():
        if word not in STOPWORDS and len(word) > 2:
            counter[word] += 1
print(table[["clean", "n_words"]])
print(counter.most_common(10))

# %% [markdown]
# ## 6. Angka dan tanggal gaya Indonesia di dalam teks
# `1,5 juta` → 1500000. Bulan Indonesia → angka, lalu `pd.to_datetime` dengan format eksplisit.

# %%
import re

import pandas as pd

MULTIPLIERS = {"ribu": 1_000, "rb": 1_000, "juta": 1_000_000, "jt": 1_000_000,
               "miliar": 1_000_000_000, "m": 1_000_000_000}                     # UBAH
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "jun": 6, "jul": 7, "agu": 8,
          "agt": 8, "sep": 9, "okt": 10, "nov": 11, "des": 12}                   # UBAH


def money_value(text):
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(ribu|rb|juta|jt|miliar|m)\b", str(text).lower())
    if match is None:
        return None
    number = float(match.group(1).replace(",", "."))
    return number * MULTIPLIERS[match.group(2)]


def indonesian_date(text):
    match = re.search(r"(\d{1,2})\s+([a-z]{3})[a-z]*\.?\s+(\d{4})", str(text).lower())
    if match is None or match.group(2) not in MONTHS:
        return pd.NaT
    day, month, year = int(match.group(1)), MONTHS[match.group(2)], int(match.group(3))
    return pd.Timestamp(year=year, month=month, day=day)


print(money_value("harga cuma 1,5 juta"), indonesian_date("tgl 3 Oktober 2026"))
table["money"] = table["text"].apply(money_value)
table["date"] = table["text"].apply(indonesian_date)
print(table[["money", "date"]])

# %% [markdown]
# ## 7. Cocokkan nama yang ditulis beda ke daftar baku (fuzzy matching)
# Skor 0–100 (`rapidfuzz`). Di bawah `MIN_SCORE` dianggap tidak cocok → dicetak untuk dicek mata.
# Tanpa rapidfuzz: `difflib.get_close_matches(nama, daftar, n=1, cutoff=0.6)` (bawaan Python).

# %%
from rapidfuzz import fuzz, process

MASTER = ["Jawa Barat", "DKI Jakarta", "Jawa Tengah", "Jawa Timur", "Banten"]   # UBAH
ALIASES = {"jabar": "Jawa Barat", "jateng": "Jawa Tengah", "jatim": "Jawa Timur"}  # UBAH
MIN_SCORE = 80                                                                     # UBAH


def match_master(name):
    name = str(name).strip()
    if name.lower() in ALIASES:            # singkatan tidak bisa ditebak dari kemiripan huruf
        return ALIASES[name.lower()], 100.0
    best, score, _ = process.extractOne(name, MASTER, scorer=fuzz.WRatio)
    if score < MIN_SCORE:
        return None, score
    return best, score


standard_names = []
scores = []
for name in table["province"]:
    standard_name, score = match_master(name)
    standard_names.append(standard_name)
    scores.append(score)
table["province_std"] = standard_names
table["province_score"] = scores
print(table[["province", "province_std", "province_score"]])
print("tidak cocok (cek mata):", table.loc[table["province_std"].isna(), "province"].tolist())

