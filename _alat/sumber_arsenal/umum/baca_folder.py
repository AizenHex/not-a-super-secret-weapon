# %% [markdown]
# # Baca banyak file sekaligus → satu tabel / satu array
#
# Hampir semua soal dimulai dari "folder berisi banyak file". Notebook ini mengubahnya menjadi
# **satu tabel pandas** (satu baris per file) supaya bisa dioper ke langkah berikutnya.
#
# ## Cara menempel ke notebook-mu
# Lihat dulu bentuk data soal (buka foldernya), lalu salin **hanya bagian yang cocok**. Tiap bagian
# berdiri sendiri: cukup sel itu + ganti baris `# UBAH`.
#
# | bentuk data soal | bagian | hasil |
# |---|---|---|
# | `train/kucing/*.jpg`, `train/anjing/*.jpg` (label = nama folder) | 1 | `table`: path, label |
# | `images/*.jpg` + `labels.csv` (nama file, label) | 2 | `table`: path, label, ... |
# | banyak CSV berkolom sama (per bulan, per toko) | 3 | satu `table` + kolom `source` |
# | banyak .txt (satu dokumen per file) | 4 | `table`: path, label, text |
# | .json / .jsonl | 5 | `table` |
# | Excel dengan banyak sheet | 6 | satu `table` + kolom `sheet` |
# | info tersimpan di nama file (`kamera01_2026-10-01.jpg`) | 7 | kolom baru dari nama file |
# | semua gambar → satu array numpy `X` (n, tinggi, lebar, 3) | 8 | `X`, `y` |
# | CSV sangat besar (RAM tidak cukup) | 9 | tabel hasil saring |
# | .zip dari soal | 10 | folder hasil ekstrak |
# | PDF (teks / tabel) | 11 | `table`: page, text |
#
# Sel **0. Data contoh** hanya membuat folder `contoh_data/` untuk mencoba. Saat lomba: lewati,
# ganti path di bagian yang dipakai ke folder soal.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

rng = np.random.default_rng(0)
for label in ["kucing", "anjing"]:
    os.makedirs(f"contoh_data/train/{label}", exist_ok=True)
    os.makedirs(f"contoh_data/teks/{label}", exist_ok=True)
    for number in range(3):
        pixels = rng.integers(0, 255, (40 + number * 10, 60, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(f"contoh_data/train/{label}/{label}_{number}.png")
        with open(f"contoh_data/teks/{label}/{number}.txt", "w", encoding="utf-8") as file:
            file.write(f"Ini dokumen {label} nomor {number}.\nBaris kedua.")
os.makedirs("contoh_data/images", exist_ok=True)
label_rows = []
for number in range(4):
    name = f"kamera0{number % 2 + 1}_2026-10-0{number + 1}_0830.png"
    Image.fromarray(rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)).save(
        f"contoh_data/images/{name}")
    label_rows.append([name, number % 2])
pd.DataFrame(label_rows, columns=["file_name", "label"]).to_csv("contoh_data/labels.csv",
                                                                index=False)
for month in [1, 2, 3]:
    sales = pd.DataFrame({"store": ["A", "B"], "amount": [10 * month, 20 * month]})
    sales.to_csv(f"contoh_data/sales_2026_{month:02d}.csv", index=False)
print("contoh_data siap:", sorted(os.listdir("contoh_data")))

# %%
import json

import pandas as pd

records = [{"id": 1, "user": {"name": "ani", "city": "Bogor"}, "tags": ["a", "b"]},
           {"id": 2, "user": {"name": "budi", "city": "Depok"}, "tags": ["c"]}]
with open("contoh_data/data.json", "w", encoding="utf-8") as file:
    json.dump(records, file)
with open("contoh_data/data.jsonl", "w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record) + "\n")
with pd.ExcelWriter("contoh_data/laporan.xlsx") as writer:
    pd.DataFrame({"item": ["x", "y"], "qty": [1, 2]}).to_excel(writer, sheet_name="Jan",
                                                                index=False)
    pd.DataFrame({"item": ["x", "z"], "qty": [3, 4]}).to_excel(writer, sheet_name="Feb",
                                                                index=False)
print("json, jsonl, xlsx siap")

# %% [markdown]
# ## 1. Gambar per folder kelas → tabel `path, label`
# **Butuh:** folder `TRAIN_FOLDER/<nama_kelas>/<file gambar>`. **Menghasilkan:** `table`.
# `width`, `height` dibaca dari header gambar (cepat, piksel tidak dimuat).

# %%
import glob
import os

import pandas as pd
from PIL import Image

TRAIN_FOLDER = "contoh_data/train"   # UBAH
EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]   # UBAH

rows = []
for path in sorted(glob.glob(f"{TRAIN_FOLDER}/*/*")):
    path = path.replace("\\", "/")
    if os.path.splitext(path)[1].lower() not in EXTENSIONS:
        continue
    label = os.path.basename(os.path.dirname(path))   # nama folder = label
    width, height = Image.open(path).size
    rows.append([path, label, width, height])
table = pd.DataFrame(rows, columns=["path", "label", "width", "height"])
print(table.head())
print(table["label"].value_counts())

# %% [markdown]
# ## 2. Gambar dalam satu folder + CSV label → tabel
# **Butuh:** `IMAGE_FOLDER`, CSV berisi nama file. **Menghasilkan:** `table` dengan kolom `path`.
# Nama kolom nama-file di CSV berbeda tiap soal (`file_name`, `id`, `image`): ubah `NAME_COLUMN`.
# Bila nama di CSV tanpa ekstensi, isi `EXTENSION`.

# %%
import os

import pandas as pd

IMAGE_FOLDER = "contoh_data/images"     # UBAH
LABEL_CSV = "contoh_data/labels.csv"    # UBAH
NAME_COLUMN = "file_name"               # UBAH
EXTENSION = ""                          # UBAH: mis. ".jpg" bila CSV hanya berisi id

table = pd.read_csv(LABEL_CSV)
paths = []
for name in table[NAME_COLUMN]:
    paths.append(f"{IMAGE_FOLDER}/{name}{EXTENSION}")
table["path"] = paths
missing = []
for path in table["path"]:
    if not os.path.exists(path):
        missing.append(path)
print(table.head())
print("file tidak ditemukan:", len(missing), missing[:3])

# %% [markdown]
# ## 3. Banyak CSV berkolom sama → satu tabel
# **Butuh:** pola nama file. **Menghasilkan:** `table` + kolom `source` (asal file).
# Kolom berbeda antar file? `pd.concat` tetap jalan; kolom yang tidak ada diisi NaN.

# %%
import glob
import os

import pandas as pd

CSV_PATTERN = "contoh_data/sales_*.csv"   # UBAH: * = bagian nama yang berubah

parts = []
for path in sorted(glob.glob(CSV_PATTERN)):
    part = pd.read_csv(path)
    part["source"] = os.path.basename(path)
    parts.append(part)
table = pd.concat(parts, ignore_index=True)
print("file:", len(parts), "| baris:", len(table))
print(table.head(6))

# %% [markdown]
# ## 4. Banyak file teks → tabel `path, label, text`
# **Butuh:** `TEXT_FOLDER/<label>/*.txt`. Tidak ada folder label? Ganti pola ke `*.txt` dan hapus
# baris `label`. File bukan UTF-8 (huruf aneh / error decode)? Ganti `ENCODING` ke `"latin-1"`.

# %%
import glob
import os

import pandas as pd

TEXT_FOLDER = "contoh_data/teks"   # UBAH
ENCODING = "utf-8"                 # UBAH

rows = []
for path in sorted(glob.glob(f"{TEXT_FOLDER}/*/*.txt")):
    path = path.replace("\\", "/")
    label = os.path.basename(os.path.dirname(path))
    with open(path, encoding=ENCODING) as file:
        text = file.read()
    rows.append([path, label, text])
table = pd.DataFrame(rows, columns=["path", "label", "text"])
table["n_chars"] = table["text"].str.len()
print(table.head())

# %% [markdown]
# ## 5. JSON / JSONL → tabel
# `.json` berisi list objek → `pd.json_normalize` meratakan isi bersarang (`user.name`).
# `.jsonl` = satu objek per baris → `pd.read_json(..., lines=True)`.
# Kolom berisi list (`tags`) → satu baris per isi list: `table.explode("tags")`.

# %%
import json

import pandas as pd

JSON_PATH = "contoh_data/data.json"     # UBAH
JSONL_PATH = "contoh_data/data.jsonl"   # UBAH

with open(JSON_PATH, encoding="utf-8") as file:
    records = json.load(file)
table = pd.json_normalize(records)            # kolom: id, tags, user.name, user.city
print(table)
lines_table = pd.read_json(JSONL_PATH, lines=True)
print(lines_table.columns.tolist())
print(table.explode("tags")[["id", "tags"]])

# %% [markdown]
# ## 6. Excel dengan banyak sheet → satu tabel
# `sheet_name=None` membaca semua sheet → dict {nama_sheet: tabel}.
# Header bukan di baris pertama? Tambah `header=2` (baris ke-3), atau `skiprows=...`.

# %%
import pandas as pd

EXCEL_PATH = "contoh_data/laporan.xlsx"   # UBAH

sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)
parts = []
for sheet_name, part in sheets.items():
    part["sheet"] = sheet_name
    parts.append(part)
table = pd.concat(parts, ignore_index=True)
print("sheet:", list(sheets.keys()))
print(table)

# %% [markdown]
# ## 7. Info dari nama file → kolom (regex)
# **Butuh:** `table` dengan kolom `path`. Pola contoh: `kamera01_2026-10-01_0830.png`.
# Tiap `(?P<nama>...)` di pola menjadi satu kolom. Cara menulis pola: lihat `teks_regex.ipynb`.

# %%
import glob
import os

import pandas as pd

IMAGE_FOLDER = "contoh_data/images"   # UBAH
NAME_PATTERN = r"(?P<camera>kamera\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hhmm>\d{4})"   # UBAH

paths = []
for path in sorted(glob.glob(f"{IMAGE_FOLDER}/*")):
    paths.append(path.replace("\\", "/"))
table = pd.DataFrame({"path": paths})
names = table["path"].apply(os.path.basename)

info = names.str.extract(NAME_PATTERN)     # satu kolom per (?P<nama>...)
table = pd.concat([table, info], axis=1)
table["date"] = pd.to_datetime(table["date"])
print(table)
print("nama yang tidak cocok pola:", table["camera"].isna().sum())

# %% [markdown]
# ## 8. Semua gambar → satu array numpy `X` (untuk model sederhana / sklearn)
# **Butuh:** `table` dengan kolom `path`, `label` (bagian 1). **Menghasilkan:** `X` bentuk
# (jumlah, `SIZE`, `SIZE`, 3) bertipe uint8, `y` (label). Ukuran gambar disamakan dulu.
# RAM: 10.000 gambar × 128 × 128 × 3 ≈ 0,5 GB. Lebih besar → pakai Dataset PyTorch (vision).

# %%
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

TRAIN_FOLDER = "contoh_data/train"   # UBAH
SIZE = 64                            # UBAH

table = pd.DataFrame({"path": sorted(glob.glob(f"{TRAIN_FOLDER}/*/*"))})
table["label"] = table["path"].apply(os.path.dirname).apply(os.path.basename)
images = []
for path in table["path"]:
    image = Image.open(path).convert("RGB")
    images.append(np.array(image.resize((SIZE, SIZE))))
X = np.stack(images)
y = table["label"].to_numpy()
print("X:", X.shape, X.dtype, "| y:", y[:4])
X_flat = X.reshape(len(X), -1) / 255     # (jumlah, SIZE*SIZE*3) untuk sklearn
print("X_flat:", X_flat.shape)

# %% [markdown]
# ## 9. CSV sangat besar: baca per potongan, ambil kolom/baris yang perlu saja
# `usecols` = kolom yang dibaca. `chunksize` = baris per potongan; tiap potongan disaring lalu
# dikumpulkan. `nrows=1000` untuk mengintip dulu tanpa membaca semua.

# %%
import pandas as pd

BIG_CSV = "contoh_data/sales_2026_01.csv"   # UBAH
COLUMNS = ["store", "amount"]               # UBAH

preview = pd.read_csv(BIG_CSV, nrows=1000)
print(preview.dtypes)
parts = []
for chunk in pd.read_csv(BIG_CSV, usecols=COLUMNS, chunksize=100_000):
    parts.append(chunk[chunk["amount"] > 5])    # UBAH: saringan
table = pd.concat(parts, ignore_index=True)
print("baris tersisa:", len(table))

# %% [markdown]
# ## 10. Ekstrak .zip
# Tujuan sebaiknya folder baru. Path di Windows panjang → error: pindahkan zip ke folder pendek.

# %%
import os
import zipfile

ZIP_PATH = "contoh_data/arsip.zip"     # UBAH
EXTRACT_FOLDER = "contoh_data/unzip"   # UBAH

with zipfile.ZipFile(ZIP_PATH, "w") as archive:   # hanya membuat contoh, hapus saat lomba
    archive.write("contoh_data/labels.csv")
with zipfile.ZipFile(ZIP_PATH) as archive:
    print(archive.namelist()[:10])
    archive.extractall(EXTRACT_FOLDER)
print(os.listdir(EXTRACT_FOLDER))

# %% [markdown]
# ## 11. PDF → teks per halaman (PDF digital, bukan hasil scan)
# PDF hasil scan = gambar: teksnya kosong → perlu OCR (`vision`). Tabel di PDF: `pdfplumber`,
# `page.extract_table()`.

# %%
import fitz   # pymupdf
import pandas as pd

PDF_PATH = "contoh_data/dokumen.pdf"   # UBAH

document = fitz.open()                 # hanya membuat contoh, hapus 4 baris ini saat lomba
page = document.new_page()
page.insert_text((72, 72), "Halaman satu: LOT AB12 EXP 03/2027")
document.save(PDF_PATH)

rows = []
for number, page in enumerate(fitz.open(PDF_PATH)):
    rows.append([number + 1, page.get_text()])
table = pd.DataFrame(rows, columns=["page", "text"])
print(table)
