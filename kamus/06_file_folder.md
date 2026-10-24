# 06 — File dan folder

Path ditulis dengan garis miring `/` (jalan di Windows juga): `"C:/GEMASTIK/soal1/train.csv"`.
Path **relatif** (`"data/train.csv"`) dihitung dari folder notebook.

| C++ | Python |
|---|---|
| `ifstream f("a.txt"); getline(f, line)` | `with open("a.txt", encoding="utf-8") as file:` ↵ `for line in file:` |
| baca seluruh isi | `text = open(path, encoding="utf-8").read()` |
| `ofstream f("a.txt"); f << s;` | `with open("a.txt", "w", encoding="utf-8") as file:` ↵ `file.write(s)` |
| tambah di akhir (`ios::app`) | `open(path, "a", ...)` |
| `filesystem::exists(p)` | `os.path.exists(p)` |
| `filesystem::create_directories(p)` | `os.makedirs(p, exist_ok=True)` |
| `directory_iterator(p)` | `os.listdir(p)` (nama saja) atau `glob.glob(f"{p}/*")` (path lengkap) |
| `recursive_directory_iterator` | `glob.glob(f"{p}/**/*.jpg", recursive=True)` |
| `p.filename()` | `os.path.basename(p)` → `"img_01.jpg"` |
| `p.stem()` | `os.path.splitext(os.path.basename(p))[0]` → `"img_01"` |
| `p.extension()` | `os.path.splitext(p)[1]` → `".jpg"` |
| `p.parent_path()` | `os.path.dirname(p)` |
| `copy_file`, `remove` | `shutil.copy(a, b)`, `os.remove(p)` |

`with open(...) as file:` = file otomatis ditutup di akhir blok (seperti destruktor `ifstream`).

```python
import glob
import os

os.makedirs("contoh_folder/sub", exist_ok=True)
for number in range(3):
    with open(f"contoh_folder/sub/catatan_{number}.txt", "w", encoding="utf-8") as file:
        file.write(f"baris pertama {number}\nbaris kedua\n")

paths = sorted(glob.glob("contoh_folder/**/*.txt", recursive=True))
for path in paths:
    name = os.path.splitext(os.path.basename(path))[0]
    lines = open(path, encoding="utf-8").read().splitlines()
    print(name, len(lines), lines[0])
```

## Satu folder → tabel (pola paling sering)

Gambar per folder kelas (`train/kucing/*.jpg`, `train/anjing/*.jpg`) → tabel `path, label`:

```python
import glob
import os

import pandas as pd

for label in ["kucing", "anjing"]:
    os.makedirs(f"contoh_kelas/{label}", exist_ok=True)
    for number in range(2):
        open(f"contoh_kelas/{label}/{number}.jpg", "w").close()   # file kosong untuk contoh

rows = []
for path in sorted(glob.glob("contoh_kelas/*/*.jpg")):
    label = os.path.basename(os.path.dirname(path))   # nama folder = label
    rows.append([path, label])
table = pd.DataFrame(rows, columns=["path", "label"])
print(table)
```

> Di Windows, `glob` mengembalikan path campuran (`contoh_kelas\anjing\0.jpg`), sedangkan library
> lain bisa menulis `contoh_kelas/anjing/0.jpg` atau path absolut. Kalau path dipakai sebagai
> **kunci** untuk mencocokkan dua tabel, samakan dulu: `path.replace("\\", "/")`.

Banyak CSV dengan kolom sama → satu tabel:

```python
import glob

import pandas as pd

for month in [1, 2]:
    pd.DataFrame({"month": [month, month], "sales": [10, 20]}).to_csv(f"contoh_{month}.csv",
                                                                     index=False)
parts = []
for path in sorted(glob.glob("contoh_*.csv")):
    part = pd.read_csv(path)
    part["source"] = path          # catat asal file
    parts.append(part)
table = pd.concat(parts, ignore_index=True)
print(table)
```

## Format lain

| format | baca | tulis |
|---|---|---|
| CSV | `pd.read_csv(path)` (pemisah lain: `sep=";"`) | `table.to_csv(path, index=False)` |
| Excel | `pd.read_excel(path, sheet_name=0)` | `table.to_excel(path, index=False)` |
| JSON | `data = json.load(open(path, encoding="utf-8"))` | `json.dump(data, open(path, "w"))` |
| JSON per baris (`.jsonl`) | `pd.read_json(path, lines=True)` | `table.to_json(path, orient="records", lines=True)` |
| Parquet | `pd.read_parquet(path)` | `table.to_parquet(path)` |
| numpy array | `np.load(path)` | `np.save(path, array)` (`.npy`) |
| objek Python apa saja | `pickle.load(open(path, "rb"))` | `pickle.dump(obj, open(path, "wb"))` |
| gambar | `Image.open(path)` | `Image.fromarray(array).save(path)` |

```python
import json

config = {"lr": 0.001, "classes": ["a", "b"]}
with open("contoh_config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
loaded = json.load(open("contoh_config.json", encoding="utf-8"))
print(loaded["classes"][1])     # b
```
