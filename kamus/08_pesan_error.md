# 08 — Membaca pesan error

## Cara membaca traceback

Baca dari **bawah**. Baris paling bawah = jenis error + pesannya. Di atasnya, cari baris terakhir
yang menunjuk **sel/kode milikmu** (bukan file di dalam `site-packages`): itu baris penyebabnya.

```
Traceback (most recent call last):
  Cell In[7], line 4                         ← sel 7, baris 4: lihat di sini
    crops["text"] = texts
  File ".../pandas/core/frame.py", line ...  ← dalam library, abaikan dulu
ValueError: Length of values (40) does not match length of index (667)   ← pesannya
```

## Error yang paling sering

| error | artinya | biasanya karena | perbaiki |
|---|---|---|---|
| `NameError: name 'x' is not defined` | variabel belum ada | sel yang membuat `x` belum dijalankan / kernel restart / salah ketik | jalankan sel pembuatnya (lihat baris **Butuh:** di arsenal) |
| `KeyError: 'label'` | kunci/kolom tidak ada | nama kolom beda (spasi, huruf besar) | `print(table.columns)` atau `print(d.keys())` |
| `IndexError: list index out of range` | indeks ≥ panjang | list kosong / loop kelebihan | `print(len(v))` |
| `TypeError: unsupported operand type(s) for +: 'int' and 'str'` | tipe tidak cocok | angka terbaca sebagai teks dari CSV | `int(x)`, `table["a"].astype(float)` |
| `TypeError: 'NoneType' object is not subscriptable` | memakai `None` seperti list | fungsi lupa `return` | cek `return` di fungsi |
| `AttributeError: 'list' object has no attribute 'shape'` | method tidak ada di tipe itu | list dikira numpy array | `np.array(v)` |
| `ValueError: ... could not convert string to float` | teks tidak bisa jadi angka | ada `"?"`, `","`, kosong di kolom | bersihkan dulu, `pd.to_numeric(col, errors="coerce")` |
| `ValueError: The truth value of an array/Series is ambiguous` | `if array:` / `and` pada array | syarat banyak nilai | pakai `.any()` / `.all()`, dan `&` `\|` untuk syarat kolom |
| `ValueError: shapes (3,4) and (5,) not aligned` / `operands could not be broadcast` | ukuran array tidak cocok | dimensi salah | `print(a.shape, b.shape)` |
| `ValueError: Length of values does not match length of index` | kolom baru beda panjang | list hasil loop tidak lengkap | `print(len(list), len(table))` |
| `FileNotFoundError` | file tidak ada | path relatif dari folder lain, `\` di path | `print(os.path.abspath(path))`, pakai `/` |
| `ModuleNotFoundError: No module named 'x'` | library belum terpasang / venv lain | kernel bukan venv proyek | pilih kernel `.venv`; `pip install x` |
| `IndentationError` | indentasi salah | campur tab dan spasi | rapikan 4 spasi |
| `SyntaxError` | tulisan tidak sah | lupa `:` setelah `if/for/def`, kurung tidak tertutup | lihat tanda `^` di pesan |
| `ZeroDivisionError` | bagi nol | | cek penyebut |
| `RecursionError` | rekursi terlalu dalam | batas default ±1000 | pakai loop |
| `torch.cuda.OutOfMemoryError: CUDA out of memory` | GPU penuh | batch/gambar terlalu besar | kecilkan `BATCH` / `IMGSZ`, restart kernel |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x512 and 256x10)` | ukuran layer tidak cocok | input layer salah | samakan angka tengah: 512 ≠ 256 |
| `RuntimeError: Expected all tensors to be on the same device` | sebagian di GPU, sebagian di CPU | lupa `.to(DEVICE)` | pindahkan model dan data ke `DEVICE` |

## Peringatan (warning) yang aman / tidak aman

| warning | arti |
|---|---|
| `SettingWithCopyWarning` | mengubah potongan tabel; tambah `.copy()` atau pakai `.loc` (03 no. 14) |
| `FutureWarning`, `DeprecationWarning` | fitur akan diganti versi depan; hasil sekarang tetap benar |
| `ConvergenceWarning` (sklearn) | model belum selesai belajar; naikkan `max_iter` |
| `UndefinedMetricWarning` | ada kelas yang tidak pernah ditebak; skor kelas itu 0 |

## Cek cepat saat bingung

```python
import numpy as np
import pandas as pd

table = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
array = np.zeros((2, 3))
print(type(table), type(array))     # jenis benda
print(table.shape, array.shape)     # ukuran
print(table.dtypes)                 # tipe tiap kolom
print(table.columns.tolist())       # nama kolom persis (spasi kelihatan)
```
