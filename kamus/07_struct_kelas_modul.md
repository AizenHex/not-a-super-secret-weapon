# 07 — struct, class, modul, library, Jupyter

## struct

Untuk data tabel, **pakai DataFrame** (05), bukan struct. Untuk satu "benda" berisi beberapa
field, pilihan paling sederhana:

| C++ | Python |
|---|---|
| `struct Box { int x1, y1, x2, y2; };` | dict: `box = {"x1": 0, "y1": 0, "x2": 5, "y2": 5}` lalu `box["x1"]` |
| struct dengan akses titik `box.x1` | `@dataclass` (contoh di bawah) |
| `pair` / `tuple` kecil | `tuple`: `box = (0, 0, 5, 5)` lalu `x1, y1, x2, y2 = box` |

```python
from dataclasses import dataclass


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "?"


box = Box(10, 20, 50, 60, "entete")
print(box.x2 - box.x1, box)        # 40 Box(x1=10, y1=20, x2=50, y2=60, label='entete')
```

## class

| C++ | Python |
|---|---|
| konstruktor `Box(int w) : w(w) {}` | `def __init__(self, w):` ↵ `self.w = w` |
| `this->w` | `self.w` (**wajib** ditulis `self.`) |
| method `int area() const` | `def area(self):` (parameter pertama selalu `self`) |
| `class Child : public Parent` | `class Child(Parent):` |
| panggil konstruktor induk | `super().__init__(...)` |
| `operator<<` / `toString` | `def __repr__(self): return "..."` |
| `private` | tidak ada; konvensi nama diawali `_` |
| method virtual | semua method otomatis bisa ditimpa |

```python
class Counter:
    def __init__(self, start):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value


counter = Counter(10)
counter.add(5)
print(counter.value)    # 15
```

Di arsenal, `class` hanya muncul untuk dua pola standar PyTorch:
`class Model(nn.Module)` (isi `__init__` + `forward`) dan `class MyDataset(Dataset)`
(isi `__len__` + `__getitem__`). Sisanya fungsi biasa.

## Modul dan library

| C++ | Python |
|---|---|
| `#include <cmath>` | `import math` lalu `math.sqrt(2)` |
| `using namespace std;` | `from math import sqrt` lalu `sqrt(2)` |
| alias | `import numpy as np`, `import pandas as pd` (nama pendek standar) |
| file header sendiri `#include "util.h"` | file `util.py` di folder yang sama → `import util` |
| package manager (vcpkg/apt) | `pip install nama` (atau `setup.bat` di repo ini) |
| cek versi | `print(np.__version__)` |
| dokumentasi offline | `help(np.where)`, atau di Jupyter: `np.where?` lalu jalankan sel |

Nama library yang sering dipakai:

| import | gunanya |
|---|---|
| `numpy as np` | array angka (04) |
| `pandas as pd` | tabel (05) |
| `matplotlib.pyplot as plt` | grafik / tampilkan gambar |
| `os`, `glob`, `shutil` | file dan folder (06) |
| `re` | regex |
| `json`, `pickle` | simpan/muat data |
| `PIL.Image`, `cv2` | gambar |
| `sklearn` | model ML klasik, split, metrik |
| `torch`, `torchvision` | deep learning |
| `tqdm.auto.tqdm` | progress bar: `for x in tqdm(items):` |

## Jupyter

| hal | penjelasan |
|---|---|
| kernel | satu proses Python yang hidup terus; variabel dari sel mana pun tetap ada |
| Shift+Enter | jalankan sel |
| nilai di baris terakhir sel | otomatis ditampilkan (tidak perlu `print`) |
| Restart | semua variabel hilang; jalankan ulang dari atas |
| Restart & Run All | cara memastikan notebook benar dari awal ke akhir |
| `!pip list` | menjalankan perintah terminal dari sel |
| `%time kode` | ukur waktu satu baris |
| kernel mati sendiri | biasanya kehabisan RAM/GPU: kecilkan batch, hapus variabel besar (`del x`) |

```python
import math
from math import sqrt

print(math.pi, sqrt(16))   # 3.141592653589793 4.0
```
