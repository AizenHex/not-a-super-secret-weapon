# 04 — numpy: array angka

`np.ndarray` ≈ array C berukuran tetap, satu tipe, memori bersambung, plus operasi
"seluruh array sekaligus" (tanpa loop, jauh lebih cepat dari loop Python).
Gambar = array `tinggi × lebar × 3` (RGB) bertipe `uint8`.

| C++ | numpy |
|---|---|
| `double a[5] = {}` | `a = np.zeros(5)` |
| `int a[3][4]` | `a = np.zeros((3, 4), dtype=int)` |
| isi semua 7 | `np.full((3, 4), 7)` |
| `{1, 2, 3}` | `np.array([1, 2, 3])` |
| `for (i = 0; i < 10; i++) a[i] = i;` | `np.arange(10)` |
| `n` titik rata dari a ke b | `np.linspace(a, b, n)` |
| ukuran tiap dimensi | `a.shape` → tuple, mis. `(3, 4)` |
| jumlah dimensi | `a.ndim` |
| tipe elemen | `a.dtype` (`uint8`, `int64`, `float32`, `float64`, `bool`) |
| ubah tipe `(float)` | `a.astype(np.float32)` |
| `a[i][j]` | `a[i, j]` |
| satu baris / satu kolom | `a[i]` atau `a[i, :]` / `a[:, j]` |
| loop `c[i] = a[i] + b[i]` | `c = a + b` |
| loop `a[i] *= 2` | `a = a * 2` |
| loop jumlah | `a.sum()`, rata-rata `a.mean()`, `a.min()`, `a.max()`, `a.std()` |
| indeks nilai terbesar | `a.argmax()` |
| jumlah per baris / per kolom | `a.sum(axis=1)` / `a.sum(axis=0)` |
| `if (a[i] > 5) cnt++` | `(a > 5).sum()` |
| ambil yang memenuhi syarat | `a[a > 5]` |
| `a[i] = (a[i] > 5) ? 1 : 0` | `np.where(a > 5, 1, 0)` |
| `sort` | `np.sort(a)`; indeks urutan: `np.argsort(a)` |
| `reshape` manual | `a.reshape(2, 6)` (`-1` = hitung sendiri) |
| transpose | `a.T` |
| perkalian matriks | `a @ b` |
| gabung array | `np.concatenate([a, b])`, `np.stack([a, b])` |
| `rand()` | `rng = np.random.default_rng(42)` lalu `rng.random(5)`, `rng.integers(0, 10, 5)` |

## axis: arah yang "dihabiskan"

`a.sum(axis=0)`: jumlah **turun ke bawah** (hasil satu nilai per kolom).
`a.sum(axis=1)`: jumlah **ke samping** (hasil satu nilai per baris).

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(a.shape, a[1, 2], a[:, 0])    # (3, 4) 6 [0 4 8]
print(a.sum(axis=0))                # [12 15 18 21]  per kolom
print(a.sum(axis=1))                # [ 6 22 38]     per baris
print(a[a > 6])                     # [ 7  8  9 10 11]
print(np.where(a % 2 == 0, "genap", "ganjil")[0])
```

## Slice array = JENDELA, bukan salinan

```python
import numpy as np

image = np.zeros((4, 4), dtype=np.uint8)
window = image[1:3, 1:3]
window[:] = 255                     # mengubah image juga!
print(image)
safe = image[1:3, 1:3].copy()       # salinan sungguhan
```

## Gambar sebagai array

| hal | kode |
|---|---|
| buka gambar → array | `np.array(Image.open(path).convert("RGB"))` |
| array → simpan | `Image.fromarray(array).save(path)` (array `uint8`) |
| ukuran | `height, width = image.shape[:2]` |
| potong kotak | `image[y1:y2, x1:x2]` (baris dulu = y, lalu kolom = x) |
| abu-abu kasar | `image.mean(axis=2)` |
| balik kiri-kanan / atas-bawah | `image[:, ::-1]` / `image[::-1]` |
| putar 90° berlawanan jarum jam | `np.rot90(image)`; 180°: `np.rot90(image, 2)` |
| ke 0–1 untuk model | `image.astype(np.float32) / 255` |

```python
import numpy as np
from PIL import Image

image = np.zeros((60, 100, 3), dtype=np.uint8)   # hitam, tinggi 60, lebar 100
image[10:30, 20:50] = [255, 0, 0]                # kotak merah
crop = image[10:30, 20:50]
print(image.shape, crop.shape)                   # (60, 100, 3) (20, 30, 3)
Image.fromarray(image).save("contoh_kotak.png")
loaded = np.array(Image.open("contoh_kotak.png").convert("RGB"))
print((loaded == image).all())                   # True
```

## Overflow uint8

`uint8` hanya 0–255 dan **berputar** seperti `unsigned char` di C++: `np.uint8(250) + 10 = 4`.
Sebelum menjumlah/mengurangi gambar: `image.astype(np.int32)` atau `astype(np.float32)`.
