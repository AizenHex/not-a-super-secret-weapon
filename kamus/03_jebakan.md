# 03 — Jebakan: yang berbeda dari C++ dan sering bikin bug

## 1. `b = a` TIDAK menyalin list / dict / array / DataFrame

Di C++, `vector<int> b = a;` membuat salinan. Di Python, `b = a` hanya memberi **nama kedua** ke
benda yang sama (seperti `vector<int>& b = a;`).

```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)          # [1, 2, 3, 4]  ← a ikut berubah
c = a.copy()      # salinan sungguhan
c.append(5)
print(a)          # [1, 2, 3, 4]
```

| benda | salin dengan |
|---|---|
| list / dict / set | `x.copy()` |
| list of list (2D) | `import copy` lalu `copy.deepcopy(x)` |
| numpy array | `x.copy()` |
| DataFrame | `df.copy()` |

Angka, string, tuple **tidak bisa diubah**, jadi `b = a` lalu `b += 1` aman (b jadi benda baru).

## 2. Parameter fungsi: list/dict/array bisa diubah dari dalam fungsi

```python
def add_item(items):
    items.append(99)       # mengubah list milik pemanggil


numbers = [1]
add_item(numbers)
print(numbers)             # [1, 99]
```

## 3. List 2D dengan `*`

```python
wrong = [[0] * 3] * 2
wrong[0][0] = 1
print(wrong)               # [[1, 0, 0], [1, 0, 0]]  ← dua baris = satu list yang sama
right = []
for _ in range(2):
    right.append([0] * 3)
right[0][0] = 1
print(right)               # [[1, 0, 0], [0, 0, 0]]
```

## 4. `range` dan slice: ujung kanan TIDAK ikut

`range(1, 5)` = 1, 2, 3, 4. `s[1:5]` = indeks 1..4. Sama seperti `for (i = 1; i < 5; i++)`.

## 5. Indeks negatif tidak error

`v[-1]` = elemen terakhir. Bug "indeks -1" di C++ jadi diam-diam membaca elemen terakhir.

## 6. Pembagian

`7 / 2 = 3.5`, `7 // 2 = 3`, `-7 // 2 = -4` (dibulatkan ke bawah, C++: -3).

## 7. String tidak bisa diubah per huruf

```python
word = "kata"
# word[0] = "K"   ← TypeError
word = "K" + word[1:]
print(word)                # Kata
letters = list(word)       # kalau perlu ubah banyak huruf: jadikan list, ubah, gabung lagi
letters[1] = "A"
print("".join(letters))    # KAta
```

## 8. Indentasi = blok

Tab dan spasi jangan dicampur. Baris yang mundur keluar dari blok `if`/`for` tanpa peringatan.

## 9. Variabel loop tetap ada setelah loop

```python
for i in range(3):
    pass
print(i)                   # 2  (C++: i sudah tidak ada)
```

## 10. Nilai "benar/salah" (truthiness)

`0`, `0.0`, `""`, `[]`, `{}`, `None` dianggap False. `if items:` = "list tidak kosong".
**Tapi** numpy array dan DataFrame tidak boleh dipakai langsung di `if` (lihat 08).

## 11. Parameter default yang bisa diubah

```python
def collect(value, bucket=[]):     # SALAH: list default dibuat SEKALI saja
    bucket.append(value)
    return bucket


print(collect(1), collect(2))      # [1, 2] [1, 2]


def collect_ok(value, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(value)
    return bucket


print(collect_ok(1), collect_ok(2))   # [1] [2]
```

## 12. Jupyter: urutan menjalankan sel = urutan program

Variabel bertahan di memori (kernel) sampai kernel restart. Sel yang dijalankan **di atas** tapi
memakai variabel dari sel **di bawah** tetap jalan kalau sel bawah pernah dijalankan; setelah
restart → `NameError`. Biasakan: Kernel → Restart & Run All sebelum yakin notebook benar.

## 13. float tidak pernah tepat

`0.1 + 0.2 == 0.3` → False (sama seperti C++). Bandingkan dengan `abs(a - b) < 1e-9`.

## 14. DataFrame: ubah kolom di potongan tabel

```python
import pandas as pd

table = pd.DataFrame({"x": [1, 2, 3]})
part = table[table["x"] > 1].copy()   # .copy() supaya jelas ini tabel baru
part["y"] = part["x"] * 10            # tanpa .copy(): SettingWithCopyWarning
print(part)
table.loc[table["x"] > 1, "x"] = 0    # mengubah tabel asli: pakai .loc[baris, kolom]
print(table)
```
