# 01 — Dasar: tipe, operator, alur, fungsi, string

Python tidak punya `{}` blok dan `;`. **Blok = indentasi** (4 spasi) setelah `:`.
Tipe tidak ditulis: variabel mengikuti nilai yang dimasukkan.

## Tipe

| C++ | Python | catatan |
|---|---|---|
| `int`, `long long` | `int` | tidak pernah overflow (besar tak terbatas) |
| `double` / `float` | `float` | selalu presisi double |
| `bool` (`true`/`false`) | `bool` (`True`/`False`) | huruf besar di depan |
| `char` | — | tidak ada; satu huruf = string panjang 1 |
| `std::string` | `str` | **tidak bisa diubah** per huruf (lihat 03) |
| `nullptr` / `NULL` | `None` | cek dengan `x is None` |
| `auto x = 5;` | `x = 5` | |
| `const int N = 10;` | `N = 10` | konvensi: nama HURUF BESAR = jangan diubah |
| `(int)x`, `(double)x` | `int(x)`, `float(x)` | `int(3.9)` = 3 (dipotong) |
| `std::to_string(x)` | `str(x)` | |
| `std::stoi(s)`, `std::stod(s)` | `int(s)`, `float(s)` | |
| `sizeof`, tipe eksplisit | `type(x)` | untuk cek saja |

## Operator

| C++ | Python | catatan |
|---|---|---|
| `a / b` (int) | `a // b` | `/` di Python **selalu** hasil float: `7 / 2 = 3.5` |
| `a % b` | `a % b` | hasil ikut tanda pembagi: `-7 % 3 = 2` (C++: -1) |
| `pow(a, b)` | `a ** b` | |
| `&&`, `\|\|`, `!` | `and`, `or`, `not` | |
| `i++`, `++i` | `i += 1` | `++` tidak ada |
| `a == b` | `a == b` | `is` hanya untuk `None` |
| `0 < x && x < 10` | `0 < x < 10` | perbandingan boleh dirantai |
| `cond ? a : b` | `a if cond else b` | |
| `&`, `\|`, `^`, `<<`, `>>`, `~` | sama | |

```python
print(7 / 2, 7 // 2, 7 % 2, 2 ** 10)   # 3.5 3 1 1024
x = 5
print(0 < x < 10)                       # True
label = "ganjil" if x % 2 else "genap"
print(label)                            # ganjil
```

## if / for / while

| C++ | Python |
|---|---|
| `if (a) {...} else if (b) {...} else {...}` | `if a:` / `elif b:` / `else:` |
| `for (int i = 0; i < n; i++)` | `for i in range(n):` |
| `for (int i = 2; i < n; i += 3)` | `for i in range(2, n, 3):` |
| `for (int i = n - 1; i >= 0; i--)` | `for i in range(n - 1, -1, -1):` |
| `for (auto x : v)` | `for x in v:` |
| `for (int i = 0; i < v.size(); i++)` pakai `i` dan `v[i]` | `for i, x in enumerate(v):` |
| loop dua vector sekaligus | `for a, b in zip(va, vb):` |
| `while (cond)` | `while cond:` |
| `break`, `continue` | sama |
| `{}` kosong | `pass` |
| `switch` | `if/elif` (atau `match` di Python ≥ 3.10) |

```python
names = ["ani", "budi", "cici"]
scores = [80, 75, 92]
for i, name in enumerate(names):
    print(i, name)
for name, score in zip(names, scores):
    if score >= 90:
        print(name, "A")
    elif score >= 80:
        print(name, "B")
    else:
        print(name, "C")
```

## Fungsi

| C++ | Python |
|---|---|
| `int add(int a, int b) { return a + b; }` | `def add(a, b):` ↵ `    return a + b` |
| parameter default `int f(int a, int b = 2)` | `def f(a, b=2):` |
| `void f()` | fungsi tanpa `return` → mengembalikan `None` |
| `pair<int,int> f()` / return banyak nilai | `return a, b` lalu `x, y = f()` |
| pass by reference `int& x` | tidak ada untuk int/str; list/dict/array **selalu** "by reference" (lihat 03) |
| memanggil dengan nama parameter | `f(b=3, a=1)` |
| `[](int x) { return x * 2; }` | `lambda x: x * 2` (hanya satu ekspresi) |

```python
def min_max(values):
    smallest = values[0]
    largest = values[0]
    for value in values:
        if value < smallest:
            smallest = value
        if value > largest:
            largest = value
    return smallest, largest


low, high = min_max([4, 9, 1, 7])
print(low, high)   # 1 9
```

## print dan input

| C++ | Python |
|---|---|
| `cout << a << " " << b << endl;` | `print(a, b)` (dipisah spasi otomatis) |
| `cout << a << b;` (tanpa spasi, tanpa enter) | `print(a, b, sep="", end="")` |
| `printf("%.3f\n", x);` | `print(f"{x:.3f}")` |
| `printf("%5d", x);` | `print(f"{x:5d}")` |
| `cin >> n;` | `n = int(input())` (jarang di notebook) |

```python
loss = 0.123456
epoch = 3
print(f"epoch {epoch} | loss {loss:.4f} | akurasi {0.9123:.1%}")
# epoch 3 | loss 0.1235 | akurasi 91.2%
```

## String

| C++ | Python | hasil |
|---|---|---|
| `s.size()` | `len(s)` | |
| `s[0]`, `s.back()` | `s[0]`, `s[-1]` | indeks negatif = dari belakang |
| `s.substr(2, 3)` | `s[2:5]` | "slice": dari 2 sampai **sebelum** 5 |
| `s + t` | `s + t` | |
| `s.find("ab")` | `s.find("ab")` | -1 kalau tidak ada (C++: `npos`) |
| `s.find("ab") != npos` | `"ab" in s` | |
| `toupper` per huruf | `s.upper()`, `s.lower()` | |
| trim spasi | `s.strip()` | |
| split manual dengan `stringstream` | `s.split(",")` | list string |
| join manual | `", ".join(list_of_str)` | |
| ganti teks | `s.replace("a", "b")` | semua kemunculan |
| `isdigit` | `s.isdigit()` | |
| `std::reverse` | `s[::-1]` | |

```python
line = "  LOT: AB12 , EXP: 03/2027  "
parts = line.strip().split(",")
print(parts)                         # ['LOT: AB12 ', ' EXP: 03/2027']
fields = []
for part in parts:
    fields.append(part.strip())
print(" | ".join(fields))            # LOT: AB12 | EXP: 03/2027
print("EXP" in line, line.strip()[:3])   # True LOT
```

## Komentar

| C++ | Python |
|---|---|
| `// satu baris` | `# satu baris` |
| `/* banyak baris */` | `#` di tiap baris (atau string `"""..."""` sebagai dokumentasi) |
