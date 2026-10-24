# 02 — Wadah: list, tuple, dict, set, deque, heap, sort

| C++ | Python | catatan |
|---|---|---|
| `vector<T>` | `list` → `[1, 2, 3]` | isi boleh campur tipe |
| `array<T, N>` / `T a[N]` | `list`, atau `np.array` untuk angka (04) | |
| `pair<A,B>`, `tuple<...>` | `tuple` → `(a, b)` | tidak bisa diubah |
| `unordered_map<K,V>` | `dict` → `{"a": 1}` | urutan = **urutan dimasukkan** |
| `map<K,V>` (terurut) | `dict` + `sorted(d)` saat perlu urut | dict **tidak** otomatis urut per kunci |
| `unordered_set<T>` / `set<T>` | `set` → `{1, 2}` | `set()` untuk set kosong (`{}` = dict) |
| `deque<T>` | `collections.deque` | |
| `priority_queue<T>` | `heapq` pada list | **min**-heap (C++: max-heap) |
| `stack<T>` | `list`: `append` + `pop()` | |
| `queue<T>` | `deque`: `append` + `popleft()` | |
| `vector<vector<int>>` | list of list, atau `np.zeros((n, m))` | |

## list ≈ vector

| C++ | Python |
|---|---|
| `vector<int> v;` | `v = []` |
| `vector<int> v(n, 0);` | `v = [0] * n` |
| `v.push_back(x)` | `v.append(x)` |
| `v.pop_back()` | `v.pop()` (sekaligus mengembalikan nilainya) |
| `v.size()` | `len(v)` |
| `v.empty()` | `len(v) == 0` (atau `not v`) |
| `v.back()` | `v[-1]` |
| `v.insert(v.begin() + i, x)` | `v.insert(i, x)` |
| `v.erase(v.begin() + i)` | `del v[i]` |
| `v.clear()` | `v.clear()` |
| `v.insert(v.end(), w.begin(), w.end())` | `v.extend(w)` atau `v = v + w` |
| `find(v.begin(), v.end(), x) != v.end()` | `x in v` |
| posisi `x` | `v.index(x)` (error kalau tidak ada) |
| `count(...)` | `v.count(x)` |
| `*min_element`, `*max_element`, `accumulate` | `min(v)`, `max(v)`, `sum(v)` |
| `vector<int>(v.begin() + 1, v.begin() + 4)` | `v[1:4]` |
| `reverse(v.begin(), v.end())` | `v.reverse()` (di tempat) atau `v[::-1]` (salinan) |
| salin vector `auto w = v;` | `w = v.copy()` — **`w = v` bukan salinan!** (03) |

```python
v = [5, 3, 8]
v.append(1)
print(v, len(v), v[-1])     # [5, 3, 8, 1] 4 1
print(v[1:3], v[:2], v[2:]) # [3, 8] [5, 3] [8, 1]
grid = []
for row in range(3):
    grid.append([0] * 4)    # 3 baris × 4 kolom, tiap baris list sendiri
grid[1][2] = 7
print(grid)
```

> Jangan membuat 2D dengan `[[0] * 4] * 3`: ketiga baris adalah list **yang sama** (03).

## tuple ≈ pair / tuple

```python
point = (3, 4)
x, y = point          # seperti structured binding: auto [x, y] = point;
print(x, y)
a, b = 1, 2
a, b = b, a           # swap
print(a, b)           # 2 1
```

## dict ≈ unordered_map

| C++ | Python |
|---|---|
| `m[k] = v` | `m[k] = v` |
| `m[k]` saat k belum ada → dibuat 0 | `m[k]` → **KeyError**. Pakai `m.get(k, 0)` |
| `m.count(k)` / `m.find(k) != m.end()` | `k in m` |
| `m.erase(k)` | `del m[k]` |
| `m.size()` | `len(m)` |
| `for (auto& [k, v] : m)` | `for k, v in m.items():` |
| hanya kunci / hanya nilai | `m.keys()`, `m.values()` |
| `map<K, vector<V>>` (kelompok) | `dict` berisi list (lihat contoh) |
| hitung frekuensi | `collections.Counter` |

```python
counts = {}
for word in ["a", "b", "a", "c", "a"]:
    counts[word] = counts.get(word, 0) + 1
print(counts)                       # {'a': 3, 'b': 1, 'c': 1}
for key in sorted(counts):          # urut kunci, seperti std::map
    print(key, counts[key])

groups = {}                         # seperti map<string, vector<int>>
for name, score in [("x", 1), ("y", 2), ("x", 3)]:
    if name not in groups:
        groups[name] = []
    groups[name].append(score)
print(groups)                       # {'x': [1, 3], 'y': [2]}

from collections import Counter
print(Counter("mississippi").most_common(2))   # [('i', 4), ('s', 4)]
```

## set

```python
seen = set()
seen.add(3)
seen.add(3)
print(len(seen), 3 in seen)         # 1 True
a = {1, 2, 3}
b = {2, 3, 4}
print(a & b, a | b, a - b)          # {2, 3} {1, 2, 3, 4} {1}
unique = sorted(set([3, 1, 3, 2]))  # seperti sort + unique
print(unique)                       # [1, 2, 3]
```

## deque, stack, queue, priority_queue

```python
from collections import deque
import heapq

queue = deque()
queue.append(1)
queue.append(2)
print(queue.popleft())              # 1  (FIFO, seperti queue.front + pop)

stack = []
stack.append(1)
stack.append(2)
print(stack.pop())                  # 2  (LIFO)

heap = []
heapq.heappush(heap, (5, "lima"))   # (prioritas, isi)
heapq.heappush(heap, (1, "satu"))
print(heapq.heappop(heap))          # (1, 'satu')  ← terkecil dulu
heapq.heappush(heap, (-9, "sembilan"))   # max-heap: simpan negatif
print(heapq.heappop(heap))          # (-9, 'sembilan')
```

## sort

| C++ | Python |
|---|---|
| `sort(v.begin(), v.end())` | `v.sort()` (di tempat) atau `w = sorted(v)` (salinan) |
| `sort(..., greater<int>())` | `v.sort(reverse=True)` |
| comparator `[](A a, A b){ return a.score < b.score; }` | `key=`: fungsi yang mengembalikan **nilai pembanding** |
| urut dua kunci | `key` mengembalikan tuple |

```python
rows = [("budi", 75), ("ani", 92), ("cici", 75)]


def by_score_then_name(row):
    return (-row[1], row[0])   # skor turun, lalu nama naik


rows.sort(key=by_score_then_name)
print(rows)     # [('ani', 92), ('budi', 75), ('cici', 75)]
```
