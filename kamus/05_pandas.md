# 05 — pandas: tabel

`pd.DataFrame` ≈ `vector<struct Row>` yang disimpan **per kolom**. Tiap kolom = `pd.Series`
(≈ `vector<T>` + label baris). Operasi kolom berlaku untuk semua baris sekaligus (tanpa loop).

```cpp
struct Row { string name; int age; double score; };
vector<Row> table = {{"ani", 20, 81.5}, {"budi", 22, 70.0}};
```
```python
import pandas as pd

table = pd.DataFrame({"name": ["ani", "budi"], "age": [20, 22], "score": [81.5, 70.0]})
```

## Membuat dan melihat

| C++ (vector<Row>) | pandas |
|---|---|
| baca CSV manual | `table = pd.read_csv("data.csv")` |
| tulis CSV manual | `table.to_csv("hasil.csv", index=False)` |
| bangun baris per baris: `push_back(Row{...})` | kumpulkan `rows.append([..])` di list, lalu `pd.DataFrame(rows, columns=[...])` (**jangan** tambah baris satu-satu ke DataFrame: lambat) |
| `table.size()` | `len(table)`; ukuran `table.shape` → (baris, kolom) |
| cetak beberapa baris | `table.head()`, `table.tail(3)`, `print(table.to_string())` |
| daftar field | `table.columns` |
| tipe tiap field | `table.dtypes` |
| ringkasan angka | `table.describe()` |
| ada nilai kosong? | `table.isna().sum()` |

## Kolom (field)

| C++ | pandas |
|---|---|
| semua `row.age` | `table["age"]` (Series) |
| beberapa field | `table[["name", "age"]]` (DataFrame) |
| loop `row.score2 = row.score * 2` | `table["score2"] = table["score"] * 2` |
| loop pakai fungsi sendiri | `table["grade"] = table["score"].apply(to_grade)` |
| hapus field | `table = table.drop(columns=["score2"])` |
| ganti nama | `table = table.rename(columns={"age": "umur"})` |
| hitung kemunculan tiap nilai | `table["label"].value_counts()` |
| nilai unik | `table["label"].unique()`, jumlahnya `table["label"].nunique()` |
| ganti nilai | `table["label"].replace({"a": "A"})`, atau `.map(dict)` |
| isi kosong | `table["age"].fillna(0)` |
| ubah tipe | `table["age"].astype(int)` |
| kolom → list / numpy | `table["age"].tolist()` / `table["age"].to_numpy()` |

## Baris (filter, akses, urut)

| C++ | pandas |
|---|---|
| `copy_if(... row.age > 20)` | `table[table["age"] > 20]` |
| dua syarat `&&` / `\|\|` | `table[(table["age"] > 20) & (table["score"] < 80)]` — pakai `&`, `\|`, `~`, dan **kurung** |
| nilai ada dalam daftar | `table[table["name"].isin(["ani", "cici"])]` |
| `table[i]` (baris ke-i) | `table.iloc[i]`; satu sel `table.iloc[i]["name"]` |
| ubah sel yang memenuhi syarat | `table.loc[table["age"] > 20, "score"] = 0` |
| sort per field | `table.sort_values("score", ascending=False)` |
| sort dua field | `table.sort_values(["label", "score"])` |
| buang duplikat | `table.drop_duplicates(["name"])` (simpan yang pertama) |
| buang baris kosong | `table.dropna()` |
| ambil acak | `table.sample(5, random_state=0)` |
| setelah filter, nomor baris jadi bolong | `table.reset_index(drop=True)` |

## Loop baris (kalau memang perlu)

```python
import pandas as pd

table = pd.DataFrame({"name": ["ani", "budi"], "age": [20, 22], "score": [81.5, 70.0]})
for row in table.itertuples():          # cepat; field diakses seperti struct
    print(row.name, row.age, row.score)
```

Mengubah DataFrame **di dalam** loop baris itu lambat dan rawan salah. Pola yang dipakai di arsenal:
kumpulkan hasil di list, lalu jadikan kolom sekali di akhir.

```python
import pandas as pd

table = pd.DataFrame({"name": ["ani", "budi"], "score": [81.5, 70.0]})
grades = []
for row in table.itertuples():
    if row.score >= 80:
        grades.append("A")
    else:
        grades.append("B")
table["grade"] = grades
print(table)
```

## groupby ≈ map<kunci, vector<Row>> lalu hitung per kelompok

```python
import pandas as pd

sales = pd.DataFrame({"city": ["A", "B", "A", "B", "A"], "amount": [10, 20, 30, 40, 50]})
print(sales.groupby("city")["amount"].sum())       # total per kota
print(sales.groupby("city")["amount"].agg(["count", "mean", "max"]))
summary = sales.groupby("city", as_index=False)["amount"].mean()   # hasil tetap tabel biasa
print(summary)
```

| ingin | kode |
|---|---|
| beberapa statistik | `.agg(["mean", "std", "min", "max"])` |
| tiap baris diberi nilai kelompoknya | `sales["city_mean"] = sales.groupby("city")["amount"].transform("mean")` |
| loop per kelompok | `for city, part in sales.groupby("city"):` (`part` = DataFrame) |

## Menggabungkan tabel

| C++ | pandas |
|---|---|
| sambung vector `v.insert(v.end(), ...)` | `pd.concat([table_a, table_b], ignore_index=True)` |
| join dua tabel lewat kunci (lookup `map`) | `table_a.merge(table_b, on="id", how="left")` |
| lookup satu kolom dari `map<id, nilai>` | `table["city"] = table["id"].map(id_to_city)` (dict) |

`how="left"`: semua baris tabel kiri tetap ada; yang tidak punya pasangan diisi NaN.
Cek jumlah baris sebelum/sesudah merge: kalau kunci kanan dobel, baris kiri ikut berlipat.

## Tanggal dan teks per kolom

```python
import pandas as pd

log = pd.DataFrame({"time": ["2026-10-01 08:30", "2026-10-02 17:45"],
                    "text": ["LOT: A1", "lot: b2"]})
log["time"] = pd.to_datetime(log["time"])
log["hour"] = log["time"].dt.hour
log["weekday"] = log["time"].dt.dayofweek          # 0 = Senin
log["upper"] = log["text"].str.upper()
log["code"] = log["upper"].str.extract(r"LOT:\s*(\w+)", expand=False)
print(log)
```

## Tabel → numpy untuk model

```python
import pandas as pd

table = pd.DataFrame({"age": [20, 22, 25], "score": [81.5, 70.0, 90.0], "label": [1, 0, 1]})
X = table[["age", "score"]].to_numpy()   # matriks fitur (baris × kolom)
y = table["label"].to_numpy()
print(X.shape, y.shape)                  # (3, 2) (3,)
```
