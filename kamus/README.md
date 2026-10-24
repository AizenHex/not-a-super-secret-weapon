# KAMUS — padanan C++ → Python

Untuk dibaca saat lupa "di Python ini namanya apa". Tiap file: tabel padanan dulu, lalu contoh
singkat. Semua contoh Python di sini sudah dijalankan (`python _alat/cek_kamus.py`).

| file | isi | buka kalau... |
|---|---|---|
| [01_dasar.md](01_dasar.md) | tipe, operator, if/for/while, fungsi, print, string | lupa sintaks dasar |
| [02_wadah.md](02_wadah.md) | list, tuple, dict, set, deque, heap, sort | butuh `vector`, `map`, `set`, `priority_queue` |
| [03_jebakan.md](03_jebakan.md) | hal yang **berbeda** dari C++ dan sering bikin bug | hasil aneh padahal kode "benar" |
| [04_numpy.md](04_numpy.md) | array angka, shape, axis, mask | olah gambar / matriks / angka banyak |
| [05_pandas.md](05_pandas.md) | DataFrame, kolom, filter, groupby, gabung tabel | olah tabel / CSV |
| [06_file_folder.md](06_file_folder.md) | baca/tulis file, isi folder, path | baca data soal, simpan hasil |
| [07_struct_kelas_modul.md](07_struct_kelas_modul.md) | struct, class, import, pip, Jupyter | bikin tipe sendiri, pakai library |
| [08_pesan_error.md](08_pesan_error.md) | cara baca traceback + error yang paling sering | notebook merah |

## Lima hal yang paling sering dicari

| C++ | Python |
|---|---|
| `vector<int> v;  v.push_back(3);` | `v = []` lalu `v.append(3)` |
| `unordered_map<string,int> m;  m["a"] = 1;` | `m = {}` lalu `m["a"] = 1` |
| `for (int i = 0; i < n; i++)` | `for i in range(n):` |
| `vector<struct Row>` (tabel) | `pd.DataFrame` (lihat 05) |
| `cout << a << " " << b << endl;` | `print(a, b)` |
