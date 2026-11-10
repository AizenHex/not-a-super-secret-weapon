# PANDUAN METODE — metode niche: kapan dipakai, untuk data seperti apa

Tujuh notebook di `contoh/text/` ini adalah **lapis kedua**. Mereka tidak menggantikan baseline.
Pakai setelah model dasar jalan dan menulis OOF (`klasifikasi_tfidf`, `baseline_oof`,
`teks_tabular`, `finetune_oof`). Peta soal → notebook: [PANDUAN.md](PANDUAN.md). Versi cetak:
[PANDUAN_METODE.pdf](PANDUAN_METODE.pdf) (sumber `.tex`, samakan kalau file ini berubah).

## 0. Aturan main (berlaku untuk ketujuhnya)

1. **Pakai hanya kalau skor jujur naik.** Tiap notebook mencetak skor jujur (lipatan luar atau
   holdout) dan memilih cara paling sederhana yang selisihnya ≤ `MIN_GAIN` dari yang terbaik.
   Ikuti pilihan itu. Di demo, cara rumit sering menang di OOF lalu kalah di test.
2. **Semuanya untuk target kelas.** Target angka → `ensemble_oof`,
   `stacking_teks_tabular_regresi`, `interval_prediksi`.
3. **Metrik harus "makin besar makin baik".** Logloss, RMSE, MAE: tulis `return -(...)`.
4. **Hapus sel `HAPUS saat lomba`.** Paling berbahaya: sel "OOF demo" di
   `bobot_ensemble_metaheuristik` dan `stacking_fwls` berisi `OOF_FOLDER = OUTPUT_FOLDER`. Kalau
   lupa dihapus, notebook melatih model demo dan membaca OOF itu, bukan OOF milikmu.
5. Data demo kecil (emot 3,9 ribu baris, baju 17,6 ribu). Selisih < 0,005 di data sekecil itu
   masih bisa derau. Cek kemenangan penting dengan `uji_beda_model`.
6. Batas angka seperti "≥ ±5.000 baris" atau "kelas ≤ ±10" adalah perkiraan, bukan hasil uji.
   Skor jujur di notebook tetap yang menentukan.

## 1. Pilih cepat

| Keadaan data / model | Buka | Syarat minimum |
|---|---|---|
| ≥ 3 OOF, metrik berundak (F1 makro, akurasi, QWK), ingin bobot lebih baik dari rata-rata | `bobot_ensemble_metaheuristik` | file `oof_/test_` dari lipatan yang sama |
| Kualitas anggota timpang, atau model unggul di jenis baris berbeda (teks pendek vs panjang, kolom kosong) | `stacking_fwls` | ≥ 2 OOF, ada kolom angka untuk ciri baris |
| Ensemble mentok karena semua anggota TF-IDF + linear (korelasi > 0,95) | `pengklasifikasi_alternatif` | teks |
| Teks + tabel untuk LightGBM, tidak ada waktu merancang fitur | `fitur_otomatis_tetangga` | kolom angka/kategori |
| Fitur buatan banyak (> 30), curiga derau, perlu slide "fitur penting" | `seleksi_fitur_metaheuristik` | target **biner** |
| Label sedikit, teks tanpa label banyak | `co_training_teks` (+ `graf_label_teks`, `semi_supervised_teks`) | teks ≥ beberapa kata |
| Juri butuh penjelasan "JIKA … MAKA" | `rulefit_aturan` | biner (multikelas: 1 lawan sisa) |

## 2. Kartu per notebook

### `bobot_ensemble_metaheuristik` — bobot gabungan tanpa turunan

- **Bagus untuk:** klasifikasi dengan metrik berundak (F1 makro, akurasi, QWK) yang membuat
  Nelder-Mead/SLSQP macet; 4–8 model yang skornya berdekatan; train ≥ ±5.000 baris.
- **Jangan dipakai kalau:** hanya 2 model (cukup rata-rata di `ensemble_oof`); satu model jauh
  lebih unggul (bobot lari ke satu model); metrik halus seperti logloss/AUC (cukup Nelder-Mead di
  `ensemble_oof`); data < 2.000 baris (bobot cocok ke derau).
- **Pilih cara:**

  | cara | pakai kalau |
  |---|---|
  | hill (Caruana) | bawaan; paling tahan overfit karena bobot = hitungan langkah |
  | hill negatif | ada model yang salahnya bisa "dikurangkan"; butuh data besar |
  | annealing | hill macet di bobot yang sama |
  | evolusi diferensial / genetik | > 6 model, ruang bobot besar |
  | Optuna | perlu jejak percobaan; hasilnya mirip evolusi |

- **Baca:** kolom `skor jujur`, bukan `skor OOF (optimis)`. Selisih keduanya = besar overfit
  bobot.
- **Demo (emot, 5 model, 1 menit):** rata 0,675 → hill negatif jujur 0,685. Di test justru hill
  biasa terbaik (0,674 vs 0,664). **Data < 5.000 baris: naikkan `MIN_GAIN` ke 0,005.**
- **Ganti:** `MODELS`, `metric`, hapus sel OOF demo.
- **Untuk juri:** "F1 makro tidak bisa diturunkan, jadi bobot dicari dengan metaheuristik dan
  dinilai di lipatan yang tidak dipakai mencari bobot."

### `stacking_fwls` — bobot yang berubah menurut ciri baris

- **Bagus untuk:** teks + kolom angka; panjang teks sangat beragam; ada kolom teks yang sering
  kosong; anggota punya kekuatan beda (huruf, kata, judul saja, tabular). Train ≥ ±5.000 baris
  (koefisien FWLS = model × kelas × ciri).
- **Jangan dipakai kalau:** anggota mirip semua; data kecil; tidak ada ciri baris yang bermakna.
- **Baca:** `skor_per_panjang.png`. Garis model **bersilangan** antar kelompok panjang → FWLS
  layak. Garis sejajar → stacking LR biasa cukup.
- **Demo (baju rating 5 kelas, 4 model, ±1,5 menit):**

  | cara | jujur | test |
  |---|---|---|
  | model tunggal terbaik | 0,434 | 0,435 |
  | rata-rata | 0,357 | 0,359 |
  | **stacking LR** (dipilih) | 0,496 | 0,509 |
  | FWLS | 0,493 | 0,490 |
  | LightGBM + ciri | 0,483 | 0,497 |

  Rata-rata merusak karena model tabular lemah (0,159). Kalau anggota timpang, stacking apa pun
  jauh di atas rata-rata. FWLS tidak menang di sini.
- **Ganti:** `MODELS`, `TEXT_COLS`, `META_COLS` (kolom yang menggambarkan *jenis* baris, bukan
  yang paling kuat menebak target), hapus sel OOF demo.

### `pengklasifikasi_alternatif` — anggota ensemble yang salahnya beda

- **Bagus untuk:** teks saja, ensemble sudah ada tapi semua anggotanya TF-IDF + linear.
  - Teks pendek penuh typo/alay → gzip-kNN, kernel RFF.
  - Dokumen panjang mirip pencarian → BM25-kNN, Rocchio.
  - Kelas tidak seimbang → Complement NB.
- **Jangan dipakai kalau:** sudah ada fine-tune transformer yang kuat (tambahan jarang membantu);
  data > 20 ribu baris untuk gzip (lambat).
- **Baca:**
  - `+ patokan`: skor rata-rata model itu dengan patokan. Ambil yang di atas patokan.
  - `korelasi`: makin kecil makin beragam.
  - `menyelamatkan`: porsi baris yang salah di patokan tapi benar di model ini.

  Keputusan akhir diambil di `ensemble_oof` / `bobot_ensemble_metaheuristik`.
- **Demo (emot, 1,5 menit):**
  - Sendirian semuanya kalah: gzip 0,365, terbaik passive-aggressive 0,675 vs patokan 0,679.
  - Dirata-rata dengan patokan, OOF naik (RFF 0,687, Rocchio 0,684) tapi test tidak.
  - Perlakukan sebagai **calon anggota**, bukan model utama.
- **Waktu:** gzip ∝ baris × `NCD_POOL` (3,9 ribu baris ≈ 1 menit, 20 ribu ≈ 5 menit).
  Kecilkan `NCD_POOL` kalau lambat.
- **Hasil:** `oof_alt_*.npy` di `OOF_FOLDER` → langsung tulis di `MODELS` notebook ensemble.

### `fitur_otomatis_tetangga` — fitur tetangga kNN + OpenFE mini

- **Bagus untuk:** teks + tabel untuk LightGBM. Ada kategori bernilai banyak dan kolom angka yang
  bisa digabung (rasio, selisih terhadap rata-rata kelompok).
- **Jangan dipakai kalau:** teks saja; sudah memakai `stacking_teks_tabular` (sudah punya fitur
  tetangga, jadi ambil kolom otomatisnya saja).
- **Baca:** `banding_fitur.csv` (dasar / + tetangga / + otomatis / + keduanya). Untung fitur
  otomatis yang kecil itu normal.
- **Demo (baju biner, 30 detik):** AUC 0,906 → + tetangga 0,914 → + keduanya 0,916. Fitur
  otomatis terbaik: `title_huruf/title_kata` (rata-rata panjang kata judul).
- **Pakai hasilnya:** `fitur_baru_train.csv` / `fitur_baru_test.csv` (ada `ID_COL`, urutan baris
  sama dengan file asli) → `pd.concat` kolomnya ke notebook LightGBM-mu.
- **Belum diuji:** multikelas (kodenya ada).

### `seleksi_fitur_metaheuristik` — buang fitur derau

- **Bagus untuk:** > 30 fitur buatan (SVD, statistik teks, encoding, agregat); ≥ 5.000 baris;
  ingin model lebih ramping/cepat; bahan slide "fitur mana penting, buktinya sisanya derau".
- **Jangan dipakai kalau:** fitur < 20; ribuan kolom TF-IDF (terlalu lambat, pakai chi²/L1);
  target multikelas/angka (notebook ini biner saja); sisa waktu < 30 menit.
- **Pilih cara:**

  | cara | sifat |
  |---|---|
  | null importance | cepat; grafiknya bagus untuk slide |
  | Boruta | paling ketat (membuang paling banyak) |
  | permutasi | paling lunak |
  | annealing / genetik | mencari kombinasi; paling mahal, untuk 30–100 fitur |

- **Demo (baju biner, 48 fitur termasuk 10 kolom acak, 1 menit):** semua cara membuang 10 kolom
  acak. Holdout tidak naik (semua 0,9066, permutasi 0,9067) → notebook tetap "semua". LightGBM
  sudah tahan derau. Gunanya terutama penjelasan dan kecepatan.

### `co_training_teks` — semi-supervised dua sudut pandang

- **Bagus untuk:** label ratusan sampai ±2 ribu, teks tanpa label jauh lebih banyak (test besar
  atau file tambahan), kelas ≤ ±10.
- **Jangan dipakai kalau:**
  - label sudah banyak (> ±50% data): untungnya hilang;
  - teks 1–3 kata: sudut pandang kata dan huruf tidak lagi saling melengkapi;
  - test bergeser dari train (cek `geser_distribusi`): label palsu salah menumpuk.
- **Pilih di antara tiga notebook semi-supervised:**

  | notebook | cocok kalau |
  |---|---|
  | `co_training_teks` | ada dua sudut pandang (kata vs huruf, teks vs tabel) |
  | `graf_label_teks` | teks membentuk kelompok/topik jelas, test besar |
  | `semi_supervised_teks` | kelas banyak, pseudo-label N teratas per kelas |

- **Demo (emot, 15% label = 544 baris, 25 detik):**

  | cara | OOF | test |
  |---|---|---|
  | hanya label | 0,548 | 0,557 |
  | self-training | 0,493 | 0,511 |
  | **co-training** | 0,569 | 0,578 |
  | tri-training | 0,519 | 0,540 |

  Pelajaran: ambang keyakinan sederhana mudah merusak. Jumlah tetap per kelas lebih aman.
- **Ganti:** `LABEL_SHARE = 1.0` saat lomba (0,15 hanya untuk demo). Baris train bertarget
  kosong otomatis masuk kolam tak berlabel. File teks tambahan tanpa label → `UNLABELED_CSV`.

### `rulefit_aturan` — aturan "JIKA … DAN … MAKA" untuk slide

- **Bagus untuk:** presentasi ke juri non-teknis; target biner atau satu kelas penting; campuran
  kata kunci + kolom tabel.
- **Jangan dipakai kalau:** sebagai model utama (CV di bawah LightGBM); multikelas banyak kelas
  (buat satu per kelas penting saja).
- **Baca:** `aturan_teratas.png`. Biru menaikkan peluang, oranye menurunkan.
  - "TIDAK kata:x DAN TIDAK kata:y" artinya kata-kata negatif tidak muncul, biasanya ulasan
    positif.
  - Pilih 3–5 aturan dengan `porsi baris` 5–60% untuk slide. Aturan yang berlaku untuk 95% baris
    kurang bercerita.
- **Demo (baju biner, 22 detik):** AUC CV RuleFit 0,873, LightGBM 0,877, L1 0,879; 53 aturan
  terpakai dari 556; test 0,881.
- **Ganti:** `POSITIVE`, `STOP_WORDS` (`None` atau daftar stopword Indonesia), `RULE_C` (kecil
  = aturan lebih sedikit).

## 3. Alur gabungan yang disarankan

- **Teks saja, multikelas:**
  1. TF-IDF + fine-tune → OOF.
  2. `pengklasifikasi_alternatif` (cari anggota yang beda).
  3. `ensemble_oof`.
  4. Metrik F1/QWK dan ≥ 4 model → `bobot_ensemble_metaheuristik`.
  5. `uji_beda_model` (menang sungguhan atau derau?).
- **Teks + tabel, biner:**
  1. `teks_tabular` / `stacking_teks_tabular`.
  2. `fitur_otomatis_tetangga`.
  3. `seleksi_fitur_metaheuristik`.
  4. LightGBM final.
  5. `rulefit_aturan` + `penjelasan_teks` untuk slide.
- **Teks + tabel, multikelas, anggota timpang:** beberapa OOF → `stacking_fwls`.
- **Label sedikit:** `co_training_teks` dan `graf_label_teks` → OOF keduanya → `ensemble_oof`.

## 4. Waktu sempit? Urutan menurut untung di demo

| urutan | langkah | untung demo | waktu demo | kapan |
|---|---|---|---|---|
| 1 | Stacking LR saat anggota timpang (`stacking_fwls` atau `ensemble_oof`) | +0,06 F1 vs model terbaik | ±1,5 menit | selalu dicoba |
| 2 | Fitur tetangga (`fitur_otomatis_tetangga`) | +0,008 AUC | 30 detik | selalu dicoba |
| 3 | Co-training | +0,02 F1 | 25 detik | hanya kalau label sedikit |
| 4 | Bobot metaheuristik, seleksi fitur, pengklasifikasi alternatif | kecil/tidak stabil | 1–1,5 menit | sisa waktu > 1 jam |
| 5 | RuleFit | – | 22 detik | khusus bahan slide |

Semua waktu di atas dari laptop dengan data demo (4–18 ribu baris). Data 5× lebih besar ≈ 5×
lebih lama, gzip-kNN dan annealing/genetik bisa lebih dari itu.
