# Laya v3: visi, fitur, requirement

Dibuat 2026-11-06, diperluas hari yang sama atas permintaan pengguna ("siapkan tank walau lawan bawa kapak
batu"). Satu-satunya sumber tujuan, fitur, dan requirement Laya v3. Status kerja ada di
[`../_memori/status.md`](../_memori/status.md). Laya v2 (`../_laya_eda_v2/`) dibekukan dan tidak dipindahkan.

## 0. Inti

Laya v3 = **kokpit lomba data mining** dalam satu aplikasi desktop (`Laya.exe`): muat data apa pun (teks, tabel,
banyak file), pahami, rapikan, latih (AutoGluon, model teks, notebook arsenal), lalu kumpulkan submission,
figur, dan berkas presentasi. Setiap perapian keluar sebagai kode yang memberi hasil persis sama di notebook.

Tiga janji:

1. **Tidak ada jalan buntu.** Apa pun yang belum ada tombolnya bisa ditulis sebagai langkah kode (Python,
   polars, pandas, SQL) di dalam Laya; langkah itu tersimpan dan ikut terekspor.
2. **Tidak menebak diam-diam.** Laya menunjukkan apa yang rusak beserta bukti (jumlah + contoh baris) dan
   mengusulkan perbaikan; pengguna yang memutuskan. Tidak ada aturan atau kamus khusus satu dataset di kode.
3. **Tidak pernah membeku, tidak pernah jatuh.** Hitungan berat jalan di latar; latih model jalan di proses
   terpisah sehingga GPU kehabisan memori atau library rusak tidak menjatuhkan Laya.

Disiplin tank: **tank yang mogok kalah oleh kapak batu.** Fitur hanya muncul di build yang dibawa ke lomba bila
sudah lulus ujinya; yang belum lulus disembunyikan, bukan dipamerkan setengah jadi. Di hari lomba Laya **bukan
jalur wajib**: notebook arsenal tetap bisa dipakai tanpa Laya.

## 1. Kenapa v1/v2 gagal, dan gantinya

| Akar | Yang dialami pengguna | Ganti di v3 |
|---|---|---|
| Visi v2 = "menganalisis data **tabular**", dibangun dan diuji di data UKT | regex tidak ada; tak berguna di data berita; fitur khas UKT | data apa pun, teks setara tabel; fitur = alat umum + kode bebas; uji terima di dataset **holdout** (bagian 6) |
| Streamlit menjalankan ulang seluruh skrip tiap klik | pindah halaman lambat, "seperti reload Facebook dari nol" | aplikasi desktop Qt: pindah tampilan hanya mengganti layar; hasil tiap langkah dihitung sekali dan disimpan |
| pandas + analisis per nilai unik | buka app 20 s, halaman 15–155 s di CSV 229 MB | polars: train GDOC 229 MB terbaca 0,5 s (diukur 2026-11-06, file di cache OS) |
| Fitur latih v2 ditulis ulang di dalam UI, setengah jadi | banyak fitur tak lengkap | latih memakai notebook arsenal yang sudah teruji + AutoGluon, dijalankan Laya di proses terpisah |
| Lulus uji otomatis dianggap selesai | "0,1–0,3 s ... bullshit" | kecepatan diukur pengguna sendiri di `Laya.exe` dengan data nyata; fitur disembunyikan sampai lulus |

## 2. Alur hari lomba yang dilayani (5 jam, 13.00–18.00)

| Waktu | Kegiatan | Modul |
|---|---|---|
| 13.00–13.30 | buka folder → peran file dikenali → temuan (duplikat, test yang ada di train, kebocoran, label timpang) → baca contoh per label | A, B, C |
| 13.30–14.00 | rapikan + ekspor; baseline cepat + validasi adversarial + label mencurigakan; **submission pertama** | D, E, F, H |
| 14.00–17.00 | latih berat di latar (fine-tune transformer, AutoGluon, ensemble OOF) sambil memperbaiki data dan analisis kesalahan | G, F |
| 17.00–17.40 | submission final tervalidasi; figur + laporan langkah | H, I |
| 17.40–18.00 | paket kumpul (kode + dokumentasi + PPT) dengan daftar periksa | I |

## 3. Katalog fitur

Kolom: **Nilai** = dampak ke final (T tinggi, S sedang, R rendah); **Biaya** (K kecil, S sedang, B besar).
**P0** = inti, wajib lulus sebelum berangkat. **P1** = sebelum berangkat setelah P0 lulus, urut nilai/biaya; yang
belum lulus pada Minggu 8 Nov siang disembunyikan. **P2** = setelah lomba.

### A. Muat dan sambung

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| A1 | Buka folder + subfolder: csv, tsv, txt-tabel, json, jsonl, parquet, xlsx (tiap sheet = tabel). Nama = path relatif. Dibaca saat pertama dibuka | T | K | P0 | `data_latihan/multitabel/berka` dan `text/absa` tampil ≤ 1 s |
| A2 | Baca tahan banting: deteksi pemisah, encoding (UTF-8, BOM, UTF-16, cp1252), kutip + baris baru dalam sel, baris tak rata; laporan baca; atur manual (pemisah, encoding, baris header, lewati baris, semua teks); gagal → 100 baris mentah + pesan | T | S | P0 | korpus file jahat (6.3) |
| A3 | Tabel turunan: tumpuk (gabung baris per nama kolom), join, dari SQL/kode | T | K | P0 | |
| A4 | Proyek `.laya.json` (path relatif + langkah), simpan otomatis, buka terakhir, pulih setelah crash | T | K | P0 | dipindah ke PC lain → jalan setelah folder diganti |
| A5 | Kenali peran file: train / test / sample_submission / label terpisah / kamus → usul kolom id, target, jenis tugas (klasifikasi, multilabel, regresi, token/NER, pasangan teks, QA) dan metrik | T | S | P1 | 17 set `data_latihan/text` dikenali benar |
| A6 | Folder dokumen `.txt` → tabel (path, label = nama folder, teks) | S | K | P1 | |
| A7 | Excel header bertingkat, sel gabungan | S | S | P1 | |
| A8 | Data > RAM: baca lazy/streaming + mode sampel | R | S | P2 | |

### B. Lihat dan jelajah

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| B1 | Grid virtual (jutaan baris lancar) + panel isi sel penuh dengan baris barunya | T | S | P0 | GDOC 52.889 baris, sel s.d. 30 rb huruf |
| B2 | Cari teks / regex / ekspresi SQL: jumlah total dan per label, sorot di panel, "jadikan langkah" | T | K | P0 | |
| B3 | Sampel acak (seed tetap) dan N contoh per label | T | K | P0 | |
| B4 | Palet perintah Ctrl+K (cari semua perintah) + pintasan keyboard | S | K | P1 | |
| B5 | Facet ala OpenRefine: klik nilai / rentang histogram → saring; gabung beberapa facet | S | S | P1 | |
| B6 | Urut, sembunyikan, sematkan kolom | S | K | P1 | |
| B7 | Edit sel / ganti label langsung di grid → tercatat sebagai langkah berbasis id | S | S | P1 | |
| B8 | Dua tabel berdampingan (train vs test, sebelum vs sesudah) | S | S | P2 | |

### C. Pahami: profil dan temuan

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| C1 | Profil kolom: tipe, kosong / string kosong / spasi, unik, nilai terbanyak; angka: kuantil + histogram; teks: panjang huruf & kata + histogram, hitungan pola | T | S | P0 | |
| C2 | Temuan integritas (gaya deepchecks): duplikat persis/ternormalisasi + label bertentangan, duplikat parsial (baris kembar di satu kolom: kolom mana yang membedakan), penanda kosong campur, tipe campur, kolom satu nilai, karakter spesial, ejaan beda untuk nilai sama, panjang teks ekstrem, kolom mirip ID | T | S | P0 | temuan emas GDOC (6.4) |
| C3 | Temuan train vs test: teks/baris test yang ada di train (+ label train konsisten?), kategori/label baru di test, beda panjang teks, rentang tanggal | T | S | P0 | |
| C4 | Temuan teks: mojibake, boilerplate awal/akhir, list-sebagai-teks, HTML/URL/mention/emoji, huruf berulang, HURUF BESAR semua | T | S | P0 | |
| C5 | Relasi antartabel: kolom bernama sama → unik?, % cocok, jenis (1:1, banyak:1, ...) | S | S | P0 | Berka: loan → account banyak:1 |
| C6 | Asosiasi kolom ↔ label: Cramér's V, mutual information, proporsi label per kategori, korelasi | T | S | P1 | GDOC: `sentimen_kutipan`, `kategori`, `sumber` muncul |
| C7 | Panjang token dengan tokenizer lokal (`pretrained/indobert`) → saran `MAX_LENGTH` + porsi teks terpotong | T | K | P1 | |
| C8 | Drift per kolom (PSI/KS) + kata test yang tidak ada di train (OOV) | S | S | P1 | |
| C9 | Deteksi bahasa/dialek (perlu `lingua`) | R | S | P2 | |

### D. Rapikan (langkah)

Langkah inti P0 (17 jenis): regex ganti (preset + uji langsung; pola dengan backreference/lookaround otomatis
pakai mesin regex Python), regex ambil → kolom baru, rapikan teks (huruf kecil, spasi, NFKC), ubah tipe (angka
`1.234,5`, tanggal campur format, kategori), nilai → kosong, isi kosong, ganti nilai (peta dibantu daftar nilai
+ jumlah), ganti nama / buang kolom, saring baris, buang duplikat, list-sebagai-teks → list, explode/unnest,
join dengan cek kunci, tumpuk, agregasi per kunci, SQL bebas, kode Python bebas.

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| D1 | Resep: nyala/mati, ubah, hapus, geser; klik langkah = data di titik itu; ringkasan (baris, kolom ±, sel berubah, kosong baru); gagal → berhenti dengan pesan asli | T | S | P0 | |
| D2 | 17 langkah inti di atas | T | B | P0 | semua langkah × semua tipe kolom: jalan atau pesan jelas |
| D3 | Satu resep untuk train **dan** test: yang belajar dari data (median, modus, peta) dihitung dari train; buang baris hanya train | T | S | P0 | test GDOC tetap 22.665 baris |
| D4 | Kamus dari CSV: alay/slang (`data_latihan/preprocessing/kamus_alay_salsabila.csv`), stopword, stemming Sastrawi dengan cache kata unik | T | S | P1 | |
| D5 | Penyatuan nilai mirip ala OpenRefine: fingerprint, n-gram fingerprint, fonetik, Levenshtein (rapidfuzz) → langkah ganti nilai | S | S | P1 | |
| D6 | Perbaiki mojibake (`ftfy`, perlu dipasang) | S | K | P1 | `¬©` → `©` |
| D7 | Pengurai Indonesia: angka + satuan (`± 4,3 jt`, `2.000rb`, `0,712 kwintal`), tanggal dengan nama hari/bulan Indonesia, durasi (`1 tahun 1 bulan`, `blm setahun`), telepon +62, email tersamar (`at`/`dot`), alamat (RT/RW/kode pos) | S | S | P1 | `kotor_SINTETIS` cocok dengan `_kunci` ≥ 95% |
| D8 | Regex dari contoh (`grex`, perlu dipasang) | S | K | P1 | |
| D9 | Multi-label `"a,b"` ↔ kolom 0/1 | S | K | P1 | |
| D10 | Paritas Power Query: pivot/unpivot, pisah kolom, isi ke bawah, transpos, kolom bersyarat | S | S | P1 | |
| D11 | Outlier (IQR / z / winsorize), skala, encoding untuk model tabular | S | S | P1 | |
| D12 | Kolom dari contoh ala FlashFill: 2–3 pasangan masukan → keluaran, Laya mencari transformasinya | S | B | P2 | |
| D13 | Data rujukan offline: wilayah Indonesia (provinsi/kabupaten + koordinat), kode pos, hari libur → langkah perkaya data | S | S | P2 | |
| D14 | Plugin: file `.py` di folder `plugin/` jadi jenis langkah/temuan baru | R | S | P2 | |

### E. Keluarkan

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| E1 | Ekspor `.py` dan `.ipynb`: kode = persis yang dijalankan Laya, `FOLDER = ...  # UBAH` di atas, akhir opsional `.to_pandas()` | T | S | P0 | skrip di proses terpisah = hasil Laya (hash sama), semua dataset uji |
| E2 | Simpan tabel bersih ke parquet/csv | T | K | P0 | |

### T. Teks lanjutan

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| T1 | Kata/n-gram khas per label (c-TF-IDF / log-odds) + awan kata | T | K | P1 | |
| T2 | Hampir-duplikat: MinHash (`datasketch`, perlu dipasang) / embedding → klaster duplikat, dipakai sebagai grup CV | T | S | P1 | |
| T3 | Fitur meta teks sebagai kolom: panjang, jumlah URL/mention/emoji/huruf besar/tanda baca, skor leksikon (InSet, unduh sebelum berangkat) | S | K | P1 | |
| T4 | Peta embedding: e5_base/minilm/indobert lokal → UMAP 2D, warna = label / train-test / klaster; klik titik = baca teks; laso = saring/ganti label | S | B | P2 | P1 bila waktu |
| T5 | Topik (BERTopic, terpasang) per label/waktu | R | S | P2 | |
| T6 | Kata kunci per dokumen | R | K | P2 | |

### F. Model cepat untuk EDA (≤ 1 menit, proses terpisah)

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| F1 | Baseline TF-IDF kata + huruf + LogisticRegression, CV berkelompok (klaster duplikat): skor, confusion matrix, laporan per kelas, kata berbobot terbesar per kelas | T | S | P1 | GDOC ≤ 60 s |
| F2 | Label mencurigakan (cleanlab `find_label_issues` dari prob OOF F1) → tinjau → langkah ganti/buang | T | K | P1 | |
| F3 | Validasi adversarial train vs test: AUC + kolom/kata/baris pembeda | T | K | P1 | |
| F4 | Uji label acak: CV dengan label diacak harus ≈ tebakan; jauh di atas → ada kebocoran | S | K | P1 | |
| F5 | Analisis kesalahan: salah per panjang/sumber/kategori, telusuri baris salah | S | S | P1 | |
| F6 | Kurva belajar (skor vs jumlah data) | R | K | P2 | |

### G. Latih berat (proses terpisah memakai venv)

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| G1 | Pelari tugas: proses terpisah dengan Python venv, antrean (1 tugas GPU sekaligus), log langsung, progres, RAM/VRAM (`nvidia-smi`), batal; tugas crash tidak menjatuhkan Laya | T | S | P1 | fondasi F, G, T4 |
| G2 | **Komandan arsenal**: buka notebook `arsenal/`/`contoh/`, baris `# UBAH` jadi formulir, jalankan di latar (nbclient), kumpulkan skor, figur, submission | T | S | P1 | `contoh/text/baseline_oof.ipynb` jalan dari Laya |
| G3 | AutoGluon Tabular 1.6.3 (terpasang): preset, `time_limit`, kolom teks via n-gram + fitur khusus teks, papan model, kepentingan fitur. Resep untuk teks + metadata: prob OOF model teks + kolom metadata → AutoGluon | T | S | P1 | |
| G4 | Fine-tune transformer lokal (IndoBERT, IndoBERTweet, XLM-R, e5) lewat notebook arsenal (G2) | T | K | P1 | notebook sudah ada |
| G5 | Ensemble dari OOF + ambang per kelas untuk macro-F1 (notebook arsenal lewat G2) | T | K | P1 | |
| G6 | Salin label untuk baris test yang identik dengan train (pasca-proses) | T | K | P1 | cek aturan panitia; jelaskan di presentasi |
| G7 | Log percobaan: konfigurasi, CV per fold, waktu, skor LB (isi manual), catatan, bandingkan | T | K | P1 | |
| G8 | Anggaran waktu: hitung mundur ke 18.00, perkiraan lama dari uji kecil, alarm tonggak | S | K | P1 | |
| G9 | Matriks model klasik: TF-IDF kata/huruf × {LR, LinearSVC, NB-SVM, LightGBM}, papan skor CV | S | S | P1 | |
| G10 | Optuna dengan batas waktu | S | S | P2 | |
| G11 | Trik juara Kaggle: pseudo-label tanpa bocor, MLM lanjutan (TAPT) di teks train+test, layer-wise LR decay, AWP, SWA, multi-seed | S | B | P2 | |
| G12 | AutoGluon MultiModal (transformer di dalam AutoGluon) | S | B | P2 | **jangan dipasang sebelum lomba**: bisa mengubah versi transformers/torch yang sudah terbukti jalan |
| G13 | AutoGluon preset extreme (TabPFN/TabICL/Mitra) | R | B | ditolak | butuh VRAM 20–32 GB dan unduh bobot saat fit; PC lomba RTX 3070 8 GB, offline; tidak membantu teks mentah |

### H. Submission

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| H1 | Validator vs `sample_submission`: id, urutan, kolom, tipe (`3` bukan `3.0`), kelas, NaN | T | K | P1 | menolak 5 submission rusak buatan |
| H2 | Versi submission + CV/LB + jumlah prediksi yang berubah antarversi + blend | S | K | P1 | |

### I. Presentasi dan dokumentasi (metode 25% + presentasi 25% nilai)

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| I1 | Figur 16:9 huruf besar (aturan figur repo): sebaran label, panjang per label, kata khas, duplikat/tumpang tindih, confusion matrix, perbandingan model | T | S | P1 | |
| I2 | Laporan langkah markdown: langkah, alasan, angka sebelum/sesudah | T | K | P1 | |
| I3 | Paket kumpul: zip kode + notebook + dokumentasi + PPT, daftar periksa wajib (gugur bila tidak ada) | T | K | P1 | |
| I4 | Draf PPTX otomatis (python-pptx terpasang) | S | S | P2 | P1 bila waktu |

### J. Lingkungan dan hari lomba

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| J1 | Cek PC (hari setup): GPU/CUDA/driver/VRAM, RAM, disk, paket venv vs `requirements.txt`, model `pretrained/`, izin tulis, exe diblokir antivirus? | T | K | P1 | dijalankan 2026-11-10 di PC panitia |
| J2 | Contekan offline di dalam Laya: regex (beda mesin Rust vs Python), polars, pandas, scikit-learn; cari teks di markdown arsenal (`PETA.md`, `KATALOG.md`, `kamus/`) | T | K | P1 | |
| J3 | Sisipkan sel arsenal sebagai langkah kode lewat pencarian | S | S | P2 | |
| J4 | Di luar Laya: Zeal (dokumentasi offline; docset Python, NumPy, pandas, dll. dicek saat setup) | S | K | P1 | hanya bila aturan panitia membolehkan |

### V. Grafik

| # | Fitur | Nilai | Biaya | P | Selesai bila / catatan |
|---|---|---|---|---|---|
| V1 | Pembuat grafik: histogram, bar, box, scatter, garis waktu, heatmap; warna per label; ekspor PNG 16:9 | S | S | P1 | |
| V2 | Grafik otomatis per tipe kolom (gaya Lux) | R | S | P2 | |
| V3 | Matriks kosong (missingno terpasang) | R | K | P2 | |

### Sengaja tidak dibangun

- Web, database sendiri, IDE sendiri: polars/SQL + langkah kode + komandan notebook (G2) menutup kebutuhannya.
- Aturan/kamus khusus satu dataset di dalam kode.
- Asisten LLM penulis kode di dalam Laya: aturan lomba "tanpa AI". `pretrained/qwen_05b` hanya boleh sebagai model
  prediksi bila panitia membolehkan.

## 4. Requirement non-fungsional (semua P0)

| # | Requirement |
|---|---|
| N1 | **Kecepatan**, diukur pengguna dengan `Laya.exe` di laptopnya, data GDOC train 229 MB: buka aplikasi ≤ 3 s; buka tabel ≤ 5 s; pindah tab/tampilan ≤ 0,2 s; gulir tanpa patah; langkah biasa ≤ 2 s; temuan lengkap ≤ 15 s di latar |
| N2 | **Tidak membeku**: hitungan di thread latar, indikator sibuk + tombol batal, UI tetap bisa diklik |
| N3 | **Tidak jatuh**: galat jadi pesan (bahasa Indonesia + pesan asli + langkah/file penyebab), dicatat ke `laya_log.txt`; model/AutoGluon/GPU di proses terpisah |
| N4 | File data asli tidak pernah ditulis |
| N5 | Offline penuh. Modul A–E jalan tanpa venv; C7, F, G, T2, T4, T5 butuh venv (`setup.bat`) dan bilang begitu bila venv tidak ada |
| N6 | Distribusi: folder `Laya/` (PyInstaller onedir, ± 300 MB, tidak di-commit karena GitHub menolak file > 100 MB, dibawa lewat flashdisk) + cadangan `jalankan.bat` dari venv bila antivirus PC panitia memblokir `.exe` |
| N7 | Hasil sama di mana pun: urutan baris stabil, semua acak pakai seed tetap |
| N8 | Fitur yang belum lulus ujinya disembunyikan di build lomba |

## 5. Teknologi dan arsitektur

- **Kokpit** `Laya.exe` = Python 3.13 + PySide6 (Qt) + polars, dikemas PyInstaller. Alasan: pengguna minta `.exe`,
  bukan web; tabel Qt hanya menggambar baris yang terlihat; polars cepat untuk teks, punya SQL dan join dengan cek
  kunci (`validate="m:1"`).
- **Mesin berat** = venv lomba (`setup.bat`: torch cu128, transformers, AutoGluon, scikit-learn, cleanlab, UMAP,
  BERTopic). Laya menjalankannya sebagai proses terpisah (G1), jadi exe tetap kecil, dan crash/kehabisan VRAM
  tidak menjatuhkan Laya.
- **Latih memakai notebook arsenal yang sudah teruji** (G2), bukan kode latih baru di UI: satu sumber kebenaran,
  dan pengguna tetap bisa menjalankan notebook yang sama tanpa Laya.
- **Langkah = potongan kode polars yang menulis ulang `df`.** Laya menjalankan (`exec`) kode yang sama dengan yang
  diekspor. Tabel lain dirujuk sebagai `T["nama"]`. Hasil tiap langkah disimpan dengan kunci hash kode sampai
  langkah itu; mengubah langkah ke-k hanya menghitung ulang dari k.
- `mesin/` tanpa Qt (diuji tanpa layar) dan `tampilan/` Qt; satu thread pekerja untuk hitungan data.
- Library baru yang perlu dipasang (kecil, masuk `requirements.txt`): `ftfy` (D6), `datasketch` (T2), `grex` (D8).
  Tidak dipasang sebelum lomba: `autogluon.multimodal` (G12).
- Ditolak: Streamlit/Flask/web (permintaan pengguna; model rerun), Tkinter (tabel lemah untuk data besar),
  Electron/Tauri (web lagi, besar), C#/.NET (kode ekspor tidak bisa Python), DuckDB (polars sudah cukup; bisa
  ditambah nanti tanpa mengubah desain).

## 6. Uji terima (seperti menguji model: ada data holdout)

1. **Dataset pengembangan**: GDOC berita (`Z:/GEMASTIK/GDOC/data`), Berka (`data_latihan/multitabel/berka`),
   `data_latihan/preprocessing/kotor_SINTETIS.csv` (+ `_kunci` = jawaban benar), `text/multilabel/casa`,
   `text/ner/nerp` (token per baris).
2. **Dataset holdout**: dipilih pengguna dan tidak dibuka agen selama pembangunan (mis. 2 dari
   `data_latihan/text` + 1 unduhan baru). Lulus = pengguna sampai tabel bersih, model baseline, dan submission
   tervalidasi tanpa jalan buntu dan tanpa membeku. Satu jalan buntu = bug P0.
3. **Korpus file jahat** (dibuat skrip): pemisah `; , \t |`, UTF-8 BOM, UTF-16, cp1252, sel berkutip berbaris
   banyak, baris tak rata, file kosong, hanya header, nama kolom ganda/kosong, kolom kosong semua, tipe campur,
   angka `1.234,5`, tanggal campur format, sel 100 rb huruf, emoji, penanda kosong, xlsx banyak sheet, JSON
   bersarang, JSONL. Tiap file: terbaca atau pesan jelas; tidak crash.
4. **Temuan emas GDOC** (diukur agen 2026-11-06 dengan polars) harus muncul tanpa menulis kode:
   - 7.214 dari 22.665 baris test (32%) punya `konten_berita` persis sama dengan train; konten kembar di train
     tidak pernah berbeda label (0 konflik)
   - 11.806 baris train berkonten kembar (5.141 grup). Di dalam grup, artikel, tanggal, tokoh, kutipan, dan label
     selalu sama; yang beda `kategori`/`subkategori` (2.104/2.500 grup) dan `katakunci` (3.758 grup): artikel yang
     sama terambil lewat beberapa kata kunci. 5.451 baris kembar persis kecuali `id`. CV harus dikelompokkan per
     konten agar skor tidak bocor; kategori/kata kunci bisa digabung per artikel jadi fitur multi-label
   - footer `COPYRIGHT ¬© ANTARA` di 18.224 artikel (boilerplate + mojibake ©), plus ratusan mojibake lain
     (`¬∞`, `¬±`, `¬∑`)
   - `nama_tokoh`, `kutipan`, `lokasi`, dll. = list Python sebagai teks (`"['Rinto']"`); `[]` = kosong
   - awalan `<Kota> (ANTARA) - ` di 9.221 artikel
   - panjang konten median 2.751 huruf, p95 6.320, maks 30.455
   - label timpang: Positif 52%, Netral 32%, Negatif 16%
   - `sentimen_kutipan` sama dengan label di 58% baris (tebakan kelas terbanyak 52%): fitur berguna, bukan kunci
   - rentang tanggal train dan test sama (2021-01-01 s.d. 2021-06-30): split acak, bukan per waktu
5. **Kesamaan**: ekspor `.py` dijalankan di proses terpisah → hasil sama persis dengan Laya, semua dataset uji.
6. **Kecepatan N1** diukur pengguna, bukan uji otomatis.
7. **Gerbang per modul**: kolom "Selesai bila" di bagian 3. Modul yang belum lulus disembunyikan (N8).

## 7. Jadwal dan garis potong

| Kapan | Tonggak | Lulus bila |
|---|---|---|
| Jum 6 Nov malam | M1 kerangka: A1, A2, A4, B1, **exe percobaan** (cek antivirus sedini mungkin). **Selesai 6 Nov**; menunggu ukur pengguna (N1) | GDOC 229 MB terbuka dan lancar di exe |
| Sab 7 Nov pagi | M2 langkah + ekspor: D1–D3, E1–E2, A3 (tabel turunan = resep yang mulai dari `T[...]`). **Selesai 6 Nov malam** (`uji.py`: matriks 351 langkah×tipe, ekspor .py/.ipynb = app; GDOC test tetap 22.665; exe `--uji` lulus); menunggu coba pengguna | 6.3, 6.5 |
| Sab 7 Nov siang | M3 pahami: C1–C5, B2–B3. **Selesai 6 Nov malam** (`mesin/temuan.py` + tab Data/Profil/Temuan; `uji.py` `uji_emas`: temuan emas GDOC + Berka loan → account muncul tanpa kode; exe `--uji` 34 temuan, 0 cek gagal); menunggu coba pengguna | 6.4 |
| Sab 7 Nov malam | M4 G1 pelari tugas + F1–F4 + T1 + H1 | F1 GDOC ≤ 60 s |
| Min 8 Nov pagi | M5 G2 komandan arsenal + G3 AutoGluon + G6–G8 + I1–I3 + J1–J2 + sisa P1 urut nilai/biaya | gerbang tiap fitur |
| Min 8 Nov siang–sore | uji holdout oleh pengguna (sekaligus latihan memakai Laya); hanya perbaikan bug | 6.2 |
| Min 8 Nov malam | beku: exe final + cadangan ke flashdisk; fitur belum lulus disembunyikan | |
| Sel 10 Nov (setup) | J1 di PC panitia; venv via `setup.bat` | |
| Setelah 12 Nov | P2 | |

**Garis potong:** kalau M2 belum lulus Sabtu malam, Laya berhenti, final memakai notebook, dan Laya dilanjutkan
setelah lomba. Sesudah Minggu siang tidak ada fitur baru, hanya perbaikan.

## 8. Keputusan

Sudah diputuskan pengguna:

- 2026-11-06: latih model **masuk** Laya ("misalnya autogluon"), lewat proses terpisah dan notebook arsenal (G).
- 2026-11-06: katalog fitur diperluas sebanyak mungkin (bagian 3); disiplin gerbang tetap berlaku (bagian 0).
- 2026-11-06: katalog, urutan, dan garis potong disetujui ("oke, mulai kerjakan"); M1 dimulai.

Menunggu pengguna (sampai dijawab, agen memakai pilihan pertama/usulan):

2. "Multi-level" diartikan: (a) folder bertingkat, (b) list/JSON bersarang dan list-sebagai-teks seperti kolom
   GDOC, (c) header Excel bertingkat, (d) data bertingkat seperti token per kalimat (NER). Ada arti lain?
3. Kode ekspor memakai polars dengan akhir `.to_pandas()` opsional. Setuju, atau wajib pandas penuh (lebih lambat)?
4. Boleh pasang `ftfy`, `datasketch`, `grex` ke venv + `requirements.txt`?
5. Tanyakan ke panitia (bukan keputusan agen): (a) boleh memakai label train untuk baris test yang identik (G6)?
   (b) boleh model LLM lokal sebagai model prediksi? (c) boleh dokumentasi offline (Zeal, J4)?

## 8b. Celah fitur preprocessing (audit 2026-11-07)

Pembanding: alat preprocessing umum (OpenRefine, Data Wrangler, KNIME, pipeline sklearn). Sudah ada: baca
format, profil, temuan (duplikat, penanda kosong/angka, outlier, tulisan beda), 20 jenis langkah termasuk outlier,
validasi CV + baseline tabular/teks, ekspor .py/.ipynb. Yang belum, urut nilai/biaya:

**Status 2026-11-08: semua butir di bawah sudah dibuat** (langkah rentang/encode/langka/penanda_kosong/tanggal/skala/mirip, isi kosong per grup, tugas fold/bocor/asosiasi, drift + skema di Temuan train-test, Ctrl+Z/Ctrl+Y resep, banding versi resep di hasil model). Batasnya di `_memori/log.md` 2026-11-08.

| Prio | Fitur | Kenapa |
|---|---|---|
| P0 | Teks rentang → angka ("Rp 1.000.000 s.d. 2.000.000" → batas bawah/tengah/atas) | data UKT 2024 penuh kolom seperti ini; sekarang hanya jadi kategori |
| P0 | Encoding kategori: one-hot, ordinal, frekuensi, target encoding **OOF** | model linear/NN butuh angka; target encoding non-OOF = bocor |
| P0 | Kategori langka → "lainnya" (ambang n/porsi dari train) | kategori test yang tak ada di train tidak merusak model |
| P0 | Kolom penanda kosong + isi kosong per grup (median per Program Studi, dsb.) | pola kosong sering informatif |
| P0 | Fitur tanggal: tahun, bulan, hari, hari-minggu, selisih hari | belum ada langkah khusus |
| P0 | Ekspor skema validasi (kolom fold) ke notebook | CV di notebook latih = CV di Laya |
| P0 | Cek kebocoran per kolom: skor 1 kolom saja vs label (AUC/F1) | kolom yang terlalu cocok dengan label ketahuan sebelum model |
| P1 | Scaling: standar, robust, log1p (statistik dari train) | untuk model linear/KNN |
| P1 | Grafik asosiasi antar kolom (korelasi, Cramér's V) dan pola kosong bersama | membaca redundansi dan pola missing |
| P1 | Drift per kolom train vs test (PSI/KS) di tab Temuan | melengkapi adversarial |
| P1 | Undo/redo resep + bandingkan skor antar versi resep | eksperimen cepat saat lomba |
| P2 | Klaster nilai mirip (key collision/fuzzy) semi-otomatis | sebagian sudah lewat temuan "ditulis beda-beda" |
| P2 | Aturan validasi skema test (tipe, rentang, kategori) | menangkap test yang formatnya beda |

## 9. Sumber riset (2026-11-06)

- Pemeriksaan integritas dan train-test: [Deepchecks (DataCamp)](https://www.datacamp.com/hi/tutorial/automating-machine-learning-testing-using-deep-checks),
  [contoh uji train-test deepchecks](https://huggingface.co/spaces/DaCrow13/Hopcroft-Skill-Classification/blob/main/tests/deepchecks/test_train_test_validation.py)
- Jenis masalah data dan label: [cleanlab Datalab issue types](https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html)
- Penyatuan nilai mirip: [OpenRefine clustering in depth](https://openrefine.org/docs/technical-reference/clustering-in-depth)
- Langkah GUI → kode pandas, kolom dari contoh: [VS Code Data Wrangler](https://code.visualstudio.com/docs/datascience/data-wrangler),
  [repo Data Wrangler](https://github.com/microsoft/vscode-data-wrangler)
- Operator pembersih teks skala besar: [Data-Juicer operators](https://modelscope.github.io/data-juicer/en/main/docs/Operators.html)
- AutoGluon: [v1.4 (preset extreme)](https://auto.gluon.ai/dev/whats_new/v1.4.0.html), [v1.5](https://auto.gluon.ai/dev/whats_new/v1.5.0.html),
  [fitur teks Tabular](https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html)
- Trik juara NLP: [CommonLit (LLRD)](https://github.com/gaetanlop/Commonlit-Readability-Competition),
  [Google QUEST (pseudo-label, LM pretraining)](https://medium.com/kaggle-blog/the-3-ingredients-to-our-success-winners-dish-on-their-solution-to-googles-quest-q-a-labeling-c1a63014b88),
  [Feedback Prize (AWP, MLM)](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx)
- Validasi adversarial dan kebocoran: [KDnuggets](https://www.kdnuggets.com/2020/02/adversarial-validation-overview.html),
  [catatan data leakage](https://livaage.github.io/ml_resources/topics/data-leakage.html)
- Regex dari contoh: [grex di PyPI](https://pypi.org/project/grex/)
- Leksikon sentimen Indonesia: [InSet](https://github.com/fajri91/InSet)
